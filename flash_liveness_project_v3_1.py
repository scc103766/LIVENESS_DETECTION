from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OPENCV_LOG_LEVEL", "SILENT")
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models

from flash_physical_features import PhysicalCueExtractor, PhysicalFeatureConfig
from scripts.collect_flash_liveness_video import (
    COLOR_SEQUENCE_RGB as COLLECT_FLASH_COLOR_SEQUENCE_RGB,
    build_frame_color_labels,
)

if hasattr(cv2, "setLogLevel"):
    try:
        cv2.setLogLevel(0)
    except Exception:
        pass


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_UNKNOWN_COLOR = 0x808080
FLASH_COLOR_PROTOCOLS = {"neutral", "collect_flash"}
LOSS_METRIC_KEYS = (
    "cls_loss",
    "depth_loss",
    "contrast_loss",
    "fft_loss",
    "weighted_depth_loss",
    "weighted_contrast_loss",
    "weighted_fft_loss",
    "aux_loss",
)
LABEL_NAME_TO_ID = {
    "live": 1,
    "real": 1,
    "bonafide": 1,
    "bona_fide": 1,
    "genuine": 1,
    "真人": 1,
    "spoof": 0,
    "fake": 0,
    "attack": 0,
    "imposter": 0,
    "print": 0,
    "replay": 0,
    "mask": 0,
    "头模": 0,
    "攻击": 0,
}


@dataclass(frozen=True)
class LivenessSample:
    media_path: str
    txt_path: str | None
    label: int
    media_type: str = "videos"
    category: str = "unknown"
    source_group: str = "unknown"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict | list) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


class CorruptedSampleRecorder:
    def __init__(self, record_path: Path | None = None) -> None:
        self.record_path = record_path
        self._seen = set()
        self._lock = threading.Lock()
        if self.record_path is not None:
            ensure_dir(self.record_path.parent)

    def record(self, video_path: str, reason: str) -> None:
        if self.record_path is None:
            return

        key = (video_path, reason)
        with self._lock:
            if key in self._seen:
                return
            self._seen.add(key)
            with self.record_path.open("a", encoding="utf-8") as file:
                file.write(f"{video_path}\t{reason}\n")


def center_crop_and_resize(frame: np.ndarray, target_size: tuple[int, int]) -> np.ndarray:
    if frame is None or frame.size == 0:
        return np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8)

    height, width = frame.shape[:2]
    side = min(height, width)
    top = max((height - side) // 2, 0)
    left = max((width - side) // 2, 0)
    cropped = frame[top:top + side, left:left + side]
    if cropped.size == 0:
        cropped = frame
    return cv2.resize(cropped, target_size)


def configure_video_capture_orientation(cap: cv2.VideoCapture) -> None:
    """Let OpenCV honor video rotation metadata when the backend supports it."""
    orientation_auto = getattr(cv2, "CAP_PROP_ORIENTATION_AUTO", None)
    if orientation_auto is None:
        return
    try:
        cap.set(orientation_auto, 1)
    except Exception:
        pass


def yaw_perspective_warp(frame: np.ndarray, yaw_ratio: float) -> np.ndarray:
    if frame is None or frame.size == 0:
        return frame

    height, width = frame.shape[:2]
    yaw_ratio = float(np.clip(yaw_ratio, -0.20, 0.20))
    if abs(yaw_ratio) < 1e-5:
        return frame

    inset_x = abs(yaw_ratio) * width
    inset_y = abs(yaw_ratio) * height * 0.18
    src = np.float32(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ]
    )
    if yaw_ratio > 0:
        dst = np.float32(
            [
                [inset_x, inset_y],
                [width - 1, 0],
                [width - 1, height - 1],
                [inset_x, height - 1 - inset_y],
            ]
        )
    else:
        dst = np.float32(
            [
                [0, 0],
                [width - 1 - inset_x, inset_y],
                [width - 1 - inset_x, height - 1 - inset_y],
                [0, height - 1],
            ]
        )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(frame, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def maybe_apply_yaw_augmentation(
    frames_bgr: list[np.ndarray],
    enabled: bool,
    probability: float,
    max_ratio: float,
) -> list[np.ndarray]:
    if not enabled or not frames_bgr or probability <= 0.0 or max_ratio <= 0.0:
        return frames_bgr
    if random.random() >= probability:
        return frames_bgr
    yaw_ratio = random.uniform(-max_ratio, max_ratio)
    if abs(yaw_ratio) < max_ratio * 0.25:
        yaw_ratio = math.copysign(max_ratio * 0.25, yaw_ratio if yaw_ratio != 0 else 1.0)
    return [yaw_perspective_warp(frame, yaw_ratio) for frame in frames_bgr]


def stabilize_physical_features(features: np.ndarray, clip_value: float = 6.0) -> np.ndarray:
    if features.size == 0:
        return features.astype(np.float32)
    stabilized = np.nan_to_num(features.astype(np.float32, copy=False), nan=0.0, posinf=1e6, neginf=-1e6)
    stabilized = np.sign(stabilized) * np.log1p(np.abs(stabilized))
    return np.clip(stabilized, -clip_value, clip_value).astype(np.float32)


def sample_frame_indices(total_frames: int, num_frames: int) -> list[int]:
    if total_frames <= 0:
        return [0] * num_frames
    if total_frames <= num_frames:
        return np.linspace(0, total_frames - 1, num_frames, dtype=int).tolist()

    start_idx = max((total_frames - num_frames) // 2, 0)
    return list(range(start_idx, start_idx + num_frames))


def infer_label_from_dir_name(name: str) -> int | None:
    normalized = name.strip().lower().replace("-", "_").replace(" ", "_")
    return LABEL_NAME_TO_ID.get(normalized)


def _find_optional_txt(media_path: Path) -> str | None:
    txt_path = media_path.with_suffix(".txt")
    if txt_path.exists():
        return str(txt_path)
    try:
        resolved_txt_path = media_path.resolve(strict=True).with_suffix(".txt")
    except OSError:
        return None
    return str(resolved_txt_path) if resolved_txt_path.exists() else None


def collect_samples_from_label_dirs(root_dir: Path) -> list[LivenessSample]:
    samples: list[LivenessSample] = []
    if not root_dir.exists():
        return samples

    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        label = infer_label_from_dir_name(child.name)
        if label is None:
            continue
        for media_path in sorted(child.rglob("*")):
            suffix = media_path.suffix.lower()
            if not media_path.is_file() or suffix not in VIDEO_EXTENSIONS | IMAGE_EXTENSIONS:
                continue
            media_type = "videos" if suffix in VIDEO_EXTENSIONS else "images"
            samples.append(
                LivenessSample(
                    media_path=str(media_path),
                    txt_path=_find_optional_txt(media_path),
                    label=label,
                    media_type=media_type,
                    category=child.name,
                    source_group="label_dir",
                )
            )
    return samples


def collect_samples_from_manifest(
    root_dir: Path,
    media_filter: str = "videos",
    require_color_txt: bool = False,
) -> list[LivenessSample]:
    manifest_path = root_dir / "manifest.tsv"
    if not manifest_path.exists():
        return []

    samples: list[LivenessSample] = []
    allowed_media = {"videos", "images"} if media_filter == "all" else {media_filter}
    with manifest_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file, delimiter="\t")
        for row in reader:
            media_type = (row.get("media_type") or "").strip()
            if media_type not in allowed_media:
                continue
            label_name = (row.get("label") or "").strip().lower()
            if label_name not in {"live", "spoof"}:
                continue
            media_path = Path(row.get("archive_path") or "")
            if not media_path.exists():
                continue
            suffix = media_path.suffix.lower()
            if media_type == "videos" and suffix not in VIDEO_EXTENSIONS:
                continue
            if media_type == "images" and suffix not in IMAGE_EXTENSIONS:
                continue
            txt_path = _find_optional_txt(media_path)
            if require_color_txt and media_type == "videos" and txt_path is None:
                continue
            samples.append(
                LivenessSample(
                    media_path=str(media_path),
                    txt_path=txt_path,
                    label=1 if label_name == "live" else 0,
                    media_type=media_type,
                    category=(row.get("category") or "unknown").strip() or "unknown",
                    source_group=(row.get("source_group") or "unknown").strip() or "unknown",
                )
            )
    return samples


def stratified_split(
    samples: list[LivenessSample],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[LivenessSample]]:
    rng = random.Random(seed)
    grouped: dict[tuple[int, str, str], list[LivenessSample]] = {}
    for sample in samples:
        grouped.setdefault((sample.label, sample.media_type, sample.category), []).append(sample)

    for group in grouped.values():
        rng.shuffle(group)

    split_result = {"train": [], "val": [], "test": []}
    for group in grouped.values():
        count = len(group)
        test_count = int(round(count * test_ratio))
        val_count = int(round(count * val_ratio))

        if count >= 3 and test_count == 0 and test_ratio > 0:
            test_count = 1
        if count - test_count >= 2 and val_count == 0 and val_ratio > 0:
            val_count = 1
        if test_count + val_count >= count:
            overflow = test_count + val_count - count + 1
            if test_count >= overflow:
                test_count -= overflow
            else:
                val_count = max(val_count - (overflow - test_count), 0)
                test_count = 0

        split_result["test"].extend(group[:test_count])
        split_result["val"].extend(group[test_count:test_count + val_count])
        split_result["train"].extend(group[test_count + val_count:])

    for split_name in split_result:
        split_result[split_name] = sorted(split_result[split_name], key=lambda item: item.media_path)
    return split_result


def discover_dataset_splits(
    data_root: str,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
    media_filter: str = "videos",
    require_color_txt: bool = False,
) -> dict[str, list[LivenessSample]]:
    root = Path(data_root)
    split_dirs = {name: root / name for name in ("train", "val", "test")}
    existing_split_dirs = {name: path for name, path in split_dirs.items() if path.exists()}

    if existing_split_dirs:
        datasets = {name: collect_samples_from_label_dirs(path) for name, path in existing_split_dirs.items()}
        if "train" not in datasets or not datasets["train"]:
            raise ValueError("使用分目录数据集时，至少需要提供 data_root/train/live|spoof 这样的结构。")

        train_samples = datasets["train"]
        missing_val = "val" not in datasets or not datasets["val"]
        missing_test = "test" not in datasets or not datasets["test"]
        if missing_val or missing_test:
            ratio_val = val_ratio if missing_val else 0.0
            ratio_test = test_ratio if missing_test else 0.0
            resplit = stratified_split(train_samples, val_ratio=ratio_val, test_ratio=ratio_test, seed=seed)
            datasets["train"] = resplit["train"]
            if missing_val:
                datasets["val"] = resplit["val"]
            if missing_test:
                datasets["test"] = resplit["test"]
        return datasets

    manifest_samples = collect_samples_from_manifest(
        root,
        media_filter=media_filter,
        require_color_txt=require_color_txt,
    )
    if manifest_samples:
        return stratified_split(manifest_samples, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)

    all_samples = collect_samples_from_label_dirs(root)
    if not all_samples:
        raise ValueError(
            "未找到可用样本。请使用 manifest.tsv 归档，或 data_root/live|spoof，或 data_root/train/live|spoof 的目录结构。"
        )
    return stratified_split(all_samples, val_ratio=val_ratio, test_ratio=test_ratio, seed=seed)


class FacePreprocessor:
    def __init__(
        self,
        detector_model_path: str | None = None,
        detector_device: str | None = None,
        target_size: tuple[int, int] = (224, 224),
        conf_threshold: float = 0.5,
        iou_threshold: float = 0.5,
    ) -> None:
        self.target_size = target_size
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.detector = None
        self.detector_load_error = None

        if detector_model_path:
            detector_device = detector_device or ("cuda:0" if torch.cuda.is_available() else "cpu")
            try:
                from Face_detection_yolo_align import YOLOv7_face_mkl

                self.detector = YOLOv7_face_mkl(
                    model_path=detector_model_path,
                    device=detector_device,
                )
            except Exception as exc:
                self.detector_load_error = str(exc)

    def _fallback_face(self, frame: np.ndarray) -> np.ndarray:
        return center_crop_and_resize(frame, self.target_size)

    def _select_best_face(self, frame: np.ndarray, result: dict) -> np.ndarray:
        if not result["bbox"]:
            return self._fallback_face(frame)

        best_index = int(np.argmax(result["confidence"]))
        aligned_faces = result.get("aligned_faces", [])
        if best_index < len(aligned_faces):
            aligned = aligned_faces[best_index]
            if aligned is not None and aligned.size > 0 and aligned.any():
                return cv2.resize(aligned, self.target_size)

        x1, y1, x2, y2 = map(int, result["bbox"][best_index])
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0:
            return self._fallback_face(frame)
        return cv2.resize(face_crop, self.target_size)

    def preprocess_frames(self, frames_bgr: list[np.ndarray], prefix: str = "frame") -> list[np.ndarray]:
        if not frames_bgr:
            return []

        if self.detector is None:
            return [self._fallback_face(frame) for frame in frames_bgr]

        try:
            names = [f"{prefix}_{idx}" for idx in range(len(frames_bgr))]
            results = self.detector.infer_batch(
                frames_bgr,
                names,
                conf_threshold=self.conf_threshold,
                iou_threshold=self.iou_threshold,
            )
            return [self._select_best_face(frame, result) for frame, result in zip(frames_bgr, results)]
        except Exception:
            return [self._fallback_face(frame) for frame in frames_bgr]


def parse_color_txt(txt_path: str | None) -> dict[int, int]:
    mapping: dict[int, int] = {}
    if not txt_path:
        return mapping
    path = Path(txt_path)
    if not path.exists():
        return mapping
    with path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            if "," in line:
                frame_str, color_str = line.split(",", 1)
                mapping[int(frame_str)] = int(color_str)
            else:
                # 兼容 tg_export 中只有颜色值的短格式
                mapping[len(mapping)] = int(line)
    return mapping


def color_int_to_feature(color_value: int, prev_color_value: int | None) -> np.ndarray:
    color_value = int(color_value)
    r = ((color_value >> 16) & 0xFF) / 255.0
    g = ((color_value >> 8) & 0xFF) / 255.0
    b = (color_value & 0xFF) / 255.0
    transition = 0.0 if prev_color_value is None or prev_color_value == color_value else 1.0
    return np.asarray([r, g, b, transition], dtype=np.float32)


class FlashLivenessDataset(Dataset):
    def __init__(
        self,
        samples: list[LivenessSample],
        transform: bool = False,
        preprocessor: FacePreprocessor | None = None,
        target_size: tuple[int, int] = (224, 224),
        corrupted_sample_recorder: CorruptedSampleRecorder | None = None,
        max_frames: int = 0,
        frame_stride: int = 1,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.preprocessor = preprocessor or FacePreprocessor(target_size=target_size)
        self.target_size = target_size
        self.corrupted_sample_recorder = corrupted_sample_recorder
        self.max_frames = max(int(max_frames), 0)
        self.frame_stride = max(int(frame_stride), 1)

    def __len__(self) -> int:
        return len(self.samples)

    def _read_all_frames_with_color(self, video_path: str, txt_path: str) -> tuple[list[np.ndarray], list[np.ndarray]]:
        cap = cv2.VideoCapture(video_path)
        color_map = parse_color_txt(txt_path)
        frames: list[np.ndarray] = []
        color_features: list[np.ndarray] = []
        current_frame_idx = 0
        prev_color_value: int | None = None

        while True:
            success, frame = cap.read()
            if not success:
                break

            should_keep_frame = (current_frame_idx % self.frame_stride) == 0
            if should_keep_frame and frame is not None and frame.size > 0:
                color_value = color_map.get(
                    current_frame_idx,
                    prev_color_value if prev_color_value is not None else DEFAULT_UNKNOWN_COLOR,
                )
                frames.append(frame.copy())
                color_features.append(color_int_to_feature(color_value, prev_color_value))
                prev_color_value = color_value
                if self.max_frames > 0 and len(frames) >= self.max_frames:
                    break

            current_frame_idx += 1

        cap.release()
        if not frames:
            if self.corrupted_sample_recorder is not None:
                self.corrupted_sample_recorder.record(video_path, "no_valid_frames_decoded_v2_sequential_read")
            return [], []
        return frames, color_features

    def process_video(self, video_path: str, txt_path: str) -> tuple[torch.Tensor, torch.Tensor]:
        frames_bgr, color_features = self._read_all_frames_with_color(video_path, txt_path)
        if not frames_bgr:
            return torch.empty(0), torch.empty(0)
        prefix = Path(video_path).stem
        processed_frames = self.preprocessor.preprocess_frames(frames_bgr, prefix=prefix)

        frames_rgb = []
        for frame in processed_frames:
            if frame.shape[:2] != (self.target_size[1], self.target_size[0]):
                frame = cv2.resize(frame, self.target_size)
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frames = np.asarray(frames_rgb, dtype=np.float32) / 255.0
        diff_frames = np.zeros_like(frames)
        if len(frames) > 1:
            diff_frames[1:] = frames[1:] - frames[:-1]
            diff_frames[0] = diff_frames[1]

        multi_modal_frames = np.concatenate([frames, diff_frames], axis=-1)
        tensor_frames = torch.from_numpy(multi_modal_frames).permute(0, 3, 1, 2).float()
        tensor_colors = torch.from_numpy(np.asarray(color_features, dtype=np.float32))
        return tensor_frames, tensor_colors

    def __getitem__(self, idx: int):
        video_path, txt_path, label = self.samples[idx]
        tensor_frames, tensor_colors = self.process_video(video_path, txt_path)
        if tensor_frames.numel() == 0:
            return None

        return tensor_frames, tensor_colors, torch.tensor(label, dtype=torch.float32)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self._build_pe(max_len)

    def _build_pe(self, max_len: int) -> None:
        pe = torch.zeros(max_len, self.d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * (-math.log(10000.0) / self.d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if "pe" in self._buffers:
            self._buffers["pe"] = pe.unsqueeze(0)
        else:
            self.register_buffer("pe", pe.unsqueeze(0))
        self.max_len = max_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            self._build_pe(seq_len)
            self.pe = self.pe.to(device=x.device, dtype=x.dtype)
        return x + self.pe[:, :seq_len, :]


class CNNTransformerLiveness(nn.Module):
    def __init__(
        self,
        num_frames: int = 16,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        use_imagenet_pretrained: bool = False,
        imagenet_pretrained_path: str | None = None,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim = embed_dim

        resnet = models.resnet18(weights=None)
        if use_imagenet_pretrained:
            load_error = None
            if imagenet_pretrained_path:
                try:
                    state_dict = torch.load(imagenet_pretrained_path, map_location="cpu")
                    resnet.load_state_dict(state_dict)
                    print(f"已从本地加载 ResNet18 预训练权重: {imagenet_pretrained_path}")
                except Exception as exc:
                    load_error = exc
            else:
                try:
                    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                    print("已通过 torchvision 加载 ResNet18 预训练权重。")
                except Exception as exc:
                    load_error = exc

            if load_error is not None:
                print(f"加载 ResNet18 预训练权重失败，回退到随机初始化。原因: {load_error}")

        original_conv1 = resnet.conv1
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_backbone[0] = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)  #直接替换为6通道输入

        with torch.no_grad():
            self.cnn_backbone[0].weight[:, :3, :, :] = original_conv1.weight
            self.cnn_backbone[0].weight[:, 3:, :, :] = original_conv1.weight

        for param in self.cnn_backbone[0:5].parameters():
            param.requires_grad = False

        # 将逐帧光色和光色切换标记映射到时序特征中，
        # 帮助模型区分光色切换造成的瞬态响应与人脸本身的 rPPG 波动。
        self.color_proj = nn.Sequential(
            nn.Linear(4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self.pos_encoder = PositionalEncoding(d_model=embed_dim, max_len=512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=0.3,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, color_features: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = x.size()
        x_cnn = x.view(batch_size * num_frames, channels, height, width)
        features = self.cnn_backbone(x_cnn)
        features = features.view(batch_size, num_frames, self.embed_dim)
        color_emb = self.color_proj(color_features)
        features = self.pos_encoder(features + color_emb)
        trans_out = self.transformer(features, src_key_padding_mask=padding_mask)

        if padding_mask is not None:
            valid_mask = (~padding_mask).unsqueeze(-1).float()
            pooled_out = (trans_out * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled_out = trans_out.mean(dim=1)
        logits = self.fc(pooled_out)
        return logits.squeeze(-1)


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_binary_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    labels = labels.astype(np.int32)
    preds = (probs >= threshold).astype(np.int32)

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())

    apcer = safe_divide(fp, int((labels == 0).sum()))
    bpcer = safe_divide(fn, int((labels == 1).sum()))
    acer = (apcer + bpcer) / 2.0
    accuracy = safe_divide(tp + tn, len(labels))

    return {
        "accuracy": accuracy,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def compute_roc_curve(labels: np.ndarray, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    labels = labels.astype(np.int32)
    probs = probs.astype(np.float64)
    order = np.argsort(-probs)
    labels_sorted = labels[order]
    probs_sorted = probs[order]

    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    if positives == 0 or negatives == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0]), np.array([1.0, 0.0])

    tpr = [0.0]
    fpr = [0.0]
    thresholds = [1.0]
    tp = 0
    fp = 0

    for index, label in enumerate(labels_sorted):
        if label == 1:
            tp += 1
        else:
            fp += 1

        if index == len(labels_sorted) - 1 or probs_sorted[index] != probs_sorted[index + 1]:
            tpr.append(tp / positives)
            fpr.append(fp / negatives)
            thresholds.append(probs_sorted[index])

    tpr.append(1.0)
    fpr.append(1.0)
    thresholds.append(0.0)
    return np.asarray(fpr), np.asarray(tpr), np.asarray(thresholds)


def compute_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    fpr, tpr, _ = compute_roc_curve(labels, probs)
    return float(np.trapz(tpr, fpr))


def compute_eer(labels: np.ndarray, probs: np.ndarray) -> tuple[float, float]:
    fpr, tpr, thresholds = compute_roc_curve(labels, probs)
    fnr = 1.0 - tpr
    index = int(np.argmin(np.abs(fnr - fpr)))
    eer = float((fnr[index] + fpr[index]) / 2.0)
    threshold = float(thresholds[index])
    return eer, threshold


def find_best_threshold(labels: np.ndarray, probs: np.ndarray) -> float:
    candidates = sorted(set(np.round(probs.astype(np.float64), 6).tolist() + [0.5]))
    best_threshold = 0.5
    best_score = -1.0

    for threshold in candidates:
        metrics = compute_binary_metrics(labels, probs, threshold)
        balanced_accuracy = 1.0 - metrics["acer"]
        if balanced_accuracy > best_score:
            best_score = balanced_accuracy
            best_threshold = float(threshold)
    return best_threshold


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float | None = None,
    window_size: int = 256,
    window_stride: int = 128,
    window_fusion: str = "quality_lower_trimmed_mean",
    window_trim_ratio: float = 0.2,
    window_min_quality: float = 0.05,
    use_amp: bool = False,
    non_blocking: bool = False,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader, start=1):
            if batch is None:
                continue
            batch_videos, batch_colors, batch_padding_mask, batch_labels = batch
            batch_videos = batch_videos.to(device, non_blocking=non_blocking)
            batch_colors = batch_colors.to(device, non_blocking=non_blocking)
            batch_padding_mask = batch_padding_mask.to(device, non_blocking=non_blocking)
            batch_labels = batch_labels.to(device, non_blocking=non_blocking)
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(batch_videos, batch_colors, batch_padding_mask)
                loss = criterion(outputs, batch_labels)
            probs = torch.sigmoid(outputs)

            total_loss += loss.item()
            all_labels.append(batch_labels.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    labels = np.concatenate(all_labels) if all_labels else np.asarray([])
    probs = np.concatenate(all_probs) if all_probs else np.asarray([])

    if labels.size == 0:
        return {"loss": 0.0, "accuracy": 0.0, "auc": 0.0, "eer": 1.0, "threshold": 0.5}

    threshold = threshold if threshold is not None else find_best_threshold(labels, probs)
    metrics = compute_binary_metrics(labels, probs, threshold)
    auc = compute_auc(labels, probs)
    eer, eer_threshold = compute_eer(labels, probs)
    metrics.update(
        {
            "loss": total_loss / max(len(data_loader), 1),
            "auc": auc,
            "eer": eer,
            "threshold": threshold,
            "eer_threshold": eer_threshold,
        }
    )
    return metrics


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    log_interval: int = 20,
    log_seconds: float = 60.0,
    window_size: int = 256,
    window_stride: int = 128,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
    non_blocking: bool = False,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    processed_items = 0
    valid_batches = 0
    dataset_size = len(getattr(data_loader, "dataset", []))
    last_log_time = time.time()

    for batch_idx, batch in enumerate(data_loader, start=1):
        if batch is None:
            continue
        batch_videos, batch_colors, batch_padding_mask, batch_labels = batch
        batch_videos = batch_videos.to(device, non_blocking=non_blocking)
        batch_colors = batch_colors.to(device, non_blocking=non_blocking)
        batch_padding_mask = batch_padding_mask.to(device, non_blocking=non_blocking)
        batch_labels = batch_labels.to(device, non_blocking=non_blocking)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(batch_videos, batch_colors, batch_padding_mask)
            loss = criterion(outputs, batch_labels)
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        valid_batches += 1
        total_loss += loss.item()
        probs = torch.sigmoid(outputs)
        all_labels.append(batch_labels.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())
        processed_items += int(batch_labels.size(0))

        now = time.time()
        should_log_by_interval = log_interval > 0 and batch_idx % log_interval == 0
        should_log_by_time = log_seconds > 0 and now - last_log_time >= log_seconds
        if should_log_by_interval or should_log_by_time:
            avg_loss = total_loss / max(valid_batches, 1)
            print(
                f"Epoch [{epoch}/{total_epochs}] "
                f"Batch [{batch_idx}/{len(data_loader)}] "
                f"processed_items={processed_items}/{dataset_size} "
                f"batch_loss={loss.item():.4f} avg_loss={avg_loss:.4f}",
                flush=True,
            )
            last_log_time = now

    labels = np.concatenate(all_labels) if all_labels else np.asarray([])
    probs = np.concatenate(all_probs) if all_probs else np.asarray([])
    threshold = find_best_threshold(labels, probs) if labels.size else 0.5
    metrics = compute_binary_metrics(labels, probs, threshold) if labels.size else {"accuracy": 0.0}
    metrics["loss"] = total_loss / max(valid_batches, 1)
    metrics["threshold"] = threshold
    metrics["processed_items"] = processed_items
    return metrics


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    args: argparse.Namespace,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    train_metrics: dict[str, float],
    pos_weight_value: float,
    split_counts: dict[str, dict[str, int]],
    best_val_auc: float,
    checkpoint_name: str,
) -> Path:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "threshold": val_metrics["threshold"],
        "config": {
            "num_frames": args.num_frames,
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "target_size": [args.image_size, args.image_size],
        },
        "resolved_pos_weight": pos_weight_value,
        "best_val_auc": best_val_auc,
        "split_counts": split_counts,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    checkpoint_path = output_dir / checkpoint_name
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def save_summary_json(output_dir: Path, payload: dict) -> None:
    summary_path = output_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def print_split_stats(split_name: str, samples: list[LivenessSample]) -> None:
    live_count = sum(1 for sample in samples if int(sample.label) == 1)
    spoof_count = sum(1 for sample in samples if int(sample.label) == 0)
    categories: dict[str, int] = {}
    for sample in samples:
        categories[sample.category] = categories.get(sample.category, 0) + 1
    preview = ", ".join(f"{name}:{count}" for name, count in sorted(categories.items())[:8])
    suffix = f" | categories={preview}" if preview else ""
    print(f"{split_name}: total={len(samples)}, live={live_count}, spoof={spoof_count}{suffix}")


def summarize_split_counts(samples: list[LivenessSample]) -> dict:
    live_count = sum(1 for sample in samples if int(sample.label) == 1)
    spoof_count = sum(1 for sample in samples if int(sample.label) == 0)
    by_category: dict[str, int] = {}
    by_media_type: dict[str, int] = {}
    for sample in samples:
        by_category[sample.category] = by_category.get(sample.category, 0) + 1
        by_media_type[sample.media_type] = by_media_type.get(sample.media_type, 0) + 1
    return {
        "total": len(samples),
        "live": live_count,
        "spoof": spoof_count,
        "by_media_type": dict(sorted(by_media_type.items())),
        "by_category": dict(sorted(by_category.items())),
    }


def summarize_category_coverage(splits: dict[str, list[LivenessSample]]) -> dict:
    all_categories = sorted({sample.category for samples in splits.values() for sample in samples})
    total_by_category: dict[str, int] = {}
    for samples in splits.values():
        for sample in samples:
            total_by_category[sample.category] = total_by_category.get(sample.category, 0) + 1

    coverage: dict[str, dict] = {
        "all_categories": all_categories,
        "total_category_count": len(all_categories),
        "splits": {},
        "insufficient_for_train_val_test": {
            category: count
            for category, count in sorted(total_by_category.items())
            if count < 3
        },
    }
    for split_name in ("train", "val", "test"):
        split_categories = sorted({sample.category for sample in splits.get(split_name, [])})
        missing = [category for category in all_categories if category not in split_categories]
        coverage["splits"][split_name] = {
            "category_count": len(split_categories),
            "missing_categories": missing,
        }
    return coverage


def print_category_coverage(coverage: dict) -> None:
    total = coverage["total_category_count"]
    for split_name in ("train", "val", "test"):
        split_info = coverage["splits"][split_name]
        missing = split_info["missing_categories"]
        print(
            f"{split_name} category coverage: "
            f"{split_info['category_count']}/{total}, missing={missing if missing else 'none'}"
        )

    insufficient = coverage["insufficient_for_train_val_test"]
    if insufficient:
        print(
            "注意: 以下 category 样本数少于 3，无法在不重复样本/不泄漏的前提下同时覆盖 train/val/test: "
            f"{insufficient}"
        )


def resolve_pos_weight(train_samples: list[LivenessSample], pos_weight_arg: str) -> float:
    if pos_weight_arg != "auto":
        return float(pos_weight_arg)

    live_count = sum(1 for sample in train_samples if int(sample.label) == 1)
    spoof_count = sum(1 for sample in train_samples if int(sample.label) == 0)
    if live_count == 0:
        return 1.0
    return max(spoof_count / live_count, 1.0)


def build_class_balanced_sampler(samples: list[LivenessSample], seed: int) -> WeightedRandomSampler:
    label_counts: dict[int, int] = {}
    for sample in samples:
        label = int(sample.label)
        label_counts[label] = label_counts.get(label, 0) + 1
    weights = [1.0 / max(label_counts.get(int(sample.label), 1), 1) for sample in samples]
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(samples),
        replacement=True,
        generator=generator,
    )


def save_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    split_counts: dict[str, dict[str, int]],
    pos_weight_value: float,
    category_coverage: dict | None = None,
) -> None:
    args_dict = vars(args).copy()
    args_dict.pop("func", None)
    args_dict["resolved_pos_weight"] = pos_weight_value
    payload = {
        "command": " ".join(str(arg) for arg in os.sys.argv),
        "arguments": args_dict,
        "split_counts": split_counts,
        "category_coverage": category_coverage,
    }
    save_json(output_dir / "run_config.json", payload)


def append_epoch_metrics(log_path: Path, epoch_payload: dict) -> None:
    with log_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(epoch_payload, ensure_ascii=False) + "\n")


def write_metrics_csv(csv_path: Path, rows: list[dict]) -> None:
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def collate_skip_none(batch):
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        return None

    max_len = max(item[0].shape[0] for item in valid_batch)
    channels = valid_batch[0][0].shape[1]
    height = valid_batch[0][0].shape[2]
    width = valid_batch[0][0].shape[3]
    color_dim = valid_batch[0][1].shape[1]

    videos = torch.zeros((len(valid_batch), max_len, channels, height, width), dtype=torch.float32)
    colors = torch.zeros((len(valid_batch), max_len, color_dim), dtype=torch.float32)
    padding_mask = torch.ones((len(valid_batch), max_len), dtype=torch.bool)
    labels = torch.stack([item[2] for item in valid_batch], dim=0)

    for idx, (video_tensor, color_tensor, _) in enumerate(valid_batch):
        seq_len = video_tensor.shape[0]
        videos[idx, :seq_len] = video_tensor
        colors[idx, :seq_len] = color_tensor
        padding_mask[idx, :seq_len] = False

    return videos, colors, padding_mask, labels


def load_history_jsonl(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []

    rows = []
    with log_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_corrupted_records(record_path: Path) -> list[tuple[str, str]]:
    if not record_path.exists():
        return []

    records = []
    with record_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) == 2:
                records.append((parts[0], parts[1]))
            else:
                records.append((parts[0], "unknown"))
    return records


def update_dataset_readme_with_corrupted_records(data_root: str, record_path: Path) -> None:
    readme_path = Path(data_root) / "README.md"
    if not readme_path.exists():
        return

    records = load_corrupted_records(record_path)
    start_marker = "<!-- corrupted-records:start -->"
    end_marker = "<!-- corrupted-records:end -->"

    if records:
        unique_records = sorted(set(records))
        section_lines = [
            "## 训练时发现的坏视频/跳过记录",
            "",
            "以下样本在训练解码阶段无法读出任何有效帧，因此会被自动跳过：",
            "",
            start_marker,
        ]
        for video_path, reason in unique_records:
            section_lines.append(f"- `{video_path}`: `{reason}`")
        section_lines.extend([end_marker, ""])
    else:
        section_lines = [
            "## 训练时发现的坏视频/跳过记录",
            "",
            "当前还没有记录到被自动跳过的坏视频样本。",
            "",
            start_marker,
            end_marker,
            "",
        ]

    content = readme_path.read_text(encoding="utf-8")
    if start_marker in content and end_marker in content:
        prefix = content.split(start_marker)[0].rstrip()
        suffix = content.split(end_marker, 1)[1].lstrip()
        new_content = prefix + "\n\n" + "\n".join(section_lines) + suffix
    else:
        new_content = content.rstrip() + "\n\n" + "\n".join(section_lines)
    readme_path.write_text(new_content, encoding="utf-8")


def maybe_resume_training(
    args: argparse.Namespace,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    history_jsonl_path: Path,
) -> tuple[int, float, list[dict], dict | None]:
    if not args.resume_checkpoint:
        return 1, -1.0, [], None

    checkpoint = torch.load(args.resume_checkpoint, map_location=device)
    load_checkpoint_state_dict(model, checkpoint["model_state_dict"])

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state:
        optimizer.load_state_dict(optimizer_state)

    start_epoch = int(checkpoint.get("epoch", 0)) + 1
    best_val_auc = float(checkpoint.get("best_val_auc", checkpoint.get("val_metrics", {}).get("auc", -1.0)))
    existing_history = load_history_jsonl(history_jsonl_path)
    best_payload = None
    if (Path(args.output_dir) / "training_summary.json").exists():
        with (Path(args.output_dir) / "training_summary.json").open("r", encoding="utf-8") as file:
            best_payload = json.load(file)

    print(f"Resumed from checkpoint: {args.resume_checkpoint}")
    print(f"Resume start epoch: {start_epoch}")
    print(f"Resume best val auc: {best_val_auc:.6f}")
    return start_epoch, best_val_auc, existing_history, best_payload


def build_model_from_args(args: argparse.Namespace) -> CNNTransformerLiveness:
    return CNNTransformerLiveness(
        num_frames=args.num_frames,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        use_imagenet_pretrained=args.use_imagenet_pretrained,
        imagenet_pretrained_path=args.imagenet_pretrained_path,
    )


def train_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    if args.cudnn_benchmark and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    splits = discover_dataset_splits(
        args.data_root,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        media_filter=args.dataset_media,
        require_color_txt=args.require_color_txt,
    )
    split_counts = {}
    for split_name in ("train", "val", "test"):
        print_split_stats(split_name, splits.get(split_name, []))
        split_counts[split_name] = summarize_split_counts(splits.get(split_name, []))
    category_coverage = summarize_category_coverage(splits)
    print_category_coverage(category_coverage)

    preprocessor = FacePreprocessor(
        detector_model_path=args.detector_model,
        detector_device=args.detector_device,
        target_size=(args.image_size, args.image_size),
        conf_threshold=args.detector_conf,
        iou_threshold=args.detector_iou,
    )
    if preprocessor.detector is None and args.detector_model:
        print(f"检测模型加载失败，已回退到中心裁剪。原因: {preprocessor.detector_load_error}")

    corrupted_record_path = output_dir / "skipped_corrupted_samples.txt"
    if corrupted_record_path.exists() and not args.resume_checkpoint:
        corrupted_record_path.unlink()
    corrupted_sample_recorder = CorruptedSampleRecorder(corrupted_record_path)

    train_dataset = FlashLivenessDataset(
        splits["train"],
        transform=True,
        preprocessor=preprocessor,
        target_size=(args.image_size, args.image_size),
        corrupted_sample_recorder=corrupted_sample_recorder,
        max_frames=args.max_train_frames,
        frame_stride=args.frame_stride,
        use_physical_features=args.use_physical_features,
        missing_color_protocol=args.missing_color_protocol,
        flash_warmup_seconds=args.flash_warmup_seconds,
        flash_hold_seconds=args.flash_hold_seconds,
        flash_restore_seconds=args.flash_restore_seconds,
        flash_tail_seconds=args.flash_tail_seconds,
        yaw_augment_prob=args.yaw_augment_prob,
        yaw_augment_max_ratio=args.yaw_augment_max_ratio,
    )
    val_dataset = FlashLivenessDataset(
        splits["val"],
        transform=False,
        preprocessor=preprocessor,
        target_size=(args.image_size, args.image_size),
        corrupted_sample_recorder=corrupted_sample_recorder,
        max_frames=args.max_eval_frames,
        frame_stride=args.frame_stride,
        use_physical_features=args.use_physical_features,
        missing_color_protocol=args.missing_color_protocol,
        flash_warmup_seconds=args.flash_warmup_seconds,
        flash_hold_seconds=args.flash_hold_seconds,
        flash_restore_seconds=args.flash_restore_seconds,
        flash_tail_seconds=args.flash_tail_seconds,
        yaw_augment_prob=0.0,
        yaw_augment_max_ratio=0.0,
    )
    test_dataset = FlashLivenessDataset(
        splits["test"],
        transform=False,
        preprocessor=preprocessor,
        target_size=(args.image_size, args.image_size),
        corrupted_sample_recorder=corrupted_sample_recorder,
        max_frames=args.max_eval_frames,
        frame_stride=args.frame_stride,
        use_physical_features=args.use_physical_features,
        missing_color_protocol=args.missing_color_protocol,
        flash_warmup_seconds=args.flash_warmup_seconds,
        flash_hold_seconds=args.flash_hold_seconds,
        flash_restore_seconds=args.flash_restore_seconds,
        flash_tail_seconds=args.flash_tail_seconds,
        yaw_augment_prob=0.0,
        yaw_augment_max_ratio=0.0,
    )

    loader_common_kwargs = {
        "num_workers": args.num_workers,
        "collate_fn": collate_skip_none,
        "pin_memory": args.pin_memory,
        "persistent_workers": args.persistent_workers and args.num_workers > 0,
    }
    if args.num_workers > 0:
        loader_common_kwargs["prefetch_factor"] = args.prefetch_factor

    train_sampler = build_class_balanced_sampler(splits["train"], args.seed) if args.balanced_train_sampler else None
    if train_sampler is not None:
        print("Using class-balanced train sampler: live/spoof are sampled with approximately equal probability.")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        **loader_common_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_common_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_common_kwargs,
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    model = build_model_from_args(args).to(device)
    if args.multi_gpu and device.type == "cuda" and torch.cuda.device_count() > 1:
        device_ids = list(range(torch.cuda.device_count()))
        model = nn.DataParallel(model, device_ids=device_ids)
        print(f"Using DataParallel on {len(device_ids)} GPUs: {device_ids}")
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    non_blocking = bool(args.pin_memory and device.type == "cuda")
    print(
        "Training performance options: "
        f"amp={use_amp}, pin_memory={args.pin_memory}, non_blocking={non_blocking}, "
        f"persistent_workers={loader_common_kwargs['persistent_workers']}, "
        f"prefetch_factor={loader_common_kwargs.get('prefetch_factor', 'disabled')}, "
        f"cudnn_benchmark={torch.backends.cudnn.benchmark}"
    )
    pos_weight_arg = args.pos_weight
    if args.balanced_train_sampler and pos_weight_arg == "auto":
        pos_weight_arg = "1.0"
        print("Class-balanced sampler is enabled; using BCE pos_weight=1.0 instead of auto to avoid double compensation.")
    pos_weight_value = resolve_pos_weight(splits["train"], pos_weight_arg)
    print(f"Using BCE positive weight for live class: {pos_weight_value:.6f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    save_run_config(output_dir, args, split_counts, pos_weight_value, category_coverage=category_coverage)

    best_val_auc = -1.0
    best_payload = None
    history_rows: list[dict] = []
    history_jsonl_path = output_dir / "metrics_history.jsonl"
    history_csv_path = output_dir / "metrics_history.csv"
    if history_jsonl_path.exists() and not args.resume_checkpoint:
        history_jsonl_path.unlink()
    if history_csv_path.exists() and not args.resume_checkpoint:
        history_csv_path.unlink()
    start_epoch = 1

    if args.resume_checkpoint:
        start_epoch, best_val_auc, history_rows, best_payload = maybe_resume_training(
            args=args,
            model=model,
            optimizer=optimizer,
            device=device,
            history_jsonl_path=history_jsonl_path,
        )
        if start_epoch > args.epochs:
            raise ValueError(
                f"resume 后起始 epoch={start_epoch} 已大于设定的 epochs={args.epochs}，请增大 --epochs。"
            )

    for epoch in range(start_epoch, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            epoch=epoch,
            total_epochs=args.epochs,
            log_interval=args.log_interval,
            log_seconds=args.log_seconds,
            window_size=args.window_size,
            window_stride=args.window_stride,
            window_fusion=args.window_fusion,
            window_trim_ratio=args.window_trim_ratio,
            window_min_quality=args.window_min_quality,
            use_amp=use_amp,
            scaler=scaler,
            non_blocking=non_blocking,
        )
        val_metrics = evaluate_model(
            model,
            val_loader,
            criterion,
            device,
            window_size=args.eval_window_size,
            window_stride=args.eval_window_stride,
            window_fusion=args.window_fusion,
            window_trim_ratio=args.window_trim_ratio,
            window_min_quality=args.window_min_quality,
            use_amp=use_amp,
            non_blocking=non_blocking,
        )
        test_metrics = evaluate_model(
            model,
            test_loader,
            criterion,
            device,
            threshold=val_metrics["threshold"],
            window_size=args.eval_window_size,
            window_stride=args.eval_window_stride,
            window_fusion=args.window_fusion,
            window_trim_ratio=args.window_trim_ratio,
            window_min_quality=args.window_min_quality,
            use_amp=use_amp,
            non_blocking=non_blocking,
        )

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"train_cls={train_metrics.get('cls_loss', 0.0):.4f} train_depth={train_metrics.get('depth_loss', 0.0):.4f} "
            f"train_contrast={train_metrics.get('contrast_loss', 0.0):.4f} train_fft={train_metrics.get('fft_loss', 0.0):.4f} "
            f"train_aux={train_metrics.get('aux_loss', 0.0):.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_auc={val_metrics['auc']:.4f} val_eer={val_metrics['eer']:.4f} val_acer={val_metrics['acer']:.4f} | "
            f"test_loss={test_metrics['loss']:.4f} test_acc={test_metrics['accuracy']:.4f} "
            f"test_auc={test_metrics['auc']:.4f} test_eer={test_metrics['eer']:.4f} test_acer={test_metrics['acer']:.4f}",
            flush=True,
        )

        epoch_payload = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_accuracy": round(train_metrics["accuracy"], 6),
            "train_cls_loss": round(train_metrics.get("cls_loss", 0.0), 6),
            "train_depth_loss": round(train_metrics.get("depth_loss", 0.0), 6),
            "train_contrast_loss": round(train_metrics.get("contrast_loss", 0.0), 6),
            "train_fft_loss": round(train_metrics.get("fft_loss", 0.0), 6),
            "train_weighted_depth_loss": round(train_metrics.get("weighted_depth_loss", 0.0), 6),
            "train_weighted_contrast_loss": round(train_metrics.get("weighted_contrast_loss", 0.0), 6),
            "train_weighted_fft_loss": round(train_metrics.get("weighted_fft_loss", 0.0), 6),
            "train_aux_loss": round(train_metrics.get("aux_loss", 0.0), 6),
            "train_threshold": round(train_metrics["threshold"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "val_auc": round(val_metrics["auc"], 6),
            "val_eer": round(val_metrics["eer"], 6),
            "val_acer": round(val_metrics["acer"], 6),
            "val_cls_loss": round(val_metrics.get("cls_loss", 0.0), 6),
            "val_depth_loss": round(val_metrics.get("depth_loss", 0.0), 6),
            "val_contrast_loss": round(val_metrics.get("contrast_loss", 0.0), 6),
            "val_fft_loss": round(val_metrics.get("fft_loss", 0.0), 6),
            "val_weighted_depth_loss": round(val_metrics.get("weighted_depth_loss", 0.0), 6),
            "val_weighted_contrast_loss": round(val_metrics.get("weighted_contrast_loss", 0.0), 6),
            "val_weighted_fft_loss": round(val_metrics.get("weighted_fft_loss", 0.0), 6),
            "val_aux_loss": round(val_metrics.get("aux_loss", 0.0), 6),
            "val_threshold": round(val_metrics["threshold"], 6),
            "test_loss": round(test_metrics["loss"], 6),
            "test_accuracy": round(test_metrics["accuracy"], 6),
            "test_auc": round(test_metrics["auc"], 6),
            "test_eer": round(test_metrics["eer"], 6),
            "test_acer": round(test_metrics["acer"], 6),
            "test_cls_loss": round(test_metrics.get("cls_loss", 0.0), 6),
            "test_depth_loss": round(test_metrics.get("depth_loss", 0.0), 6),
            "test_contrast_loss": round(test_metrics.get("contrast_loss", 0.0), 6),
            "test_fft_loss": round(test_metrics.get("fft_loss", 0.0), 6),
            "test_weighted_depth_loss": round(test_metrics.get("weighted_depth_loss", 0.0), 6),
            "test_weighted_contrast_loss": round(test_metrics.get("weighted_contrast_loss", 0.0), 6),
            "test_weighted_fft_loss": round(test_metrics.get("weighted_fft_loss", 0.0), 6),
            "test_aux_loss": round(test_metrics.get("aux_loss", 0.0), 6),
            "test_threshold": round(test_metrics["threshold"], 6),
        }
        history_rows.append(epoch_payload)
        append_epoch_metrics(history_jsonl_path, epoch_payload)

        last_checkpoint_path = save_checkpoint(
            output_dir=output_dir,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            args=args,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            train_metrics=train_metrics,
            pos_weight_value=pos_weight_value,
            split_counts=split_counts,
            best_val_auc=max(best_val_auc, val_metrics["auc"]),
            checkpoint_name="last_flash_liveness_model.pth",
        )

        if val_metrics["auc"] >= best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_payload = {
                "best_epoch": epoch,
                "resolved_pos_weight": pos_weight_value,
                "split_counts": split_counts,
                "train_metrics": train_metrics,
                "val_metrics": val_metrics,
                "test_metrics": test_metrics,
                "last_checkpoint_path": str(last_checkpoint_path),
            }
            save_summary_json(output_dir, best_payload)

            checkpoint_path = save_checkpoint(
                output_dir=output_dir,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                args=args,
                val_metrics=val_metrics,
                test_metrics=test_metrics,
                train_metrics=train_metrics,
                pos_weight_value=pos_weight_value,
                split_counts=split_counts,
                best_val_auc=best_val_auc,
                checkpoint_name="best_flash_liveness_model.pth",
            )
            best_payload["checkpoint_path"] = str(checkpoint_path)
            save_summary_json(output_dir, best_payload)

    if best_payload is None:
        raise RuntimeError("训练未能产生有效结果，请检查数据集是否为空。")

    write_metrics_csv(history_csv_path, history_rows)
    best_payload["last_checkpoint_path"] = str(output_dir / "last_flash_liveness_model.pth")
    best_payload["skipped_corrupted_samples_path"] = str(corrupted_record_path)
    save_summary_json(output_dir, best_payload)
    update_dataset_readme_with_corrupted_records(args.data_root, corrupted_record_path)
    print(f"Best model saved to: {best_payload['checkpoint_path']}")
    print(json.dumps(best_payload["test_metrics"], indent=2, ensure_ascii=False))


def load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[CNNTransformerLiveness, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = CNNTransformerLiveness(
        num_frames=config["num_frames"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_video(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    threshold = args.threshold if args.threshold is not None else float(checkpoint.get("threshold", 0.5))
    txt_path = args.txt_path if args.txt_path else str(Path(args.video_path).with_suffix(".txt"))

    preprocessor = FacePreprocessor(
        detector_model_path=args.detector_model,
        detector_device=args.detector_device,
        target_size=tuple(config["target_size"]),
        conf_threshold=args.detector_conf,
        iou_threshold=args.detector_iou,
    )
    dataset = FlashLivenessDataset(
        samples=[
            LivenessSample(
                media_path=args.video_path,
                txt_path=txt_path,
                label=0,
                media_type="videos",
                category="inference",
                source_group="single_video",
            )
        ],
        transform=False,
        preprocessor=preprocessor,
        target_size=tuple(config["target_size"]),
    )
    frames_tensor, color_tensor = dataset.process_video(args.video_path, txt_path)
    if frames_tensor.numel() == 0:
        raise RuntimeError(f"无法从视频中读取有效帧: {args.video_path}")
    frames_tensor = frames_tensor.unsqueeze(0).to(device)
    color_tensor = color_tensor.unsqueeze(0).to(device)
    padding_mask = torch.zeros((1, frames_tensor.shape[1]), dtype=torch.bool, device=device)

    with torch.no_grad():
        logit = model(frames_tensor, color_tensor, padding_mask)
        probability = torch.sigmoid(logit).item()

    prediction = "live" if probability >= threshold else "spoof"
    print(
        json.dumps(
            {
                "video_path": args.video_path,
                "probability_live": round(probability, 6),
                "threshold": round(threshold, 6),
                "prediction": prediction,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def _fft_target_from_frames(frames_rgb: np.ndarray, target_size: int = 32) -> np.ndarray:
    targets = []
    for frame_rgb in frames_rgb:
        gray = cv2.cvtColor((np.clip(frame_rgb, 0.0, 1.0) * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray = cv2.resize(gray, (target_size, target_size)).astype(np.float32) / 255.0
        spectrum = np.fft.fftshift(np.fft.fft2(gray))
        mag = np.log1p(np.abs(spectrum)).astype(np.float32)
        mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-6)
        targets.append(mag[None, :, :])
    if not targets:
        return np.zeros((0, 1, target_size, target_size), dtype=np.float32)
    return np.stack(targets, axis=0).astype(np.float32)


class FlashLivenessDataset(Dataset):
    def __init__(
        self,
        samples: list[LivenessSample],
        transform: bool = False,
        preprocessor: FacePreprocessor | None = None,
        target_size: tuple[int, int] = (224, 224),
        corrupted_sample_recorder: CorruptedSampleRecorder | None = None,
        max_frames: int = 0,
        frame_stride: int = 1,
        use_physical_features: bool = True,
        fft_target_size: int = 32,
        missing_color_protocol: str = "collect_flash",
        flash_warmup_seconds: float = 1.0,
        flash_hold_seconds: float = 0.35,
        flash_restore_seconds: float = 0.15,
        flash_tail_seconds: float = 0.5,
        yaw_augment_prob: float = 0.35,
        yaw_augment_max_ratio: float = 0.10,
    ) -> None:
        self.samples = samples
        self.transform = transform
        self.preprocessor = preprocessor or FacePreprocessor(target_size=target_size)
        self.target_size = target_size
        self.corrupted_sample_recorder = corrupted_sample_recorder
        self.max_frames = max(int(max_frames), 0)
        self.frame_stride = max(int(frame_stride), 1)
        self.use_physical_features = use_physical_features
        self.fft_target_size = int(fft_target_size)
        if missing_color_protocol not in FLASH_COLOR_PROTOCOLS:
            raise ValueError(f"missing_color_protocol must be one of {sorted(FLASH_COLOR_PROTOCOLS)}")
        self.missing_color_protocol = missing_color_protocol
        self.flash_warmup_seconds = float(flash_warmup_seconds)
        self.flash_hold_seconds = float(flash_hold_seconds)
        self.flash_restore_seconds = float(flash_restore_seconds)
        self.flash_tail_seconds = float(flash_tail_seconds)
        self.yaw_augment_prob = float(np.clip(yaw_augment_prob, 0.0, 1.0))
        self.yaw_augment_max_ratio = float(np.clip(yaw_augment_max_ratio, 0.0, 0.2))
        self.physical_extractor = PhysicalCueExtractor(
            PhysicalFeatureConfig(
                use_frequency=use_physical_features,
                use_flash_response=use_physical_features,
                use_rppg=use_physical_features,
                use_depth_normal=False,
            )
        )
        self.physical_dim = self.physical_extractor.feature_dim if use_physical_features else 0

    def __len__(self) -> int:
        return len(self.samples)

    def _build_missing_video_color_map(self, cap: cv2.VideoCapture) -> dict[int, int]:
        if self.missing_color_protocol == "neutral":
            return {}

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 0:
            return {}
        if fps <= 1e-6 or math.isnan(fps):
            fps = 30.0
        labels = build_frame_color_labels(
            frame_count=frame_count,
            fps=fps,
            warmup_seconds=self.flash_warmup_seconds,
            hold_seconds=self.flash_hold_seconds,
            restore_seconds=self.flash_restore_seconds,
            tail_seconds=self.flash_tail_seconds,
            color_sequence=COLLECT_FLASH_COLOR_SEQUENCE_RGB,
        )
        return {idx: color for idx, color in enumerate(labels)}

    def _read_all_frames_with_color(self, video_path: str, txt_path: str | None) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
        cap = cv2.VideoCapture(video_path)
        configure_video_capture_orientation(cap)
        color_map = parse_color_txt(txt_path)
        if not color_map:
            color_map = self._build_missing_video_color_map(cap)
        expected_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        max_decode_failures = max(8, min(90, int(max(expected_frame_count, 1) * 0.02)))
        decode_failures = 0
        consecutive_decode_failures = 0
        frames: list[np.ndarray] = []
        color_features: list[np.ndarray] = []
        color_values: list[int] = []
        current_frame_idx = 0
        prev_color_value: int | None = None

        while True:
            success, frame = cap.read()
            if not success:
                if (
                    expected_frame_count > 0
                    and current_frame_idx + 1 < expected_frame_count
                    and consecutive_decode_failures < max_decode_failures
                ):
                    decode_failures += 1
                    consecutive_decode_failures += 1
                    current_frame_idx += 1
                    cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame_idx)
                    continue
                break

            consecutive_decode_failures = 0

            should_keep_frame = (current_frame_idx % self.frame_stride) == 0
            if should_keep_frame and frame is not None and frame.size > 0:
                color_value = color_map.get(
                    current_frame_idx,
                    prev_color_value if prev_color_value is not None else DEFAULT_UNKNOWN_COLOR,
                )
                frames.append(frame.copy())
                color_features.append(color_int_to_feature(color_value, prev_color_value))
                color_values.append(int(color_value))
                prev_color_value = int(color_value)
                if self.max_frames > 0 and len(frames) >= self.max_frames:
                    break

            current_frame_idx += 1

        cap.release()
        if not frames:
            if self.corrupted_sample_recorder is not None:
                self.corrupted_sample_recorder.record(video_path, "no_valid_frames_decoded_v3_sequential_read")
            return [], [], []
        if decode_failures > 0 and self.corrupted_sample_recorder is not None:
            self.corrupted_sample_recorder.record(video_path, f"decode_failures_skipped={decode_failures}")
        return frames, color_features, color_values

    def _read_image_with_color(self, image_path: str, txt_path: str | None) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
        frame = cv2.imread(image_path)
        if frame is None or frame.size == 0:
            if self.corrupted_sample_recorder is not None:
                self.corrupted_sample_recorder.record(image_path, "no_valid_image_decoded_v3")
            return [], [], []
        color_map = parse_color_txt(txt_path)
        color_value = color_map.get(0, DEFAULT_UNKNOWN_COLOR)
        return [frame], [color_int_to_feature(color_value, None)], [int(color_value)]

    def _sample_frames(self, sample: LivenessSample) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
        if sample.media_type == "images":
            return self._read_image_with_color(sample.media_path, sample.txt_path)
        return self._read_all_frames_with_color(sample.media_path, sample.txt_path)

    def process_sample(self, sample: LivenessSample) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        frames_bgr, color_features, color_values = self._sample_frames(sample)
        if not frames_bgr:
            return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
        prefix = Path(sample.media_path).stem
        processed_frames = self.preprocessor.preprocess_frames(frames_bgr, prefix=prefix)
        processed_frames = maybe_apply_yaw_augmentation(
            processed_frames,
            enabled=self.transform,
            probability=self.yaw_augment_prob,
            max_ratio=self.yaw_augment_max_ratio,
        )

        frames_rgb = []
        for frame in processed_frames:
            if frame.shape[:2] != (self.target_size[1], self.target_size[0]):
                frame = cv2.resize(frame, self.target_size)
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frames = np.asarray(frames_rgb, dtype=np.float32) / 255.0
        diff_frames = np.zeros_like(frames)
        if len(frames) > 1:
            diff_frames[1:] = frames[1:] - frames[:-1]
            diff_frames[0] = diff_frames[1]

        multi_modal_frames = np.concatenate([frames, diff_frames], axis=-1)
        tensor_frames = torch.from_numpy(multi_modal_frames).permute(0, 3, 1, 2).float()
        tensor_colors = torch.from_numpy(np.asarray(color_features, dtype=np.float32))
        if self.use_physical_features and self.physical_dim > 0:
            physical_np = self.physical_extractor.per_frame(frames, np.asarray(color_values, dtype=np.int64))
            physical_np = stabilize_physical_features(physical_np)
        else:
            physical_np = np.zeros((len(frames), 0), dtype=np.float32)
        tensor_physical = torch.from_numpy(physical_np.astype(np.float32))
        tensor_fft = torch.from_numpy(_fft_target_from_frames(frames, self.fft_target_size))
        return tensor_frames, tensor_colors, tensor_physical, tensor_fft

    def process_video(self, video_path: str, txt_path: str | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        sample = LivenessSample(
            media_path=video_path,
            txt_path=txt_path,
            label=0,
            media_type="videos",
            category="inference",
            source_group="single_video",
        )
        return self.process_sample(sample)

    def process_video_legacy(self, video_path: str, txt_path: str | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        frames_bgr, color_features, color_values = self._read_all_frames_with_color(video_path, txt_path)
        if not frames_bgr:
            return torch.empty(0), torch.empty(0), torch.empty(0), torch.empty(0)
        prefix = Path(video_path).stem
        processed_frames = self.preprocessor.preprocess_frames(frames_bgr, prefix=prefix)

        frames_rgb = []
        for frame in processed_frames:
            if frame.shape[:2] != (self.target_size[1], self.target_size[0]):
                frame = cv2.resize(frame, self.target_size)
            frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        frames = np.asarray(frames_rgb, dtype=np.float32) / 255.0
        diff_frames = np.zeros_like(frames)
        if len(frames) > 1:
            diff_frames[1:] = frames[1:] - frames[:-1]
            diff_frames[0] = diff_frames[1]

        multi_modal_frames = np.concatenate([frames, diff_frames], axis=-1)
        tensor_frames = torch.from_numpy(multi_modal_frames).permute(0, 3, 1, 2).float()
        tensor_colors = torch.from_numpy(np.asarray(color_features, dtype=np.float32))
        if self.use_physical_features and self.physical_dim > 0:
            physical_np = self.physical_extractor.per_frame(frames, np.asarray(color_values, dtype=np.int64))
        else:
            physical_np = np.zeros((len(frames), 0), dtype=np.float32)
        tensor_physical = torch.from_numpy(physical_np.astype(np.float32))
        tensor_fft = torch.from_numpy(_fft_target_from_frames(frames, self.fft_target_size))
        return tensor_frames, tensor_colors, tensor_physical, tensor_fft

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        tensor_frames, tensor_colors, tensor_physical, tensor_fft = self.process_sample(sample)
        if tensor_frames.numel() == 0:
            return None

        return tensor_frames, tensor_colors, tensor_physical, tensor_fft, torch.tensor(sample.label, dtype=torch.float32)


class Conv2dCD(nn.Module):
    def __init__(self, *args, theta: float = 0.7, **kwargs) -> None:
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        self.theta = float(theta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out_normal = self.conv(x)
        if abs(self.theta) < 1e-8 or self.conv.weight.shape[2] == 1:
            return out_normal
        kernel_diff = self.conv.weight.sum(dim=(2, 3), keepdim=True)
        out_diff = F.conv2d(
            x,
            kernel_diff,
            bias=None,
            stride=self.conv.stride,
            padding=0,
            dilation=self.conv.dilation,
            groups=self.conv.groups,
        )
        return out_normal - self.theta * out_diff


class CDCTextureBranch(nn.Module):
    def __init__(self, in_channels: int = 6, embed_dim: int = 512, theta: float = 0.7) -> None:
        super().__init__()
        self.features = nn.Sequential(
            Conv2dCD(in_channels, 64, kernel_size=3, stride=2, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Conv2dCD(64, 128, kernel_size=3, stride=2, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Conv2dCD(128, embed_dim, kernel_size=3, stride=2, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.depth_head = nn.Sequential(
            nn.Conv2d(embed_dim, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature_map = self.features(x)
        embedding = self.pool(feature_map).flatten(1)
        depth_map = F.interpolate(self.depth_head(feature_map), size=(32, 32), mode="bilinear", align_corners=False)
        return embedding, depth_map


class CNNTransformerLiveness(nn.Module):
    def __init__(
        self,
        num_frames: int = 16,
        embed_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        use_imagenet_pretrained: bool = False,
        imagenet_pretrained_path: str | None = None,
        physical_dim: int = 19,
        cdc_theta: float = 0.7,
        lambda_depth: float = 0.05,
        lambda_contrast: float = 0.1,
        lambda_fft: float = 0.05,
        aux_loss_max_ratio: float = 0.35,
        aux_loss_total_max_ratio: float = 0.7,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.embed_dim = embed_dim
        self.physical_dim = int(physical_dim)
        self.lambda_depth = float(lambda_depth)
        self.lambda_contrast = float(lambda_contrast)
        self.lambda_fft = float(lambda_fft)
        self.aux_loss_max_ratio = float(aux_loss_max_ratio)
        self.aux_loss_total_max_ratio = float(aux_loss_total_max_ratio)

        resnet = models.resnet18(weights=None)
        if use_imagenet_pretrained:
            load_error = None
            if imagenet_pretrained_path:
                try:
                    state_dict = torch.load(imagenet_pretrained_path, map_location="cpu")
                    resnet.load_state_dict(state_dict)
                    print(f"已从本地加载 ResNet18 预训练权重: {imagenet_pretrained_path}")
                except Exception as exc:
                    load_error = exc
            else:
                try:
                    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
                    print("已通过 torchvision 加载 ResNet18 预训练权重。")
                except Exception as exc:
                    load_error = exc
            if load_error is not None:
                print(f"加载 ResNet18 预训练权重失败，回退到随机初始化。原因: {load_error}")

        original_conv1 = resnet.conv1
        self.cnn_backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.cnn_backbone[0] = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.cnn_backbone[0].weight[:, :3, :, :] = original_conv1.weight
            self.cnn_backbone[0].weight[:, 3:, :, :] = original_conv1.weight
        for param in self.cnn_backbone[0:5].parameters():
            param.requires_grad = False

        self.frame_proj = nn.Identity() if embed_dim == 512 else nn.Linear(512, embed_dim)
        self.texture_branch = CDCTextureBranch(in_channels=6, embed_dim=embed_dim, theta=cdc_theta)
        self.color_proj = nn.Sequential(nn.Linear(4, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
        self.physical_proj = (
            nn.Sequential(nn.Linear(self.physical_dim, embed_dim), nn.LayerNorm(embed_dim), nn.ReLU())
            if self.physical_dim > 0
            else None
        )
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )
        self.fft_head = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 32 * 32),
            nn.Sigmoid(),
        )
        self.pos_encoder = PositionalEncoding(d_model=embed_dim, max_len=512)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=0.3,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Sequential(nn.Linear(embed_dim, 128), nn.ReLU(), nn.Dropout(0.5), nn.Linear(128, 1))

    def forward(
        self,
        x: torch.Tensor,
        color_features: torch.Tensor,
        physical_features: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size, num_frames, channels, height, width = x.size()
        x_cnn = x.view(batch_size * num_frames, channels, height, width)
        frame_emb = self.cnn_backbone(x_cnn).flatten(1)
        frame_emb = self.frame_proj(frame_emb).view(batch_size, num_frames, self.embed_dim)
        texture_emb, depth_maps = self.texture_branch(x_cnn)
        texture_emb = texture_emb.view(batch_size, num_frames, self.embed_dim)
        depth_maps = depth_maps.view(batch_size, num_frames, 1, 32, 32)

        color_emb = self.color_proj(color_features)
        if self.physical_proj is not None and physical_features is not None and physical_features.shape[-1] == self.physical_dim:
            physical_emb = self.physical_proj(physical_features)
        else:
            physical_emb = torch.zeros_like(frame_emb)

        fused = self.fusion(torch.cat([frame_emb + color_emb, texture_emb, physical_emb], dim=-1))
        fft_maps = self.fft_head(texture_emb).view(batch_size, num_frames, 1, 32, 32)
        features = self.pos_encoder(fused)
        trans_out = self.transformer(features, src_key_padding_mask=padding_mask)

        if padding_mask is not None:
            valid_mask = (~padding_mask).unsqueeze(-1).float()
            pooled_out = (trans_out * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp_min(1.0)
        else:
            pooled_out = trans_out.mean(dim=1)
        logits = self.fc(pooled_out).squeeze(-1)
        return {"logits": logits, "depth_maps": depth_maps, "fft_maps": fft_maps}


def collate_skip_none(batch):
    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        return None

    max_len = max(item[0].shape[0] for item in valid_batch)
    channels = valid_batch[0][0].shape[1]
    height = valid_batch[0][0].shape[2]
    width = valid_batch[0][0].shape[3]
    color_dim = valid_batch[0][1].shape[1]
    physical_dim = valid_batch[0][2].shape[1]
    fft_size = valid_batch[0][3].shape[-1]

    videos = torch.zeros((len(valid_batch), max_len, channels, height, width), dtype=torch.float32)
    colors = torch.zeros((len(valid_batch), max_len, color_dim), dtype=torch.float32)
    physical = torch.zeros((len(valid_batch), max_len, physical_dim), dtype=torch.float32)
    fft_targets = torch.zeros((len(valid_batch), max_len, 1, fft_size, fft_size), dtype=torch.float32)
    padding_mask = torch.ones((len(valid_batch), max_len), dtype=torch.bool)
    labels = torch.stack([item[4] for item in valid_batch], dim=0)

    for idx, (video_tensor, color_tensor, physical_tensor, fft_tensor, _) in enumerate(valid_batch):
        seq_len = video_tensor.shape[0]
        videos[idx, :seq_len] = video_tensor
        colors[idx, :seq_len] = color_tensor
        physical[idx, :seq_len] = physical_tensor
        fft_targets[idx, :seq_len] = fft_tensor
        padding_mask[idx, :seq_len] = False

    return videos, colors, physical, fft_targets, padding_mask, labels


def contrast_depth_loss(pred_depth: torch.Tensor, target_depth: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    kernels = torch.tensor(
        [
            [[0, 0, 0], [0, 1, -1], [0, 0, 0]],
            [[0, 0, 0], [-1, 1, 0], [0, 0, 0]],
            [[0, -1, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, -1, 0]],
            [[-1, 0, 0], [0, 1, 0], [0, 0, 0]],
            [[0, 0, -1], [0, 1, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [-1, 0, 0]],
            [[0, 0, 0], [0, 1, 0], [0, 0, -1]],
        ],
        dtype=pred_depth.dtype,
        device=pred_depth.device,
    ).unsqueeze(1)
    batch_size, num_frames = pred_depth.shape[:2]
    pred_flat = pred_depth.reshape(batch_size * num_frames, 1, 32, 32)
    target_flat = target_depth.reshape(batch_size * num_frames, 1, 32, 32)
    pred_diff = F.conv2d(pred_flat, kernels, padding=1)
    target_diff = F.conv2d(target_flat, kernels, padding=1)
    per_frame = F.mse_loss(pred_diff, target_diff, reduction="none").mean(dim=(1, 2, 3)).view(batch_size, num_frames)
    return (per_frame * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(1.0)


def make_loss_metric_totals() -> dict[str, float]:
    return {key: 0.0 for key in LOSS_METRIC_KEYS}


def balance_auxiliary_losses(
    cls_loss: torch.Tensor,
    weighted_terms: list[torch.Tensor],
    per_term_max_ratio: float,
    total_max_ratio: float,
) -> list[torch.Tensor]:
    balanced_terms = weighted_terms
    if per_term_max_ratio > 0:
        per_term_cap = cls_loss.detach().clamp_min(1e-6) * per_term_max_ratio
        capped_terms = []
        for term in balanced_terms:
            scale = (per_term_cap / term.detach().abs().clamp_min(1e-8)).clamp(max=1.0)
            capped_terms.append(term * scale)
        balanced_terms = capped_terms

    if total_max_ratio > 0 and balanced_terms:
        aux_sum = torch.stack(balanced_terms).sum()
        total_cap = cls_loss.detach().clamp_min(1e-6) * total_max_ratio
        total_scale = (total_cap / aux_sum.detach().abs().clamp_min(1e-8)).clamp(max=1.0)
        balanced_terms = [term * total_scale for term in balanced_terms]

    return balanced_terms


def compute_v3_loss(
    outputs: dict[str, torch.Tensor],
    labels: torch.Tensor,
    padding_mask: torch.Tensor,
    fft_targets: torch.Tensor,
    criterion: nn.Module,
) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs["logits"]
    cls_loss = criterion(logits, labels)
    valid_mask = ~padding_mask
    depth_targets = labels.view(-1, 1, 1, 1, 1).expand_as(outputs["depth_maps"]).float()
    depth_per_frame = F.mse_loss(outputs["depth_maps"], depth_targets, reduction="none").mean(dim=(2, 3, 4))
    depth_loss = (depth_per_frame * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(1.0)
    cdl = contrast_depth_loss(outputs["depth_maps"], depth_targets, valid_mask)
    fft_per_frame = F.mse_loss(outputs["fft_maps"], fft_targets, reduction="none").mean(dim=(2, 3, 4))
    fft_loss = (fft_per_frame * valid_mask.float()).sum() / valid_mask.float().sum().clamp_min(1.0)
    model = getattr(outputs["logits"], "_v3_model", None)
    lambda_depth = float(getattr(model, "lambda_depth", 0.05))
    lambda_contrast = float(getattr(model, "lambda_contrast", 0.1))
    lambda_fft = float(getattr(model, "lambda_fft", 0.05))
    aux_loss_max_ratio = float(getattr(model, "aux_loss_max_ratio", 0.35))
    aux_loss_total_max_ratio = float(getattr(model, "aux_loss_total_max_ratio", 0.7))
    weighted_depth, weighted_contrast, weighted_fft = balance_auxiliary_losses(
        cls_loss,
        [lambda_depth * depth_loss, lambda_contrast * cdl, lambda_fft * fft_loss],
        per_term_max_ratio=aux_loss_max_ratio,
        total_max_ratio=aux_loss_total_max_ratio,
    )
    aux_loss = weighted_depth + weighted_contrast + weighted_fft
    total_loss = cls_loss + aux_loss
    return total_loss, {
        "cls_loss": float(cls_loss.detach().cpu()),
        "depth_loss": float(depth_loss.detach().cpu()),
        "contrast_loss": float(cdl.detach().cpu()),
        "fft_loss": float(fft_loss.detach().cpu()),
        "weighted_depth_loss": float(weighted_depth.detach().cpu()),
        "weighted_contrast_loss": float(weighted_contrast.detach().cpu()),
        "weighted_fft_loss": float(weighted_fft.detach().cpu()),
        "aux_loss": float(aux_loss.detach().cpu()),
    }


def tensor_diagnostic_stats(name: str, tensor: torch.Tensor) -> dict[str, object]:
    detached = tensor.detach()
    finite_mask = torch.isfinite(detached)
    stats: dict[str, object] = {
        "name": name,
        "shape": list(detached.shape),
        "finite": bool(finite_mask.all().item()),
    }
    if finite_mask.any():
        finite_values = detached[finite_mask].float()
        stats.update(
            {
                "min": float(finite_values.min().cpu()),
                "max": float(finite_values.max().cpu()),
                "mean": float(finite_values.mean().cpu()),
            }
        )
    return stats


def assert_finite_window_loss(
    loss: torch.Tensor,
    aux: dict[str, float],
    outputs: dict[str, torch.Tensor],
    frames: torch.Tensor,
    colors: torch.Tensor,
    physical: torch.Tensor,
    fft_target: torch.Tensor,
    context: dict[str, object],
) -> None:
    aux_is_finite = all(math.isfinite(float(value)) for value in aux.values())
    tensors_to_check = {
        "loss": loss,
        "logits": outputs["logits"],
        "depth_maps": outputs["depth_maps"],
        "fft_maps": outputs["fft_maps"],
        "frames": frames,
        "colors": colors,
        "physical": physical,
        "fft_target": fft_target,
    }
    tensors_are_finite = all(bool(torch.isfinite(tensor.detach()).all().item()) for tensor in tensors_to_check.values())
    if aux_is_finite and tensors_are_finite:
        return

    diagnostics = {
        "context": context,
        "aux": aux,
        "tensors": [tensor_diagnostic_stats(name, tensor) for name, tensor in tensors_to_check.items()],
    }
    raise FloatingPointError("Non-finite V3_1 training loss detected:\n" + json.dumps(diagnostics, indent=2, ensure_ascii=False))


def _attach_model_to_outputs(outputs: dict[str, torch.Tensor], model: nn.Module) -> dict[str, torch.Tensor]:
    outputs["logits"]._v3_model = unwrap_parallel_model(model)
    return outputs


def unwrap_parallel_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def get_checkpoint_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    return unwrap_parallel_model(model).state_dict()


def load_checkpoint_state_dict(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> None:
    target_model = unwrap_parallel_model(model)
    if any(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    target_model.load_state_dict(state_dict)


def iter_window_ranges(seq_len: int, window_size: int, window_stride: int) -> list[tuple[int, int]]:
    if seq_len <= 0:
        return []
    if window_size <= 0 or seq_len <= window_size:
        return [(0, seq_len)]

    stride = window_stride if window_stride > 0 else window_size
    ranges: list[tuple[int, int]] = []
    start = 0
    while start + window_size < seq_len:
        ranges.append((start, start + window_size))
        start += stride

    final_start = max(seq_len - window_size, 0)
    if not ranges or ranges[-1][0] != final_start:
        ranges.append((final_start, seq_len))
    return ranges


def make_window_mask(length: int, device: torch.device) -> torch.Tensor:
    return torch.zeros((1, length), dtype=torch.bool, device=device)


def estimate_window_quality(frames: torch.Tensor) -> float:
    if frames.numel() == 0:
        return 0.0

    with torch.no_grad():
        diff_motion = float(frames[:, :, 3:6].abs().mean().detach().cpu())
        rgb = frames[:, :, :3]
        brightness = rgb.mean(dim=2)
        temporal_std = float(brightness.mean(dim=(2, 3)).std().detach().cpu()) if brightness.shape[1] > 1 else 0.0

    motion_penalty = max(diff_motion - 0.08, 0.0) / 0.12
    flicker_penalty = max(temporal_std - 0.18, 0.0) / 0.18
    quality = 1.0 / (1.0 + motion_penalty + flicker_penalty)
    return float(np.clip(quality, 0.05, 1.0))


def fuse_window_probabilities(
    probabilities: list[float],
    qualities: list[float] | None = None,
    method: str = "quality_lower_trimmed_mean",
    trim_ratio: float = 0.2,
    min_quality: float = 0.05,
) -> float:
    if not probabilities:
        return 0.0

    probs = np.asarray(probabilities, dtype=np.float32)
    if qualities is None or len(qualities) != len(probabilities):
        weights = np.ones_like(probs, dtype=np.float32)
    else:
        weights = np.asarray(qualities, dtype=np.float32)
        weights = np.clip(weights, min_quality, 1.0)

    if method == "mean":
        return float(np.average(probs, weights=weights))

    if method in {"trimmed_mean", "quality_trimmed_mean"}:
        keep_count = max(1, int(math.ceil(probs.size * (1.0 - max(trim_ratio, 0.0)))))
        keep_indices = np.argsort(probs)[-keep_count:]
        keep_probs = probs[keep_indices]
        keep_weights = weights[keep_indices] if method == "quality_trimmed_mean" else np.ones_like(keep_probs)
        return float(np.average(keep_probs, weights=keep_weights))

    if method in {"lower_trimmed_mean", "quality_lower_trimmed_mean"}:
        keep_count = max(1, int(math.ceil(probs.size * (1.0 - max(trim_ratio, 0.0)))))
        keep_indices = np.argsort(probs)[:keep_count]
        keep_probs = probs[keep_indices]
        keep_weights = weights[keep_indices] if method == "quality_lower_trimmed_mean" else np.ones_like(keep_probs)
        return float(np.average(keep_probs, weights=keep_weights))

    if method == "low_percentile":
        percentile = float(np.clip(trim_ratio, 0.0, 1.0) * 100.0)
        return float(np.percentile(probs, percentile))

    if method == "min":
        return float(np.min(probs))

    if method == "median":
        return float(np.median(probs))

    return float(np.average(probs, weights=weights))


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    log_interval: int = 20,
    log_seconds: float = 60.0,
    window_size: int = 256,
    window_stride: int = 128,
    window_fusion: str = "quality_lower_trimmed_mean",
    window_trim_ratio: float = 0.2,
    window_min_quality: float = 0.05,
    use_amp: bool = False,
    scaler: torch.cuda.amp.GradScaler | None = None,
    non_blocking: bool = False,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    aux_totals = make_loss_metric_totals()
    all_labels = []
    all_probs = []
    processed_items = 0
    valid_batches = 0
    dataset_size = len(getattr(data_loader, "dataset", []))
    last_log_time = time.time()

    for batch_idx, batch in enumerate(data_loader, start=1):
        if batch is None:
            continue
        batch_videos, batch_colors, batch_physical, batch_fft, batch_padding_mask, batch_labels = batch
        optimizer.zero_grad()
        batch_window_count = 0
        batch_loss_total = 0.0
        batch_aux_totals = make_loss_metric_totals()
        batch_video_probs = []
        batch_video_labels = []

        for sample_idx in range(batch_labels.size(0)):
            valid_len = int((~batch_padding_mask[sample_idx]).sum().item())
            if valid_len <= 0:
                continue

            ranges = iter_window_ranges(valid_len, window_size, window_stride)
            if not ranges:
                continue

            label = batch_labels[sample_idx:sample_idx + 1].to(device, non_blocking=non_blocking)
            window_probs = []
            window_qualities = []
            for start, end in ranges:
                frames = batch_videos[sample_idx:sample_idx + 1, start:end].to(device, non_blocking=non_blocking)
                colors = batch_colors[sample_idx:sample_idx + 1, start:end].to(device, non_blocking=non_blocking)
                physical = batch_physical[sample_idx:sample_idx + 1, start:end].to(device, non_blocking=non_blocking)
                fft_target = batch_fft[sample_idx:sample_idx + 1, start:end].to(device, non_blocking=non_blocking)
                padding_mask = make_window_mask(end - start, device)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    outputs = _attach_model_to_outputs(model(frames, colors, physical, padding_mask), model)
                    loss, aux = compute_v3_loss(outputs, label, padding_mask, fft_target, criterion)
                    assert_finite_window_loss(
                        loss,
                        aux,
                        outputs,
                        frames,
                        colors,
                        physical,
                        fft_target,
                        {
                            "phase": "train",
                            "epoch": epoch,
                            "total_epochs": total_epochs,
                            "batch_idx": batch_idx,
                            "sample_idx": sample_idx,
                            "window_start": start,
                            "window_end": end,
                            "valid_len": valid_len,
                            "label": float(label.detach().cpu().item()),
                        },
                    )
                    scaled_loss = loss / max(len(ranges), 1)

                if scaler is not None and use_amp:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                batch_window_count += 1
                batch_loss_total += float(loss.detach().cpu())
                for key in batch_aux_totals:
                    batch_aux_totals[key] += aux[key]
                window_probs.append(float(torch.sigmoid(outputs["logits"]).detach().cpu().item()))
                window_qualities.append(estimate_window_quality(frames))

            if window_probs:
                batch_video_probs.append(
                    fuse_window_probabilities(
                        window_probs,
                        window_qualities,
                        method=window_fusion,
                        trim_ratio=window_trim_ratio,
                        min_quality=window_min_quality,
                    )
                )
                batch_video_labels.append(float(batch_labels[sample_idx].item()))

        if batch_window_count == 0:
            optimizer.zero_grad()
            continue

        if scaler is not None and use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        valid_batches += 1
        batch_loss = batch_loss_total / max(batch_window_count, 1)
        total_loss += batch_loss
        for key in aux_totals:
            aux_totals[key] += batch_aux_totals[key] / max(batch_window_count, 1)
        if batch_video_labels:
            all_labels.append(np.asarray(batch_video_labels, dtype=np.float32))
            all_probs.append(np.asarray(batch_video_probs, dtype=np.float32))
            processed_items += len(batch_video_labels)

        now = time.time()
        should_log_by_interval = log_interval > 0 and batch_idx % log_interval == 0
        should_log_by_time = log_seconds > 0 and now - last_log_time >= log_seconds
        if should_log_by_interval or should_log_by_time:
            avg_loss = total_loss / max(valid_batches, 1)
            print(
                f"Epoch [{epoch}/{total_epochs}] Batch [{batch_idx}/{len(data_loader)}] "
                f"processed_items={processed_items}/{dataset_size} windows={batch_window_count} "
                f"batch_loss={batch_loss:.4f} avg_loss={avg_loss:.4f} "
                f"cls={batch_aux_totals['cls_loss'] / max(batch_window_count, 1):.4f} "
                f"depth={batch_aux_totals['depth_loss'] / max(batch_window_count, 1):.4f} "
                f"contrast={batch_aux_totals['contrast_loss'] / max(batch_window_count, 1):.4f} "
                f"fft={batch_aux_totals['fft_loss'] / max(batch_window_count, 1):.4f} "
                f"weighted_aux={batch_aux_totals['aux_loss'] / max(batch_window_count, 1):.4f}",
                flush=True,
            )
            last_log_time = now

    labels = np.concatenate(all_labels) if all_labels else np.asarray([])
    probs = np.concatenate(all_probs) if all_probs else np.asarray([])
    threshold = find_best_threshold(labels, probs) if labels.size else 0.5
    metrics = compute_binary_metrics(labels, probs, threshold) if labels.size else {"accuracy": 0.0}
    metrics["loss"] = total_loss / max(valid_batches, 1)
    metrics["threshold"] = threshold
    metrics["processed_items"] = processed_items
    for key, value in aux_totals.items():
        metrics[key] = value / max(valid_batches, 1)
    return metrics


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float | None = None,
    window_size: int = 256,
    window_stride: int = 128,
    window_fusion: str = "quality_lower_trimmed_mean",
    window_trim_ratio: float = 0.2,
    window_min_quality: float = 0.05,
    use_amp: bool = False,
    non_blocking: bool = False,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    aux_totals = make_loss_metric_totals()
    valid_batches = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader, start=1):
            if batch is None:
                continue
            batch_videos, batch_colors, batch_physical, batch_fft, batch_padding_mask, batch_labels = batch
            batch_loss_total = 0.0
            batch_window_count = 0
            batch_aux_totals = make_loss_metric_totals()
            batch_video_probs = []
            batch_video_labels = []

            for sample_idx in range(batch_labels.size(0)):
                valid_len = int((~batch_padding_mask[sample_idx]).sum().item())
                if valid_len <= 0:
                    continue
                ranges = iter_window_ranges(valid_len, window_size, window_stride)
                if not ranges:
                    continue

                label = batch_labels[sample_idx:sample_idx + 1].to(device, non_blocking=non_blocking)
                window_probs = []
                window_qualities = []
                for start, end in ranges:
                    frames = batch_videos[sample_idx:sample_idx + 1, start:end].to(device, non_blocking=non_blocking)
                    colors = batch_colors[sample_idx:sample_idx + 1, start:end].to(device, non_blocking=non_blocking)
                    physical = batch_physical[sample_idx:sample_idx + 1, start:end].to(device, non_blocking=non_blocking)
                    fft_target = batch_fft[sample_idx:sample_idx + 1, start:end].to(device, non_blocking=non_blocking)
                    padding_mask = make_window_mask(end - start, device)
                    with torch.cuda.amp.autocast(enabled=use_amp):
                        outputs = _attach_model_to_outputs(model(frames, colors, physical, padding_mask), model)
                        loss, aux = compute_v3_loss(outputs, label, padding_mask, fft_target, criterion)
                        assert_finite_window_loss(
                            loss,
                            aux,
                            outputs,
                            frames,
                            colors,
                            physical,
                            fft_target,
                            {
                                "phase": "eval",
                                "batch_idx": batch_idx,
                                "sample_idx": sample_idx,
                                "window_start": start,
                                "window_end": end,
                                "valid_len": valid_len,
                                "label": float(label.detach().cpu().item()),
                            },
                        )
                    batch_window_count += 1
                    batch_loss_total += float(loss.detach().cpu())
                    for key in batch_aux_totals:
                        batch_aux_totals[key] += aux[key]
                    window_probs.append(float(torch.sigmoid(outputs["logits"]).detach().cpu().item()))
                    window_qualities.append(estimate_window_quality(frames))

                if window_probs:
                    batch_video_probs.append(
                        fuse_window_probabilities(
                            window_probs,
                            window_qualities,
                            method=window_fusion,
                            trim_ratio=window_trim_ratio,
                            min_quality=window_min_quality,
                        )
                    )
                    batch_video_labels.append(float(batch_labels[sample_idx].item()))

            if batch_window_count == 0:
                continue

            valid_batches += 1
            total_loss += batch_loss_total / max(batch_window_count, 1)
            for key in aux_totals:
                aux_totals[key] += batch_aux_totals[key] / max(batch_window_count, 1)
            if batch_video_labels:
                all_labels.append(np.asarray(batch_video_labels, dtype=np.float32))
                all_probs.append(np.asarray(batch_video_probs, dtype=np.float32))

    labels = np.concatenate(all_labels) if all_labels else np.asarray([])
    probs = np.concatenate(all_probs) if all_probs else np.asarray([])
    if labels.size == 0:
        return {"loss": 0.0, "accuracy": 0.0, "auc": 0.0, "eer": 1.0, "threshold": 0.5}

    threshold = threshold if threshold is not None else find_best_threshold(labels, probs)
    metrics = compute_binary_metrics(labels, probs, threshold)
    auc = compute_auc(labels, probs)
    eer, eer_threshold = compute_eer(labels, probs)
    metrics.update(
        {
            "loss": total_loss / max(valid_batches, 1),
            "auc": auc,
            "eer": eer,
            "threshold": threshold,
            "eer_threshold": eer_threshold,
        }
    )
    for key, value in aux_totals.items():
        metrics[key] = value / max(valid_batches, 1)
    return metrics


def build_model_from_args(args: argparse.Namespace) -> CNNTransformerLiveness:
    physical_dim = 19 if getattr(args, "use_physical_features", True) else 0
    return CNNTransformerLiveness(
        num_frames=args.num_frames,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        use_imagenet_pretrained=args.use_imagenet_pretrained,
        imagenet_pretrained_path=args.imagenet_pretrained_path,
        physical_dim=physical_dim,
        cdc_theta=getattr(args, "cdc_theta", 0.7),
        lambda_depth=getattr(args, "lambda_depth", 0.05),
        lambda_contrast=getattr(args, "lambda_contrast", 0.1),
        lambda_fft=getattr(args, "lambda_fft", 0.05),
        aux_loss_max_ratio=getattr(args, "aux_loss_max_ratio", 0.35),
        aux_loss_total_max_ratio=getattr(args, "aux_loss_total_max_ratio", 0.7),
    )


def save_checkpoint(
    output_dir: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    epoch: int,
    args: argparse.Namespace,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    train_metrics: dict[str, float],
    pos_weight_value: float,
    split_counts: dict[str, dict[str, int]],
    best_val_auc: float,
    checkpoint_name: str,
) -> Path:
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": get_checkpoint_state_dict(model),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "threshold": val_metrics["threshold"],
        "config": {
            "num_frames": args.num_frames,
            "embed_dim": args.embed_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "target_size": [args.image_size, args.image_size],
            "physical_dim": getattr(unwrap_parallel_model(model), "physical_dim", 19),
            "cdc_theta": getattr(args, "cdc_theta", 0.7),
            "lambda_depth": getattr(args, "lambda_depth", 0.05),
            "lambda_contrast": getattr(args, "lambda_contrast", 0.1),
            "lambda_fft": getattr(args, "lambda_fft", 0.05),
            "aux_loss_max_ratio": getattr(args, "aux_loss_max_ratio", 0.35),
            "aux_loss_total_max_ratio": getattr(args, "aux_loss_total_max_ratio", 0.7),
            "missing_color_protocol": getattr(args, "missing_color_protocol", "collect_flash"),
            "flash_warmup_seconds": getattr(args, "flash_warmup_seconds", 1.0),
            "flash_hold_seconds": getattr(args, "flash_hold_seconds", 0.35),
            "flash_restore_seconds": getattr(args, "flash_restore_seconds", 0.15),
            "flash_tail_seconds": getattr(args, "flash_tail_seconds", 0.5),
            "yaw_augment_prob": getattr(args, "yaw_augment_prob", 0.35),
            "yaw_augment_max_ratio": getattr(args, "yaw_augment_max_ratio", 0.10),
            "window_size": getattr(args, "window_size", 256),
            "window_stride": getattr(args, "window_stride", 128),
            "eval_window_size": getattr(args, "eval_window_size", getattr(args, "window_size", 256)),
            "eval_window_stride": getattr(args, "eval_window_stride", getattr(args, "window_stride", 128)),
            "window_fusion": getattr(args, "window_fusion", "quality_lower_trimmed_mean"),
            "window_trim_ratio": getattr(args, "window_trim_ratio", 0.2),
            "window_min_quality": getattr(args, "window_min_quality", 0.05),
        },
        "resolved_pos_weight": pos_weight_value,
        "best_val_auc": best_val_auc,
        "split_counts": split_counts,
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    checkpoint_path = output_dir / checkpoint_name
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


def load_checkpoint(checkpoint_path: str, device: torch.device) -> tuple[CNNTransformerLiveness, dict]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    model = CNNTransformerLiveness(
        num_frames=config["num_frames"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        physical_dim=config.get("physical_dim", 19),
        cdc_theta=config.get("cdc_theta", 0.7),
        lambda_depth=config.get("lambda_depth", 0.05),
        lambda_contrast=config.get("lambda_contrast", 0.1),
        lambda_fft=config.get("lambda_fft", 0.05),
        aux_loss_max_ratio=config.get("aux_loss_max_ratio", 0.35),
        aux_loss_total_max_ratio=config.get("aux_loss_total_max_ratio", 0.7),
    ).to(device)
    load_checkpoint_state_dict(model, checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_video(args: argparse.Namespace) -> None:
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    threshold = args.threshold if args.threshold is not None else float(checkpoint.get("threshold", 0.5))
    txt_path = args.txt_path if args.txt_path else str(Path(args.video_path).with_suffix(".txt"))

    preprocessor = FacePreprocessor(
        detector_model_path=args.detector_model,
        detector_device=args.detector_device,
        target_size=tuple(config["target_size"]),
        conf_threshold=args.detector_conf,
        iou_threshold=args.detector_iou,
    )
    dataset = FlashLivenessDataset(
        samples=[
            LivenessSample(
                media_path=args.video_path,
                txt_path=txt_path,
                label=0,
                media_type="videos",
                category="inference",
                source_group="single_video",
            )
        ],
        transform=False,
        preprocessor=preprocessor,
        target_size=tuple(config["target_size"]),
        use_physical_features=config.get("physical_dim", 19) > 0,
        missing_color_protocol=config.get("missing_color_protocol", "collect_flash"),
        flash_warmup_seconds=config.get("flash_warmup_seconds", 1.0),
        flash_hold_seconds=config.get("flash_hold_seconds", 0.35),
        flash_restore_seconds=config.get("flash_restore_seconds", 0.15),
        flash_tail_seconds=config.get("flash_tail_seconds", 0.5),
    )
    frames_tensor, color_tensor, physical_tensor, _ = dataset.process_video(args.video_path, txt_path)
    if frames_tensor.numel() == 0:
        raise RuntimeError(f"无法从视频中读取有效帧: {args.video_path}")

    window_size = int(args.window_size or config.get("eval_window_size", config.get("window_size", 256)))
    window_stride = int(args.window_stride or config.get("eval_window_stride", config.get("window_stride", 128)))
    window_fusion = args.window_fusion or config.get("window_fusion", "quality_lower_trimmed_mean")
    window_trim_ratio = float(args.window_trim_ratio if args.window_trim_ratio is not None else config.get("window_trim_ratio", 0.2))
    window_min_quality = float(args.window_min_quality if args.window_min_quality is not None else config.get("window_min_quality", 0.05))
    ranges = iter_window_ranges(int(frames_tensor.shape[0]), window_size, window_stride)
    probabilities = []
    qualities = []
    with torch.no_grad():
        for start, end in ranges:
            frames = frames_tensor[start:end].unsqueeze(0).to(device)
            colors = color_tensor[start:end].unsqueeze(0).to(device)
            physical = physical_tensor[start:end].unsqueeze(0).to(device)
            padding_mask = make_window_mask(end - start, device)
            outputs = model(frames, colors, physical, padding_mask)
            probabilities.append(float(torch.sigmoid(outputs["logits"]).detach().cpu().item()))
            qualities.append(estimate_window_quality(frames))

    probability = fuse_window_probabilities(
        probabilities,
        qualities,
        method=window_fusion,
        trim_ratio=window_trim_ratio,
        min_quality=window_min_quality,
    )

    prediction = "live" if probability >= threshold else "spoof"
    print(
        json.dumps(
            {
                "video_path": args.video_path,
                "num_frames": int(frames_tensor.shape[0]),
                "window_size": window_size,
                "window_stride": window_stride,
                "window_fusion": window_fusion,
                "window_trim_ratio": window_trim_ratio,
                "num_windows": len(ranges),
                "mean_window_quality": round(float(np.mean(qualities)) if qualities else 0.0, 6),
                "min_window_quality": round(float(np.min(qualities)) if qualities else 0.0, 6),
                "probability_live": round(probability, 6),
                "threshold": round(threshold, 6),
                "prediction": prediction,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flash liveness training and inference V3 (CDC/FFT/physical/depth enhanced)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate the liveness model")
    train_parser.add_argument("--data-root", required=True, help="数据根目录")
    train_parser.add_argument(
        "--dataset-media",
        choices=["videos", "images", "all"],
        default="videos",
        help="manifest.tsv 归档输入时使用的视频/图片范围。默认 videos，优先训练炫彩时序样本。",
    )
    train_parser.add_argument(
        "--require-color-txt",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="仅使用带同名颜色 txt 的视频。默认关闭，缺失 txt 时按 missing-color-protocol 补齐颜色标签。",
    )
    train_parser.add_argument(
        "--missing-color-protocol",
        choices=sorted(FLASH_COLOR_PROTOCOLS),
        default="collect_flash",
        help="视频缺少同名 txt 时的颜色协议。collect_flash 会复用 collect_flash_liveness_video.py 的三色炫光时序。",
    )
    train_parser.add_argument("--flash-warmup-seconds", type=float, default=1.0, help="生成 collect_flash 缺省颜色标签时的原色视频预热秒数。")
    train_parser.add_argument("--flash-hold-seconds", type=float, default=0.35, help="生成 collect_flash 缺省颜色标签时每种颜色保持秒数。")
    train_parser.add_argument("--flash-restore-seconds", type=float, default=0.15, help="生成 collect_flash 缺省颜色标签时每次闪光后的原色视频恢复秒数。")
    train_parser.add_argument("--flash-tail-seconds", type=float, default=0.5, help="生成 collect_flash 缺省颜色标签时的原色视频收尾秒数。")
    train_parser.add_argument("--output-dir", default="./flash_liveness_runs", help="训练输出目录")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=1)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--prefetch-factor", type=int, default=4, help="DataLoader 每个 worker 预取 batch 数，仅 num_workers>0 时生效。")
    train_parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True, help="启用 DataLoader pin_memory，加速 CPU 到 GPU 拷贝。")
    train_parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True, help="保持 DataLoader workers 跨 epoch 存活，减少反复启动开销。")
    train_parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True, help="CUDA 下启用 AMP 混合精度训练以提升吞吐。")
    train_parser.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=True, help="输入尺寸固定时启用 cuDNN benchmark。")
    train_parser.add_argument("--num-frames", type=int, default=0, help="V2 中该参数已不再抽帧，训练默认读取视频全部可解码帧。")
    train_parser.add_argument("--frame-stride", type=int, default=1, help="V2 顺序读帧时每隔多少帧保留一帧，1 表示不跳帧。")
    train_parser.add_argument("--max-train-frames", type=int, default=0, help="训练时每个视频最多保留多少帧，0 表示不限制。")
    train_parser.add_argument("--max-eval-frames", type=int, default=0, help="验证/测试时每个视频最多保留多少帧，0 表示不限制。")
    train_parser.add_argument("--window-size", type=int, default=256, help="V3_1 滑动窗口训练长度。0 表示整段一次前向，不推荐长视频使用。")
    train_parser.add_argument("--window-stride", type=int, default=128, help="V3_1 滑动窗口训练步长。0 表示不重叠。")
    train_parser.add_argument("--eval-window-size", type=int, default=256, help="V3_1 验证/测试滑动窗口长度。")
    train_parser.add_argument("--eval-window-stride", type=int, default=128, help="V3_1 验证/测试滑动窗口步长。")
    train_parser.add_argument(
        "--window-fusion",
        choices=[
            "mean",
            "trimmed_mean",
            "quality_trimmed_mean",
            "lower_trimmed_mean",
            "quality_lower_trimmed_mean",
            "low_percentile",
            "min",
            "median",
        ],
        default="quality_lower_trimmed_mean",
        help="窗口概率到视频概率的融合方式。quality_lower_trimmed_mean 更偏向保留低 live 可疑窗口。",
    )
    train_parser.add_argument("--window-trim-ratio", type=float, default=0.2, help="稳健融合比例。legacy trimmed_mean 丢弃低 live；lower_trimmed_mean 丢弃高 live。")
    train_parser.add_argument("--window-min-quality", type=float, default=0.05, help="质量加权融合时窗口最低权重。")
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--use-physical-features", action=argparse.BooleanOptionalAction, default=True, help="V3 启用 flash_physical_features.py 中的频域/闪光/rPPG token。")
    train_parser.add_argument("--cdc-theta", type=float, default=0.7, help="Central Difference Convolution 的 theta。")
    train_parser.add_argument("--lambda-depth", type=float, default=0.05, help="pseudo-depth MSE 辅助损失权重。pseudo-depth 是弱监督，默认保持偏低。")
    train_parser.add_argument("--lambda-contrast", type=float, default=0.1, help="contrast depth loss 辅助损失权重。")
    train_parser.add_argument("--lambda-fft", type=float, default=0.05, help="FFT map 辅助损失权重。")
    train_parser.add_argument("--aux-loss-max-ratio", type=float, default=0.35, help="单个加权辅助 loss 相对 cls loss 的最大比例，<=0 表示不限制。")
    train_parser.add_argument("--aux-loss-total-max-ratio", type=float, default=0.7, help="所有加权辅助 loss 总和相对 cls loss 的最大比例，<=0 表示不限制。")
    train_parser.add_argument("--yaw-augment-prob", type=float, default=0.35, help="训练时对整段人脸序列应用轻量左右 yaw 透视增强的概率。验证/测试不启用。")
    train_parser.add_argument("--yaw-augment-max-ratio", type=float, default=0.10, help="左右 yaw 透视增强最大水平压缩比例，不做上下翻转或左右镜像。")
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-2)
    train_parser.add_argument("--log-interval", type=int, default=20, help="每隔多少个 batch 打印一次训练进度。")
    train_parser.add_argument("--log-seconds", type=float, default=60.0, help="每隔多少秒打印一次 batch 训练进度，0 表示关闭按时间打印。")
    train_parser.add_argument(
        "--pos-weight",
        default="auto",
        help="BCE 正类权重。默认 auto，会按 train 中 spoof/live 比例自动计算。",
    )
    train_parser.add_argument(
        "--balanced-train-sampler",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="训练集使用类均衡采样。启用且 pos-weight=auto 时，会自动使用 pos_weight=1.0，避免过采样和 BCE 权重双重补偿。",
    )
    train_parser.add_argument(
        "--resume-checkpoint",
        default=None,
        help="可选。传入 last/best checkpoint 路径后，从该权重和优化器状态继续训练。",
    )
    train_parser.add_argument("--val-ratio", type=float, default=0.2)
    train_parser.add_argument("--test-ratio", type=float, default=0.1)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--device", default=None)
    train_parser.add_argument(
        "--multi-gpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="CUDA 下可见 GPU 数大于 1 时自动使用 nn.DataParallel。用 CUDA_VISIBLE_DEVICES 控制具体显卡。",
    )
    train_parser.add_argument("--detector-model", default=None, help="YOLOv7 人脸检测权重路径")
    train_parser.add_argument("--detector-device", default=None)
    train_parser.add_argument("--detector-conf", type=float, default=0.5)
    train_parser.add_argument("--detector-iou", type=float, default=0.5)
    train_parser.add_argument("--embed-dim", type=int, default=512)
    train_parser.add_argument("--num-heads", type=int, default=8)
    train_parser.add_argument("--num-layers", type=int, default=2)
    train_parser.add_argument(
        "--use-imagenet-pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="启用 ResNet18 ImageNet 预训练权重。若提供本地权重路径则优先从本地加载。",
    )
    train_parser.add_argument(
        "--imagenet-pretrained-path",
        default="/supercloud/llm-code/scc/scc/Liveness_Detection/resnet18-f37072fd.pth",
        help="本地 ResNet18 ImageNet 预训练权重路径。默认使用项目根目录下的 resnet18-f37072fd.pth。",
    )
    train_parser.set_defaults(func=train_model)

    infer_parser = subparsers.add_parser("infer", help="Run inference on a single video")
    infer_parser.add_argument("--checkpoint", required=True, help="训练得到的模型权重")
    infer_parser.add_argument("--video-path", required=True, help="待推理视频路径")
    infer_parser.add_argument("--txt-path", default=None, help="可选。与视频对应的逐帧颜色 txt 路径。默认取同名 txt。")
    infer_parser.add_argument("--threshold", type=float, default=None, help="可选阈值，默认使用训练保存的最佳阈值")
    infer_parser.add_argument("--window-size", type=int, default=0, help="滑动窗口推理长度，默认读取 checkpoint 中 eval_window_size。")
    infer_parser.add_argument("--window-stride", type=int, default=0, help="滑动窗口推理步长，默认读取 checkpoint 中 eval_window_stride。")
    infer_parser.add_argument(
        "--window-fusion",
        choices=[
            "mean",
            "trimmed_mean",
            "quality_trimmed_mean",
            "lower_trimmed_mean",
            "quality_lower_trimmed_mean",
            "low_percentile",
            "min",
            "median",
        ],
        default=None,
        help="窗口概率到视频概率的融合方式，默认读取 checkpoint 配置。",
    )
    infer_parser.add_argument("--window-trim-ratio", type=float, default=None, help="稳健融合比例。legacy trimmed_mean 丢弃低 live；lower_trimmed_mean 丢弃高 live。")
    infer_parser.add_argument("--window-min-quality", type=float, default=None, help="质量加权融合时窗口最低权重。")
    infer_parser.add_argument("--device", default=None)
    infer_parser.add_argument("--detector-model", default=None, help="YOLOv7 人脸检测权重路径")
    infer_parser.add_argument("--detector-device", default=None)
    infer_parser.add_argument("--detector-conf", type=float, default=0.5)
    infer_parser.add_argument("--detector-iou", type=float, default=0.5)
    infer_parser.set_defaults(func=predict_video)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
