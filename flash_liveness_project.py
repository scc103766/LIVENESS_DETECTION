from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models

if hasattr(cv2, "setLogLevel"):
    try:
        cv2.setLogLevel(0)
    except Exception:
        pass


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
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


def collect_samples_from_label_dirs(root_dir: Path) -> list[tuple[str, int]]:
    samples: list[tuple[str, int]] = []
    if not root_dir.exists():
        return samples

    for child in sorted(root_dir.iterdir()):
        if not child.is_dir():
            continue
        label = infer_label_from_dir_name(child.name)
        if label is None:
            continue
        for video_path in sorted(child.rglob("*")):
            if video_path.is_file() and video_path.suffix.lower() in VIDEO_EXTENSIONS:
                samples.append((str(video_path), label))
    return samples


def stratified_split(
    samples: list[tuple[str, int]],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> dict[str, list[tuple[str, int]]]:
    rng = random.Random(seed)
    grouped: dict[int, list[tuple[str, int]]] = {0: [], 1: []}
    for sample in samples:
        grouped[sample[1]].append(sample)

    for group in grouped.values():
        rng.shuffle(group)

    split_result = {"train": [], "val": [], "test": []}
    for label, group in grouped.items():
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
        split_result[split_name] = sorted(split_result[split_name], key=lambda item: item[0])
    return split_result


def discover_dataset_splits(
    data_root: str,
    val_ratio: float = 0.2,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> dict[str, list[tuple[str, int]]]:
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

    all_samples = collect_samples_from_label_dirs(root)
    if not all_samples:
        raise ValueError(
            "未找到可用视频。请使用 data_root/live|spoof 或 data_root/train/live|spoof 的目录结构。"
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


class FlashLivenessDataset(Dataset):
    def __init__(
        self,
        samples: list[tuple[str, int]],
        num_frames: int = 16,
        transform: bool = False,
        preprocessor: FacePreprocessor | None = None,
        target_size: tuple[int, int] = (224, 224),
        corrupted_sample_recorder: CorruptedSampleRecorder | None = None,
    ) -> None:
        self.samples = samples
        self.num_frames = num_frames
        self.transform = transform
        self.preprocessor = preprocessor or FacePreprocessor(target_size=target_size)
        self.target_size = target_size
        self.corrupted_sample_recorder = corrupted_sample_recorder

    def __len__(self) -> int:
        return len(self.samples)

    def _read_frame_with_fallback(
        self,
        cap: cv2.VideoCapture,
        frame_idx: int,
        total_frames: int,
    ) -> np.ndarray | None:
        candidate_indices = [frame_idx]
        for offset in range(1, 4):
            prev_idx = max(frame_idx - offset, 0)
            next_idx = min(frame_idx + offset, max(total_frames - 1, 0))
            if prev_idx not in candidate_indices:
                candidate_indices.append(prev_idx)
            if next_idx not in candidate_indices:
                candidate_indices.append(next_idx)

        for candidate_idx in candidate_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, candidate_idx)
            success, frame = cap.read()
            if success and frame is not None and frame.size > 0:
                return frame
        return None

    def _read_sampled_frames(self, video_path: str) -> list[np.ndarray]:
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = sample_frame_indices(total_frames, self.num_frames)
        frames: list[np.ndarray] = []

        for frame_idx in frame_indices:
            frame = self._read_frame_with_fallback(cap, frame_idx, total_frames)
            if frame is not None:
                frames.append(frame)

        cap.release()

        if not frames:
            if self.corrupted_sample_recorder is not None:
                self.corrupted_sample_recorder.record(video_path, "no_valid_frames_decoded")
            return []

        while len(frames) < self.num_frames:
            fallback = frames[-1] if frames else np.zeros((self.target_size[1], self.target_size[0], 3), dtype=np.uint8)
            frames.append(fallback.copy())
        return frames

    def process_video(self, video_path: str) -> torch.Tensor:
        frames_bgr = self._read_sampled_frames(video_path)
        if not frames_bgr:
            return torch.empty(0)
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
        return tensor_frames

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        video_path, label = self.samples[idx]
        tensor_frames = self.process_video(video_path)
        if tensor_frames.numel() == 0:
            return None

        if self.transform and torch.rand(1).item() > 0.5:
            tensor_frames = torch.flip(tensor_frames, dims=[3])

        return tensor_frames, torch.tensor(label, dtype=torch.float32)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 50):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
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
        self.cnn_backbone[0] = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

        with torch.no_grad():
            self.cnn_backbone[0].weight[:, :3, :, :] = original_conv1.weight
            self.cnn_backbone[0].weight[:, 3:, :, :] = original_conv1.weight

        for param in self.cnn_backbone[0:5].parameters():
            param.requires_grad = False

        self.pos_encoder = PositionalEncoding(d_model=embed_dim, max_len=num_frames)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_frames, channels, height, width = x.size()
        x_cnn = x.view(batch_size * num_frames, channels, height, width)
        features = self.cnn_backbone(x_cnn)
        features = features.view(batch_size, num_frames, self.embed_dim)
        features = self.pos_encoder(features)
        trans_out = self.transformer(features)
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
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in data_loader:
            if batch is None:
                continue
            batch_videos, batch_labels = batch
            batch_videos = batch_videos.to(device)
            batch_labels = batch_labels.to(device)
            outputs = model(batch_videos)
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
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    all_labels = []
    all_probs = []
    processed_items = 0
    valid_batches = 0
    dataset_size = len(getattr(data_loader, "dataset", []))

    for batch_idx, batch in enumerate(data_loader, start=1):
        if batch is None:
            continue
        batch_videos, batch_labels = batch
        batch_videos = batch_videos.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()
        outputs = model(batch_videos)
        loss = criterion(outputs, batch_labels)
        loss.backward()
        optimizer.step()

        valid_batches += 1
        total_loss += loss.item()
        probs = torch.sigmoid(outputs)
        all_labels.append(batch_labels.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())
        processed_items += int(batch_labels.size(0))

        if valid_batches == 1 or batch_idx % log_interval == 0:
            avg_loss = total_loss / max(valid_batches, 1)
            print(
                f"Epoch [{epoch}/{total_epochs}] "
                f"Batch [{batch_idx}/{len(data_loader)}] "
                f"processed_items={processed_items}/{dataset_size} "
                f"batch_loss={loss.item():.4f} avg_loss={avg_loss:.4f}"
            )

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


def print_split_stats(split_name: str, samples: list[tuple[str, int]]) -> None:
    live_count = sum(1 for _, label in samples if label == 1)
    spoof_count = sum(1 for _, label in samples if label == 0)
    print(f"{split_name}: total={len(samples)}, live={live_count}, spoof={spoof_count}")


def summarize_split_counts(samples: list[tuple[str, int]]) -> dict[str, int]:
    live_count = sum(1 for _, label in samples if label == 1)
    spoof_count = sum(1 for _, label in samples if label == 0)
    return {
        "total": len(samples),
        "live": live_count,
        "spoof": spoof_count,
    }


def resolve_pos_weight(train_samples: list[tuple[str, int]], pos_weight_arg: str) -> float:
    if pos_weight_arg != "auto":
        return float(pos_weight_arg)

    live_count = sum(1 for _, label in train_samples if label == 1)
    spoof_count = sum(1 for _, label in train_samples if label == 0)
    if live_count == 0:
        return 1.0
    return max(spoof_count / live_count, 1.0)


def save_run_config(
    output_dir: Path,
    args: argparse.Namespace,
    split_counts: dict[str, dict[str, int]],
    pos_weight_value: float,
) -> None:
    args_dict = vars(args).copy()
    args_dict.pop("func", None)
    args_dict["resolved_pos_weight"] = pos_weight_value
    payload = {
        "command": " ".join(str(arg) for arg in os.sys.argv),
        "arguments": args_dict,
        "split_counts": split_counts,
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

    videos = torch.stack([item[0] for item in valid_batch], dim=0)
    labels = torch.stack([item[1] for item in valid_batch], dim=0)
    return videos, labels


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
    model.load_state_dict(checkpoint["model_state_dict"])

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
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    splits = discover_dataset_splits(
        args.data_root,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    split_counts = {}
    for split_name in ("train", "val", "test"):
        print_split_stats(split_name, splits.get(split_name, []))
        split_counts[split_name] = summarize_split_counts(splits.get(split_name, []))

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
        num_frames=args.num_frames,
        transform=True,
        preprocessor=preprocessor,
        target_size=(args.image_size, args.image_size),
        corrupted_sample_recorder=corrupted_sample_recorder,
    )
    val_dataset = FlashLivenessDataset(
        splits["val"],
        num_frames=args.num_frames,
        transform=False,
        preprocessor=preprocessor,
        target_size=(args.image_size, args.image_size),
        corrupted_sample_recorder=corrupted_sample_recorder,
    )
    test_dataset = FlashLivenessDataset(
        splits["test"],
        num_frames=args.num_frames,
        transform=False,
        preprocessor=preprocessor,
        target_size=(args.image_size, args.image_size),
        corrupted_sample_recorder=corrupted_sample_recorder,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_skip_none,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_skip_none,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_skip_none,
    )

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    model = build_model_from_args(args).to(device)
    pos_weight_value = resolve_pos_weight(splits["train"], args.pos_weight)
    print(f"Using BCE positive weight for live class: {pos_weight_value:.6f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_value], device=device))
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    save_run_config(output_dir, args, split_counts, pos_weight_value)

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
        )
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        test_metrics = evaluate_model(model, test_loader, criterion, device, threshold=val_metrics["threshold"])

        print(
            f"Epoch [{epoch}/{args.epochs}] "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} val_auc={val_metrics['auc']:.4f} val_eer={val_metrics['eer']:.4f} | "
            f"test_acc={test_metrics['accuracy']:.4f} test_auc={test_metrics['auc']:.4f} test_eer={test_metrics['eer']:.4f}"
        )

        epoch_payload = {
            "epoch": epoch,
            "train_loss": round(train_metrics["loss"], 6),
            "train_accuracy": round(train_metrics["accuracy"], 6),
            "train_threshold": round(train_metrics["threshold"], 6),
            "val_loss": round(val_metrics["loss"], 6),
            "val_accuracy": round(val_metrics["accuracy"], 6),
            "val_auc": round(val_metrics["auc"], 6),
            "val_eer": round(val_metrics["eer"], 6),
            "val_acer": round(val_metrics["acer"], 6),
            "val_threshold": round(val_metrics["threshold"], 6),
            "test_loss": round(test_metrics["loss"], 6),
            "test_accuracy": round(test_metrics["accuracy"], 6),
            "test_auc": round(test_metrics["auc"], 6),
            "test_eer": round(test_metrics["eer"], 6),
            "test_acer": round(test_metrics["acer"], 6),
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

    preprocessor = FacePreprocessor(
        detector_model_path=args.detector_model,
        detector_device=args.detector_device,
        target_size=tuple(config["target_size"]),
        conf_threshold=args.detector_conf,
        iou_threshold=args.detector_iou,
    )
    dataset = FlashLivenessDataset(
        samples=[(args.video_path, 0)],
        num_frames=config["num_frames"],
        transform=False,
        preprocessor=preprocessor,
        target_size=tuple(config["target_size"]),
    )
    frames_tensor = dataset.process_video(args.video_path).unsqueeze(0).to(device)

    with torch.no_grad():
        logit = model(frames_tensor)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Flash liveness training and inference")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate the liveness model")
    train_parser.add_argument("--data-root", required=True, help="数据根目录")
    train_parser.add_argument("--output-dir", default="./flash_liveness_runs", help="训练输出目录")
    train_parser.add_argument("--epochs", type=int, default=20)
    train_parser.add_argument("--batch-size", type=int, default=2)
    train_parser.add_argument("--num-workers", type=int, default=0)
    train_parser.add_argument("--num-frames", type=int, default=16)
    train_parser.add_argument("--image-size", type=int, default=224)
    train_parser.add_argument("--learning-rate", type=float, default=1e-4)
    train_parser.add_argument("--weight-decay", type=float, default=1e-2)
    train_parser.add_argument("--log-interval", type=int, default=20, help="每隔多少个 batch 打印一次训练进度。")
    train_parser.add_argument(
        "--pos-weight",
        default="auto",
        help="BCE 正类权重。默认 auto，会按 train 中 spoof/live 比例自动计算。",
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
    infer_parser.add_argument("--threshold", type=float, default=None, help="可选阈值，默认使用训练保存的最佳阈值")
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
