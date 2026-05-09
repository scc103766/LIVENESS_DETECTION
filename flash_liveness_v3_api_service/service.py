from __future__ import annotations

import json
import math
import os
import shutil
import sys
import time
import uuid
import zipfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flash_liveness_project_v3 import (  # noqa: E402
    CNNTransformerLiveness,
    FacePreprocessor,
    FlashLivenessDataset,
    load_checkpoint_state_dict,
)
from server_storage_manager import StorageRetentionConfig, StorageRetentionManager  # noqa: E402


DEFAULT_CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "weights/flash_liveness_v3_fixed_protocol/best_flash_liveness_model.pth"
)
DEFAULT_OUTPUT_DIR = Path("/raid/scc/data/liveness_v3_server_result")

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
ARCHIVE_EXTENSIONS = {".zip"}
FUSION_METHODS = {
    "mean",
    "trimmed_mean",
    "quality_trimmed_mean",
    "lower_trimmed_mean",
    "quality_lower_trimmed_mean",
    "low_percentile",
    "min",
    "median",
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


@contextmanager
def suppress_native_stderr(enabled: bool):
    if not enabled:
        yield
        return

    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        os.dup2(devnull.fileno(), stderr_fd)
        try:
            yield
        finally:
            os.dup2(saved_fd, stderr_fd)
            os.close(saved_fd)


def copy_file(src: Path, dst: Path) -> Path:
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return dst


def infer_upload_type(file_name: str) -> str:
    suffix = Path(file_name).suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in ARCHIVE_EXTENSIONS:
        return "archive"
    raise ValueError(
        "unsupported_file_type:"
        f"{file_name}. V3 service accepts video files or .zip archives containing video and optional txt."
    )


def load_v3_checkpoint(checkpoint_path: str | Path, device: torch.device) -> tuple[CNNTransformerLiveness, dict[str, Any]]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    config = checkpoint["config"]
    model = CNNTransformerLiveness(
        num_frames=config["num_frames"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_layers=config["num_layers"],
        physical_dim=config.get("physical_dim", 19),
        cdc_theta=config.get("cdc_theta", 0.7),
        lambda_depth=config.get("lambda_depth", 0.1),
        lambda_contrast=config.get("lambda_contrast", 0.1),
        lambda_fft=config.get("lambda_fft", 0.05),
    )
    load_checkpoint_state_dict(model, checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    checkpoint.pop("optimizer_state_dict", None)
    checkpoint.pop("model_state_dict", None)
    return model, checkpoint


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


class FlashLivenessV3ApiService:
    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT_PATH,
        output_dir: str | Path = DEFAULT_OUTPUT_DIR,
        threshold: float | None = None,
        device: str | None = None,
        inference_mode: str = "window",
        max_seq_len: int = 512,
        window_size: int = 256,
        window_stride: int = 128,
        window_fusion: str = "quality_lower_trimmed_mean",
        window_trim_ratio: float = 0.2,
        window_min_quality: float = 0.05,
        require_color_txt: bool = False,
        detector_model_path: str | None = None,
        detector_device: str | None = None,
        detector_conf: float = 0.5,
        detector_iou: float = 0.5,
        show_decoder_warnings: bool = False,
        return_window_details: bool = False,
        storage_max_videos: int = 2000,
        storage_backup_dir: str | Path | None = None,
        storage_cleanup_batch_size: int = 200,
        storage_retention_enabled: bool = True,
    ) -> None:
        if inference_mode not in {"window", "full_sequence"}:
            raise ValueError("inference_mode must be one of: window, full_sequence")
        if window_fusion not in FUSION_METHODS:
            raise ValueError(f"window_fusion must be one of: {sorted(FUSION_METHODS)}")

        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.output_dir = Path(output_dir).resolve()
        ensure_dir(self.output_dir)
        storage_backup_path = (
            Path(storage_backup_dir).resolve()
            if storage_backup_dir is not None
            else (self.output_dir / "_retention_backups").resolve()
        )
        self.storage_retention = StorageRetentionManager(
            StorageRetentionConfig(
                output_dir=self.output_dir,
                backup_dir=storage_backup_path,
                max_videos=int(storage_max_videos),
                cleanup_batch_size=int(storage_cleanup_batch_size),
                enabled=bool(storage_retention_enabled),
            )
        )
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        self.inference_mode = inference_mode
        self.max_seq_len = min(max(int(max_seq_len), 1), 512)
        self.window_size = min(max(int(window_size), 1), 512)
        self.window_stride = int(window_stride)
        self.window_fusion = window_fusion
        self.window_trim_ratio = float(window_trim_ratio)
        self.window_min_quality = float(window_min_quality)
        self.require_color_txt = bool(require_color_txt)
        self.show_decoder_warnings = bool(show_decoder_warnings)
        self.return_window_details = bool(return_window_details)

        self.model, self.checkpoint = load_v3_checkpoint(self.checkpoint_path, self.device)
        self.config = self.checkpoint["config"]
        self.threshold = float(threshold) if threshold is not None else float(self.checkpoint.get("threshold", 0.5))

        target_size = tuple(self.config.get("target_size", (224, 224)))
        preprocessor = FacePreprocessor(
            detector_model_path=detector_model_path,
            detector_device=detector_device,
            target_size=target_size,
            conf_threshold=detector_conf,
            iou_threshold=detector_iou,
        )
        self.dataset = FlashLivenessDataset(
            samples=[],
            transform=False,
            preprocessor=preprocessor,
            target_size=target_size,
            max_frames=0,
            frame_stride=1,
            use_physical_features=self.config.get("physical_dim", 19) > 0,
            missing_color_protocol=self.config.get("missing_color_protocol", "collect_flash"),
            flash_warmup_seconds=self.config.get("flash_warmup_seconds", 1.0),
            flash_hold_seconds=self.config.get("flash_hold_seconds", 0.35),
            flash_tail_seconds=self.config.get("flash_tail_seconds", 0.5),
        )

    def _request_dir(self, request_id: str) -> Path:
        return self.output_dir / request_id

    def _safe_extract_zip(self, archive_path: Path, extract_dir: Path) -> None:
        ensure_dir(extract_dir)
        root = extract_dir.resolve()
        with zipfile.ZipFile(archive_path) as archive:
            for member in archive.infolist():
                target = (root / member.filename).resolve()
                if root not in target.parents and target != root:
                    raise RuntimeError(f"unsafe_zip_member:{member.filename}")
                if member.is_dir():
                    ensure_dir(target)
                    continue
                ensure_dir(target.parent)
                with archive.open(member) as source, target.open("wb") as output:
                    output.write(source.read())

    def _find_video_and_txt(self, root: Path) -> tuple[Path, Path | None]:
        video_files = sorted(path for path in root.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)
        if not video_files:
            raise RuntimeError("archive_has_no_video")
        for video_path in video_files:
            same_stem_txt = video_path.with_suffix(".txt")
            if same_stem_txt.exists():
                return video_path, same_stem_txt
        txt_files = sorted(root.rglob("*.txt"))
        if len(video_files) == 1 and len(txt_files) == 1:
            return video_files[0], txt_files[0]
        return video_files[0], None

    def _process_video_tensors(self, video_path: Path, txt_path: Path | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.require_color_txt and txt_path is None:
            raise RuntimeError("missing_color_txt")
        with suppress_native_stderr(not self.show_decoder_warnings):
            frames_tensor, color_tensor, physical_tensor, _ = self.dataset.process_video(
                str(video_path),
                str(txt_path) if txt_path is not None else None,
            )
        if frames_tensor.numel() == 0:
            raise RuntimeError("no_valid_frames_decoded")
        return frames_tensor, color_tensor, physical_tensor

    def _infer_tensors(
        self,
        frames_tensor: torch.Tensor,
        color_tensor: torch.Tensor,
        physical_tensor: torch.Tensor,
    ) -> dict[str, Any]:
        num_frames = int(frames_tensor.shape[0])
        probabilities: list[float] = []
        qualities: list[float] = []

        with torch.inference_mode():
            if self.inference_mode == "full_sequence":
                end = min(num_frames, self.max_seq_len)
                frames = frames_tensor[:end].unsqueeze(0).to(self.device)
                colors = color_tensor[:end].unsqueeze(0).to(self.device)
                physical = physical_tensor[:end].unsqueeze(0).to(self.device)
                padding_mask = torch.zeros((1, end), dtype=torch.bool, device=self.device)
                outputs = self.model(frames, colors, physical, padding_mask)
                probability_live = float(torch.sigmoid(outputs["logits"]).detach().cpu().item())
                probabilities.append(probability_live)
                num_frames_used = int(end)
                return {
                    "probability_live": probability_live,
                    "num_frames": num_frames,
                    "num_frames_used": num_frames_used,
                    "truncated_to_model_max_len": int(num_frames_used < num_frames),
                    "num_windows": 1,
                    "mean_window_quality": 0.0,
                    "min_window_quality": 0.0,
                    "max_window_quality": 0.0,
                    "window_probabilities": probabilities,
                    "window_qualities": qualities,
                }

            ranges = iter_window_ranges(num_frames, self.window_size, self.window_stride)
            for start, end in ranges:
                frames = frames_tensor[start:end].unsqueeze(0).to(self.device)
                colors = color_tensor[start:end].unsqueeze(0).to(self.device)
                physical = physical_tensor[start:end].unsqueeze(0).to(self.device)
                padding_mask = torch.zeros((1, end - start), dtype=torch.bool, device=self.device)
                outputs = self.model(frames, colors, physical, padding_mask)
                probabilities.append(float(torch.sigmoid(outputs["logits"]).detach().cpu().item()))
                qualities.append(estimate_window_quality(frames))

        probability_live = fuse_window_probabilities(
            probabilities,
            qualities,
            method=self.window_fusion,
            trim_ratio=self.window_trim_ratio,
            min_quality=self.window_min_quality,
        )
        return {
            "probability_live": float(probability_live),
            "num_frames": num_frames,
            "num_frames_used": num_frames,
            "truncated_to_model_max_len": 0,
            "num_windows": len(probabilities),
            "mean_window_quality": float(np.mean(qualities)) if qualities else 0.0,
            "min_window_quality": float(np.min(qualities)) if qualities else 0.0,
            "max_window_quality": float(np.max(qualities)) if qualities else 0.0,
            "window_probabilities": probabilities,
            "window_qualities": qualities,
        }

    def _build_payload(
        self,
        request_id: str,
        started: float,
        video_path: Path,
        txt_path: Path | None,
        input_type: str,
        source_details: dict[str, Any],
    ) -> dict[str, Any]:
        frames_tensor, color_tensor, physical_tensor = self._process_video_tensors(video_path, txt_path)
        result = self._infer_tensors(frames_tensor, color_tensor, physical_tensor)
        probability_live = float(result["probability_live"])
        prediction_id = int(probability_live >= self.threshold)
        prediction_name = "live" if prediction_id == 1 else "spoof"
        payload: dict[str, Any] = {
            "request_id": request_id,
            "result": prediction_name,
            "prediction_id": prediction_id,
            "probability_live": probability_live,
            "threshold": self.threshold,
            "input_type": input_type,
            "video_path": str(video_path),
            "txt_path": str(txt_path) if txt_path is not None else "",
            "txt_source": "provided" if txt_path is not None else "generated_collect_flash_protocol",
            "checkpoint_path": str(self.checkpoint_path),
            "checkpoint_epoch": self.checkpoint.get("epoch"),
            "device": str(self.device),
            "inference_mode": self.inference_mode,
            "window_size": self.window_size if self.inference_mode == "window" else None,
            "window_stride": self.window_stride if self.inference_mode == "window" else None,
            "window_fusion": self.window_fusion if self.inference_mode == "window" else None,
            "window_trim_ratio": self.window_trim_ratio if self.inference_mode == "window" else None,
            "window_min_quality": self.window_min_quality if self.inference_mode == "window" else None,
            "max_seq_len": self.max_seq_len if self.inference_mode == "full_sequence" else None,
            "num_frames": int(result["num_frames"]),
            "num_frames_used": int(result["num_frames_used"]),
            "truncated_to_model_max_len": int(result["truncated_to_model_max_len"]),
            "num_windows": int(result["num_windows"]),
            "mean_window_quality": float(result["mean_window_quality"]),
            "min_window_quality": float(result["min_window_quality"]),
            "max_window_quality": float(result["max_window_quality"]),
            "flash_protocol": {
                "missing_color_protocol": self.config.get("missing_color_protocol", "collect_flash"),
                "warmup_seconds": self.config.get("flash_warmup_seconds", 1.0),
                "hold_seconds": self.config.get("flash_hold_seconds", 0.35),
                "tail_seconds": self.config.get("flash_tail_seconds", 0.5),
                "color_order_rgb": [[255, 20, 255], [20, 255, 20], [255, 20, 20]],
                "color_order_packed": [16717055, 1376020, 16716820],
            },
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 2),
            **source_details,
        }
        if self.return_window_details:
            payload["window_probabilities"] = [float(item) for item in result["window_probabilities"]]
            payload["window_qualities"] = [float(item) for item in result["window_qualities"]]

        metadata_path = self._request_dir(request_id) / "metadata.json"
        payload["metadata_json"] = str(metadata_path)
        save_json(metadata_path, payload)
        cleanup = self.storage_retention.enforce(exclude_request_id=request_id)
        if cleanup.get("action") == "cleanup":
            payload["storage_retention"] = cleanup
            save_json(metadata_path, payload)
        return payload

    def predict_video(
        self,
        video_path: Path,
        txt_path: Path | None = None,
        request_id: str | None = None,
        uploaded_filename: str = "",
    ) -> dict[str, Any]:
        started = time.perf_counter()
        request_id = request_id or uuid.uuid4().hex[:12]
        request_dir = self._request_dir(request_id)
        ensure_dir(request_dir)

        video_path = Path(video_path)
        stored_video = copy_file(video_path, request_dir / f"upload{video_path.suffix.lower()}")
        stored_txt: Path | None = None
        if txt_path is not None:
            stored_txt = copy_file(Path(txt_path), stored_video.with_suffix(".txt"))
        elif stored_video.with_suffix(".txt").exists():
            stored_txt = stored_video.with_suffix(".txt")

        return self._build_payload(
            request_id=request_id,
            started=started,
            video_path=stored_video,
            txt_path=stored_txt,
            input_type="video",
            source_details={
                "uploaded_filename": uploaded_filename,
                "stored_video": str(stored_video),
                "stored_txt": str(stored_txt) if stored_txt is not None else "",
            },
        )

    def predict_archive(
        self,
        archive_path: Path,
        request_id: str | None = None,
        uploaded_filename: str = "",
    ) -> dict[str, Any]:
        started = time.perf_counter()
        request_id = request_id or uuid.uuid4().hex[:12]
        request_dir = self._request_dir(request_id)
        ensure_dir(request_dir)
        stored_archive = copy_file(Path(archive_path), request_dir / "upload.zip")
        extract_dir = request_dir / "uploaded_archive"
        self._safe_extract_zip(stored_archive, extract_dir)
        video_path, txt_path = self._find_video_and_txt(extract_dir)
        return self._build_payload(
            request_id=request_id,
            started=started,
            video_path=video_path,
            txt_path=txt_path,
            input_type="archive",
            source_details={
                "uploaded_filename": uploaded_filename,
                "stored_archive": str(stored_archive),
                "extract_dir": str(extract_dir),
            },
        )

    def predict_path(
        self,
        video_path: Path,
        txt_path: Path | None = None,
        request_id: str | None = None,
    ) -> dict[str, Any]:
        started = time.perf_counter()
        request_id = request_id or uuid.uuid4().hex[:12]
        video_path = Path(video_path).resolve()
        txt_path = Path(txt_path).resolve() if txt_path is not None else None
        if not video_path.exists():
            raise RuntimeError(f"video_path_not_found:{video_path}")
        if txt_path is not None and not txt_path.exists():
            raise RuntimeError(f"txt_path_not_found:{txt_path}")
        ensure_dir(self._request_dir(request_id))
        return self._build_payload(
            request_id=request_id,
            started=started,
            video_path=video_path,
            txt_path=txt_path,
            input_type="path",
            source_details={"source_video_path": str(video_path), "source_txt_path": str(txt_path) if txt_path else ""},
        )
