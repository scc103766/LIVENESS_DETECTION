from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch

from flash_liveness_project import CorruptedSampleRecorder, FacePreprocessor, FlashLivenessDataset, load_checkpoint


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
FLASH_COLOR_MAP = {
    "normal": None,
    "red": np.asarray([1.0, 0.25, 0.25], dtype=np.float32),
    "green": np.asarray([0.25, 1.0, 0.25], dtype=np.float32),
    "blue": np.asarray([0.25, 0.25, 1.0], dtype=np.float32),
}


def parse_flash_colors(value: str) -> list[str]:
    colors = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [color for color in colors if color not in FLASH_COLOR_MAP]
    if unknown:
        raise ValueError(f"未知闪光颜色: {unknown}，支持: {sorted(FLASH_COLOR_MAP)}")
    if not colors:
        raise ValueError("--flash-colors 至少需要一个颜色")
    return colors


def infer_label_from_name(path: Path, real_token: str) -> tuple[int, str]:
    label_id = 1 if real_token.lower() in path.stem.lower() else 0
    return label_id, "live" if label_id == 1 else "spoof"


def detect_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    raise ValueError(f"unsupported_media_extension: {suffix}")


def build_static_frame_sequence(
    face_rgb: np.ndarray,
    num_frames: int,
    mode: str,
    flash_group_size: int,
    flash_colors: list[str],
    flash_alpha: float,
) -> np.ndarray:
    if mode == "copy":
        return np.repeat(face_rgb[None, ...], repeats=num_frames, axis=0)

    if flash_group_size <= 0:
        raise ValueError("--flash-group-size 必须大于 0")

    flash_alpha = float(np.clip(flash_alpha, 0.0, 1.0))
    frames = []
    for frame_index in range(num_frames):
        group_index = frame_index // flash_group_size
        color_name = flash_colors[group_index % len(flash_colors)]
        color = FLASH_COLOR_MAP[color_name]
        if color is None:
            flashed = face_rgb
        else:
            flashed = face_rgb * (1.0 - flash_alpha) + color.reshape(1, 1, 3) * flash_alpha
        frames.append(np.clip(flashed, 0.0, 1.0))
    return np.asarray(frames, dtype=np.float32)


def image_bgr_to_liveness_tensor(
    image_bgr: np.ndarray,
    preprocessor: FacePreprocessor,
    num_frames: int,
    target_size: tuple[int, int],
    static_sequence_mode: str,
    flash_group_size: int,
    flash_colors: list[str],
    flash_alpha: float,
    prefix: str,
) -> torch.Tensor:
    if image_bgr is None or image_bgr.size == 0:
        raise RuntimeError("image_decode_failed")

    processed = preprocessor.preprocess_frames([image_bgr], prefix=prefix)
    if not processed:
        raise RuntimeError("image_preprocess_failed")

    face_bgr = processed[0]
    if face_bgr.shape[:2] != (target_size[1], target_size[0]):
        face_bgr = cv2.resize(face_bgr, target_size)

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    frames = build_static_frame_sequence(
        face_rgb=face_rgb,
        num_frames=num_frames,
        mode=static_sequence_mode,
        flash_group_size=flash_group_size,
        flash_colors=flash_colors,
        flash_alpha=flash_alpha,
    )

    diff_frames = np.zeros_like(frames)
    if num_frames > 1:
        diff_frames[1:] = frames[1:] - frames[:-1]
        diff_frames[0] = diff_frames[1]

    multi_modal_frames = np.concatenate([frames, diff_frames], axis=-1)
    return torch.from_numpy(multi_modal_frames).permute(0, 3, 1, 2).float()


def image_path_to_liveness_tensor(
    image_path: str | Path,
    preprocessor: FacePreprocessor,
    num_frames: int,
    target_size: tuple[int, int],
    static_sequence_mode: str,
    flash_group_size: int,
    flash_colors: list[str],
    flash_alpha: float,
) -> torch.Tensor:
    path = Path(image_path)
    image_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return image_bgr_to_liveness_tensor(
        image_bgr=image_bgr,
        preprocessor=preprocessor,
        num_frames=num_frames,
        target_size=target_size,
        static_sequence_mode=static_sequence_mode,
        flash_group_size=flash_group_size,
        flash_colors=flash_colors,
        flash_alpha=flash_alpha,
        prefix=path.stem,
    )


def image_bytes_to_liveness_tensor(
    payload: bytes,
    filename_hint: str,
    preprocessor: FacePreprocessor,
    num_frames: int,
    target_size: tuple[int, int],
    static_sequence_mode: str,
    flash_group_size: int,
    flash_colors: list[str],
    flash_alpha: float,
) -> torch.Tensor:
    image_array = np.frombuffer(payload, dtype=np.uint8)
    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image_bgr_to_liveness_tensor(
        image_bgr=image_bgr,
        preprocessor=preprocessor,
        num_frames=num_frames,
        target_size=target_size,
        static_sequence_mode=static_sequence_mode,
        flash_group_size=flash_group_size,
        flash_colors=flash_colors,
        flash_alpha=flash_alpha,
        prefix=Path(filename_hint).stem or "image",
    )


def save_debug_grid(output_path: Path, tensor_frames: torch.Tensor) -> None:
    rgb_frames = tensor_frames[:, :3].permute(0, 2, 3, 1).cpu().numpy()
    rgb_frames = np.clip(rgb_frames * 255.0, 0, 255).astype(np.uint8)
    bgr_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in rgb_frames]
    if not bgr_frames:
        return

    tile_h, tile_w = bgr_frames[0].shape[:2]
    cols = min(4, len(bgr_frames))
    rows = int(np.ceil(len(bgr_frames) / cols))
    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for index, frame in enumerate(bgr_frames):
        row = index // cols
        col = index % cols
        canvas[row * tile_h:(row + 1) * tile_h, col * tile_w:(col + 1) * tile_w] = frame
    cv2.imwrite(str(output_path), canvas)


def predict_from_tensor(
    model: torch.nn.Module,
    tensor_frames: torch.Tensor,
    device: torch.device,
    threshold: float,
) -> dict:
    with torch.no_grad():
        logits = model(tensor_frames.unsqueeze(0).to(device))
        probability = float(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)[0])
    prediction_id = int(probability >= threshold)
    return {
        "probability_live_raw": probability,
        "score_live": probability,
        "threshold": float(threshold),
        "prediction_id": prediction_id,
        "prediction_name": "live" if prediction_id == 1 else "spoof",
    }


def load_flash_liveness_bundle(
    checkpoint_path: str,
    device_spec: str | None,
    detector_model: str | None,
    detector_device: str | None,
    detector_conf: float,
    detector_iou: float,
    threshold_override: float | None = None,
    corrupted_record_path: str | Path | None = None,
) -> dict:
    device = torch.device(device_spec if device_spec else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = load_checkpoint(checkpoint_path, device)
    config = checkpoint["config"]
    threshold = threshold_override if threshold_override is not None else float(checkpoint.get("threshold", 0.5))
    target_size = tuple(config["target_size"])
    preprocessor = FacePreprocessor(
        detector_model_path=detector_model,
        detector_device=detector_device,
        target_size=target_size,
        conf_threshold=detector_conf,
        iou_threshold=detector_iou,
    )
    dataset = FlashLivenessDataset(
        samples=[],
        num_frames=int(config["num_frames"]),
        transform=False,
        preprocessor=preprocessor,
        target_size=target_size,
        corrupted_sample_recorder=CorruptedSampleRecorder(corrupted_record_path) if corrupted_record_path else None,
    )
    return {
        "device": device,
        "model": model,
        "checkpoint": checkpoint,
        "threshold": threshold,
        "num_frames": int(config["num_frames"]),
        "target_size": target_size,
        "preprocessor": preprocessor,
        "dataset": dataset,
    }


def tensor_from_media_path(
    media_path: str | Path,
    bundle: dict,
    static_sequence_mode: str,
    flash_group_size: int,
    flash_colors: list[str],
    flash_alpha: float,
) -> tuple[torch.Tensor, str]:
    path = Path(media_path)
    media_type = detect_media_type(path)
    if media_type == "video":
        tensor = bundle["dataset"].process_video(str(path))
        if tensor.numel() == 0:
            raise RuntimeError("no_valid_frames_decoded")
        return tensor, media_type
    tensor = image_path_to_liveness_tensor(
        image_path=path,
        preprocessor=bundle["preprocessor"],
        num_frames=bundle["num_frames"],
        target_size=bundle["target_size"],
        static_sequence_mode=static_sequence_mode,
        flash_group_size=flash_group_size,
        flash_colors=flash_colors,
        flash_alpha=flash_alpha,
    )
    return tensor, media_type


def tensor_from_uploaded_media(
    payload: bytes,
    filename_hint: str,
    bundle: dict,
    static_sequence_mode: str,
    flash_group_size: int,
    flash_colors: list[str],
    flash_alpha: float,
) -> tuple[torch.Tensor, str]:
    path = Path(filename_hint or "upload.bin")
    media_type = detect_media_type(path)
    if media_type == "image":
        tensor = image_bytes_to_liveness_tensor(
            payload=payload,
            filename_hint=filename_hint,
            preprocessor=bundle["preprocessor"],
            num_frames=bundle["num_frames"],
            target_size=bundle["target_size"],
            static_sequence_mode=static_sequence_mode,
            flash_group_size=flash_group_size,
            flash_colors=flash_colors,
            flash_alpha=flash_alpha,
        )
        return tensor, media_type

    suffix = path.suffix if path.suffix else ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
        temp_file.write(payload)
        temp_file.flush()
        tensor = bundle["dataset"].process_video(temp_file.name)
    if tensor.numel() == 0:
        raise RuntimeError("no_valid_frames_decoded")
    return tensor, media_type
