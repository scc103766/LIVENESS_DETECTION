from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flash_liveness_project_v2 import (  # noqa: E402
    FacePreprocessor,
    color_int_to_feature,
    compute_auc,
    compute_binary_metrics,
    compute_eer,
    find_best_threshold,
    load_checkpoint,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
DEFAULT_FLASH_COLORS = [0xFF14FF, 0x14FF14, 0xFF1414]


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict | list) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def parse_flash_colors(value: str) -> list[int]:
    colors = []
    for item in value.split(","):
        raw = item.strip()
        if not raw:
            continue
        colors.append(int(raw, 0))
    if not colors:
        raise ValueError("--flash-colors 至少要有一个颜色值。")
    return colors


def detect_media_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    raise ValueError(f"unsupported_media_extension:{suffix}")


def collect_media(input_root: Path) -> list[Path]:
    valid_exts = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    return sorted(path for path in input_root.rglob("*") if path.is_file() and path.suffix.lower() in valid_exts)


def infer_label_from_path(media_path: Path) -> tuple[int | None, str]:
    for parent in media_path.parents:
        name = parent.name.lower()
        if name == "live":
            return 1, "path_dir_live"
        if name == "spoof":
            return 0, "path_dir_spoof"
    return None, "unresolved"


def build_color_tensor(seq_len: int, group_size: int, color_values: list[int]) -> torch.Tensor:
    features = []
    prev_color: int | None = None
    for frame_index in range(seq_len):
        color = color_values[(frame_index // group_size) % len(color_values)]
        features.append(color_int_to_feature(color, prev_color))
        prev_color = color
    return torch.from_numpy(np.asarray(features, dtype=np.float32)).float()


def frame_sequence_to_tensor(processed_frames_bgr: list[np.ndarray], target_size: tuple[int, int]) -> torch.Tensor:
    frames_rgb = []
    for frame in processed_frames_bgr:
        if frame.shape[:2] != (target_size[1], target_size[0]):
            frame = cv2.resize(frame, target_size)
        frames_rgb.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    frames = np.asarray(frames_rgb, dtype=np.float32) / 255.0
    diff_frames = np.zeros_like(frames)
    if len(frames) > 1:
        diff_frames[1:] = frames[1:] - frames[:-1]
        diff_frames[0] = diff_frames[1]
    multi_modal_frames = np.concatenate([frames, diff_frames], axis=-1)
    return torch.from_numpy(multi_modal_frames).permute(0, 3, 1, 2).float()


def load_video_frames(
    video_path: Path,
    preprocessor: FacePreprocessor,
    target_size: tuple[int, int],
    max_frames: int,
    frame_stride: int,
) -> torch.Tensor:
    cap = cv2.VideoCapture(str(video_path))
    frames_bgr: list[np.ndarray] = []
    frame_index = 0
    while True:
        success, frame = cap.read()
        if not success:
            break
        if frame_index % frame_stride == 0 and frame is not None and frame.size > 0:
            frames_bgr.append(frame.copy())
            if max_frames > 0 and len(frames_bgr) >= max_frames:
                break
        frame_index += 1
    cap.release()
    if not frames_bgr:
        raise RuntimeError("no_valid_frames_decoded")
    processed = preprocessor.preprocess_frames(frames_bgr, prefix=video_path.stem)
    if not processed:
        raise RuntimeError("video_preprocess_failed")
    return frame_sequence_to_tensor(processed, target_size=target_size)


def load_image_frames(
    image_path: Path,
    preprocessor: FacePreprocessor,
    target_size: tuple[int, int],
    num_frames: int,
    flash_colors: list[int],
    flash_alpha: float,
    flash_group_size: int,
) -> torch.Tensor:
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None or image_bgr.size == 0:
        raise RuntimeError("image_decode_failed")
    processed = preprocessor.preprocess_frames([image_bgr], prefix=image_path.stem)
    if not processed:
        raise RuntimeError("image_preprocess_failed")
    face_bgr = processed[0]
    if face_bgr.shape[:2] != (target_size[1], target_size[0]):
        face_bgr = cv2.resize(face_bgr, target_size)
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    frames = []
    for frame_index in range(num_frames):
        color = flash_colors[(frame_index // flash_group_size) % len(flash_colors)]
        rgb = np.asarray(
            [((color >> 16) & 0xFF), ((color >> 8) & 0xFF), (color & 0xFF)],
            dtype=np.float32,
        ) / 255.0
        flashed = face_rgb * (1.0 - flash_alpha) + rgb.reshape(1, 1, 3) * flash_alpha
        frames.append(np.clip(flashed, 0.0, 1.0))

    diff_frames = np.zeros_like(frames)
    frames_np = np.asarray(frames, dtype=np.float32)
    if len(frames_np) > 1:
        diff_frames[1:] = frames_np[1:] - frames_np[:-1]
        diff_frames[0] = diff_frames[1]
    multi_modal_frames = np.concatenate([frames_np, diff_frames], axis=-1)
    return torch.from_numpy(multi_modal_frames).permute(0, 3, 1, 2).float()


def evaluate_rows(rows: list[dict], threshold: float | None = None) -> dict | None:
    labeled_rows = [row for row in rows if row["status"] == "ok" and row["label_id"] != ""]
    if not labeled_rows:
        return None

    labels = np.asarray([int(row["label_id"]) for row in labeled_rows], dtype=np.int32)
    probs = np.asarray([float(row["probability_live"]) for row in labeled_rows], dtype=np.float64)
    threshold = float(threshold) if threshold is not None else float(find_best_threshold(labels, probs))
    metrics = compute_binary_metrics(labels, probs, threshold)
    auc = compute_auc(labels, probs)
    eer, eer_threshold = compute_eer(labels, probs)
    metrics.update(
        {
            "threshold": threshold,
            "auc": float(auc),
            "eer": float(eer),
            "eer_threshold": float(eer_threshold),
            "total_labeled": int(labels.size),
            "live_count": int((labels == 1).sum()),
            "spoof_count": int((labels == 0).sum()),
        }
    )
    return metrics


def evaluate_by_subset(rows: list[dict], field: str) -> list[dict]:
    values = sorted(set(row[field] for row in rows if row.get("status") == "ok"))
    result = []
    for value in values:
        subset = [row for row in rows if row.get(field) == value]
        metrics = evaluate_rows(subset)
        if metrics is None:
            continue
        result.append({"subset_field": field, "subset_value": value, **metrics})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a V2 flash-liveness checkpoint on a mixed new-domain dataset.")
    parser.add_argument("--checkpoint", required=True, help="V2 best_flash_liveness_model.pth")
    parser.add_argument("--input-root", required=True, help="已映射成 live/spoof 目录结构的数据根目录。")
    parser.add_argument("--output-dir", required=True, help="评估结果输出目录。")
    parser.add_argument("--device", default=None, help="例如 cuda:1 或 cpu")
    parser.add_argument("--threshold", type=float, default=None, help="不传则读取 checkpoint 阈值。")
    parser.add_argument("--image-num-frames", type=int, default=16, help="图片合成序列时使用多少帧。")
    parser.add_argument("--max-video-frames", type=int, default=0, help="视频最多取多少帧，0 表示全部。")
    parser.add_argument("--frame-stride", type=int, default=1, help="视频每隔多少帧取一帧。")
    parser.add_argument("--flash-group-size", type=int, default=5, help="合成颜色序列每多少帧切换一次颜色。")
    parser.add_argument("--flash-colors", default="0xFF14FF,0x14FF14,0xFF1414", help="合成颜色序列，默认紫红/绿/红。")
    parser.add_argument("--flash-alpha", type=float, default=0.25, help="图片合成闪光叠加强度。")
    parser.add_argument("--detector-model", default=None)
    parser.add_argument("--detector-device", default=None)
    parser.add_argument("--detector-conf", type=float, default=0.5)
    parser.add_argument("--detector-iou", type=float, default=0.5)
    parser.add_argument("--show-decoder-warnings", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    threshold = args.threshold if args.threshold is not None else float(checkpoint.get("threshold", 0.5))
    target_size = tuple(config["target_size"])
    flash_colors = parse_flash_colors(args.flash_colors)

    preprocessor = FacePreprocessor(
        detector_model_path=args.detector_model,
        detector_device=args.detector_device,
        target_size=target_size,
        conf_threshold=args.detector_conf,
        iou_threshold=args.detector_iou,
    )

    media_paths = collect_media(Path(args.input_root).resolve())
    rows: list[dict] = []

    model.eval()
    with torch.no_grad():
        for index, media_path in enumerate(media_paths, start=1):
            print(f"[{index}/{len(media_paths)}] evaluating {media_path}", flush=True)
            label_id, label_source = infer_label_from_path(media_path)
            media_type = detect_media_type(media_path)
            category = media_path.parent.name
            row = {
                "index": index,
                "media_path": str(media_path),
                "media_type": media_type,
                "category": category,
                "label_id": "" if label_id is None else int(label_id),
                "label_name": "" if label_id is None else ("live" if label_id == 1 else "spoof"),
                "label_source": label_source,
                "status": "failed",
                "error": "",
            }
            try:
                with suppress_native_stderr(not args.show_decoder_warnings):
                    if media_type == "video":
                        frames_tensor = load_video_frames(
                            video_path=media_path,
                            preprocessor=preprocessor,
                            target_size=target_size,
                            max_frames=args.max_video_frames,
                            frame_stride=args.frame_stride,
                        )
                    else:
                        frames_tensor = load_image_frames(
                            image_path=media_path,
                            preprocessor=preprocessor,
                            target_size=target_size,
                            num_frames=args.image_num_frames,
                            flash_colors=flash_colors,
                            flash_alpha=args.flash_alpha,
                            flash_group_size=args.flash_group_size,
                        )
                color_tensor = build_color_tensor(
                    seq_len=int(frames_tensor.shape[0]),
                    group_size=args.flash_group_size,
                    color_values=flash_colors,
                )
                padding_mask = torch.zeros((1, frames_tensor.shape[0]), dtype=torch.bool, device=device)
                logit = model(
                    frames_tensor.unsqueeze(0).to(device),
                    color_tensor.unsqueeze(0).to(device),
                    padding_mask,
                )
                probability = float(torch.sigmoid(logit).detach().cpu().numpy().reshape(-1)[0])
                prediction_id = int(probability >= threshold)
                row.update(
                    {
                        "num_frames": int(frames_tensor.shape[0]),
                        "probability_live": round(probability, 8),
                        "threshold": round(float(threshold), 8),
                        "prediction_id": prediction_id,
                        "prediction_name": "live" if prediction_id == 1 else "spoof",
                        "correct": "" if label_id is None else int(prediction_id == int(label_id)),
                        "status": "ok",
                    }
                )
            except Exception as exc:
                row["error"] = repr(exc)
            rows.append(row)

    overall_metrics = evaluate_rows(rows, threshold=float(threshold))
    media_metrics = evaluate_by_subset(rows, "media_type")
    category_metrics = evaluate_by_subset(rows, "category")
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "input_root": str(Path(args.input_root).resolve()),
        "output_dir": str(output_dir),
        "device": str(device),
        "threshold_used": float(threshold),
        "total_media": len(media_paths),
        "ok_media": sum(1 for row in rows if row["status"] == "ok"),
        "failed_media": sum(1 for row in rows if row["status"] != "ok"),
        "notes": [
            "该评估是对新增非闪光数据的跨域泛化测试，不是与训练同协议的闪光视频测试。",
            "新增数据大多没有逐帧颜色 txt，因此视频与图片都使用固定三色合成颜色序列来构造 V2 color_tensor。",
            "图片样本使用静态图像加合成闪光序列近似输入，因此图片结果只能作为参考，不等同于真实闪光视频结果。",
        ],
        "overall_metrics": overall_metrics,
        "media_type_metrics": media_metrics,
        "category_metrics": category_metrics,
    }

    write_csv(output_dir / "mixed_v2_eval_rows.csv", rows)
    write_csv(output_dir / "mixed_v2_eval_media_type_metrics.csv", media_metrics)
    write_csv(output_dir / "mixed_v2_eval_category_metrics.csv", category_metrics)
    save_json(output_dir / "mixed_v2_eval_summary.json", summary)
    print(f"Saved rows: {output_dir / 'mixed_v2_eval_rows.csv'}")
    print(f"Saved summary: {output_dir / 'mixed_v2_eval_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
