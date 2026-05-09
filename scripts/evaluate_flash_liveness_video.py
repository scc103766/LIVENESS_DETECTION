from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flash_liveness_infer_utils import (  # noqa: E402
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    detect_media_type,
    infer_label_from_name,
    load_flash_liveness_bundle,
    parse_flash_colors,
    predict_from_tensor,
    save_debug_grid,
    tensor_from_media_path,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_json(path: Path, payload: dict | list) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


@contextmanager
def suppress_native_stderr(enabled: bool):
    """Temporarily hide FFmpeg/MJPEG native stderr warnings."""
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


def build_metrics(rows: list[dict]) -> dict | None:
    labeled_rows = [row for row in rows if row["status"] == "ok" and row["label_id"] != ""]
    if not labeled_rows:
        return None

    labels = np.asarray([int(row["label_id"]) for row in labeled_rows], dtype=np.int32)
    preds = np.asarray([int(row["prediction_id"]) for row in labeled_rows], dtype=np.int32)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    total = int(labels.size)
    correct = tp + tn
    return {
        "total_labeled": total,
        "correct": correct,
        "incorrect": total - correct,
        "live_count": int((labels == 1).sum()),
        "spoof_count": int((labels == 0).sum()),
        "accuracy": float(correct / max(total, 1)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def collect_media(input_root: Path) -> list[Path]:
    valid_exts = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
    return sorted(path for path in input_root.rglob("*") if path.is_file() and path.suffix.lower() in valid_exts)


def infer_label_from_path_then_name(media_path: Path, real_name_token: str) -> tuple[int | None, str]:
    for parent in media_path.parents:
        name = parent.name.lower()
        if name == "live":
            return 1, "path_dir_live"
        if name == "spoof":
            return 0, "path_dir_spoof"
    try:
        label_id, _label_name = infer_label_from_name(media_path, real_name_token)
        return label_id, "name_token_fallback"
    except Exception:
        return None, "unresolved"


def resolve_inputs(args: argparse.Namespace) -> list[Path]:
    if args.video_path:
        return [Path(args.video_path).resolve()]
    if args.image_path:
        return [Path(args.image_path).resolve()]
    if args.input_root:
        root = Path(args.input_root).resolve()
        if root.is_file():
            return [root]
        return collect_media(root)
    raise ValueError("请提供 --video-path、--image-path 或 --input-root。")


def build_row(
    index: int,
    media_path: Path,
    media_type: str,
    txt_path: Path | None,
    label_id: int | None,
    prediction: dict | None,
    error: str = "",
) -> dict:
    label_name = "" if label_id is None else ("live" if int(label_id) == 1 else "spoof")
    if prediction is None:
        return {
            "index": index,
            "media_type": media_type,
            "media_path": str(media_path),
            "video_path": str(media_path) if media_type == "video" else "",
            "image_path": str(media_path) if media_type == "image" else "",
            "txt_path": str(txt_path) if txt_path else "",
            "label_id": "" if label_id is None else int(label_id),
            "label_name": label_name,
            "probability_live_raw": "",
            "score_live": "",
            "threshold": "",
            "prediction_id": "",
            "prediction_name": "",
            "correct": "",
            "status": "failed",
            "error": error,
        }

    prediction_id = int(prediction["prediction_id"])
    return {
        "index": index,
        "media_type": media_type,
        "media_path": str(media_path),
        "video_path": str(media_path) if media_type == "video" else "",
        "image_path": str(media_path) if media_type == "image" else "",
        "txt_path": str(txt_path) if txt_path else "",
        "label_id": "" if label_id is None else int(label_id),
        "label_name": label_name,
        "probability_live_raw": round(float(prediction["probability_live_raw"]), 12),
        "score_live": round(float(prediction["score_live"]), 8),
        "threshold": round(float(prediction["threshold"]), 8),
        "prediction_id": prediction_id,
        "prediction_name": prediction["prediction_name"],
        "correct": "" if label_id is None else int(prediction_id == int(label_id)),
        "status": "ok",
        "error": "",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate flash-liveness checkpoint on one or many videos/images with image compatibility support."
    )
    parser.add_argument("--checkpoint", required=True, help="best/last_flash_liveness_model.pth")
    parser.add_argument("--video-path", default=None, help="单视频路径")
    parser.add_argument("--image-path", default=None, help="单图片路径")
    parser.add_argument("--input-root", default=None, help="批量输入目录，递归扫描视频与图片")
    parser.add_argument("--txt-path", default=None, help="单视频可选 txt 路径，仅记录到结果中")
    parser.add_argument("--label", type=int, choices=[0, 1], default=None, help="单样本可选标签，0=spoof，1=live")
    parser.add_argument("--label-from-name", action="store_true", help="批量模式自动标注：优先按路径中的 live/spoof 目录判断，文件名关键字仅作兜底")
    parser.add_argument("--real-name-token", default="true", help="文件名中表示真人的关键字，默认 true")
    parser.add_argument("--output-dir", required=True, help="结果输出目录")
    parser.add_argument("--threshold", type=float, default=None, help="默认读取 checkpoint 中保存的阈值")
    parser.add_argument("--device", default=None, help="例如 cuda:0/cpu")
    parser.add_argument("--detector-model", default=None, help="YOLOv7 人脸检测权重路径；为空则中心裁剪")
    parser.add_argument("--detector-device", default=None)
    parser.add_argument("--detector-conf", type=float, default=0.5)
    parser.add_argument("--detector-iou", type=float, default=0.5)
    parser.add_argument("--save-debug-samples", type=int, default=12, help="最多保存多少条调试帧拼图")
    parser.add_argument(
        "--static-sequence-mode",
        choices=["copy", "flash"],
        default="copy",
        help="图片输入转视频序列的方式：copy 为复制静态图，flash 为合成闪光序列。",
    )
    parser.add_argument("--flash-group-size", type=int, default=4, help="flash 模式下每多少帧切换一次颜色")
    parser.add_argument("--flash-colors", default="normal,red,green,blue", help="flash 模式颜色序列")
    parser.add_argument("--flash-alpha", type=float, default=0.25, help="flash 模式颜色叠加强度")
    parser.add_argument(
        "--show-decoder-warnings",
        action="store_true",
        help="显示 OpenCV/FFmpeg 的 MJPEG 解码告警；默认隐藏。",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    debug_dir = output_dir / "debug_sample_frames"
    ensure_dir(output_dir)
    ensure_dir(debug_dir)

    corrupted_path = output_dir / "skipped_or_failed_samples.txt"
    bundle = load_flash_liveness_bundle(
        checkpoint_path=args.checkpoint,
        device_spec=args.device,
        detector_model=args.detector_model,
        detector_device=args.detector_device,
        detector_conf=args.detector_conf,
        detector_iou=args.detector_iou,
        threshold_override=args.threshold,
        corrupted_record_path=corrupted_path,
    )
    flash_colors = parse_flash_colors(args.flash_colors)
    inputs = resolve_inputs(args)
    started_at = time.time()
    rows: list[dict] = []

    csv_path = output_dir / "predictions.csv"
    jsonl_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"

    print(
        json.dumps(
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "device": str(bundle["device"]),
                "threshold": bundle["threshold"],
                "num_frames": bundle["num_frames"],
                "target_size": list(bundle["target_size"]),
                "input_count": len(inputs),
                "input_root": str(Path(args.input_root).resolve()) if args.input_root else "",
                "static_sequence_mode": args.static_sequence_mode,
                "flash_colors": flash_colors if args.static_sequence_mode == "flash" else [],
                "output_dir": str(output_dir.resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for index, media_path in enumerate(inputs, start=1):
        media_type = detect_media_type(media_path)
        txt_path = None
        if media_type == "video":
            txt_candidate = Path(args.txt_path).resolve() if args.txt_path and len(inputs) == 1 else media_path.with_suffix(".txt")
            txt_path = txt_candidate if txt_candidate.exists() else txt_candidate

        label_id = None
        label_source = ""
        if len(inputs) == 1 and args.label is not None:
            label_id = int(args.label)
            label_source = "explicit_label"
        elif args.label_from_name:
            label_id, label_source = infer_label_from_path_then_name(media_path, args.real_name_token)

        try:
            with suppress_native_stderr(not args.show_decoder_warnings):
                tensor_frames, resolved_media_type = tensor_from_media_path(
                    media_path=media_path,
                    bundle=bundle,
                    static_sequence_mode=args.static_sequence_mode,
                    flash_group_size=args.flash_group_size,
                    flash_colors=flash_colors,
                    flash_alpha=args.flash_alpha,
                )
            if index <= args.save_debug_samples:
                save_debug_grid(debug_dir / f"{index:04d}_{media_path.stem}.jpg", tensor_frames)
            prediction = predict_from_tensor(
                model=bundle["model"],
                tensor_frames=tensor_frames,
                device=bundle["device"],
                threshold=bundle["threshold"],
            )
            row = build_row(index, media_path, resolved_media_type, txt_path, label_id, prediction)
            row["label_source"] = label_source
            rows.append(row)
        except Exception as exc:
            row = build_row(index, media_path, media_type, txt_path, label_id, None, str(exc))
            row["label_source"] = label_source
            rows.append(row)

        if index == 1 or index % 20 == 0 or index == len(inputs):
            print(f"processed {index}/{len(inputs)} items", flush=True)

    metrics = build_metrics(rows)
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(bundle["device"]),
        "threshold": bundle["threshold"],
        "num_frames": bundle["num_frames"],
        "target_size": list(bundle["target_size"]),
        "static_sequence_mode": args.static_sequence_mode,
        "flash_group_size": args.flash_group_size if args.static_sequence_mode == "flash" else "",
        "flash_colors": flash_colors if args.static_sequence_mode == "flash" else [],
        "flash_alpha": args.flash_alpha if args.static_sequence_mode == "flash" else "",
        "label_from_name": bool(args.label_from_name),
        "real_name_token": args.real_name_token if args.label_from_name else "",
        "total_items": len(inputs),
        "processed_items": int(sum(1 for row in rows if row["status"] == "ok")),
        "failed_items": int(sum(1 for row in rows if row["status"] != "ok")),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "result_files": {
            "csv": str(csv_path.resolve()),
            "jsonl": str(jsonl_path.resolve()),
            "summary": str(summary_path.resolve()),
            "debug_sample_frames": str(debug_dir.resolve()),
            "skipped_or_failed": str(corrupted_path.resolve()),
        },
    }
    if args.video_path:
        summary["video_path"] = str(Path(args.video_path).resolve())
    if args.image_path:
        summary["image_path"] = str(Path(args.image_path).resolve())
    if args.input_root:
        summary["input_root"] = str(Path(args.input_root).resolve())
    if metrics is not None:
        summary["metrics"] = metrics

    write_csv(csv_path, rows)
    write_jsonl(jsonl_path, rows)
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
