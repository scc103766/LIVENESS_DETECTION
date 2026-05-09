from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flash_liveness_project import (  # noqa: E402
    CorruptedSampleRecorder,
    FacePreprocessor,
    FlashLivenessDataset,
    compute_auc,
    compute_binary_metrics,
    compute_eer,
    discover_dataset_splits,
    infer_label_from_dir_name,
    load_checkpoint,
)
from scripts.collect_flash_liveness_video import build_frame_color_labels  # noqa: E402


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
LABEL_TOKEN_TO_ID = {
    "live": 1,
    "real": 1,
    "true": 1,
    "bonafide": 1,
    "genuine": 1,
    "zhengchang": 1,
    "spoof": 0,
    "fake": 0,
    "attack": 0,
    "replay": 0,
    "print": 0,
    "mask": 0,
    "toumo": 0,
    "headmodel": 0,
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict | list) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test the V1 flash-liveness checkpoint on collected face videos "
            "recorded with the collect_flash_liveness_video.py protocol."
        )
    )
    parser.add_argument(
        "--checkpoint",
        default=str(
            PROJECT_ROOT
            / "flash_liveness_runs/flash_liveness_gpu1_localresnet_e20_manual/best_flash_liveness_model.pth"
        ),
        help="V1 checkpoint path.",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Single video path or a directory. Directory can be train|val|test/live|spoof structured or a collected-video folder.",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=["train", "val", "test", "all", "auto"],
        help="Used when --input is a dataset root with train|val|test.",
    )
    parser.add_argument("--label", type=int, choices=[0, 1], default=None, help="Optional label for single video.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--threshold", type=float, default=None, help="Override checkpoint threshold.")
    parser.add_argument("--device", default=None, help="cuda:0/cpu etc.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--detector-model", default=None, help="Optional YOLOv7 face detector weights.")
    parser.add_argument("--detector-device", default=None)
    parser.add_argument("--detector-conf", type=float, default=0.5)
    parser.add_argument("--detector-iou", type=float, default=0.5)
    parser.add_argument("--save-debug-samples", type=int, default=8)
    parser.add_argument(
        "--infer-label-from-filename",
        action="store_true",
        help="When scanning a collected folder, also infer labels from filename tokens like live/real/toumo/spoof.",
    )
    parser.add_argument(
        "--color-hold-seconds",
        type=float,
        default=0.35,
        help="Used only to synthesize collection-style frame color labels if txt is missing.",
    )
    parser.add_argument(
        "--color-warmup-seconds",
        type=float,
        default=1.0,
        help="Used only to synthesize collection-style frame color labels if txt is missing.",
    )
    parser.add_argument(
        "--color-tail-seconds",
        type=float,
        default=0.5,
        help="Used only to synthesize collection-style frame color labels if txt is missing.",
    )
    return parser.parse_args()


def infer_label_from_filename(path: Path) -> int | None:
    normalized = path.stem.lower().replace("-", "_").replace(" ", "_")
    for token, label_id in LABEL_TOKEN_TO_ID.items():
        if token in normalized:
            return label_id
    return None


def infer_label_from_path(path: Path, allow_filename: bool) -> int | None:
    for parent in [path.parent, path.parent.parent]:
        if parent is None:
            continue
        label_id = infer_label_from_dir_name(parent.name)
        if label_id is not None:
            return label_id
    if allow_filename:
        return infer_label_from_filename(path)
    return None


def resolve_collected_samples(root: Path, allow_filename: bool) -> list[tuple[str, int | None]]:
    samples: list[tuple[str, int | None]] = []
    for video_path in sorted(root.rglob("*")):
        if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        samples.append((str(video_path), infer_label_from_path(video_path, allow_filename)))
    return samples


def resolve_samples(args: argparse.Namespace) -> tuple[list[tuple[str, int | None]], str]:
    input_path = Path(args.input)
    if input_path.is_file():
        return [(str(input_path), args.label)], "single_video"

    if not input_path.exists():
        raise FileNotFoundError(f"Input not found: {input_path}")

    split_dirs = [input_path / name for name in ("train", "val", "test")]
    label_dirs = [input_path / name for name in ("live", "spoof")]
    has_split_dataset = any(path.exists() for path in split_dirs)
    has_label_dataset = any(path.exists() for path in label_dirs)
    if has_split_dataset or has_label_dataset:
        split_map = discover_dataset_splits(str(input_path))
        split_name = args.split
        if split_name == "auto":
            split_name = "test" if split_map.get("test") else "val" if split_map.get("val") else "train"
        if split_name == "all":
            samples = split_map.get("train", []) + split_map.get("val", []) + split_map.get("test", [])
        else:
            samples = split_map.get(split_name, [])
        return [(path, int(label_id)) for path, label_id in samples], f"dataset:{split_name}"

    return resolve_collected_samples(input_path, args.infer_label_from_filename), "collected_folder"


def apply_sample_limits(
    samples: list[tuple[str, int | None]],
    limit: int | None,
    samples_per_class: int | None,
) -> list[tuple[str, int | None]]:
    result = samples
    if samples_per_class is not None:
        live_samples = [item for item in result if item[1] == 1][:samples_per_class]
        spoof_samples = [item for item in result if item[1] == 0][:samples_per_class]
        unknown_samples = [item for item in result if item[1] is None]
        result = live_samples + spoof_samples + unknown_samples
    if limit is not None:
        result = result[:limit]
    return result


def get_video_meta(video_path: Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"frame_count": 0, "fps": 0.0}
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    cap.release()
    return {"frame_count": frame_count, "fps": fps}


def read_color_txt_info(txt_path: Path) -> tuple[bool, int]:
    if not txt_path.exists():
        return False, 0
    try:
        line_count = sum(1 for _ in txt_path.open("r", encoding="utf-8"))
    except Exception:
        line_count = 0
    return True, line_count


def save_debug_grid(output_path: Path, tensor_frames: torch.Tensor) -> None:
    rgb_frames = tensor_frames[:, :3].permute(0, 2, 3, 1).cpu().numpy()
    rgb_frames = np.clip(rgb_frames * 255.0, 0, 255).astype(np.uint8)
    if len(rgb_frames) == 0:
        return
    bgr_frames = [cv2.cvtColor(frame, cv2.COLOR_RGB2BGR) for frame in rgb_frames]
    tile_h, tile_w = bgr_frames[0].shape[:2]
    cols = min(4, len(bgr_frames))
    rows = int(np.ceil(len(bgr_frames) / cols))
    canvas = np.zeros((rows * tile_h, cols * tile_w, 3), dtype=np.uint8)
    for index, frame in enumerate(bgr_frames):
        row = index // cols
        col = index % cols
        canvas[row * tile_h:(row + 1) * tile_h, col * tile_w:(col + 1) * tile_w] = frame
    cv2.imwrite(str(output_path), canvas)


def flush_batch(
    model: torch.nn.Module,
    device: torch.device,
    batch_tensors: list[torch.Tensor],
    batch_meta: list[dict],
    threshold: float,
) -> list[dict]:
    if not batch_tensors:
        return []
    batch = torch.stack(batch_tensors, dim=0).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(float)

    rows: list[dict] = []
    for meta, prob in zip(batch_meta, probs):
        prediction_id = int(prob >= threshold)
        rows.append(
            {
                "index": meta["index"],
                "video_path": meta["video_path"],
                "txt_path": meta["txt_path"],
                "has_color_txt": meta["has_color_txt"],
                "color_label_count": meta["color_label_count"],
                "synthesized_color_label_count": meta["synthesized_color_label_count"],
                "frame_count": meta["frame_count"],
                "fps": meta["fps"],
                "label_id": "" if meta["label_id"] is None else int(meta["label_id"]),
                "label_name": "" if meta["label_id"] is None else ("live" if int(meta["label_id"]) == 1 else "spoof"),
                "probability_live": round(float(prob), 8),
                "threshold": round(float(threshold), 8),
                "prediction_id": prediction_id,
                "prediction_name": "live" if prediction_id == 1 else "spoof",
                "correct": "" if meta["label_id"] is None else int(prediction_id == int(meta["label_id"])),
                "status": "ok",
                "error": "",
            }
        )
    return rows


def build_summary(
    rows: list[dict],
    samples: list[tuple[str, int | None]],
    checkpoint_path: str,
    output_dir: Path,
    source_mode: str,
    threshold: float,
    config: dict,
    device: torch.device,
    elapsed_seconds: float,
) -> dict:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    labeled_rows = [row for row in ok_rows if row["label_id"] != ""]
    summary = {
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "input_mode": source_mode,
        "device": str(device),
        "threshold": float(threshold),
        "num_frames": int(config["num_frames"]),
        "target_size": config["target_size"],
        "total_samples": len(samples),
        "processed_samples": len(ok_rows),
        "failed_samples": len(rows) - len(ok_rows),
        "labeled_samples": len(labeled_rows),
        "with_color_txt_samples": int(sum(int(row["has_color_txt"]) for row in ok_rows)),
        "without_color_txt_samples": int(sum(int(not row["has_color_txt"]) for row in ok_rows)),
        "elapsed_seconds": round(elapsed_seconds, 3),
        "result_files": {
            "csv": str(output_dir / "predictions.csv"),
            "jsonl": str(output_dir / "predictions.jsonl"),
            "summary": str(output_dir / "summary.json"),
            "debug_sample_frames": str(output_dir / "debug_sample_frames"),
            "skipped_or_failed": str(output_dir / "skipped_or_failed_samples.txt"),
        },
    }

    if labeled_rows:
        labels = np.asarray([int(row["label_id"]) for row in labeled_rows], dtype=np.int32)
        probs = np.asarray([float(row["probability_live"]) for row in labeled_rows], dtype=np.float64)
        metrics = compute_binary_metrics(labels, probs, threshold)
        metrics["auc"] = compute_auc(labels, probs)
        eer, eer_threshold = compute_eer(labels, probs)
        metrics["eer"] = eer
        metrics["eer_threshold"] = eer_threshold
        summary["metrics"] = metrics

        by_label: dict[str, dict] = {}
        for label_name, label_id in (("live", 1), ("spoof", 0)):
            subset = [row for row in labeled_rows if int(row["label_id"]) == label_id]
            if not subset:
                continue
            by_label[label_name] = {
                "count": len(subset),
                "pred_live": int(sum(int(row["prediction_id"]) == 1 for row in subset)),
                "pred_spoof": int(sum(int(row["prediction_id"]) == 0 for row in subset)),
                "correct": int(sum(int(row["correct"]) for row in subset)),
                "accuracy": round(sum(int(row["correct"]) for row in subset) / len(subset), 6),
                "mean_probability_live": round(
                    float(np.mean([float(row["probability_live"]) for row in subset])),
                    6,
                ),
            }
        summary["per_label"] = by_label

    return summary


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    debug_dir = output_dir / "debug_sample_frames"
    ensure_dir(output_dir)
    ensure_dir(debug_dir)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    threshold = args.threshold if args.threshold is not None else float(checkpoint.get("threshold", 0.5))

    samples, source_mode = resolve_samples(args)
    samples = apply_sample_limits(samples, args.limit, args.samples_per_class)

    preprocessor = FacePreprocessor(
        detector_model_path=args.detector_model,
        detector_device=args.detector_device,
        target_size=tuple(config["target_size"]),
        conf_threshold=args.detector_conf,
        iou_threshold=args.detector_iou,
    )
    corrupted_path = output_dir / "skipped_or_failed_samples.txt"
    dataset = FlashLivenessDataset(
        samples=[],
        num_frames=int(config["num_frames"]),
        transform=False,
        preprocessor=preprocessor,
        target_size=tuple(config["target_size"]),
        corrupted_sample_recorder=CorruptedSampleRecorder(corrupted_path),
    )

    csv_path = output_dir / "predictions.csv"
    jsonl_path = output_dir / "predictions.jsonl"
    for stale_path in (csv_path, jsonl_path):
        if stale_path.exists():
            stale_path.unlink()
    rows: list[dict] = []
    batch_tensors: list[torch.Tensor] = []
    batch_meta: list[dict] = []
    start_time = time.time()

    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "input": args.input,
                "input_mode": source_mode,
                "sample_count": len(samples),
                "device": str(device),
                "threshold": threshold,
                "num_frames": config["num_frames"],
                "target_size": config["target_size"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for index, (video_path_str, label_id) in enumerate(samples, start=1):
        video_path = Path(video_path_str)
        txt_path = video_path.with_suffix(".txt")
        try:
            meta = get_video_meta(video_path)
            has_color_txt, color_label_count = read_color_txt_info(txt_path)
            synthesized_color_count = 0
            if not has_color_txt and meta["frame_count"] > 0 and meta["fps"] > 1e-6:
                synthesized_color_count = len(
                    build_frame_color_labels(
                        frame_count=int(meta["frame_count"]),
                        fps=float(meta["fps"]),
                        warmup_seconds=args.color_warmup_seconds,
                        hold_seconds=args.color_hold_seconds,
                        tail_seconds=args.color_tail_seconds,
                    )
                )

            tensor_frames = dataset.process_video(str(video_path))
            if tensor_frames.numel() == 0:
                raise RuntimeError("no_valid_frames_decoded")

            if index <= args.save_debug_samples:
                save_debug_grid(debug_dir / f"{index:04d}_{video_path.stem}.jpg", tensor_frames)

            batch_tensors.append(tensor_frames)
            batch_meta.append(
                {
                    "index": index,
                    "video_path": str(video_path),
                    "txt_path": str(txt_path) if txt_path.exists() else "",
                    "has_color_txt": int(has_color_txt),
                    "color_label_count": int(color_label_count),
                    "synthesized_color_label_count": int(synthesized_color_count),
                    "frame_count": int(meta["frame_count"]),
                    "fps": round(float(meta["fps"]), 4),
                    "label_id": label_id,
                }
            )

            if len(batch_tensors) >= args.batch_size:
                flushed = flush_batch(model, device, batch_tensors, batch_meta, threshold)
                rows.extend(flushed)
                write_csv(csv_path, rows)
                append_jsonl(jsonl_path, flushed)
                batch_tensors.clear()
                batch_meta.clear()
        except Exception as exc:
            failed_row = {
                "index": index,
                "video_path": str(video_path),
                "txt_path": str(txt_path) if txt_path.exists() else "",
                "has_color_txt": int(txt_path.exists()),
                "color_label_count": 0,
                "synthesized_color_label_count": 0,
                "frame_count": 0,
                "fps": 0.0,
                "label_id": "" if label_id is None else int(label_id),
                "label_name": "" if label_id is None else ("live" if int(label_id) == 1 else "spoof"),
                "probability_live": "",
                "threshold": round(float(threshold), 8),
                "prediction_id": "",
                "prediction_name": "",
                "correct": "",
                "status": "failed",
                "error": str(exc),
            }
            rows.append(failed_row)
            write_csv(csv_path, rows)
            append_jsonl(jsonl_path, [failed_row])

        if index == 1 or index % 20 == 0 or index == len(samples):
            print(f"processed {index}/{len(samples)} videos", flush=True)

    if batch_tensors:
        flushed = flush_batch(model, device, batch_tensors, batch_meta, threshold)
        rows.extend(flushed)
        write_csv(csv_path, rows)
        append_jsonl(jsonl_path, flushed)

    summary = build_summary(
        rows=rows,
        samples=samples,
        checkpoint_path=args.checkpoint,
        output_dir=output_dir,
        source_mode=source_mode,
        threshold=threshold,
        config=config,
        device=device,
        elapsed_seconds=time.time() - start_time,
    )
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
