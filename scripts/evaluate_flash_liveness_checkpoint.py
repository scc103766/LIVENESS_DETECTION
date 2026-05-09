from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from contextlib import contextmanager
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
    load_checkpoint,
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict | list) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def append_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def append_jsonl(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


@contextmanager
def suppress_native_stderr(enabled: bool):
    """Temporarily hide FFmpeg/MJPEG decoder warnings emitted from native code."""
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


def save_debug_grid(output_path: Path, tensor_frames: torch.Tensor) -> None:
    """Save the sampled RGB frames so the inference input can be inspected."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a flash_liveness_project.py checkpoint on split folders or one video."
    )
    parser.add_argument("--checkpoint", required=True, help="best/last_flash_liveness_model.pth")
    parser.add_argument("--data-root", help="数据集根目录，结构为 train|val|test/live|spoof")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"], help="要评估的数据划分")
    parser.add_argument("--video-path", help="单视频推理路径。提供后不再扫描 data-root。")
    parser.add_argument("--label", type=int, choices=[0, 1], help="单视频可选标签，0=spoof/fake，1=live/real")
    parser.add_argument("--output-dir", required=True, help="结果输出目录")
    parser.add_argument("--threshold", type=float, default=None, help="可选阈值，默认读取 checkpoint 中保存的阈值")
    parser.add_argument("--batch-size", type=int, default=8, help="模型前向 batch size")
    parser.add_argument("--limit", type=int, default=None, help="仅测试前 N 条，用于快速冒烟测试")
    parser.add_argument("--samples-per-class", type=int, default=None, help="每类最多抽取 N 条，适合快速平衡测试")
    parser.add_argument("--device", default=None, help="例如 cuda:0/cuda:1/cpu")
    parser.add_argument("--detector-model", default=None, help="YOLOv7 人脸检测权重路径；为空则中心裁剪")
    parser.add_argument("--detector-device", default=None)
    parser.add_argument("--detector-conf", type=float, default=0.5)
    parser.add_argument("--detector-iou", type=float, default=0.5)
    parser.add_argument("--save-debug-samples", type=int, default=8, help="保存前 N 条输入帧拼图")
    parser.add_argument(
        "--show-decoder-warnings",
        action="store_true",
        help="显示 OpenCV/FFmpeg 的 MJPEG 解码告警；默认隐藏，失败样本仍会写入结果文件。",
    )
    return parser.parse_args()


def resolve_samples(args: argparse.Namespace) -> list[tuple[str, int | None]]:
    if args.video_path:
        return [(args.video_path, args.label)]

    if not args.data_root:
        raise ValueError("批量测试需要提供 --data-root，单视频推理需要提供 --video-path。")

    split_map = discover_dataset_splits(args.data_root)
    if args.split == "all":
        samples = split_map.get("train", []) + split_map.get("val", []) + split_map.get("test", [])
    else:
        samples = split_map.get(args.split, [])
    if args.samples_per_class is not None:
        live_samples = [sample for sample in samples if int(sample[1]) == 1][:args.samples_per_class]
        spoof_samples = [sample for sample in samples if int(sample[1]) == 0][:args.samples_per_class]
        samples = live_samples + spoof_samples
    if args.limit is not None:
        samples = samples[:args.limit]
    return [(path, label) for path, label in samples]


def flush_batch(
    model: torch.nn.Module,
    device: torch.device,
    batch_tensors: list[torch.Tensor],
    batch_meta: list[dict],
    threshold: float,
    rows: list[dict],
) -> list[dict]:
    if not batch_tensors:
        return []

    batch = torch.stack(batch_tensors, dim=0).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(float)

    flushed_rows = []
    for meta, prob in zip(batch_meta, probs):
        pred_id = int(prob >= threshold)
        row = {
            "index": meta["index"],
            "video_path": meta["video_path"],
            "label_id": "" if meta["label_id"] is None else int(meta["label_id"]),
            "label_name": meta["label_name"],
            "probability_live": round(float(prob), 8),
            "threshold": round(float(threshold), 8),
            "prediction_id": pred_id,
            "prediction_name": "live" if pred_id == 1 else "spoof",
            "correct": "" if meta["label_id"] is None else int(pred_id == int(meta["label_id"])),
            "status": "ok",
            "error": "",
        }
        rows.append(row)
        flushed_rows.append(row)
    return flushed_rows


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    debug_dir = output_dir / "debug_sample_frames"
    ensure_dir(debug_dir)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    threshold = args.threshold if args.threshold is not None else float(checkpoint.get("threshold", 0.5))

    corrupted_path = output_dir / "skipped_or_failed_samples.txt"
    preprocessor = FacePreprocessor(
        detector_model_path=args.detector_model,
        detector_device=args.detector_device,
        target_size=tuple(config["target_size"]),
        conf_threshold=args.detector_conf,
        iou_threshold=args.detector_iou,
    )
    dataset = FlashLivenessDataset(
        samples=[],
        num_frames=int(config["num_frames"]),
        transform=False,
        preprocessor=preprocessor,
        target_size=tuple(config["target_size"]),
        corrupted_sample_recorder=CorruptedSampleRecorder(corrupted_path),
    )

    samples = resolve_samples(args)
    start_time = time.time()
    rows: list[dict] = []
    batch_tensors: list[torch.Tensor] = []
    batch_meta: list[dict] = []
    csv_path = output_dir / "predictions.csv"
    jsonl_path = output_dir / "predictions.jsonl"
    for stale_path in (csv_path, jsonl_path):
        if stale_path.exists():
            stale_path.unlink()

    print(
        json.dumps(
            {
                "checkpoint": args.checkpoint,
                "device": str(device),
                "threshold": threshold,
                "num_frames": config["num_frames"],
                "target_size": config["target_size"],
                "sample_count": len(samples),
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        )
    )

    for index, (video_path, label_id) in enumerate(samples, start=1):
        try:
            with suppress_native_stderr(not args.show_decoder_warnings):
                tensor_frames = dataset.process_video(video_path)
            if tensor_frames.numel() == 0:
                raise RuntimeError("no_valid_frames_decoded")

            if index <= args.save_debug_samples:
                save_debug_grid(debug_dir / f"{index:04d}_{Path(video_path).stem}.jpg", tensor_frames)

            batch_tensors.append(tensor_frames)
            batch_meta.append(
                {
                    "index": index,
                    "video_path": video_path,
                    "label_id": label_id,
                    "label_name": "" if label_id is None else ("live" if int(label_id) == 1 else "spoof"),
                }
            )

            if len(batch_tensors) >= args.batch_size:
                flushed_rows = flush_batch(model, device, batch_tensors, batch_meta, threshold, rows)
                append_csv(csv_path, flushed_rows)
                append_jsonl(jsonl_path, flushed_rows)
                batch_tensors.clear()
                batch_meta.clear()
        except Exception as exc:
            failed_row = {
                "index": index,
                "video_path": video_path,
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
            append_csv(csv_path, [failed_row])
            append_jsonl(jsonl_path, [failed_row])

        if index == 1 or index % 20 == 0 or index == len(samples):
            print(f"processed {index}/{len(samples)} videos", flush=True)

    flushed_rows = flush_batch(model, device, batch_tensors, batch_meta, threshold, rows)
    append_csv(csv_path, flushed_rows)
    append_jsonl(jsonl_path, flushed_rows)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    labeled_rows = [row for row in ok_rows if row["label_id"] != ""]
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_root": str(Path(args.data_root).resolve()) if args.data_root else "",
        "split": args.split if args.data_root else "",
        "video_path": str(Path(args.video_path).resolve()) if args.video_path else "",
        "device": str(device),
        "threshold": threshold,
        "num_frames": int(config["num_frames"]),
        "target_size": config["target_size"],
        "total_samples": len(samples),
        "processed_samples": len(ok_rows),
        "failed_samples": len(rows) - len(ok_rows),
        "elapsed_seconds": round(time.time() - start_time, 3),
        "result_files": {
            "csv": str(csv_path),
            "jsonl": str(jsonl_path),
            "summary": str(output_dir / "summary.json"),
            "debug_sample_frames": str(debug_dir),
            "skipped_or_failed": str(corrupted_path),
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

    write_csv(csv_path, rows)
    append_jsonl(jsonl_path, [])
    save_json(output_dir / "summary.json", summary)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
