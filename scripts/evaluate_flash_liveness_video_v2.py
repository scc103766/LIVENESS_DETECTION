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
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flash_liveness_project_v2 import (  # noqa: E402
    FacePreprocessor,
    FlashLivenessDataset,
    compute_auc,
    compute_binary_metrics,
    compute_eer,
    discover_dataset_splits,
    load_checkpoint,
)
from flash_liveness_project_v3 import discover_dataset_splits as discover_v3_dataset_splits  # noqa: E402


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


def infer_source_key(video_path: Path) -> str:
    name = video_path.name
    if "__" in name:
        return name.split("__", 1)[0]
    return video_path.parent.name


def sample_to_eval_tuple(sample) -> tuple[str, str, int, str, str]:
    if hasattr(sample, "media_path"):
        return (
            str(sample.media_path),
            str(sample.txt_path) if sample.txt_path else str(Path(sample.media_path).with_suffix(".txt")),
            int(sample.label),
            str(getattr(sample, "category", Path(sample.media_path).parent.name)),
            str(getattr(sample, "source_group", "")),
        )
    video_path, txt_path, label = sample
    video_path_obj = Path(video_path)
    return str(video_path), str(txt_path), int(label), video_path_obj.parent.name, infer_source_key(video_path_obj)


def resolve_samples(args: argparse.Namespace) -> list[tuple[str, str, int, str, str]]:
    if args.video_path:
        video_path = str(Path(args.video_path).resolve())
        txt_path = str(Path(args.txt_path).resolve()) if args.txt_path else str(Path(video_path).with_suffix(".txt"))
        return [(video_path, txt_path, args.label, "single_video", "single_video")]

    if not args.data_root:
        raise ValueError("批量测试请提供 --data-root，单视频请提供 --video-path。")

    if args.split_source == "v3":
        split_map = discover_v3_dataset_splits(
            args.data_root,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            media_filter="videos",
            require_color_txt=True,
        )
    else:
        split_map = discover_dataset_splits(args.data_root)

    if args.split == "all":
        samples = split_map.get("train", []) + split_map.get("val", []) + split_map.get("test", [])
    else:
        samples = split_map.get(args.split, [])
    if args.samples_per_class is not None:
        def label_of(sample) -> int:
            return int(sample.label) if hasattr(sample, "label") else int(sample[2])

        live_samples = [sample for sample in samples if label_of(sample) == 1][:args.samples_per_class]
        spoof_samples = [sample for sample in samples if label_of(sample) == 0][:args.samples_per_class]
        samples = live_samples + spoof_samples
    if args.limit is not None:
        samples = samples[:args.limit]
    return [sample_to_eval_tuple(sample) for sample in samples]


def evaluate_rows(rows: list[dict], threshold: float) -> dict | None:
    labeled_rows = [row for row in rows if row["status"] == "ok" and row["label_id"] != ""]
    if not labeled_rows:
        return None

    labels = np.asarray([int(row["label_id"]) for row in labeled_rows], dtype=np.int32)
    probs = np.asarray([float(row["probability_live"]) for row in labeled_rows], dtype=np.float64)
    metrics = compute_binary_metrics(labels, probs, threshold)
    auc = compute_auc(labels, probs)
    eer, eer_threshold = compute_eer(labels, probs)
    metrics.update(
        {
            "threshold": float(threshold),
            "auc": float(auc),
            "eer": float(eer),
            "eer_threshold": float(eer_threshold),
            "total_labeled": int(labels.size),
            "live_count": int((labels == 1).sum()),
            "spoof_count": int((labels == 0).sum()),
        }
    )
    return metrics


def evaluate_by_subset(rows: list[dict], field: str, threshold: float) -> list[dict]:
    values = sorted(set(row[field] for row in rows if row.get("status") == "ok"))
    result = []
    for value in values:
        subset = [row for row in rows if row.get(field) == value]
        metrics = evaluate_rows(subset, threshold=threshold)
        if metrics is None:
            continue
        result.append({"subset_field": field, "subset_value": value, **metrics})
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pure-video official evaluator for flash_liveness_project_v2.py.")
    parser.add_argument("--checkpoint", required=True, help="V2 best/last checkpoint path")
    parser.add_argument("--data-root", default=None, help="数据集根目录。split-source=v2 时为 train|val|test/live|spoof；split-source=v3 时为 manifest 归档。")
    parser.add_argument("--split-source", choices=["v2", "v3"], default="v2", help="样本划分来源。v3 会复用 flash_liveness_project_v3.py 的 manifest/category split。")
    parser.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    parser.add_argument("--video-path", default=None, help="单视频路径")
    parser.add_argument("--txt-path", default=None, help="单视频对应 txt；默认同名 txt")
    parser.add_argument("--label", type=int, choices=[0, 1], default=None, help="单视频标签")
    parser.add_argument("--output-dir", required=True, help="结果输出目录")
    parser.add_argument("--threshold", type=float, default=None, help="默认使用 checkpoint 内阈值")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2, help="split-source=v3 时复现 V3 split 的 val ratio。")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="split-source=v3 时复现 V3 split 的 test ratio。")
    parser.add_argument("--seed", type=int, default=42, help="split-source=v3 时复现 V3 split 的随机种子。")
    parser.add_argument("--max-seq-len", type=int, default=0, help="0 表示自动按模型支持长度裁剪")
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
    model_max_seq_len = int(args.max_seq_len) if int(args.max_seq_len) > 0 else 512
    target_size = tuple(config["target_size"])

    preprocessor = FacePreprocessor(
        detector_model_path=args.detector_model,
        detector_device=args.detector_device,
        target_size=target_size,
        conf_threshold=args.detector_conf,
        iou_threshold=args.detector_iou,
    )
    dataset = FlashLivenessDataset(
        samples=[],
        transform=False,
        preprocessor=preprocessor,
        target_size=target_size,
    )

    samples = resolve_samples(args)
    rows: list[dict] = []
    started_at = time.time()

    model.eval()
    with torch.no_grad():
        for index, (video_path, txt_path, label_id, category, source_group) in enumerate(samples, start=1):
            video_path_obj = Path(video_path)
            row = {
                "index": index,
                "video_path": video_path,
                "txt_path": txt_path,
                "label_id": "" if label_id is None else int(label_id),
                "label_name": "" if label_id is None else ("live" if int(label_id) == 1 else "spoof"),
                "source_key": infer_source_key(video_path_obj),
                "category": category,
                "source_group": source_group,
                "status": "failed",
                "error": "",
            }
            try:
                with suppress_native_stderr(not args.show_decoder_warnings):
                    frames_tensor, color_tensor = dataset.process_video(video_path, txt_path)
                if frames_tensor.numel() == 0:
                    raise RuntimeError("no_valid_frames_decoded")

                original_seq_len = int(frames_tensor.shape[0])
                if original_seq_len > model_max_seq_len:
                    frames_tensor = frames_tensor[:model_max_seq_len]
                    color_tensor = color_tensor[:model_max_seq_len]

                padding_mask = torch.zeros((1, frames_tensor.shape[0]), dtype=torch.bool, device=device)
                logits = model(
                    frames_tensor.unsqueeze(0).to(device),
                    color_tensor.unsqueeze(0).to(device),
                    padding_mask,
                )
                probability = float(torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)[0])
                pred_id = int(probability >= threshold)
                row.update(
                    {
                        "num_frames_original": original_seq_len,
                        "num_frames_used": int(frames_tensor.shape[0]),
                        "truncated_to_model_max_len": int(original_seq_len > model_max_seq_len),
                        "probability_live": round(probability, 8),
                        "threshold": round(float(threshold), 8),
                        "prediction_id": pred_id,
                        "prediction_name": "live" if pred_id == 1 else "spoof",
                        "correct": "" if label_id is None else int(pred_id == int(label_id)),
                        "status": "ok",
                    }
                )
            except Exception as exc:
                row["error"] = repr(exc)
            rows.append(row)

            if index == 1 or index % 20 == 0 or index == len(samples):
                print(f"processed {index}/{len(samples)} videos", flush=True)

    overall_metrics = evaluate_rows(rows, threshold=float(threshold))
    source_metrics = evaluate_by_subset(rows, "source_key", threshold=float(threshold))
    category_metrics = evaluate_by_subset(rows, "category", threshold=float(threshold))
    label_metrics = evaluate_by_subset(rows, "label_name", threshold=float(threshold))

    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_root": str(Path(args.data_root).resolve()) if args.data_root else "",
        "split_source": args.split_source,
        "split": args.split if args.data_root else "",
        "video_path": str(Path(args.video_path).resolve()) if args.video_path else "",
        "device": str(device),
        "threshold_used": float(threshold),
        "model_max_seq_len_used": model_max_seq_len,
        "total_samples": len(samples),
        "ok_samples": sum(1 for row in rows if row["status"] == "ok"),
        "failed_samples": sum(1 for row in rows if row["status"] != "ok"),
        "elapsed_seconds": round(time.time() - started_at, 3),
        "notes": [
            "该脚本只针对纯视频正式测评，不处理图片。",
            "如果视频帧数超过 V2 位置编码支持长度，会自动裁剪到 model_max_seq_len。",
            "对 protocol 数据集而言，txt 可能是按固定三色协议自动合成，用于模拟 V2 训练时的输入格式。",
        ],
        "overall_metrics": overall_metrics,
        "source_metrics": source_metrics,
        "category_metrics": category_metrics,
        "label_metrics": label_metrics,
        "result_files": {
            "rows_csv": str(output_dir / "v2_video_eval_rows.csv"),
            "source_metrics_csv": str(output_dir / "v2_video_eval_source_metrics.csv"),
            "category_metrics_csv": str(output_dir / "v2_video_eval_category_metrics.csv"),
            "label_metrics_csv": str(output_dir / "v2_video_eval_label_metrics.csv"),
            "summary_json": str(output_dir / "v2_video_eval_summary.json"),
        },
    }

    write_csv(output_dir / "v2_video_eval_rows.csv", rows)
    write_csv(output_dir / "v2_video_eval_source_metrics.csv", source_metrics)
    write_csv(output_dir / "v2_video_eval_category_metrics.csv", category_metrics)
    write_csv(output_dir / "v2_video_eval_label_metrics.csv", label_metrics)
    save_json(output_dir / "v2_video_eval_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
