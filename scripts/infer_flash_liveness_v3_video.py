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

from flash_liveness_project_v3 import (  # noqa: E402
    FacePreprocessor,
    FlashLivenessDataset,
    LivenessSample,
    compute_auc,
    compute_binary_metrics,
    compute_eer,
    discover_dataset_splits,
    load_checkpoint,
)
from flash_liveness_project_v3_1 import (  # noqa: E402
    estimate_window_quality,
    fuse_window_probabilities,
    iter_window_ranges,
    make_window_mask,
)


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


def sample_to_eval_tuple(sample: LivenessSample, split_name: str) -> tuple[str, str | None, int, str, str, str]:
    video_path_obj = Path(sample.media_path)
    return (
        str(sample.media_path),
        str(sample.txt_path) if sample.txt_path else None,
        int(sample.label),
        str(sample.category),
        str(sample.source_group),
        split_name if split_name else video_path_obj.parents[1].name,
    )


def resolve_samples(args: argparse.Namespace) -> list[tuple[str, str | None, int | None, str, str, str]]:
    if args.video_path:
        video_path = str(Path(args.video_path).resolve())
        txt_path = str(Path(args.txt_path).resolve()) if args.txt_path else str(Path(video_path).with_suffix(".txt"))
        return [(video_path, txt_path, args.label, "single_video", "single_video", "single")]

    if not args.data_root:
        raise ValueError("批量测试请提供 --data-root，单视频请提供 --video-path。")

    split_map = discover_dataset_splits(
        args.data_root,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        media_filter="videos",
        require_color_txt=False,
    )

    split_names = ["train", "val", "test"] if args.split == "all" else [args.split]
    samples: list[tuple[str, str | None, int | None, str, str, str]] = []
    for split_name in split_names:
        for sample in split_map.get(split_name, []):
            samples.append(sample_to_eval_tuple(sample, split_name))

    if args.samples_per_class is not None:
        filtered_samples = []
        for split_name in split_names:
            split_rows = [row for row in samples if row[5] == split_name]
            live_rows = [row for row in split_rows if row[2] == 1][:args.samples_per_class]
            spoof_rows = [row for row in split_rows if row[2] == 0][:args.samples_per_class]
            filtered_samples.extend(live_rows + spoof_rows)
        samples = filtered_samples

    if args.limit is not None:
        samples = samples[:args.limit]
    return samples


def infer_single_video(
    model,
    dataset: FlashLivenessDataset,
    device: torch.device,
    video_path: str,
    txt_path: str | None,
    inference_mode: str,
    max_seq_len: int,
    window_size: int,
    window_stride: int,
    window_fusion: str,
    window_trim_ratio: float,
    window_min_quality: float,
    show_decoder_warnings: bool,
) -> dict:
    with suppress_native_stderr(not show_decoder_warnings):
        frames_tensor, color_tensor, physical_tensor, _ = dataset.process_video(video_path, txt_path)
    if frames_tensor.numel() == 0:
        raise RuntimeError("no_valid_frames_decoded")

    num_frames = int(frames_tensor.shape[0])
    if inference_mode == "full_sequence":
        frames_for_infer = frames_tensor
        colors_for_infer = color_tensor
        physical_for_infer = physical_tensor
        if max_seq_len > 0 and num_frames > max_seq_len:
            frames_for_infer = frames_for_infer[:max_seq_len]
            colors_for_infer = colors_for_infer[:max_seq_len]
            physical_for_infer = physical_for_infer[:max_seq_len]

        with torch.no_grad():
            frames = frames_for_infer.unsqueeze(0).to(device)
            colors = colors_for_infer.unsqueeze(0).to(device)
            physical = physical_for_infer.unsqueeze(0).to(device)
            padding_mask = torch.zeros((1, frames_for_infer.shape[0]), dtype=torch.bool, device=device)
            outputs = model(frames, colors, physical, padding_mask)
            probability_live = float(torch.sigmoid(outputs["logits"]).detach().cpu().item())

        num_frames_used = int(frames_for_infer.shape[0])
        return {
            "probability_live": float(probability_live),
            "num_frames": num_frames,
            "num_frames_used": num_frames_used,
            "truncated_to_model_max_len": int(num_frames_used < num_frames),
            "num_windows": 1,
            "mean_window_quality": 0.0,
            "min_window_quality": 0.0,
            "max_window_quality": 0.0,
            "window_probabilities": [float(probability_live)],
            "window_qualities": [],
        }

    ranges = iter_window_ranges(num_frames, window_size, window_stride)
    probabilities: list[float] = []
    qualities: list[float] = []

    with torch.no_grad():
        for start, end in ranges:
            frames = frames_tensor[start:end].unsqueeze(0).to(device)
            colors = color_tensor[start:end].unsqueeze(0).to(device)
            physical = physical_tensor[start:end].unsqueeze(0).to(device)
            padding_mask = make_window_mask(end - start, device)
            outputs = model(frames, colors, physical, padding_mask)
            probabilities.append(float(torch.sigmoid(outputs["logits"]).detach().cpu().item()))
            qualities.append(estimate_window_quality(frames))

    probability_live = fuse_window_probabilities(
        probabilities,
        qualities,
        method=window_fusion,
        trim_ratio=window_trim_ratio,
        min_quality=window_min_quality,
    )
    return {
        "probability_live": float(probability_live),
        "num_frames": num_frames,
        "num_frames_used": num_frames,
        "truncated_to_model_max_len": 0,
        "num_windows": len(ranges),
        "mean_window_quality": float(np.mean(qualities)) if qualities else 0.0,
        "min_window_quality": float(np.min(qualities)) if qualities else 0.0,
        "max_window_quality": float(np.max(qualities)) if qualities else 0.0,
        "window_probabilities": [float(item) for item in probabilities],
        "window_qualities": [float(item) for item in qualities],
    }


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
            "mean_probability_live": float(probs.mean()) if probs.size else 0.0,
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


def render_markdown_report(
    args: argparse.Namespace,
    checkpoint_path: str,
    threshold: float,
    overall_metrics: dict | None,
    split_metrics: list[dict],
    source_metrics: list[dict],
    category_metrics: list[dict],
    rows: list[dict],
) -> str:
    ok_rows = [row for row in rows if row["status"] == "ok"]
    failed_rows = [row for row in rows if row["status"] != "ok"]
    lines = [
        "# Flash Liveness V3 Fixed-Protocol Inference Report",
        "",
        "## Run Summary",
        f"- checkpoint: `{checkpoint_path}`",
        f"- data_root: `{args.data_root or args.video_path}`",
        f"- split: `{args.split}`",
        f"- threshold: `{threshold:.6f}`",
        f"- device: `{args.device or 'auto'}`",
        f"- total_samples: `{len(rows)}`",
        f"- success_samples: `{len(ok_rows)}`",
        f"- failed_samples: `{len(failed_rows)}`",
        f"- inference_mode: `{args.inference_mode}`",
        "",
        "## Overall Metrics",
    ]

    if args.inference_mode == "window":
        lines[10:10] = [
            f"- window_size: `{args.window_size}`",
            f"- window_stride: `{args.window_stride}`",
            f"- window_fusion: `{args.window_fusion}`",
            f"- window_trim_ratio: `{args.window_trim_ratio}`",
            f"- window_min_quality: `{args.window_min_quality}`",
        ]
    else:
        lines[10:10] = [f"- model_max_seq_len: `{args.max_seq_len if args.max_seq_len > 0 else 512}`"]

    if overall_metrics is None:
        lines.extend(["- 无可计算指标的标注样本。", ""])
    else:
        lines.extend(
            [
                f"- accuracy: `{overall_metrics['accuracy']:.6f}`",
                f"- auc: `{overall_metrics['auc']:.6f}`",
                f"- eer: `{overall_metrics['eer']:.6f}`",
                f"- apcer: `{overall_metrics['apcer']:.6f}`",
                f"- bpcer: `{overall_metrics['bpcer']:.6f}`",
                f"- acer: `{overall_metrics['acer']:.6f}`",
                f"- tp/tn/fp/fn: `{overall_metrics['tp']}/{overall_metrics['tn']}/{overall_metrics['fp']}/{overall_metrics['fn']}`",
                "",
            ]
        )

    def append_table(title: str, metric_rows: list[dict], subset_key: str = "subset_value") -> None:
        lines.append(f"## {title}")
        if not metric_rows:
            lines.extend(["- 无数据", ""])
            return
        lines.append("| subset | total | accuracy | auc | eer | acer |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
        for row in metric_rows:
            lines.append(
                f"| {row[subset_key]} | {row['total_labeled']} | "
                f"{row['accuracy']:.6f} | {row['auc']:.6f} | {row['eer']:.6f} | {row['acer']:.6f} |"
            )
        lines.append("")

    append_table("Per Split Metrics", split_metrics)
    append_table("Per Source Metrics", source_metrics)
    append_table("Per Category Metrics", category_metrics)

    if failed_rows:
        lines.append("## Failed Samples")
        for row in failed_rows[:50]:
            lines.append(f"- `{row['video_path']}` -> `{row['error']}`")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Video inference and dataset evaluation for flash_liveness_project_v3 checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="V3 best/last checkpoint path")
    parser.add_argument("--data-root", default=None, help="数据集根目录")
    parser.add_argument("--split", default="all", choices=["train", "val", "test", "all"])
    parser.add_argument("--video-path", default=None, help="单视频路径")
    parser.add_argument("--txt-path", default=None, help="单视频同名 txt，可选")
    parser.add_argument("--label", type=int, choices=[0, 1], default=None, help="单视频标签")
    parser.add_argument("--output-dir", required=True, help="结果输出目录")
    parser.add_argument("--threshold", type=float, default=None, help="默认使用 checkpoint 内阈值")
    parser.add_argument("--device", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--samples-per-class", type=int, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--inference-mode",
        choices=["window", "full_sequence"],
        default="window",
        help="window 保持当前 V3.1 滑窗融合；full_sequence 按 V2 风格整段直接前向。",
    )
    parser.add_argument("--max-seq-len", type=int, default=0, help="full_sequence 模式下的最大序列长度；0 表示按 V2 兼容方式使用 512。")
    parser.add_argument("--window-size", type=int, default=256, help="窗口长度，建议与训练 max_eval_frames 对齐")
    parser.add_argument("--window-stride", type=int, default=128, help="窗口滑动步长")
    parser.add_argument(
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
        help="视频级融合策略，默认保留低 live 可疑窗口，避免局部高 live 分数盖掉攻击证据。",
    )
    parser.add_argument("--window-trim-ratio", type=float, default=0.2, help="trimmed/lower_trimmed 融合比例")
    parser.add_argument("--window-min-quality", type=float, default=0.05, help="质量权重最小值")
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

    preprocessor = FacePreprocessor(
        detector_model_path=args.detector_model,
        detector_device=args.detector_device,
        target_size=tuple(config["target_size"]),
        conf_threshold=args.detector_conf,
        iou_threshold=args.detector_iou,
    )
    dataset = FlashLivenessDataset(
        samples=[],
        transform=False,
        preprocessor=preprocessor,
        target_size=tuple(config["target_size"]),
        max_frames=0,
        frame_stride=1,
        use_physical_features=config.get("physical_dim", 19) > 0,
        missing_color_protocol=config.get("missing_color_protocol", "collect_flash"),
        flash_warmup_seconds=config.get("flash_warmup_seconds", 1.0),
        flash_hold_seconds=config.get("flash_hold_seconds", 0.35),
        flash_tail_seconds=config.get("flash_tail_seconds", 0.5),
    )

    samples = resolve_samples(args)
    rows: list[dict] = []
    started_at = time.time()

    model.eval()
    for index, (video_path, txt_path, label_id, category, source_group, split_name) in enumerate(samples, start=1):
        row = {
            "index": index,
            "split": split_name,
            "video_path": video_path,
            "txt_path": txt_path or "",
            "label_id": "" if label_id is None else int(label_id),
            "label_name": "" if label_id is None else ("live" if int(label_id) == 1 else "spoof"),
            "category": category,
            "source_group": source_group,
            "status": "failed",
            "error": "",
        }
        try:
            result = infer_single_video(
                model=model,
                dataset=dataset,
                device=device,
                video_path=video_path,
                txt_path=txt_path,
                inference_mode=args.inference_mode,
                max_seq_len=model_max_seq_len,
                window_size=args.window_size,
                window_stride=args.window_stride,
                window_fusion=args.window_fusion,
                window_trim_ratio=args.window_trim_ratio,
                window_min_quality=args.window_min_quality,
                show_decoder_warnings=args.show_decoder_warnings,
            )
            probability_live = float(result["probability_live"])
            pred_id = int(probability_live >= threshold)
            row.update(
                {
                    "status": "ok",
                    "probability_live": round(probability_live, 8),
                    "threshold": round(float(threshold), 8),
                    "prediction_id": pred_id,
                    "prediction_name": "live" if pred_id == 1 else "spoof",
                    "is_correct": "" if label_id is None else int(pred_id == int(label_id)),
                    "num_frames": int(result["num_frames"]),
                    "num_frames_used": int(result["num_frames_used"]),
                    "truncated_to_model_max_len": int(result["truncated_to_model_max_len"]),
                    "num_windows": int(result["num_windows"]),
                    "mean_window_quality": round(float(result["mean_window_quality"]), 8),
                    "min_window_quality": round(float(result["min_window_quality"]), 8),
                    "max_window_quality": round(float(result["max_window_quality"]), 8),
                }
            )
        except Exception as exc:
            row["error"] = str(exc)
        rows.append(row)

        if index % 20 == 0 or index == len(samples):
            elapsed = time.time() - started_at
            print(
                f"[{index}/{len(samples)}] processed, success={sum(item['status'] == 'ok' for item in rows)}, "
                f"failed={sum(item['status'] != 'ok' for item in rows)}, elapsed={elapsed:.1f}s",
                flush=True,
            )

    overall_metrics = evaluate_rows(rows, threshold=float(threshold))
    split_metrics = evaluate_by_subset(rows, "split", threshold=float(threshold))
    source_metrics = evaluate_by_subset(rows, "source_group", threshold=float(threshold))
    category_metrics = evaluate_by_subset(rows, "category", threshold=float(threshold))
    label_metrics = evaluate_by_subset(rows, "label_name", threshold=float(threshold))

    result_summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "data_root": str(Path(args.data_root).resolve()) if args.data_root else None,
        "single_video": str(Path(args.video_path).resolve()) if args.video_path else None,
        "threshold_used": float(threshold),
        "device": str(device),
        "inference_mode": args.inference_mode,
        "model_max_seq_len_used": model_max_seq_len if args.inference_mode == "full_sequence" else None,
        "window_size": int(args.window_size) if args.inference_mode == "window" else None,
        "window_stride": int(args.window_stride) if args.inference_mode == "window" else None,
        "window_fusion": args.window_fusion if args.inference_mode == "window" else None,
        "window_trim_ratio": float(args.window_trim_ratio) if args.inference_mode == "window" else None,
        "window_min_quality": float(args.window_min_quality) if args.inference_mode == "window" else None,
        "elapsed_seconds": round(time.time() - started_at, 4),
        "total_rows": len(rows),
        "success_rows": sum(row["status"] == "ok" for row in rows),
        "failed_rows": sum(row["status"] != "ok" for row in rows),
        "overall_metrics": overall_metrics,
        "split_metrics": split_metrics,
        "source_metrics": source_metrics,
        "category_metrics": category_metrics,
        "label_metrics": label_metrics,
    }

    rows_csv_path = output_dir / "predictions.csv"
    summary_json_path = output_dir / "summary.json"
    summary_md_path = output_dir / "summary.md"
    write_csv(rows_csv_path, rows)
    save_json(summary_json_path, result_summary)
    summary_md_path.write_text(
        render_markdown_report(
            args=args,
            checkpoint_path=result_summary["checkpoint"],
            threshold=float(threshold),
            overall_metrics=overall_metrics,
            split_metrics=split_metrics,
            source_metrics=source_metrics,
            category_metrics=category_metrics,
            rows=rows,
        ),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "rows_csv": str(rows_csv_path),
                "summary_json": str(summary_json_path),
                "summary_md": str(summary_md_path),
                "overall_metrics": overall_metrics,
                "elapsed_seconds": result_summary["elapsed_seconds"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
