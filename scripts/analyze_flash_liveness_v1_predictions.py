from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flash_liveness_project import compute_auc, compute_binary_metrics, compute_eer, find_best_threshold


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze V1 flash-liveness predictions and sweep thresholds.")
    parser.add_argument("--predictions-csv", required=True, help="Path to predictions.csv from test_flash_liveness_v1_collect_protocol.py")
    parser.add_argument("--output-dir", required=True, help="Directory for analysis outputs.")
    parser.add_argument(
        "--grid-size",
        type=int,
        default=2001,
        help="Number of evenly spaced thresholds in [0,1] for rescanning.",
    )
    parser.add_argument(
        "--optimize-metric",
        choices=["acer", "accuracy"],
        default="acer",
        help="Metric used to pick the best threshold from the rescanned grid.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def to_labeled_ok_rows(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if row.get("status") == "ok" and row.get("label_id", "") != "" and row.get("probability_live", "") != ""
    ]


def build_misclassified_rows(rows: list[dict], threshold: float) -> list[dict]:
    misclassified: list[dict] = []
    for row in rows:
        prob = float(row["probability_live"])
        label_id = int(row["label_id"])
        pred_id = int(prob >= threshold)
        if pred_id == label_id:
            continue
        misclassified.append(
            {
                "video_path": row["video_path"],
                "txt_path": row.get("txt_path", ""),
                "label_id": label_id,
                "label_name": "live" if label_id == 1 else "spoof",
                "pred_id": pred_id,
                "pred_name": "live" if pred_id == 1 else "spoof",
                "probability_live": round(prob, 8),
                "threshold": round(float(threshold), 8),
                "error_type": "fp_spoof_to_live" if label_id == 0 else "fn_live_to_spoof",
                "frame_count": row.get("frame_count", ""),
                "fps": row.get("fps", ""),
                "has_color_txt": row.get("has_color_txt", ""),
            }
        )
    misclassified.sort(key=lambda item: (-item["probability_live"], item["video_path"]))
    return misclassified


def evaluate_thresholds(labels: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> list[dict]:
    results: list[dict] = []
    for threshold in thresholds:
        metrics = compute_binary_metrics(labels, probs, float(threshold))
        results.append(
            {
                "threshold": round(float(threshold), 8),
                "accuracy": round(float(metrics["accuracy"]), 8),
                "apcer": round(float(metrics["apcer"]), 8),
                "bpcer": round(float(metrics["bpcer"]), 8),
                "acer": round(float(metrics["acer"]), 8),
                "tp": int(metrics["tp"]),
                "tn": int(metrics["tn"]),
                "fp": int(metrics["fp"]),
                "fn": int(metrics["fn"]),
            }
        )
    return results


def pick_best_threshold(rows: list[dict], optimize_metric: str) -> dict:
    if optimize_metric == "acer":
        return min(rows, key=lambda item: (item["acer"], -item["accuracy"], item["threshold"]))
    return max(rows, key=lambda item: (item["accuracy"], -item["acer"], -item["threshold"]))


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = load_rows(Path(args.predictions_csv))
    labeled_rows = to_labeled_ok_rows(all_rows)
    labels = np.asarray([int(row["label_id"]) for row in labeled_rows], dtype=np.int32)
    probs = np.asarray([float(row["probability_live"]) for row in labeled_rows], dtype=np.float64)

    unique_threshold = float(find_best_threshold(labels, probs))
    eer, eer_threshold = compute_eer(labels, probs)
    auc = compute_auc(labels, probs)

    grid_thresholds = np.linspace(0.0, 1.0, args.grid_size, dtype=np.float64)
    threshold_rows = evaluate_thresholds(labels, probs, grid_thresholds)
    best_grid = pick_best_threshold(threshold_rows, args.optimize_metric)
    best_unique_metrics = compute_binary_metrics(labels, probs, unique_threshold)

    default_threshold = float(all_rows[0]["threshold"]) if all_rows else 0.5
    default_metrics = compute_binary_metrics(labels, probs, default_threshold)
    best_grid_misclassified = build_misclassified_rows(labeled_rows, float(best_grid["threshold"]))
    default_misclassified = build_misclassified_rows(labeled_rows, default_threshold)

    summary = {
        "predictions_csv": str(Path(args.predictions_csv).resolve()),
        "sample_count": len(labeled_rows),
        "default_threshold": default_threshold,
        "default_metrics": {**default_metrics, "auc": auc, "eer": eer, "eer_threshold": eer_threshold},
        "best_threshold_from_unique_probs": {
            "threshold": unique_threshold,
            "optimize_metric": "balanced_accuracy_via_find_best_threshold",
            **best_unique_metrics,
        },
        "best_threshold_from_grid": {
            **best_grid,
            "optimize_metric": args.optimize_metric,
        },
        "misclassified_count_default": len(default_misclassified),
        "misclassified_count_best_grid": len(best_grid_misclassified),
        "files": {
            "threshold_scan_csv": str(output_dir / "threshold_scan.csv"),
            "misclassified_default_csv": str(output_dir / "misclassified_default_threshold.csv"),
            "misclassified_best_csv": str(output_dir / "misclassified_best_threshold.csv"),
            "summary_json": str(output_dir / "analysis_summary.json"),
        },
    }

    write_csv(output_dir / "threshold_scan.csv", threshold_rows)
    write_csv(output_dir / "misclassified_default_threshold.csv", default_misclassified)
    write_csv(output_dir / "misclassified_best_threshold.csv", best_grid_misclassified)
    save_json(output_dir / "analysis_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
