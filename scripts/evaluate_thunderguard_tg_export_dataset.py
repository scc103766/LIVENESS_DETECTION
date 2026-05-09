from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnxruntime


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def read_colors(file_path: Path) -> list[tuple[int, int, int]]:
    line_list: list[tuple[int, int, int]] = []
    with file_path.open("r", encoding="utf-8") as file:
        for raw in file:
            line = raw.strip()
            if not line:
                continue
            color = int(line)
            c1 = ((color & 0x00FF0000) >> 16)
            c2 = ((color & 0x0000FF00) >> 8)
            c3 = (color & 0x000000FF)
            line_list.append((c1, c2, c3))
    return line_list


def only_load_image(jpg_path: Path) -> np.ndarray:
    img = cv2.imread(str(jpg_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"failed_to_read_image:{jpg_path}")
    normal_cue_list = []
    for i in range(3):
        for j in range(2):
            nid = i * 2 + j
            cur_img = img[nid * 288 + 16 : (nid + 1) * 288 - 16, 16 : 288 - 16, :].transpose((2, 0, 1)).astype(
                np.float32
            )
            cur_img = cur_img.reshape(1, 3, 256, 256)
            normal_cue_list.append(cur_img)
    return np.concatenate(normal_cue_list, axis=0)


def load_sample(sample_txt: Path) -> tuple[np.ndarray, np.ndarray]:
    sc = np.array(read_colors(sample_txt), dtype=np.int32)
    if sc.shape != (3, 3):
        raise RuntimeError(f"unexpected_color_shape:{sample_txt}:{sc.shape}")
    normal_cues = only_load_image(sample_txt.with_suffix(".jpg"))
    sc_flag = []
    for i in range(3):
        sc_flag.append(np.where(sc[i] > sc[i, [1, 2, 0]], 1, 0) + np.where(sc[i] < sc[i, [1, 2, 0]], -1, 0))
        sc_flag.append(sc_flag[-1])
    return normal_cues, np.array(sc_flag, dtype=np.int32)


def evaluate_color_validation(color_output: np.ndarray, sc_flag: np.ndarray) -> tuple[int, list[dict[str, Any]]]:
    sc = np.reshape(color_output, [-1, 2, 3])
    flags = np.reshape(sc_flag, [-1, 2, 3])
    details: list[dict[str, Any]] = []
    passed = 1
    for pair_index, (sc_pair, flag_pair) in enumerate(zip(sc, flags)):
        error_time = 0
        pair_detail = {"pair_index": pair_index, "sub_checks": []}
        for i in range(2):
            sp = sc_pair[i]
            fl = flag_pair[i]
            violation = bool(np.min(fl * (sp - 0.5)) < 0)
            if violation:
                error_time += 1
            pair_detail["sub_checks"].append(
                {
                    "sub_index": i,
                    "score_triplet": np.asarray(sp).round(6).tolist(),
                    "flag_triplet": np.asarray(fl).tolist(),
                    "violation": violation,
                }
            )
        pair_detail["pair_pass"] = bool(error_time < 2)
        details.append(pair_detail)
        if error_time == 2:
            passed = 0
    return passed, details


def confusion_metrics(labels: np.ndarray, preds: np.ndarray) -> dict[str, Any]:
    labels = labels.astype(np.int32)
    preds = preds.astype(np.int32)
    tp = int(np.sum((labels == 1) & (preds == 1)))
    tn = int(np.sum((labels == 0) & (preds == 0)))
    fp = int(np.sum((labels == 0) & (preds == 1)))
    fn = int(np.sum((labels == 1) & (preds == 0)))
    total = int(labels.size)
    live_count = int(np.sum(labels == 1))
    spoof_count = int(np.sum(labels == 0))
    accuracy = float((tp + tn) / total) if total else 0.0
    apcer = float(fp / spoof_count) if spoof_count else 0.0
    bpcer = float(fn / live_count) if live_count else 0.0
    acer = float((apcer + bpcer) / 2.0)
    live_pass_rate = float(tp / live_count) if live_count else 0.0
    spoof_block_rate = float(tn / spoof_count) if spoof_count else 0.0
    overall_pass_rate = float(np.sum(preds == 1) / total) if total else 0.0
    return {
        "count": total,
        "live_count": live_count,
        "spoof_count": spoof_count,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": acer,
        "live_pass_rate": live_pass_rate,
        "spoof_block_rate": spoof_block_rate,
        "overall_pass_rate": overall_pass_rate,
    }


class ONNXModel:
    def __init__(self, onnx_path: Path) -> None:
        session_options = onnxruntime.SessionOptions()
        session_options.log_severity_level = 3
        self.session = onnxruntime.InferenceSession(str(onnx_path), sess_options=session_options)
        self.input_name = self.session.get_inputs()[0].name
        outputs = self.session.get_outputs()
        self.output_names = [node.name for node in outputs]
        if len(self.output_names) != 2:
            raise RuntimeError(f"unexpected_output_count:{len(self.output_names)}")

    def forward(self, normal_cues: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        color_pred, score_pred = self.session.run(self.output_names, {self.input_name: (normal_cues - 127.5) / 128.0})
        return color_pred, score_pred


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ThunderGuard ONNX on a tg_export-style dataset via manifest.tsv.")
    parser.add_argument(
        "--manifest",
        default="/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/tg_export_from_current_dataset/manifest.tsv",
        help="Manifest TSV generated during tg_export_from_current_dataset export.",
    )
    parser.add_argument(
        "--onnx-path",
        default="/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/ThunderGuard/resources/MoEA_score.onnx",
        help="ThunderGuard score ONNX model path.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9061252,
        help="Live score threshold. Default uses model_best.pth.tar val_threshold.",
    )
    parser.add_argument(
        "--output-dir",
        default="/supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_runs/thunderguard_eval_tg_export_from_current_dataset",
        help="Directory to save report files.",
    )
    parser.add_argument(
        "--split",
        default="all",
        choices=["all", "train", "val", "test"],
        help="Which manifest split to evaluate.",
    )
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    manifest_path = Path(args.manifest).resolve()
    onnx_path = Path(args.onnx_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    worker = ONNXModel(onnx_path)
    manifest_rows = read_manifest(manifest_path)
    selected_rows = [
        row
        for row in manifest_rows
        if row.get("status") == "ok" and (args.split == "all" or row.get("split") == args.split)
    ]

    detailed_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for row in selected_rows:
        sample_txt = Path(row["sample_txt"])
        label_id = int(row["label_id"])
        try:
            if len(detailed_rows) % 25 == 0:
                print(
                    f"[{len(detailed_rows) + len(failures) + 1}/{len(selected_rows)}] evaluating {sample_txt.name}",
                    flush=True,
                )
            normal_cues, sc_flag = load_sample(sample_txt)
            color_pred, score_pred = worker.forward(normal_cues)
            score = float(np.reshape(score_pred, [-1])[0])
            color_pass, color_details = evaluate_color_validation(color_pred, sc_flag)
            prediction_raw = 1 if score > args.threshold else 0
            prediction_with_color = prediction_raw if color_pass == 1 else 0
            detailed_rows.append(
                {
                    "index": int(row["index"]),
                    "split": row["split"],
                    "label_id": label_id,
                    "label_name": row["label_name"],
                    "video_path": row["video_path"],
                    "sample_jpg": row["sample_jpg"],
                    "sample_txt": row["sample_txt"],
                    "score": round(score, 8),
                    "threshold": round(float(args.threshold), 8),
                    "prediction_raw_id": prediction_raw,
                    "prediction_raw_name": "live" if prediction_raw == 1 else "spoof",
                    "prediction_with_color_id": prediction_with_color,
                    "prediction_with_color_name": "live" if prediction_with_color == 1 else "spoof",
                    "color_validation_pass": int(color_pass),
                    "raw_correct": int(prediction_raw == label_id),
                    "color_correct": int(prediction_with_color == label_id),
                    "color_head_output": np.reshape(color_pred, [-1, 3]).round(6).tolist(),
                    "color_validation_details": color_details,
                }
            )
        except Exception as exc:
            failures.append(
                {
                    "index": row.get("index"),
                    "split": row.get("split"),
                    "sample_txt": row.get("sample_txt"),
                    "video_path": row.get("video_path"),
                    "error": repr(exc),
                }
            )

    raw_labels = np.asarray([row["label_id"] for row in detailed_rows], dtype=np.int32)
    raw_preds = np.asarray([row["prediction_raw_id"] for row in detailed_rows], dtype=np.int32)
    color_preds = np.asarray([row["prediction_with_color_id"] for row in detailed_rows], dtype=np.int32)
    color_pass = np.asarray([row["color_validation_pass"] for row in detailed_rows], dtype=np.int32)

    overall_raw = confusion_metrics(raw_labels, raw_preds)
    overall_with_color = confusion_metrics(raw_labels, color_preds)

    split_reports = {}
    for split_name in sorted(set(row["split"] for row in detailed_rows)):
        subset = [row for row in detailed_rows if row["split"] == split_name]
        labels = np.asarray([row["label_id"] for row in subset], dtype=np.int32)
        preds_raw = np.asarray([row["prediction_raw_id"] for row in subset], dtype=np.int32)
        preds_color = np.asarray([row["prediction_with_color_id"] for row in subset], dtype=np.int32)
        split_reports[split_name] = {
            "raw_score_only": confusion_metrics(labels, preds_raw),
            "score_plus_color_validation": confusion_metrics(labels, preds_color),
            "color_validation_pass_rate": float(np.mean([row["color_validation_pass"] for row in subset])) if subset else 0.0,
        }

    summary = {
        "manifest": str(manifest_path),
        "onnx_path": str(onnx_path),
        "threshold_used": float(args.threshold),
        "selected_split": args.split,
        "manifest_rows_total": len(manifest_rows),
        "selected_manifest_rows": len(selected_rows),
        "evaluated_ok_rows": len(detailed_rows),
        "runtime_failures": len(failures),
        "color_validation_pass_rate": float(color_pass.mean()) if color_pass.size else 0.0,
        "raw_score_only": overall_raw,
        "score_plus_color_validation": overall_with_color,
        "per_split": split_reports,
        "notes": [
            "raw_score_only 表示只用 MoEA_score.onnx 的 score 和阈值做 live/spoof 判决。",
            "score_plus_color_validation 表示在 raw_score_only 基础上，再要求 ThunderGuard 颜色校验通过；未通过时强制判为 spoof。",
            "tg_export_from_current_dataset 的 _d.jpg 是占位深度图，不是原始 FaceAlign 深度图，因此该评测更适合看当前导出格式与旧模型的兼容性，而不是与原始 tg_export 横向等价。",
        ],
    }

    write_csv(
        output_dir / "thunderguard_eval_rows.csv",
        [
            {
                k: v
                for k, v in row.items()
                if k not in {"color_head_output", "color_validation_details"}
            }
            for row in detailed_rows
        ],
    )
    save_json(output_dir / "thunderguard_eval_detailed.json", detailed_rows)
    save_json(output_dir / "thunderguard_eval_failures.json", failures)
    save_json(output_dir / "thunderguard_eval_summary.json", summary)

    report_lines = [
        "ThunderGuard tg_export_from_current_dataset Evaluation Report",
        "",
        f"Manifest: {manifest_path}",
        f"ONNX: {onnx_path}",
        f"Threshold: {args.threshold:.7f}",
        f"Selected split: {args.split}",
        f"Selected rows: {len(selected_rows)}",
        f"Evaluated rows: {len(detailed_rows)}",
        f"Runtime failures: {len(failures)}",
        f"Color validation pass rate: {summary['color_validation_pass_rate']:.4f}",
        "",
        "Raw score only metrics:",
        json.dumps(overall_raw, indent=2, ensure_ascii=False),
        "",
        "Score + color validation metrics:",
        json.dumps(overall_with_color, indent=2, ensure_ascii=False),
        "",
        "Per-split metrics:",
        json.dumps(split_reports, indent=2, ensure_ascii=False),
    ]
    (output_dir / "thunderguard_eval_report.txt").write_text("\n".join(report_lines), encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved report to: {output_dir / 'thunderguard_eval_report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
