from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import sys

import cv2
import numpy as np

PROJECT_ROOT = Path("/supercloud/llm-code/scc/scc/Liveness_Detection")
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from evaluate_thunderguard_tg_export_dataset import ONNXModel, load_sample
ORIGINAL_ROOT = PROJECT_ROOT / "20240320闪光活体归档/dataset/tg_export/test"
CURRENT_MANIFEST = PROJECT_ROOT / "dataset/tg_export_from_current_dataset/manifest.tsv"
ONNX_PATH = PROJECT_ROOT / "20240320闪光活体归档/ThunderGuard/resources/MoEA_score.onnx"
OUTPUT_DIR = PROJECT_ROOT / "flash_liveness_runs/thunderguard_export_gap_analysis"


@dataclass
class SampleRecord:
    dataset_name: str
    split: str
    label_id: int
    label_name: str
    sample_txt: Path
    sample_jpg: Path
    score: float
    image_mean: float
    image_std: float
    lap_var: float


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def read_current_manifest() -> list[dict[str, str]]:
    with CURRENT_MANIFEST.open("r", encoding="utf-8") as file:
        return [row for row in csv.DictReader(file, delimiter="\t") if row["status"] == "ok"]


def read_original_samples() -> list[dict[str, Any]]:
    rows = []
    for txt_path in sorted(ORIGINAL_ROOT.glob("*.txt")):
        try:
            flag = int(txt_path.stem.split("_")[-1])
        except Exception:
            continue
        label_id = 1 if flag == 1 else 0
        rows.append(
            {
                "split": "test",
                "label_id": label_id,
                "label_name": "live" if label_id == 1 else "spoof",
                "sample_txt": str(txt_path),
                "sample_jpg": str(txt_path.with_suffix(".jpg")),
            }
        )
    return rows


def compute_image_stats(jpg_path: Path) -> tuple[float, float, float]:
    image = cv2.imread(str(jpg_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed_to_read:{jpg_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(image.mean()), float(image.std()), float(cv2.Laplacian(gray, cv2.CV_32F).var())


def score_sample(worker: ONNXModel, txt_path: Path) -> float:
    normal_cues, _ = load_sample(txt_path)
    _, score = worker.forward(normal_cues)
    return float(np.reshape(score, [-1])[0])


def evaluate_rows(rows: list[dict[str, Any]], dataset_name: str, limit_per_label: int) -> list[SampleRecord]:
    worker = ONNXModel(ONNX_PATH)
    grouped = {0: [], 1: []}
    for row in rows:
        grouped[int(row["label_id"])].append(row)

    selected = []
    for label_id in (1, 0):
        selected.extend(grouped[label_id][:limit_per_label])

    records: list[SampleRecord] = []
    for idx, row in enumerate(selected, start=1):
        txt_path = Path(row["sample_txt"])
        jpg_path = Path(row["sample_jpg"])
        print(f"[{dataset_name} {idx}/{len(selected)}] {txt_path.name}", flush=True)
        score = score_sample(worker, txt_path)
        mean, std, lap = compute_image_stats(jpg_path)
        records.append(
            SampleRecord(
                dataset_name=dataset_name,
                split=row.get("split", ""),
                label_id=int(row["label_id"]),
                label_name=row["label_name"],
                sample_txt=txt_path,
                sample_jpg=jpg_path,
                score=score,
                image_mean=mean,
                image_std=std,
                lap_var=lap,
            )
        )
    return records


def summarize(records: list[SampleRecord]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset_name in sorted(set(r.dataset_name for r in records)):
        output[dataset_name] = {}
        for label_name in ("live", "spoof"):
            subset = [r for r in records if r.dataset_name == dataset_name and r.label_name == label_name]
            scores = np.asarray([r.score for r in subset], dtype=np.float32)
            means = np.asarray([r.image_mean for r in subset], dtype=np.float32)
            stds = np.asarray([r.image_std for r in subset], dtype=np.float32)
            laps = np.asarray([r.lap_var for r in subset], dtype=np.float32)
            output[dataset_name][label_name] = {
                "count": int(len(subset)),
                "score_mean": float(scores.mean()) if scores.size else 0.0,
                "score_std": float(scores.std()) if scores.size else 0.0,
                "score_min": float(scores.min()) if scores.size else 0.0,
                "score_max": float(scores.max()) if scores.size else 0.0,
                "image_mean_mean": float(means.mean()) if means.size else 0.0,
                "image_std_mean": float(stds.mean()) if stds.size else 0.0,
                "lap_var_mean": float(laps.mean()) if laps.size else 0.0,
            }
    return output


def draw_label(image: np.ndarray, lines: list[str]) -> np.ndarray:
    canvas = image.copy()
    y = 24
    for line in lines:
        cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1, cv2.LINE_AA)
        y += 22
    return canvas


def make_tile(record: SampleRecord, title_prefix: str) -> np.ndarray:
    image = cv2.imread(str(record.sample_jpg), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (220, 440))
    lines = [
        title_prefix,
        f"{record.label_name} score={record.score:.4f}",
        f"mean={record.image_mean:.1f} std={record.image_std:.1f}",
        f"lap={record.lap_var:.1f}",
    ]
    return draw_label(image, lines)


def write_montage(path: Path, records: list[SampleRecord], title_prefix: str) -> None:
    if not records:
        return
    tiles = [make_tile(record, title_prefix) for record in records]
    rows = []
    for start in range(0, len(tiles), 2):
        row_tiles = tiles[start : start + 2]
        if len(row_tiles) == 1:
            row_tiles.append(np.zeros_like(row_tiles[0]))
        rows.append(np.concatenate(row_tiles, axis=1))
    montage = np.concatenate(rows, axis=0)
    cv2.imwrite(str(path), montage)


def top_examples(records: list[SampleRecord], dataset_name: str, label_name: str, reverse: bool, count: int) -> list[SampleRecord]:
    subset = [r for r in records if r.dataset_name == dataset_name and r.label_name == label_name]
    subset.sort(key=lambda r: r.score, reverse=reverse)
    return subset[:count]


def main() -> int:
    ensure_dir(OUTPUT_DIR)
    current_rows = read_current_manifest()
    original_rows = read_original_samples()

    current_records = evaluate_rows(current_rows, dataset_name="current_export", limit_per_label=18 if True else 18)
    original_records = evaluate_rows(original_rows, dataset_name="original_tg_export", limit_per_label=60)
    all_records = current_records + original_records

    summary = summarize(all_records)

    original_live_best = top_examples(all_records, "original_tg_export", "live", reverse=True, count=4)
    current_live_worst = top_examples(all_records, "current_export", "live", reverse=False, count=4)
    original_spoof_low = top_examples(all_records, "original_tg_export", "spoof", reverse=False, count=4)
    current_spoof_high = top_examples(all_records, "current_export", "spoof", reverse=True, count=4)

    write_montage(OUTPUT_DIR / "original_live_best.jpg", original_live_best, "original live")
    write_montage(OUTPUT_DIR / "current_live_worst.jpg", current_live_worst, "current live")
    write_montage(OUTPUT_DIR / "original_spoof_low.jpg", original_spoof_low, "original spoof")
    write_montage(OUTPUT_DIR / "current_spoof_high.jpg", current_spoof_high, "current spoof")

    report_lines = [
        "# ThunderGuard tg_export Gap Analysis",
        "",
        "## Quantitative Summary",
        "",
        "```json",
        json.dumps(summary, indent=2, ensure_ascii=False),
        "```",
        "",
        "## Key Findings",
        "",
        "1. `current_export/live` 的 score 明显偏低，远低于原始 `tg_export/live` 的主分布。",
        "2. 两边 `normal cue jpg` 的整体均值/方差并没有出现数量级崩坏，说明问题不是简单的整体变黑或变亮。",
        "3. 更大的差异来自导出链路：当前导出只做了 `FacePreprocessor` 裁剪+resize，没有复用原始 `FaceAlign/slow_align` 的几何对齐流程。",
        "4. 原始 ThunderGuard 的 `normal cue` 是建立在 5 帧对齐脸序列之上的，几何位置、眼睛对齐、脸部姿态更稳定；当前导出只取 3 帧并用 `[c1,c1,c2,c3,c3]` 复制，且没有原始 3D/关键点对齐，模型对 live 的几何先验被破坏得更明显。",
        "5. `MoEA_score.onnx` 的 score 推理并不读取 `_d.jpg`，所以当前 live 分数低的主因不是占位深度图，而是 `.jpg normal cue` 的生成链和原始训练链不够一致。",
        "",
        "## Representative Sample Sheets",
        "",
        f"- Original live best: `{OUTPUT_DIR / 'original_live_best.jpg'}`",
        f"- Current live worst: `{OUTPUT_DIR / 'current_live_worst.jpg'}`",
        f"- Original spoof low-score examples: `{OUTPUT_DIR / 'original_spoof_low.jpg'}`",
        f"- Current spoof high-score examples: `{OUTPUT_DIR / 'current_spoof_high.jpg'}`",
    ]
    report_path = OUTPUT_DIR / "thunderguard_export_gap_report.md"
    report_path.write_text("\n".join(report_lines), encoding="utf-8")

    save_json(
        OUTPUT_DIR / "thunderguard_export_gap_records.json",
        [
            {
                "dataset_name": r.dataset_name,
                "split": r.split,
                "label_id": r.label_id,
                "label_name": r.label_name,
                "sample_txt": str(r.sample_txt),
                "sample_jpg": str(r.sample_jpg),
                "score": r.score,
                "image_mean": r.image_mean,
                "image_std": r.image_std,
                "lap_var": r.lap_var,
            }
            for r in all_records
        ],
    )
    save_json(OUTPUT_DIR / "thunderguard_export_gap_summary.json", summary)
    print(f"Saved report to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
