from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fused_face_liveness_api import (  # noqa: E402
    DEFAULT_BANK_DIR,
    DEFAULT_FLASH_ONNX_PATH,
    FusionLivenessService,
    read_image_from_path,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_TEST_ROOT = PROJECT_ROOT / "Face-Anti-Spoofing-using-DeePixBiS/data/test_fake"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict | list) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def list_images(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def resolve_label_dir(root: Path, preferred_name: str, fallback_name: str) -> Path:
    preferred = root / preferred_name
    if preferred.exists():
        return preferred
    fallback = root / fallback_name
    if fallback.exists():
        return fallback
    return preferred


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_metrics(rows: list[dict]) -> dict:
    valid_rows = [row for row in rows if row["status"] == "ok"]
    labels = np.asarray([int(row["label_id"]) for row in valid_rows], dtype=np.int32)
    preds = np.asarray([int(row["prediction_id"]) for row in valid_rows], dtype=np.int32)

    if labels.size == 0:
        return {
            "accuracy": 0.0,
            "apcer": 0.0,
            "bpcer": 0.0,
            "acer": 0.0,
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
        }

    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    apcer = safe_divide(fp, int((labels == 0).sum()))
    bpcer = safe_divide(fn, int((labels == 1).sum()))
    return {
        "accuracy": safe_divide(tp + tn, int(labels.size)),
        "apcer": apcer,
        "bpcer": bpcer,
        "acer": (apcer + bpcer) / 2.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate fused_face_liveness_api.py ThunderGuard API logic on local image folders."
    )
    parser.add_argument("--test-root", default=str(DEFAULT_TEST_ROOT), help="包含 local/gongji 或 face_1_local/face_1_url_gongji 的根目录")
    parser.add_argument("--local-dir", default=None, help="真人图片目录；默认自动查找 local 或 face_1_local")
    parser.add_argument("--gongji-dir", default=None, help="攻击图片目录；默认自动查找 gongji 或 face_1_url_gongji")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "Face-Anti-Spoofing-using-DeePixBiS/fusion_api_v1_thunderguard_eval_test_fake"))
    parser.add_argument("--gallery-dir", default=str(DEFAULT_BANK_DIR), help="API 使用的 bank 底库目录")
    parser.add_argument("--fr-threshold", type=float, default=0.49)
    parser.add_argument("--flash-threshold", type=float, default=0.93)
    parser.add_argument("--flash-onnx-path", default=str(DEFAULT_FLASH_ONNX_PATH))
    parser.add_argument("--flash-second-view-mode", default="blur", choices=["same", "blur", "brighten"])
    parser.add_argument("--yolo-path", default=str(PROJECT_ROOT / "yolov7_face/yolov7-w6-face.pt"))
    parser.add_argument("--arcface-path", default=str(PROJECT_ROOT / "model_16.pt"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--return-images", action="store_true", help="是否在 JSONL 中写入 base64 图片；默认不写，避免结果文件过大")
    return parser.parse_args()


def build_samples(args: argparse.Namespace) -> list[tuple[Path, int, str]]:
    root = Path(args.test_root)
    local_dir = Path(args.local_dir) if args.local_dir else resolve_label_dir(root, "local", "face_1_local")
    gongji_dir = Path(args.gongji_dir) if args.gongji_dir else resolve_label_dir(root, "gongji", "face_1_url_gongji")

    samples = []
    for image_path in list_images(local_dir):
        samples.append((image_path, 1, "local"))
    for image_path in list_images(gongji_dir):
        samples.append((image_path, 0, "gongji"))
    return samples


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    service = FusionLivenessService(args)
    samples = build_samples(args)
    rows = []
    jsonl_path = output_dir / "predictions.jsonl"
    started_at = time.time()

    if jsonl_path.exists():
        jsonl_path.unlink()

    print(
        json.dumps(
            {
                "sample_count": len(samples),
                "gallery_dir": args.gallery_dir,
                "flash_onnx_path": args.flash_onnx_path,
                "flash_threshold": args.flash_threshold,
                "output_dir": str(output_dir),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    with jsonl_path.open("a", encoding="utf-8") as jsonl_file:
        for index, (image_path, label_id, label_name) in enumerate(samples, start=1):
            row = {
                "index": index,
                "image_path": str(image_path),
                "label_id": label_id,
                "label_name": label_name,
                "prediction_id": "",
                "prediction_name": "",
                "correct": "",
                "result": "",
                "binary_result": "",
                "flash_score": "",
                "flash_threshold": args.flash_threshold,
                "fr_min_score": "",
                "fr_threshold": args.fr_threshold,
                "best_bank_match": "",
                "best_bank_label": "",
                "annotated_image_path": "",
                "aligned_face_path": "",
                "thunderguard_sample_jpg": "",
                "thunderguard_sample_txt": "",
                "status": "ok",
                "error": "",
            }
            try:
                image_bgr = read_image_from_path(image_path)
                result = service.predict(
                    image_bgr=image_bgr,
                    source_type="local_eval_path",
                    source_value=str(image_path),
                    return_images=args.return_images,
                )
                prediction_id = 1 if result["binary_result"] == "real" else 0
                row.update(
                    {
                        "prediction_id": prediction_id,
                        "prediction_name": result["binary_result"],
                        "correct": int(prediction_id == label_id),
                        "result": result["result"],
                        "binary_result": result["binary_result"],
                        "flash_score": result["flash_liveness"]["score"],
                        "flash_threshold": result["flash_liveness"]["threshold"],
                        "fr_min_score": result["fr"]["min_score"],
                        "fr_threshold": result["fr"]["threshold"],
                        "best_bank_match": result.get("best_bank_match"),
                        "best_bank_label": result.get("best_bank_label"),
                        "annotated_image_path": result.get("annotated_image_path"),
                        "aligned_face_path": result.get("aligned_face_path"),
                        "thunderguard_sample_jpg": result.get("thunderguard_sample_jpg"),
                        "thunderguard_sample_txt": result.get("thunderguard_sample_txt"),
                    }
                )
                jsonl_file.write(json.dumps(result, ensure_ascii=False) + "\n")
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
                jsonl_file.write(
                    json.dumps(
                        {
                            "image_path": str(image_path),
                            "label_id": label_id,
                            "label_name": label_name,
                            "status": "failed",
                            "error": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            rows.append(row)

            print(
                f"[{index}/{len(samples)}] {label_name} {image_path.name} -> "
                f"{row['binary_result'] or 'failed'} correct={row['correct']}",
                flush=True,
            )

    metrics = compute_metrics(rows)
    ok_count = sum(1 for row in rows if row["status"] == "ok")
    failed_count = len(rows) - ok_count
    summary = {
        "test_root": str(Path(args.test_root)),
        "local_dir": str(Path(args.local_dir) if args.local_dir else resolve_label_dir(Path(args.test_root), "local", "face_1_local")),
        "gongji_dir": str(Path(args.gongji_dir) if args.gongji_dir else resolve_label_dir(Path(args.test_root), "gongji", "face_1_url_gongji")),
        "gallery_dir": args.gallery_dir,
        "flash_onnx_path": args.flash_onnx_path,
        "fr_threshold": args.fr_threshold,
        "flash_threshold": args.flash_threshold,
        "total_samples": len(rows),
        "processed_samples": ok_count,
        "failed_samples": failed_count,
        "metrics": metrics,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "result_files": {
            "csv": str(output_dir / "predictions.csv"),
            "jsonl": str(jsonl_path),
            "summary": str(output_dir / "summary.json"),
            "api_prediction_log": str(service.log_path),
            "annotated_dir": str(service.annotated_dir),
            "aligned_dir": str(service.aligned_dir),
            "thunderguard_sample_dir": str(service.thunderguard_sample_dir),
        },
    }

    write_csv(output_dir / "predictions.csv", rows)
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
