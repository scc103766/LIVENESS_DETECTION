from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flash_liveness_project import FacePreprocessor, load_checkpoint  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
EPSILON = 1e-12
FLASH_COLOR_MAP = {
    "normal": None,
    "red": np.asarray([1.0, 0.25, 0.25], dtype=np.float32),
    "green": np.asarray([0.25, 1.0, 0.25], dtype=np.float32),
    "blue": np.asarray([0.25, 0.25, 1.0], dtype=np.float32),
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def collect_images(image_root: Path) -> list[Path]:
    if image_root.is_file():
        return [image_root] if image_root.suffix.lower() in IMAGE_EXTENSIONS else []
    return sorted(path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)


def load_image_list(list_path: Path) -> list[Path]:
    images: list[Path] = []
    base_dir = list_path.parent
    with list_path.open("r", encoding="utf-8") as file:
        for line in file:
            value = line.strip()
            if not value or value.startswith("#"):
                continue
            path = Path(value)
            if not path.is_absolute():
                path = base_dir / path
            images.append(path)
    return images


def image_to_liveness_tensor(
    image_path: Path,
    preprocessor: FacePreprocessor,
    num_frames: int,
    target_size: tuple[int, int],
    static_sequence_mode: str,
    flash_group_size: int,
    flash_colors: list[str],
    flash_alpha: float,
) -> tuple[torch.Tensor, np.ndarray]:
    frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame_bgr is None or frame_bgr.size == 0:
        raise RuntimeError("image_decode_failed")

    processed = preprocessor.preprocess_frames([frame_bgr], prefix=image_path.stem)
    if not processed:
        raise RuntimeError("image_preprocess_failed")

    face_bgr = processed[0]
    if face_bgr.shape[:2] != (target_size[1], target_size[0]):
        face_bgr = cv2.resize(face_bgr, target_size)

    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    frames = build_static_frame_sequence(
        face_rgb=face_rgb,
        num_frames=num_frames,
        mode=static_sequence_mode,
        flash_group_size=flash_group_size,
        flash_colors=flash_colors,
        flash_alpha=flash_alpha,
    )

    diff_frames = np.zeros_like(frames)
    if num_frames > 1:
        diff_frames[1:] = frames[1:] - frames[:-1]
        diff_frames[0] = diff_frames[1]

    multi_modal_frames = np.concatenate([frames, diff_frames], axis=-1)
    tensor_frames = torch.from_numpy(multi_modal_frames).permute(0, 3, 1, 2).float()
    return tensor_frames, face_bgr


def parse_flash_colors(value: str) -> list[str]:
    colors = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = [color for color in colors if color not in FLASH_COLOR_MAP]
    if unknown:
        raise ValueError(f"未知闪光颜色: {unknown}，支持: {sorted(FLASH_COLOR_MAP)}")
    if not colors:
        raise ValueError("--flash-colors 至少需要一个颜色")
    return colors


def build_static_frame_sequence(
    face_rgb: np.ndarray,
    num_frames: int,
    mode: str,
    flash_group_size: int,
    flash_colors: list[str],
    flash_alpha: float,
) -> np.ndarray:
    if mode == "copy":
        return np.repeat(face_rgb[None, ...], repeats=num_frames, axis=0)

    if flash_group_size <= 0:
        raise ValueError("--flash-group-size 必须大于 0")

    flash_alpha = float(np.clip(flash_alpha, 0.0, 1.0))
    frames = []
    for frame_index in range(num_frames):
        group_index = frame_index // flash_group_size
        color_name = flash_colors[group_index % len(flash_colors)]
        color = FLASH_COLOR_MAP[color_name]
        if color is None:
            flashed = face_rgb
        else:
            flashed = face_rgb * (1.0 - flash_alpha) + color.reshape(1, 1, 3) * flash_alpha
        frames.append(np.clip(flashed, 0.0, 1.0))
    return np.asarray(frames, dtype=np.float32)


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


def infer_label_from_name(image_path: Path, real_token: str) -> tuple[int, str]:
    label_id = 1 if real_token.lower() in image_path.stem.lower() else 0
    return label_id, "live" if label_id == 1 else "spoof"


def minmax_normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    min_value = float(values.min())
    max_value = float(values.max())
    if max_value - min_value < EPSILON:
        return np.full_like(values, 0.5, dtype=np.float64)
    return (values - min_value) / (max_value - min_value)


def find_best_threshold_for_scores(labels: np.ndarray, scores: np.ndarray) -> tuple[float, dict[str, float]]:
    candidates = sorted(set(np.round(scores.astype(np.float64), 8).tolist() + [0.5]))
    best_threshold = 0.5
    best_accuracy = -1.0
    best_correct = -1

    for threshold in candidates:
        preds = (scores >= threshold).astype(np.int32)
        correct = int((preds == labels).sum())
        accuracy = correct / max(len(labels), 1)
        if accuracy > best_accuracy or (accuracy == best_accuracy and correct > best_correct):
            best_accuracy = accuracy
            best_correct = correct
            best_threshold = float(threshold)

    preds = (scores >= best_threshold).astype(np.int32)
    tp = int(((preds == 1) & (labels == 1)).sum())
    tn = int(((preds == 0) & (labels == 0)).sum())
    fp = int(((preds == 1) & (labels == 0)).sum())
    fn = int(((preds == 0) & (labels == 1)).sum())
    correct = tp + tn
    total = int(len(labels))
    metrics = {
        "total_labeled": total,
        "correct": correct,
        "incorrect": total - correct,
        "live_count": int((labels == 1).sum()),
        "spoof_count": int((labels == 0).sum()),
        "accuracy": best_accuracy,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
    return best_threshold, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run flash liveness inference on one image or an image folder.")
    parser.add_argument("--checkpoint", required=True, help="best/last_flash_liveness_model.pth")
    parser.add_argument("--image-root", help="单张图片或图片目录，会递归扫描常见图片格式")
    parser.add_argument("--image-list", help="可选 txt 文件，每行一个图片路径；提供后优先使用它")
    parser.add_argument("--output-dir", required=True, help="结果输出目录")
    parser.add_argument("--threshold", type=float, default=None, help="可选阈值，默认读取 checkpoint 中保存的阈值")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default=None, help="例如 cuda:0/cuda:1/cpu")
    parser.add_argument("--detector-model", default=None, help="YOLOv7 人脸检测权重路径；为空则中心裁剪")
    parser.add_argument("--detector-device", default=None)
    parser.add_argument("--detector-conf", type=float, default=0.5)
    parser.add_argument("--detector-iou", type=float, default=0.5)
    parser.add_argument("--save-preprocessed", action="store_true", help="保存模型实际使用的 224x224 人脸图")
    parser.add_argument(
        "--static-sequence-mode",
        choices=["copy", "flash"],
        default="copy",
        help="单张图片转 16 帧的方式：copy 为简单复制；flash 为合成闪光三色序列。",
    )
    parser.add_argument("--flash-group-size", type=int, default=4, help="flash 模式下每多少帧切换一次颜色，默认每 4 帧。")
    parser.add_argument(
        "--flash-colors",
        default="normal,red,green,blue",
        help="flash 模式颜色序列，默认 normal,red,green,blue，前 4 帧保持原始光色。",
    )
    parser.add_argument("--flash-alpha", type=float, default=0.25, help="flash 模式颜色叠加强度，范围 0-1。")
    parser.add_argument(
        "--label-from-name",
        action="store_true",
        help="按文件名推断标签：文件名包含 --real-name-token 为 live，否则为 spoof。",
    )
    parser.add_argument("--real-name-token", default="true", help="文件名中表示真人/活体的关键字，默认 true")
    parser.add_argument(
        "--score-mode",
        choices=["probability", "minmax"],
        default="probability",
        help="输出分数模式：probability 为模型 sigmoid 概率；minmax 为本批图片内 0-1 归一化分数。",
    )
    parser.add_argument(
        "--auto-threshold",
        action="store_true",
        help="有标签时按当前输出分数自动选择准确率最高的阈值。",
    )
    parser.add_argument(
        "--auto-orient-score",
        action="store_true",
        help="有标签时自动校准分数方向，保证真人样本均分高于假人样本。",
    )
    return parser.parse_args()


def flush_batch(
    model: torch.nn.Module,
    device: torch.device,
    batch_tensors: list[torch.Tensor],
    batch_meta: list[dict],
    threshold: float,
    rows: list[dict],
) -> None:
    if not batch_tensors:
        return

    batch = torch.stack(batch_tensors, dim=0).to(device)
    with torch.no_grad():
        logits = model(batch)
        probs = torch.sigmoid(logits).detach().cpu().numpy().astype(float)

    for meta, prob in zip(batch_meta, probs):
        rows.append(
            {
                "index": meta["index"],
                "image_path": meta["image_path"],
                "label_id": meta["label_id"],
                "label_name": meta["label_name"],
                "probability_live_raw": round(float(prob), 12),
                "score_live": "",
                "threshold": round(float(threshold), 8),
                "prediction_id": "",
                "prediction_name": "",
                "correct": "",
                "status": "ok",
                "error": "",
            }
        )


def finalize_rows(
    rows: list[dict],
    threshold: float,
    score_mode: str,
    auto_threshold: bool,
    auto_orient_score: bool,
) -> tuple[float, dict | None, str]:
    ok_indices = [index for index, row in enumerate(rows) if row["status"] == "ok"]
    if not ok_indices:
        return threshold, None, "raw"

    raw_probs = np.asarray([float(rows[index]["probability_live_raw"]) for index in ok_indices], dtype=np.float64)
    if score_mode == "minmax":
        scores = minmax_normalize(raw_probs)
    else:
        scores = np.clip(raw_probs, 0.0, 1.0)

    labeled_ok = [index for index in ok_indices if rows[index]["label_id"] != ""]
    score_orientation = "raw"
    if auto_orient_score and labeled_ok:
        label_positions = [ok_indices.index(index) for index in labeled_ok]
        labels = np.asarray([int(rows[index]["label_id"]) for index in labeled_ok], dtype=np.int32)
        live_scores = scores[label_positions][labels == 1]
        spoof_scores = scores[label_positions][labels == 0]
        if live_scores.size and spoof_scores.size and float(live_scores.mean()) < float(spoof_scores.mean()):
            scores = 1.0 - scores
            score_orientation = "inverted"

    metrics = None
    if auto_threshold and labeled_ok:
        label_positions = [ok_indices.index(index) for index in labeled_ok]
        labels = np.asarray([int(rows[index]["label_id"]) for index in labeled_ok], dtype=np.int32)
        threshold, metrics = find_best_threshold_for_scores(labels, scores[label_positions])

    for row_index, score in zip(ok_indices, scores):
        pred_id = int(float(score) >= threshold)
        row = rows[row_index]
        row["score_live"] = round(float(score), 8)
        row["threshold"] = round(float(threshold), 8)
        row["prediction_id"] = pred_id
        row["prediction_name"] = "live" if pred_id == 1 else "spoof"
        if row["label_id"] != "":
            row["correct"] = int(pred_id == int(row["label_id"]))

    if labeled_ok and metrics is None:
        labels = np.asarray([int(rows[index]["label_id"]) for index in labeled_ok], dtype=np.int32)
        score_values = np.asarray([float(rows[index]["score_live"]) for index in labeled_ok], dtype=np.float64)
        preds = (score_values >= threshold).astype(np.int32)
        metrics = {
            "total_labeled": int(len(labels)),
            "correct": int((preds == labels).sum()),
            "incorrect": int((preds != labels).sum()),
            "live_count": int((labels == 1).sum()),
            "spoof_count": int((labels == 0).sum()),
            "accuracy": float((preds == labels).sum() / max(len(labels), 1)),
            "tp": int(((preds == 1) & (labels == 1)).sum()),
            "tn": int(((preds == 0) & (labels == 0)).sum()),
            "fp": int(((preds == 1) & (labels == 0)).sum()),
            "fn": int(((preds == 0) & (labels == 1)).sum()),
        }
    return threshold, metrics, score_orientation


def main() -> int:
    args = parse_args()
    if not args.image_root and not args.image_list:
        raise ValueError("请提供 --image-root 或 --image-list。")

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)
    preprocessed_dir = output_dir / "preprocessed_224"
    if args.save_preprocessed:
        ensure_dir(preprocessed_dir)

    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    config = checkpoint["config"]
    threshold = args.threshold if args.threshold is not None else float(checkpoint.get("threshold", 0.5))
    num_frames = int(config["num_frames"])
    target_size = tuple(config["target_size"])
    flash_colors = parse_flash_colors(args.flash_colors)

    preprocessor = FacePreprocessor(
        detector_model_path=args.detector_model,
        detector_device=args.detector_device,
        target_size=target_size,
        conf_threshold=args.detector_conf,
        iou_threshold=args.detector_iou,
    )

    image_paths = load_image_list(Path(args.image_list)) if args.image_list else collect_images(Path(args.image_root))
    rows: list[dict] = []
    batch_tensors: list[torch.Tensor] = []
    batch_meta: list[dict] = []

    print(
        json.dumps(
            {
                "checkpoint": str(Path(args.checkpoint).resolve()),
                "device": str(device),
                "threshold": threshold,
                "num_frames": num_frames,
                "target_size": list(target_size),
                "image_count": len(image_paths),
                "output_dir": str(output_dir.resolve()),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for index, image_path in enumerate(image_paths, start=1):
        try:
            tensor_frames, face_bgr = image_to_liveness_tensor(
                image_path=image_path,
                preprocessor=preprocessor,
                num_frames=num_frames,
                target_size=target_size,
                static_sequence_mode=args.static_sequence_mode,
                flash_group_size=args.flash_group_size,
                flash_colors=flash_colors,
                flash_alpha=args.flash_alpha,
            )
            if args.save_preprocessed:
                safe_name = f"{index:06d}_{image_path.stem}.jpg"
                cv2.imwrite(str(preprocessed_dir / safe_name), face_bgr)

            label_id = ""
            label_name = ""
            if args.label_from_name:
                label_id, label_name = infer_label_from_name(image_path, args.real_name_token)

            batch_tensors.append(tensor_frames)
            batch_meta.append(
                {
                    "index": index,
                    "image_path": str(image_path),
                    "label_id": label_id,
                    "label_name": label_name,
                }
            )

            if len(batch_tensors) >= args.batch_size:
                flush_batch(model, device, batch_tensors, batch_meta, threshold, rows)
                batch_tensors.clear()
                batch_meta.clear()
        except Exception as exc:
            rows.append(
                {
                    "index": index,
                    "image_path": str(image_path),
                    "label_id": "",
                    "label_name": "",
                    "probability_live_raw": "",
                    "score_live": "",
                    "threshold": round(float(threshold), 8),
                    "prediction_id": "",
                    "prediction_name": "",
                    "correct": "",
                    "status": "failed",
                    "error": str(exc),
                }
            )

        if index == 1 or index % 50 == 0 or index == len(image_paths):
            print(f"processed {index}/{len(image_paths)} images", flush=True)

    flush_batch(model, device, batch_tensors, batch_meta, threshold, rows)
    threshold, metrics, score_orientation = finalize_rows(
        rows=rows,
        threshold=threshold,
        score_mode=args.score_mode,
        auto_threshold=args.auto_threshold,
        auto_orient_score=args.auto_orient_score,
    )

    csv_path = output_dir / "image_predictions.csv"
    jsonl_path = output_dir / "image_predictions.jsonl"
    summary_path = output_dir / "summary.json"
    write_csv(csv_path, rows)
    write_jsonl(jsonl_path, rows)

    ok_rows = [row for row in rows if row["status"] == "ok"]
    summary = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "device": str(device),
        "threshold": threshold,
        "score_mode": args.score_mode,
        "score_orientation": score_orientation,
        "static_sequence_mode": args.static_sequence_mode,
        "flash_group_size": args.flash_group_size if args.static_sequence_mode == "flash" else "",
        "flash_colors": flash_colors if args.static_sequence_mode == "flash" else [],
        "flash_alpha": args.flash_alpha if args.static_sequence_mode == "flash" else "",
        "auto_threshold": bool(args.auto_threshold),
        "auto_orient_score": bool(args.auto_orient_score),
        "label_from_name": bool(args.label_from_name),
        "real_name_token": args.real_name_token if args.label_from_name else "",
        "num_frames": num_frames,
        "target_size": list(target_size),
        "total_images": len(image_paths),
        "processed_images": len(ok_rows),
        "failed_images": len(rows) - len(ok_rows),
        "result_files": {
            "csv": str(csv_path.resolve()),
            "jsonl": str(jsonl_path.resolve()),
            "summary": str(summary_path.resolve()),
            "preprocessed_224": str(preprocessed_dir.resolve()) if args.save_preprocessed else "",
        },
    }
    if metrics is not None:
        summary["metrics"] = metrics
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




'''
/home/scc/anaconda3/envs/anti-spoofing_scc_175/bin/python scripts/infer_flash_liveness_images.py \
  --checkpoint /supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_runs/flash_liveness_gpu1_localresnet_e20_manual/best_flash_liveness_model.pth \
  --image-root /supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_runs/flash_liveness_gpu1_localresnet_e20_manual/batch_image_outputs/preprocessed_224 \
  --output-dir /supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_runs/flash_liveness_gpu1_localresnet_e20_manual/preprocessed_224_flash_normal_rgb_eval \
  --device cuda:1 \
  --batch-size 32 \
  --static-sequence-mode flash \
  --flash-group-size 4 \
  --flash-colors normal,red,green,blue \
  --flash-alpha 0.25 \
  --label-from-name \
  --real-name-token true\
  --score-mode minmax \
  --auto-orient-score \
  --auto-threshold

'''
