from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
import onnxruntime

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from face_interactive_liveness import LocalFaceComparator  # noqa: E402
from fused_face_liveness_api import (  # noqa: E402
    DEFAULT_FLASH_ONNX_PATH,
    THUNDERGUARD_COLOR_TRIPLETS,
    build_depth_placeholder,
    build_thunderguard_sc_flags,
    center_crop_and_resize,
    encode_image_file,
    read_image_from_path,
    rgb_to_packed_int,
    tint_image,
    validate_thunderguard_color,
)

THUNDERGUARD_ROOT = PROJECT_ROOT / "20240320闪光活体归档/ThunderGuard"
TG_PROCESS_ROOT = THUNDERGUARD_ROOT / "tg_process"
DEFAULT_TG_MASK_PATH = PROJECT_ROOT / "20240320闪光活体归档/dataset/dataset/raw/ovalMask.jpg"
if str(TG_PROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(TG_PROCESS_ROOT))

from libtg import TG  # noqa: E402


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
        return {"accuracy": 0.0, "apcer": 0.0, "bpcer": 0.0, "acer": 0.0, "tp": 0, "tn": 0, "fp": 0, "fn": 0}

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


def read_colors(txt_path: Path) -> list[tuple[int, int, int]]:
    colors = []
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        packed = int(line.strip())
        colors.append(((packed & 0x00FF0000) >> 16, (packed & 0x0000FF00) >> 8, packed & 0x000000FF))
    return colors


def load_thunderguard_onnx_input(sample_jpg: Path, sample_txt: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match ThunderGuard/tg_infer/infer_one_raw.py exactly."""
    color_triplets = read_colors(sample_txt)
    normal_cue = cv2.imread(str(sample_jpg), cv2.IMREAD_UNCHANGED)
    if normal_cue is None or normal_cue.ndim != 3 or normal_cue.shape[2] != 3:
        raise ValueError(f"invalid_thunderguard_normal_cue: {sample_jpg}")
    if normal_cue.shape[0] < 6 * 288 or normal_cue.shape[1] < 288:
        raise ValueError(f"unexpected_thunderguard_normal_cue_shape: {normal_cue.shape}")

    normal_cue_list = []
    cropped_tiles = []
    for idx in range(6):
        cur_img = normal_cue[idx * 288 + 16:(idx + 1) * 288 - 16, 16:288 - 16, :]
        cropped_tiles.append(cur_img)
        cur_chw = cur_img.transpose((2, 0, 1)).astype(np.float32)
        normal_cue_list.append(cur_chw.reshape(1, 3, 256, 256))
    normal_cues = np.concatenate(normal_cue_list, axis=0)
    normalized_input = ((normal_cues - 127.5) / 128.0).astype(np.float32)
    cut_visual = np.concatenate(cropped_tiles, axis=0)
    return normalized_input, build_thunderguard_sc_flags(color_triplets), cut_visual


def build_five_frame_flash_stack(aligned_face_bgr: np.ndarray) -> tuple[np.ndarray, list[int], list[tuple[int, int, int]]]:
    """Create a 5-frame synthetic capture so TG can emit its original 6-block normal-cue map."""
    base_480 = center_crop_and_resize(aligned_face_bgr, size=480)
    flash_triplets = list(THUNDERGUARD_COLOR_TRIPLETS)
    context_triplets = [flash_triplets[0], *flash_triplets, flash_triplets[-1]]
    frames = [tint_image(base_480, rgb) for rgb in context_triplets]
    packed_colors = [rgb_to_packed_int(rgb) for rgb in context_triplets]
    return np.concatenate(frames, axis=0), packed_colors, flash_triplets


def save_thunderguard_input_package(
    aligned_face_bgr: np.ndarray,
    sample_dir: Path,
    request_id: str,
    tg_processor: TG,
) -> dict:
    ensure_dir(sample_dir)
    flash_stack, packed_colors_5, color_triplets = build_five_frame_flash_stack(aligned_face_bgr)
    normal_cue = tg_processor.cal_normal_cues_map(flash_stack, packed_colors_5)

    sample_jpg = sample_dir / f"{request_id}.jpg"
    sample_txt = sample_dir / f"{request_id}.txt"
    sample_depth = sample_dir / f"{request_id}_d.jpg"
    sample_cut = sample_dir / f"{request_id}_cut.jpg"
    onnx_input_npy = sample_dir / f"{request_id}_onnx_input.npy"
    sc_flag_npy = sample_dir / f"{request_id}_sc_flag.npy"
    metadata_json = sample_dir / f"{request_id}_metadata.json"

    encode_image_file(normal_cue, sample_jpg)
    sample_txt.write_text("\n".join(str(rgb_to_packed_int(rgb)) for rgb in color_triplets) + "\n", encoding="utf-8")
    encode_image_file(build_depth_placeholder(aligned_face_bgr), sample_depth)
    normalized_input, sc_flag, cut_visual = load_thunderguard_onnx_input(sample_jpg, sample_txt)
    encode_image_file(cut_visual, sample_cut)
    np.save(onnx_input_npy, normalized_input.astype(np.float32))
    np.save(sc_flag_npy, sc_flag.astype(np.int32))

    metadata = {
        "sample_jpg": str(sample_jpg),
        "sample_txt": str(sample_txt),
        "sample_depth_path": str(sample_depth),
        "normal_visual_path": str(sample_cut),
        "sample_cut_path": str(sample_cut),
        "onnx_input_npy": str(onnx_input_npy),
        "sc_flag_npy": str(sc_flag_npy),
        "sample_jpg_shape": list(normal_cue.shape),
        "sample_depth_shape": list(build_depth_placeholder(aligned_face_bgr).shape),
        "sample_cut_shape": list(cut_visual.shape),
        "onnx_input_shape": list(normalized_input.shape),
        "onnx_input_dtype": str(normalized_input.dtype),
        "onnx_input_min": float(normalized_input.min()),
        "onnx_input_max": float(normalized_input.max()),
        "color_triplets_rgb": [list(rgb) for rgb in color_triplets],
        "color_triplets_packed": [rgb_to_packed_int(rgb) for rgb in color_triplets],
        "synthetic_five_frame_colors_packed": packed_colors_5,
        "sc_flag": sc_flag.tolist(),
        "format_basis": [
            "sample_jpg is the original ThunderGuard normal-cue map with shape 1728x288x3",
            "sample_jpg is generated by TG.cal_normal_cues_map from a 5x480 synthetic flash stack",
            "sample_txt stores three packed RGB integers, one per flash color",
            "sample_depth_path follows ThunderGuard *_d.jpg naming and 3x288 stacked depth placeholder",
            "onnx_input_npy is reloaded from sample_jpg/sample_txt using ThunderGuard/tg_infer/infer_one_raw.py crop rules",
            "normal_visual_path points to the 1536x256 cropped cue stack used for ONNX tensor inspection",
        ],
    }
    save_json(metadata_json, metadata)
    metadata["metadata_json"] = str(metadata_json)
    metadata["normalized_input"] = normalized_input
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate ThunderGuard MoEA_score.onnx model only, without bank-assisted decisions."
    )
    parser.add_argument("--test-root", default=str(DEFAULT_TEST_ROOT), help="包含 local/gongji 或 face_1_local/face_1_url_gongji 的根目录")
    parser.add_argument("--local-dir", default=None, help="真人图片目录；默认自动查找 local 或 face_1_local")
    parser.add_argument("--gongji-dir", default=None, help="攻击图片目录；默认自动查找 gongji 或 face_1_url_gongji")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "Face-Anti-Spoofing-using-DeePixBiS/thunderguard_model_only_eval_test_fake_original_format"))
    parser.add_argument("--flash-onnx-path", default=str(DEFAULT_FLASH_ONNX_PATH))
    parser.add_argument("--flash-threshold", type=float, default=0.93)
    parser.add_argument("--tg-mask-path", default=str(DEFAULT_TG_MASK_PATH), help="ThunderGuard TG normal-cue 计算使用的 ovalMask.jpg")
    parser.add_argument("--yolo-path", default=str(PROJECT_ROOT / "yolov7_face/yolov7-w6-face.pt"))
    parser.add_argument("--arcface-path", default=str(PROJECT_ROOT / "model_16.pt"))
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    aligned_dir = output_dir / "aligned"
    sample_dir = output_dir / "thunderguard_onnx_inputs"
    ensure_dir(output_dir)
    ensure_dir(aligned_dir)
    ensure_dir(sample_dir)

    engine = LocalFaceComparator(args.yolo_path, args.arcface_path)
    tg_processor = TG(str(args.tg_mask_path))
    session = onnxruntime.InferenceSession(str(args.flash_onnx_path))
    input_name = session.get_inputs()[0].name
    output_names = [node.name for node in session.get_outputs()]
    samples = build_samples(args)
    rows = []
    jsonl_path = output_dir / "predictions.jsonl"
    started_at = time.time()

    if jsonl_path.exists():
        jsonl_path.unlink()

    print(
        json.dumps(
            {
                "mode": "ThunderGuard model only, no bank decision",
                "sample_count": len(samples),
                "flash_onnx_path": args.flash_onnx_path,
                "flash_threshold": args.flash_threshold,
                "output_dir": str(output_dir),
                "onnx_input_name": input_name,
                "onnx_output_names": output_names,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    with jsonl_path.open("a", encoding="utf-8") as jsonl_file:
        for index, (image_path, label_id, label_name) in enumerate(samples, start=1):
            request_id = f"{index:04d}_{uuid.uuid4().hex[:8]}"
            row = {
                "index": index,
                "image_path": str(image_path),
                "label_id": label_id,
                "label_name": label_name,
                "prediction_id": "",
                "prediction_name": "",
                "correct": "",
                "flash_score": "",
                "flash_threshold": args.flash_threshold,
                "color_validation_pass": "",
                "aligned_face_path": "",
                "sample_jpg": "",
                "sample_txt": "",
                "sample_depth_path": "",
                "normal_visual_path": "",
                "onnx_input_npy": "",
                "metadata_json": "",
                "onnx_input_shape": "",
                "status": "ok",
                "error": "",
            }
            try:
                image_bgr = read_image_from_path(image_path)
                if image_bgr is None:
                    raise ValueError("image_decode_failed")

                _embedding, _bbox, aligned_face = engine.process_image_details(image_bgr, request_id)
                if aligned_face is None:
                    raise ValueError("no_face_or_alignment_failed")

                aligned_path = aligned_dir / f"{request_id}.jpg"
                encode_image_file(aligned_face, aligned_path)

                package = save_thunderguard_input_package(
                    aligned_face_bgr=aligned_face,
                    sample_dir=sample_dir,
                    request_id=request_id,
                    tg_processor=tg_processor,
                )
                normalized_input = package.pop("normalized_input")
                outputs = session.run(output_names, {input_name: normalized_input})
                if len(outputs) != 2:
                    raise RuntimeError(f"Unexpected ThunderGuard ONNX outputs: {output_names}")
                color_pred, score_pred = outputs
                score = float(np.reshape(score_pred, [-1])[0])
                color_pass = validate_thunderguard_color(color_pred, [tuple(rgb) for rgb in package["color_triplets_rgb"]])
                prediction_id = 1 if score > args.flash_threshold else 0
                prediction_name = "real" if prediction_id == 1 else "fake"

                row.update(
                    {
                        "prediction_id": prediction_id,
                        "prediction_name": prediction_name,
                        "correct": int(prediction_id == label_id),
                        "flash_score": score,
                        "color_validation_pass": bool(color_pass),
                        "aligned_face_path": str(aligned_path),
                        "sample_jpg": package["sample_jpg"],
                        "sample_txt": package["sample_txt"],
                        "sample_depth_path": package["sample_depth_path"],
                        "normal_visual_path": package["normal_visual_path"],
                        "onnx_input_npy": package["onnx_input_npy"],
                        "metadata_json": package["metadata_json"],
                        "onnx_input_shape": package["onnx_input_shape"],
                    }
                )
                jsonl_file.write(
                    json.dumps(
                        {
                            **row,
                            "onnx_input_name": input_name,
                            "onnx_output_names": output_names,
                            "color_head_output": np.round(np.reshape(color_pred, [-1, 3]), 6).tolist(),
                            "format_metadata": {k: v for k, v in package.items() if k != "metadata_json"},
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
            except Exception as exc:
                row["status"] = "failed"
                row["error"] = str(exc)
                jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")

            rows.append(row)
            print(
                f"[{index}/{len(samples)}] {label_name} {image_path.name} -> "
                f"{row['prediction_name'] or 'failed'} score={row['flash_score']} correct={row['correct']}",
                flush=True,
            )

    metrics = compute_metrics(rows)
    ok_count = sum(1 for row in rows if row["status"] == "ok")
    summary = {
        "mode": "ThunderGuard model only, no bank decision",
        "test_root": str(Path(args.test_root)),
        "local_dir": str(Path(args.local_dir) if args.local_dir else resolve_label_dir(Path(args.test_root), "local", "face_1_local")),
        "gongji_dir": str(Path(args.gongji_dir) if args.gongji_dir else resolve_label_dir(Path(args.test_root), "gongji", "face_1_url_gongji")),
        "flash_onnx_path": args.flash_onnx_path,
        "flash_threshold": args.flash_threshold,
        "tg_mask_path": args.tg_mask_path,
        "onnx_input_name": input_name,
        "onnx_output_names": output_names,
        "total_samples": len(rows),
        "processed_samples": ok_count,
        "failed_samples": len(rows) - ok_count,
        "metrics": metrics,
        "elapsed_seconds": round(time.time() - started_at, 3),
        "result_files": {
            "csv": str(output_dir / "predictions.csv"),
            "jsonl": str(jsonl_path),
            "summary": str(output_dir / "summary.json"),
            "aligned_dir": str(aligned_dir),
            "thunderguard_onnx_inputs": str(sample_dir),
        },
    }

    write_csv(output_dir / "predictions.csv", rows)
    save_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
