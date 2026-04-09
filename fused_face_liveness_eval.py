import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import torch


PROJECT_ROOT = Path("/supercloud/llm-code/scc/scc/Liveness_Detection")
DEEPIXBIS_ROOT = PROJECT_ROOT / "Face-Anti-Spoofing-using-DeePixBiS"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DEEPIXBIS_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEPIXBIS_ROOT))

from face_interactive_liveness import LocalFaceComparator  # noqa: E402
from infer_images import build_model, build_transforms, infer_face_bgr  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fuse ArcFace gallery similarity and DeePixBiS anti-spoofing on still images."
    )
    parser.add_argument(
        "--gallery-dir",
        default=str(DEEPIXBIS_ROOT / "data/test_1_compare/bank"),
        help="Bank folder used to build labeled reference identities.",
    )
    parser.add_argument(
        "--query-dirs",
        nargs="+",
        default=[
            str(DEEPIXBIS_ROOT / "data/test_1_compare/bank"),
            str(DEEPIXBIS_ROOT / "data/test_1_compare/test"),
        ],
        help="Folders to evaluate.",
    )
    parser.add_argument(
        "--gallery-count",
        type=int,
        default=0,
        help="Unused in bank mode. Kept for CLI compatibility.",
    )
    parser.add_argument(
        "--spoof-threshold",
        type=float,
        default=0.62,
        help="If bank is not matched and DeePixBiS pixel_score > 0.62, output real.",
    )
    parser.add_argument(
        "--fr-threshold",
        type=float,
        default=0.49,
        help="Bank match threshold for the best labeled bank embedding.",
    )
    parser.add_argument(
        "--yolo-path",
        default=str(PROJECT_ROOT / "yolov7_face/yolov7-w6-face.pt"),
        help="YOLO face detector weights.",
    )
    parser.add_argument(
        "--arcface-path",
        default=str(PROJECT_ROOT / "model_16.pt"),
        help="ArcFace weights.",
    )
    parser.add_argument(
        "--deepixbis-weights",
        default=str(DEEPIXBIS_ROOT / "DeePixBiS.pth"),
        help="DeePixBiS weights.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEEPIXBIS_ROOT / "fusion_eval_outputs"),
        help="Directory for logs and annotated outputs.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device, for example cuda:0 or cpu.",
    )
    return parser.parse_args()


def list_images(folder: Path):
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]
    )


def infer_ground_truth(image_path: Path):
    lowered = image_path.name.lower()
    if any(token in lowered for token in ("toumo", "head", "model", "fake", "spoof")):
        return "fake"
    if any(token in lowered for token in ("true", "genuine", "real", "local", "live")):
        return "real"
    return "unknown"


def infer_bank_label_from_name(image_name: str):
    lowered = image_name.lower()
    if any(token in lowered for token in ("toumo", "head", "model", "fake", "spoof")):
        return "fake"
    if any(token in lowered for token in ("true", "genuine", "real", "local", "live")):
        return "real"
    raise ValueError(f"Cannot infer bank label from filename: {image_name}")


def hash_decoded_image(image_bgr):
    payload = image_bgr.tobytes() + str(image_bgr.shape).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_bank(engine, bank_paths):
    bank_records = []
    for image_path in bank_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        embedding, _bbox, _aligned_face = engine.process_image_details(image, image_path.name)
        if embedding is None:
            continue
        bank_records.append(
            {
                "path": image_path,
                "label": infer_bank_label_from_name(image_path.name),
                "embedding": embedding,
                "image_hash": hash_decoded_image(image),
            }
        )
    return bank_records


def compare_against_bank(engine, query_embedding, bank_records):
    best_record = None
    best_score = None
    for record in bank_records:
        score = float(engine.compare_with_faiss(record["embedding"], query_embedding))
        if best_score is None or score > best_score:
            best_score = score
            best_record = record
    return best_record, best_score


def fuse_decision(exact_bank_hit, matched_record, matched_score, fr_threshold, spoof_pixel_score, spoof_threshold):
    spoof_is_real = spoof_pixel_score > spoof_threshold
    fr_pass = matched_record is not None and matched_score is not None and matched_score > fr_threshold

    if exact_bank_hit is not None and fr_pass:
        bank_label = exact_bank_hit["label"]
        final_label = "bank_real_exact" if bank_label == "real" else "bank_fake_exact"
        binary_label = bank_label
        reason = "Exact bank image matched, and FR score is greater than 0.49."
    elif fr_pass:
        bank_label = matched_record["label"]
        final_label = "bank_real_match" if bank_label == "real" else "bank_fake_match"
        binary_label = bank_label
        reason = (
            "Matched the prepared real person in bank."
            if bank_label == "real"
            else "Matched the prepared head model in bank."
        )
    elif spoof_is_real:
        final_label = "live_not_in_bank"
        binary_label = "real"
        reason = "Did not match the prepared person or head models in bank, and DeePixBiS > 0.62."
    else:
        final_label = "fake_not_in_bank"
        binary_label = "fake"
        reason = "Did not match bank, and DeePixBiS <= 0.62."

    basis = [
        f"Best bank score={matched_score:.6f}" if matched_score is not None else "Best bank score unavailable",
        f"FR min_score {'>' if fr_pass else '<='} threshold={fr_threshold:.2f}",
        f"DeePixBiS pixel_score={spoof_pixel_score:.6f} {'>' if spoof_is_real else '<='} threshold={spoof_threshold:.2f}",
        reason,
    ]
    return final_label, binary_label, basis


def annotate_result(image, bbox, final_label, binary_label, fr_min_score, spoof_pixel_score, gt_label):
    annotated = image.copy()
    color = (0, 255, 0) if binary_label == "real" else (0, 0, 255)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    lines = [
        f"FINAL: {final_label.upper()}",
        f"GT: {gt_label.upper()}",
        f"FR min: {fr_min_score:.3f}",
        f"Spoof pixel: {spoof_pixel_score:.3f}",
    ]
    for idx, text in enumerate(lines):
        cv2.putText(
            annotated,
            text,
            (x1, max(25, y1 - 40 + idx * 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return annotated


def main():
    args = parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    annotated_dir = output_dir / "annotated"
    aligned_dir = output_dir / "aligned_faces"
    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir.mkdir(parents=True, exist_ok=True)
    aligned_dir.mkdir(parents=True, exist_ok=True)

    engine = LocalFaceComparator(args.yolo_path, args.arcface_path)
    spoof_model = build_model(args.deepixbis_weights, device)
    spoof_tfms = build_transforms()

    bank_dir = Path(args.gallery_dir)
    bank_candidates = list_images(bank_dir)
    bank_records = build_bank(engine, bank_candidates)

    if not bank_records:
        raise RuntimeError(f"No valid bank embeddings were built from {bank_dir}")

    bank_hash_index = {record["image_hash"]: record for record in bank_records}

    query_paths = []
    for query_dir in args.query_dirs:
        for image_path in list_images(Path(query_dir)):
            query_paths.append(image_path)

    results = []
    for image_path in query_paths:
        image = cv2.imread(str(image_path))
        gt_label = infer_ground_truth(image_path)
        record = {
            "image": str(image_path),
            "ground_truth": gt_label,
            "bank_images": [str(item["path"]) for item in bank_records],
            "fr_threshold": args.fr_threshold,
            "spoof_threshold": args.spoof_threshold,
        }

        if image is None:
            record.update(
                {
                    "status": "error",
                    "reason": "image_not_readable",
                }
            )
            results.append(record)
            continue

        embedding, bbox, aligned_face = engine.process_image_details(image, image_path.name)
        if embedding is None or bbox is None or aligned_face is None:
            record.update(
                {
                    "status": "error",
                    "reason": "no_face_or_alignment_failed",
                }
            )
            results.append(record)
            continue

        exact_bank_hit = bank_hash_index.get(hash_decoded_image(image))
        matched_record, matched_score = compare_against_bank(engine, embedding, bank_records)
        spoof_result = infer_face_bgr(aligned_face, spoof_model, spoof_tfms, device, args.spoof_threshold)
        final_label, binary_label, fusion_basis = fuse_decision(
            exact_bank_hit=exact_bank_hit,
            matched_record=matched_record,
            matched_score=matched_score,
            fr_threshold=args.fr_threshold,
            spoof_pixel_score=spoof_result["pixel_score"],
            spoof_threshold=args.spoof_threshold,
        )

        annotated = annotate_result(
            image=image,
            bbox=bbox,
            final_label=final_label,
            binary_label=binary_label,
            fr_min_score=matched_score if matched_score is not None else -1.0,
            spoof_pixel_score=spoof_result["pixel_score"],
            gt_label=gt_label,
        )
        annotated_path = annotated_dir / f"{image_path.stem}_fused.jpg"
        aligned_path = aligned_dir / f"{image_path.stem}_aligned.jpg"
        cv2.imwrite(str(annotated_path), annotated)
        cv2.imwrite(str(aligned_path), aligned_face)

        record.update(
            {
                "status": "ok",
                "bbox_xyxy": [int(v) for v in bbox],
                "fr": {
                    "min_score": float(matched_score) if matched_score is not None else None,
                    "max_score": float(matched_score) if matched_score is not None else None,
                    "label": matched_record["label"] if matched_record is not None and matched_score is not None and matched_score >= args.fr_threshold else "unmatched",
                    "basis": [
                        "Bank matching uses the best labeled bank match instead of treating FR as direct liveness.",
                        f"bank_size={len(bank_records)}",
                    ],
                },
                "spoof": spoof_result,
                "final_label": final_label,
                "binary_final_label": binary_label,
                "best_bank_match": str(matched_record["path"]) if matched_record is not None else None,
                "best_bank_label": matched_record["label"] if matched_record is not None else None,
                "exact_bank_hit": str(exact_bank_hit["path"]) if exact_bank_hit is not None else None,
                "is_correct": gt_label == binary_label if gt_label != "unknown" else None,
                "basis": fusion_basis,
                "annotated_image": str(annotated_path),
                "aligned_face_image": str(aligned_path),
            }
        )
        results.append(record)

    valid_results = [item for item in results if item.get("status") == "ok" and item.get("is_correct") is not None]
    total = len(valid_results)
    correct = sum(1 for item in valid_results if item["is_correct"])
    real_total = sum(1 for item in valid_results if item["ground_truth"] == "real")
    fake_total = sum(1 for item in valid_results if item["ground_truth"] == "fake")
    real_correct = sum(1 for item in valid_results if item["ground_truth"] == "real" and item["is_correct"])
    fake_correct = sum(1 for item in valid_results if item["ground_truth"] == "fake" and item["is_correct"])

    summary = {
        "bank_dir": str(bank_dir),
        "bank_images": [str(item["path"]) for item in bank_records],
        "query_count": len(query_paths),
        "evaluated_count": total,
        "overall_accuracy": (correct / total) if total else None,
        "real_accuracy": (real_correct / real_total) if real_total else None,
        "fake_accuracy": (fake_correct / fake_total) if fake_total else None,
        "correct": correct,
        "total": total,
        "real_correct": real_correct,
        "real_total": real_total,
        "fake_correct": fake_correct,
        "fake_total": fake_total,
    }

    json_path = output_dir / "fusion_results.json"
    txt_path = output_dir / "fusion_results.log"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("Fusion Face Liveness Evaluation Log\n")
        f.write("=" * 80 + "\n")
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
        f.write("\n\n")
        for item in results:
            f.write("-" * 80 + "\n")
            f.write(f"image: {item['image']}\n")
            f.write(f"ground_truth: {item.get('ground_truth')}\n")
            f.write(f"status: {item.get('status')}\n")
            if item.get("status") != "ok":
                f.write(f"reason: {item.get('reason')}\n")
                continue
            f.write(f"fr_label: {item['fr']['label']}\n")
            f.write(f"fr_min_score: {item['fr']['min_score'] if item['fr']['min_score'] is not None else 'None'}\n")
            f.write(f"fr_max_score: {item['fr']['max_score'] if item['fr']['max_score'] is not None else 'None'}\n")
            f.write(f"spoof_label: {item['spoof']['label']}\n")
            f.write(f"spoof_pixel_score: {item['spoof']['pixel_score']:.6f}\n")
            f.write(f"spoof_binary_score: {item['spoof']['binary_score']:.6f}\n")
            f.write(f"spoof_combined_score: {item['spoof']['combined_score']:.6f}\n")
            f.write(f"final_label: {item['final_label']}\n")
            f.write(f"binary_final_label: {item['binary_final_label']}\n")
            f.write(f"is_correct: {item['is_correct']}\n")
            f.write("basis:\n")
            for basis_item in item["basis"]:
                f.write(f"  - {basis_item}\n")
            f.write(f"annotated_image: {item['annotated_image']}\n")
            f.write(f"aligned_face_image: {item['aligned_face_image']}\n")

    print(json.dumps({"summary": summary, "json_report": str(json_path), "txt_report": str(txt_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
