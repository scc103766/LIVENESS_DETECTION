import argparse
import base64
import hashlib
import json
import sys
import time
import urllib.request
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse


PROJECT_ROOT = Path("/supercloud/llm-code/scc/scc/Liveness_Detection")
DEEPIXBIS_ROOT = PROJECT_ROOT / "Face-Anti-Spoofing-using-DeePixBiS"
DEFAULT_GALLERY_DIR = PROJECT_ROOT / "data/gallery_real_faces"
FALLBACK_GALLERY_DIR = DEEPIXBIS_ROOT / "data/test_fake/face_1_local"
DEFAULT_BANK_DIR = DEEPIXBIS_ROOT / "data/test_1_compare/bank"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(DEEPIXBIS_ROOT) not in sys.path:
    sys.path.insert(0, str(DEEPIXBIS_ROOT))

from face_interactive_liveness import LocalFaceComparator  # noqa: E402
from infer_images import build_model, build_transforms, infer_face_bgr  # noqa: E402


def resolve_default_gallery_dir() -> Path:
    if DEFAULT_BANK_DIR.exists():
        return DEFAULT_BANK_DIR
    if DEFAULT_GALLERY_DIR.exists():
        return DEFAULT_GALLERY_DIR
    return FALLBACK_GALLERY_DIR


def parse_args():
    parser = argparse.ArgumentParser(description="FastAPI fused face recognition and anti-spoofing service.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=18119, help="Bind port.")
    parser.add_argument(
        "--gallery-dir",
        default=str(resolve_default_gallery_dir()),
        help="Bank folder with labeled real/head-model reference images.",
    )
    parser.add_argument(
        "--fr-threshold",
        type=float,
        default=0.49,
        help="Gallery similarity threshold.",
    )
    parser.add_argument(
        "--spoof-threshold",
        type=float,
        default=0.68,
        help="Fallback DeePixBiS threshold. If bank is not matched and pixel_score > 0.68, output real.",
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
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device for DeePixBiS.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEEPIXBIS_ROOT / "fusion_api_outputs"),
        help="Directory for prediction logs and optional debug images.",
    )
    return parser.parse_args()


def list_images(folder: Path):
    if not folder.exists():
        return []
    return sorted(
        [
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        ]
    )


def infer_bank_label_from_name(image_name: str):
    lowered = image_name.lower()
    if any(token in lowered for token in ("toumo", "head", "model", "fake", "spoof")):
        return "fake"
    if any(token in lowered for token in ("true", "genuine", "real", "local", "live")):
        return "real"
    raise ValueError(f"Cannot infer bank label from filename: {image_name}")


def read_image_from_path(image_path: Path):
    try:
        data = np.fromfile(str(image_path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def read_image_from_bytes(image_bytes: bytes):
    if not image_bytes:
        return None
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def image_to_base64(image_bgr):
    ok, encoded = cv2.imencode(".jpg", image_bgr)
    if not ok:
        return None
    return base64.b64encode(encoded.tobytes()).decode("utf-8")


def hash_decoded_image(image_bgr):
    payload = image_bgr.tobytes() + str(image_bgr.shape).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def decode_base64_image(image_base64: str):
    if not image_base64:
        return None
    if "," in image_base64 and image_base64.strip().lower().startswith("data:image"):
        image_base64 = image_base64.split(",", 1)[1]
    missing_padding = len(image_base64) % 4
    if missing_padding:
        image_base64 += "=" * (4 - missing_padding)
    try:
        return read_image_from_bytes(base64.b64decode(image_base64))
    except Exception:
        return None


def read_image_from_url(image_url: str):
    try:
        with urllib.request.urlopen(image_url, timeout=10) as response:
            return read_image_from_bytes(response.read())
    except Exception:
        return None


def build_bank(engine, bank_dir: Path):
    bank_records = []
    for image_path in list_images(bank_dir):
        image = read_image_from_path(image_path)
        if image is None:
            continue
        embedding, _bbox, _aligned = engine.process_image_details(image, image_path.name)
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


def decide_with_bank(
    exact_bank_hit,
    matched_record,
    matched_score,
    fr_threshold,
    spoof_label,
    spoof_pixel_score,
    spoof_threshold,
):
    spoof_is_real = spoof_pixel_score > spoof_threshold
    fr_pass = matched_record is not None and matched_score is not None and matched_score > fr_threshold

    if exact_bank_hit is not None and fr_pass:
        bank_label = exact_bank_hit["label"]
        final_label = "bank_real_exact" if bank_label == "real" else "bank_fake_exact"
        binary_label = bank_label
        reason = "Exact bank image matched, and FR score is greater than 0.49."
        return final_label, binary_label, [
            f"Exact bank image hash matched: {exact_bank_hit['path'].name}",
            f"Bank label={bank_label}",
            f"FR min_score={matched_score:.6f} > threshold={fr_threshold:.2f}",
            f"DeePixBiS pixel_score={spoof_pixel_score:.6f}",
            reason,
        ]

    bank_hit = fr_pass
    if bank_hit:
        bank_label = matched_record["label"]
        if bank_label == "real":
            final_label = "bank_real_match"
            binary_label = "real"
            reason = "Matched the prepared real person in bank."
        else:
            final_label = "bank_fake_match"
            binary_label = "fake"
            reason = "Matched the prepared head model in bank."
        return final_label, binary_label, [
            f"Best bank match={matched_record['path'].name}",
            f"Best bank score={matched_score:.6f} >= threshold={fr_threshold:.2f}",
            f"Matched bank label={bank_label}",
            f"DeePixBiS pixel_score={spoof_pixel_score:.6f}",
            reason,
        ]

    if spoof_is_real:
        final_label = "live_not_in_bank"
        binary_label = "real"
        reason = "Did not match the prepared person or head models in bank, and DeePixBiS > 0.68."
    else:
        final_label = "fake_not_in_bank"
        binary_label = "fake"
        reason = "Did not match bank, and DeePixBiS <= 0.68."

    return final_label, binary_label, [
        f"No bank match above threshold={fr_threshold:.2f}",
        f"Best bank score={matched_score:.6f}" if matched_score is not None else "Best bank score unavailable",
        f"DeePixBiS pixel_score={spoof_pixel_score:.6f} {'>' if spoof_is_real else '<='} threshold={spoof_threshold:.2f}",
        reason,
    ]


def annotate_result(image, bbox, final_label, binary_label, fr_min_score, spoof_pixel_score):
    annotated = image.copy()
    color = (0, 255, 0) if binary_label == "real" else (0, 0, 255)
    x1, y1, x2, y2 = bbox
    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
    lines = [
        f"FINAL: {final_label.upper()}",
        f"FR min: {fr_min_score:.3f}",
        f"Spoof pixel: {spoof_pixel_score:.3f}",
    ]
    for idx, text in enumerate(lines):
        cv2.putText(
            annotated,
            text,
            (x1, max(25, y1 - 30 + idx * 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )
    return annotated


class FusionLivenessService:
    def __init__(self, args):
        self.args = args
        self.device = torch.device(args.device)
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.annotated_dir = self.output_dir / "annotated"
        self.aligned_dir = self.output_dir / "aligned"
        self.annotated_dir.mkdir(parents=True, exist_ok=True)
        self.aligned_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.output_dir / "prediction_log.jsonl"

        self.engine = LocalFaceComparator(args.yolo_path, args.arcface_path)
        self.spoof_model = build_model(args.deepixbis_weights, self.device)
        self.spoof_tfms = build_transforms()

        self.bank_dir = Path(args.gallery_dir)
        self.bank_records = build_bank(self.engine, self.bank_dir)
        self.bank_hash_index = {record["image_hash"]: record for record in self.bank_records}
        if not self.bank_records:
            raise RuntimeError(f"No valid bank embeddings were built from {self.bank_dir}")

    def predict(self, image_bgr, source_type: str, source_value: str, return_images: bool = False):
        started_at = time.time()
        request_id = uuid.uuid4().hex[:12]

        if image_bgr is None:
            raise ValueError("image_decode_failed")

        embedding, bbox, aligned_face = self.engine.process_image_details(image_bgr, request_id)
        if embedding is None or bbox is None or aligned_face is None:
            raise ValueError("no_face_or_alignment_failed")

        exact_bank_hit = self.bank_hash_index.get(hash_decoded_image(image_bgr))
        matched_record, matched_score = compare_against_bank(self.engine, embedding, self.bank_records)
        spoof_result = infer_face_bgr(
            aligned_face, self.spoof_model, self.spoof_tfms, self.device, self.args.spoof_threshold
        )
        final_label, binary_label, fusion_basis = decide_with_bank(
            exact_bank_hit=exact_bank_hit,
            matched_record=matched_record,
            matched_score=matched_score,
            fr_threshold=self.args.fr_threshold,
            spoof_label=spoof_result["label"],
            spoof_pixel_score=spoof_result["pixel_score"],
            spoof_threshold=self.args.spoof_threshold,
        )

        annotated = annotate_result(
            image=image_bgr,
            bbox=bbox,
            final_label=final_label,
            binary_label=binary_label,
            fr_min_score=matched_score if matched_score is not None else -1.0,
            spoof_pixel_score=spoof_result["pixel_score"],
        )
        annotated_path = self.annotated_dir / f"{request_id}.jpg"
        aligned_path = self.aligned_dir / f"{request_id}.jpg"
        cv2.imwrite(str(annotated_path), annotated)
        cv2.imwrite(str(aligned_path), aligned_face)

        result = {
            "request_id": request_id,
            "source_type": source_type,
            "source_value": source_value,
            "result": final_label,
            "binary_result": binary_label,
            "basis": fusion_basis,
            "fr": {
                "label": matched_record["label"] if matched_record is not None and matched_score is not None and matched_score >= self.args.fr_threshold else "unmatched",
                "min_score": float(matched_score) if matched_score is not None else None,
                "max_score": float(matched_score) if matched_score is not None else None,
                "threshold": self.args.fr_threshold,
                "basis": [
                    "Bank matching uses the best labeled bank match instead of treating FR as direct liveness.",
                    f"bank_size={len(self.bank_records)}",
                ],
            },
            "spoof": {
                **spoof_result,
                "basis": spoof_result["basis"],
            },
            "face_box_xyxy": [int(v) for v in bbox],
            "bank_dir": str(self.bank_dir),
            "bank_images": [str(record["path"]) for record in self.bank_records],
            "best_bank_match": str(matched_record["path"]) if matched_record is not None else None,
            "best_bank_label": matched_record["label"] if matched_record is not None else None,
            "exact_bank_hit": str(exact_bank_hit["path"]) if exact_bank_hit is not None else None,
            "annotated_image_path": str(annotated_path),
            "aligned_face_path": str(aligned_path),
            "latency_ms": round((time.time() - started_at) * 1000.0, 2),
        }

        if return_images:
            result["annotated_image_base64"] = image_to_base64(annotated)
            result["aligned_face_base64"] = image_to_base64(aligned_face)

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")

        return result


async def extract_request_payload(request: Request):
    content_type = request.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        return await request.json()

    if content_type.startswith("image/"):
        return {"raw_bytes": await request.body()}

    return {}


def create_app(args):
    app = FastAPI(title="Fused Face Liveness API", version="1.0.0")
    service = FusionLivenessService(args)

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "bank_dir": str(service.bank_dir),
            "bank_size": len(service.bank_records),
            "device": args.device,
            "spoof_threshold": args.spoof_threshold,
        }

    @app.post("/predict")
    async def predict(
        request: Request,
        file: Optional[UploadFile] = File(default=None),
        image_base64: Optional[str] = Form(default=None),
        image_path: Optional[str] = Form(default=None),
        image_url: Optional[str] = Form(default=None),
        return_images: Optional[bool] = Form(default=False),
    ):
        payload = await extract_request_payload(request)

        if not image_base64:
            image_base64 = payload.get("image_base64") or request.query_params.get("image_base64")
        if not image_path:
            image_path = payload.get("image_path") or request.query_params.get("image_path")
        if not image_url:
            image_url = payload.get("image_url") or request.query_params.get("image_url")
        if "return_images" in payload and payload.get("return_images") is not None:
            return_images = bool(payload.get("return_images"))

        source_type = None
        source_value = None
        image_bgr = None

        if file is not None:
            source_type = "upload_file"
            source_value = file.filename or "uploaded_image"
            image_bgr = read_image_from_bytes(await file.read())
        elif payload.get("raw_bytes"):
            source_type = "raw_bytes"
            source_value = "request_body_image"
            image_bgr = read_image_from_bytes(payload["raw_bytes"])
        elif image_base64:
            source_type = "base64"
            source_value = "inline_base64"
            image_bgr = decode_base64_image(image_base64)
        elif image_path:
            source_type = "path"
            source_value = image_path
            image_bgr = read_image_from_path(Path(image_path))
        elif image_url:
            source_type = "url"
            source_value = image_url
            image_bgr = read_image_from_url(image_url)

        if image_bgr is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No valid image input found. Supported inputs: multipart file, "
                    "JSON/form/query image_base64, image_path, image_url, or raw image bytes."
                ),
            )

        try:
            result = service.predict(
                image_bgr=image_bgr,
                source_type=source_type or "unknown",
                source_value=source_value or "unknown",
                return_images=bool(return_images),
            )
            return JSONResponse(content=result)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return app


def main():
    args = parse_args()
    app = create_app(args)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
