import argparse
import json
from pathlib import Path

import cv2 as cv
import torch
from torchvision import transforms

from Model import DeePixBiS


def parse_args():
    parser = argparse.ArgumentParser(description="Infer face anti-spoofing labels for still images.")
    parser.add_argument("images", nargs="+", help="Input image paths")
    parser.add_argument(
        "--weights",
        default="./DeePixBiS.pth",
        help="Path to model weights",
    )
    parser.add_argument(
        "--face-cascade",
        default="./Classifiers/haarface.xml",
        help="Path to Haar face cascade",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold",
    )
    parser.add_argument(
        "--output-dir",
        default="./inference_outputs",
        help="Directory to save reports and annotated images",
    )
    return parser.parse_args()


def build_model(weights_path: str, device: torch.device) -> DeePixBiS:
    model = DeePixBiS(pretrained=False)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def build_transforms():
    return transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )


def select_face(faces):
    if len(faces) == 0:
        return None
    return max(faces, key=lambda box: box[2] * box[3])


def clamp_box(x, y, w, h, width, height):
    x = max(0, x)
    y = max(0, y)
    w = min(w, width - x)
    h = min(h, height - y)
    return x, y, w, h


def infer_face_bgr(face_bgr, model, tfms, device, threshold):
    face_region_rgb = cv.cvtColor(face_bgr, cv.COLOR_BGR2RGB)
    face_tensor = tfms(face_region_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        mask, binary = model(face_tensor)
        pixel_score = torch.mean(mask).item()
        binary_score = torch.flatten(binary)[0].item()
        combined_score = (pixel_score + binary_score) / 2.0

    label = "real" if pixel_score >= threshold else "fake"
    confidence = pixel_score if label == "real" else (1.0 - pixel_score)
    return {
        "threshold": threshold,
        "pixel_score": pixel_score,
        "binary_score": binary_score,
        "combined_score": combined_score,
        "label": label,
        "confidence": confidence,
        "basis": [
            "project Test.py uses mean(mask) as the practical liveness score",
            "README states the mean of the output feature map is used as the test-time score",
            f"decision made by pixel_score {'>=' if label == 'real' else '<'} threshold",
        ],
    }


def infer_one_image(image_path, model, face_classifier, tfms, device, threshold, output_dir: Path):
    image = cv.imread(str(image_path))
    if image is None:
        return {
            "image": str(image_path),
            "status": "error",
            "reason": "image_not_readable",
        }

    grey = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(grey, scaleFactor=1.1, minNeighbors=4)
    selected = select_face(faces)

    if selected is None:
        height, width = image.shape[:2]
        side = min(width, height)
        x = (width - side) // 2
        y = (height - side) // 2
        w = side
        h = side
        face_source = "center_crop_fallback"
    else:
        x, y, w, h = [int(v) for v in selected]
        face_source = "haar_largest_face"

    x, y, w, h = clamp_box(x, y, w, h, image.shape[1], image.shape[0])
    face_region = image[y : y + h, x : x + w]
    infer_result = infer_face_bgr(face_region, model, tfms, device, threshold)
    pixel_score = infer_result["pixel_score"]
    binary_score = infer_result["binary_score"]
    combined_score = infer_result["combined_score"]
    label = infer_result["label"]
    confidence = infer_result["confidence"]

    annotated = image.copy()
    color = (0, 255, 0) if label == "real" else (0, 0, 255)
    cv.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
    cv.putText(
        annotated,
        f"{label.upper()} pixel={pixel_score:.3f}",
        (x, max(25, y - 10)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )

    out_image = output_dir / f"{image_path.stem}_annotated.jpg"
    cv.imwrite(str(out_image), annotated)

    return {
        "image": str(image_path),
        "status": "ok",
        "face_source": face_source,
        "face_box_xywh": [int(x), int(y), int(w), int(h)],
        **infer_result,
        "annotated_image": str(out_image),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.weights, device)
    face_classifier = cv.CascadeClassifier(args.face_cascade)
    tfms = build_transforms()

    results = []
    for image_path in args.images:
        result = infer_one_image(
            Path(image_path),
            model,
            face_classifier,
            tfms,
            device,
            args.threshold,
            output_dir,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    report_path = output_dir / "inference_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps({"report": str(report_path), "count": len(results)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
