import argparse
import json
from collections import defaultdict
from pathlib import Path
import sys
from typing import List, Dict

import cv2
import numpy as np

try:
    import onnxruntime
except ImportError:
    onnxruntime = None

import torch

PYTG_ROOT = Path(__file__).resolve().parents[1] / "pytg"
sys.path.insert(0, str(PYTG_ROOT))
import networks  # noqa: E402
from networks import load_checkpoint  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="Infer liveness on the current 3-image-per-sample directory layout.")
    parser.add_argument(
        "--onnx-path",
        default="../resources/MoEA_score.onnx",
        help="Exported MoEA score ONNX model path",
    )
    parser.add_argument(
        "--input-root",
        required=True,
        help="Directory like data/1 with class subdirectories",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold for real/fake decision",
    )
    parser.add_argument(
        "--output-dir",
        default="./infer_outputs_current_dir",
        help="Directory to save JSON report",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="../resources/MoEA/model_best.pth.tar",
        help="Fallback PyTorch checkpoint when onnxruntime is unavailable",
    )
    return parser.parse_args()


class ONNXModel:
    def __init__(self, onnx_path: str):
        self.session = onnxruntime.InferenceSession(onnx_path)
        self.input_names = [node.name for node in self.session.get_inputs()]
        self.output_names = [node.name for node in self.session.get_outputs()]

    def forward(self, tensor: np.ndarray):
        outputs = self.session.run(self.output_names, {self.input_names[0]: tensor})
        if len(outputs) != 2:
            raise RuntimeError(f"unexpected ONNX output count: {len(outputs)}")
        return outputs[0], outputs[1]


class TorchModel:
    def __init__(self, checkpoint_path: str):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        network = getattr(networks, "MoEA")
        self.model = network.MoEA(infer_type="score").to(self.device)
        load_checkpoint(checkpoint_path, self.model)
        self.model.eval()

    def forward(self, tensor: np.ndarray):
        with torch.no_grad():
            x = torch.from_numpy(tensor).to(self.device)
            color_pred, score_pred = self.model(x)
        return color_pred.detach().cpu().numpy(), score_pred.detach().cpu().numpy()


def center_crop_to_square(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    side = min(height, width)
    top = (height - side) // 2
    left = (width - side) // 2
    return image[top:top + side, left:left + side]


def preprocess_image(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"failed to read image: {image_path}")
    image = center_crop_to_square(image)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_LINEAR)
    image = image.astype(np.float32)
    image = (image - 127.5) / 128.0
    return image.transpose(2, 0, 1)


def build_sample_tensor(image_paths: List[Path]) -> np.ndarray:
    if len(image_paths) != 3:
        raise RuntimeError(f"each sample must contain exactly 3 images, got {len(image_paths)}")
    frames = []
    for image_path in sorted(image_paths, key=lambda p: int(p.stem.rsplit("_", 2)[1])):
        frame = preprocess_image(image_path)
        # The training path uses 6 frames. We duplicate each of the 3 cue images once to match that shape.
        frames.append(frame)
        frames.append(frame.copy())
    return np.stack(frames, axis=0).astype(np.float32)


def parse_groups(input_root: Path):
    grouped = []
    label_map = {"真人": "real", "攻击": "attack"}
    for class_dir in sorted(input_root.iterdir()):
        if not class_dir.is_dir():
            continue
        groups = defaultdict(list)
        for image_path in sorted(class_dir.glob("*.png")):
            base, _, _ = image_path.stem.rsplit("_", 2)
            groups[base].append(image_path)
        for sample_id, paths in sorted(groups.items()):
            grouped.append(
                {
                    "class_dir": class_dir.name,
                    "expected_label": label_map.get(class_dir.name, class_dir.name),
                    "sample_id": sample_id,
                    "image_paths": paths,
                }
            )
    return grouped


def infer_one(worker: ONNXModel, sample: dict, threshold: float):
    tensor = build_sample_tensor(sample["image_paths"])
    color_pred, score_pred = worker.forward(tensor)
    score = float(np.reshape(score_pred, [-1])[0])
    color_pred = np.reshape(color_pred, [-1, 3]).tolist()
    predicted_label = "real" if score > threshold else "fake"
    return {
        "sample_id": sample["sample_id"],
        "expected_label": sample["expected_label"],
        "source_dir": sample["class_dir"],
        "image_paths": [str(p) for p in sample["image_paths"]],
        "input_shape": list(tensor.shape),
        "score": score,
        "threshold": threshold,
        "predicted_label": predicted_label,
        "color_head_output": color_pred,
        "basis": [
            "used ThunderGuard MoEA score ONNX exported from pytg/resources/MoEA/model_best.pth.tar",
            "adapted current directory by grouping 3 PNGs per sample_id and duplicating each frame once to form 6-frame input",
            "used center-crop + resize to 256x256 + training-style normalization ((img - 127.5) / 128.0)",
            "color validation was skipped because the current directory does not contain ThunderGuard-required flash color metadata",
            f"decision made by score {'>' if predicted_label == 'real' else '<='} threshold",
        ],
    }


def summarize(results: List[Dict]):
    total = len(results)
    correct = 0
    for item in results:
        expected_binary = "real" if item["expected_label"] == "real" else "fake"
        if item["predicted_label"] == expected_binary:
            correct += 1
    return {
        "count": total,
        "correct": correct,
        "accuracy_vs_directory_label": (correct / total) if total else 0.0,
    }


def main():
    args = parse_args()
    input_root = Path(args.input_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if onnxruntime is not None:
        worker = ONNXModel(args.onnx_path)
        backend = "onnxruntime"
    else:
        worker = TorchModel(args.checkpoint_path)
        backend = "torch_fallback"
    samples = parse_groups(input_root)
    results = [infer_one(worker, sample, args.threshold) for sample in samples]
    report = {
        "input_root": str(input_root),
        "onnx_path": str(Path(args.onnx_path).resolve()),
        "checkpoint_path": str(Path(args.checkpoint_path).resolve()),
        "backend": backend,
        "summary": summarize(results),
        "results": results,
    }

    report_path = output_dir / "current_dir_infer_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(json.dumps(report["summary"], ensure_ascii=False))
    for item in results:
        print(json.dumps(item, ensure_ascii=False))
    print(json.dumps({"report": str(report_path)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
