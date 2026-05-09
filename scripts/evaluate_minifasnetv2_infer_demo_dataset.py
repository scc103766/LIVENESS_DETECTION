#!/usr/bin/env python3
"""Evaluate infer_demo.py's MiniFASNetV2 logic on a live/spoof video dataset.

The original Face-Anti-Spoofing-using-DeePixBiS/infer_demo.py is a single-video
demo with a hard-coded model path.  This script keeps the same model structure
and inference rule, but adds batch evaluation, metrics, and clear artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from facenet_pytorch import MTCNN
from tqdm import tqdm


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


class MiniFASNetV2(nn.Module):
    """Same lightweight anti-spoofing model defined in infer_demo.py."""

    def __init__(self) -> None:
        super().__init__()
        self.feature = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.feature(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


@dataclass
class VideoResult:
    path: str
    split: str
    label_name: str
    label: int
    pred_name: str
    pred: int
    fake_score: Optional[float]
    sampled_frames: int
    detected_faces: int
    status: str
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/flash_liveness_new_domain_video_protocol_v1v2"),
        help="Dataset root containing split/live and split/spoof folders.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/Face-Anti-Spoofing-using-DeePixBiS/models/anti_spoof.pth"),
        help="MiniFASNetV2 checkpoint used by infer_demo.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_runs/minifasnetv2_infer_demo_eval"),
        help="Directory for CSV/JSON/TXT outputs.",
    )
    parser.add_argument("--splits", nargs="+", default=["test", "val"], help="Dataset splits to evaluate.")
    parser.add_argument("--target-fps", type=float, default=3.0, help="Same sampling intent as infer_demo.py.")
    parser.add_argument("--threshold", type=float, default=0.50, help="Fake score > threshold means spoof.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def iter_videos(input_root: Path, splits: Iterable[str]) -> Iterable[Tuple[Path, str, str, int]]:
    for split in splits:
        for label_name, label in (("live", 0), ("spoof", 1)):
            folder = input_root / split / label_name
            if not folder.exists():
                continue
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                    yield path, split, label_name, label


def load_model(model_path: Path, device: torch.device) -> MiniFASNetV2:
    if not model_path.exists():
        raise FileNotFoundError(
            f"MiniFASNetV2 checkpoint not found: {model_path}. "
            "infer_demo.py expects this file as models/anti_spoof.pth."
        )
    model = MiniFASNetV2().to(device)
    state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


def preprocess_face(face: torch.Tensor, device: torch.device) -> torch.Tensor:
    # infer_demo.py converts MTCNN CHW output to HWC, resizes to 80x80, then CHW.
    face_np = face.squeeze(0).detach().cpu().numpy()
    face_np = np.transpose(face_np, (1, 2, 0))
    face_np = cv2.resize(face_np, (80, 80))
    face_np = face_np.astype(np.float32) / 255.0
    face_np = np.transpose(face_np, (2, 0, 1))
    return torch.from_numpy(face_np).unsqueeze(0).to(device)


def infer_video(
    video_path: Path,
    model: MiniFASNetV2,
    mtcnn: MTCNN,
    device: torch.device,
    target_fps: float,
    threshold: float,
) -> Tuple[Optional[float], int, int, int, str, str]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, 0, 0, 0, "error", "could not open video"

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = target_fps
    step = max(1, int(fps / target_fps))

    probabilities: List[float] = []
    frame_idx = 0
    sampled_frames = 0
    detected_faces = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % step != 0:
            frame_idx += 1
            continue

        sampled_frames += 1
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face = mtcnn(rgb)
        if face is not None:
            detected_faces += 1
            face_tensor = preprocess_face(face, device)
            with torch.no_grad():
                out = model(face_tensor)
                fake_prob = F.softmax(out, dim=1)[0][0].item()
            probabilities.append(fake_prob)
        frame_idx += 1

    cap.release()

    if not probabilities:
        return None, sampled_frames, detected_faces, -1, "no_face", "no face frames detected"

    score = float(np.median(probabilities))
    pred = 1 if score > threshold else 0
    return score, sampled_frames, detected_faces, pred, "ok", ""


def compute_metrics(results: List[VideoResult]) -> dict:
    valid = [r for r in results if r.status == "ok"]
    tp = sum(1 for r in valid if r.label == 0 and r.pred == 0)
    tn = sum(1 for r in valid if r.label == 1 and r.pred == 1)
    fp = sum(1 for r in valid if r.label == 1 and r.pred == 0)
    fn = sum(1 for r in valid if r.label == 0 and r.pred == 1)
    total = len(valid)
    accuracy = (tp + tn) / total if total else 0.0
    apcer = fp / (fp + tn) if (fp + tn) else 0.0
    bpcer = fn / (fn + tp) if (fn + tp) else 0.0
    acer = (apcer + bpcer) / 2.0
    return {
        "total_files": len(results),
        "valid_files": total,
        "failed_or_no_face": len(results) - total,
        "accuracy": accuracy,
        "APCER": apcer,
        "BPCER": bpcer,
        "ACER": acer,
        "TP_live_as_live": tp,
        "TN_spoof_as_spoof": tn,
        "FP_spoof_as_live": fp,
        "FN_live_as_spoof": fn,
    }


def write_outputs(output_dir: Path, results: List[VideoResult], metrics: dict, args: argparse.Namespace) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "per_video_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(results[0]).keys()) if results else ["path"])
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))

    summary = {
        "input_root": str(args.input_root),
        "model_path": str(args.model_path),
        "splits": args.splits,
        "threshold": args.threshold,
        "target_fps": args.target_fps,
        "metrics": metrics,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = [
        "# MiniFASNetV2 infer_demo Dataset Evaluation",
        "",
        f"- input_root: {args.input_root}",
        f"- model_path: {args.model_path}",
        f"- splits: {', '.join(args.splits)}",
        f"- threshold: fake_score > {args.threshold} => spoof",
        f"- target_fps: {args.target_fps}",
        "",
        "## Metrics",
    ]
    for key, value in metrics.items():
        lines.append(f"- {key}: {value:.6f}" if isinstance(value, float) else f"- {key}: {value}")
    lines.extend(
        [
            "",
            "## Metric Meaning",
            "- TP_live_as_live: live video predicted as live.",
            "- TN_spoof_as_spoof: spoof video predicted as spoof.",
            "- FP_spoof_as_live: spoof video missed as live, equal to APCER numerator.",
            "- FN_live_as_spoof: live video rejected as spoof, equal to BPCER numerator.",
            "- APCER: spoof attack pass rate, lower is safer.",
            "- BPCER: live user reject rate, lower is friendlier.",
            "- ACER: mean of APCER and BPCER.",
        ]
    )
    (output_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    device = torch.device(args.device)

    try:
        model = load_model(args.model_path, device)
    except FileNotFoundError as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "report.txt").write_text(str(exc) + "\n", encoding="utf-8")
        print(str(exc), file=sys.stderr)
        return 2

    mtcnn = MTCNN(keep_all=False, device=str(device), image_size=256, margin=20)
    videos = list(iter_videos(args.input_root, args.splits))
    results: List[VideoResult] = []

    for path, split, label_name, label in tqdm(videos, desc="MiniFASNetV2 eval"):
        score, sampled, faces, pred, status, error = infer_video(
            path, model, mtcnn, device, args.target_fps, args.threshold
        )
        pred_name = "unknown" if pred < 0 else ("spoof" if pred == 1 else "live")
        results.append(
            VideoResult(
                path=str(path),
                split=split,
                label_name=label_name,
                label=label,
                pred_name=pred_name,
                pred=pred,
                fake_score=score,
                sampled_frames=sampled,
                detected_faces=faces,
                status=status,
                error=error,
            )
        )

    metrics = compute_metrics(results)
    write_outputs(args.output_dir, results, metrics, args)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"report_written: {args.output_dir / 'report.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
