#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import importlib.util
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flash_liveness_project_v2 import FacePreprocessor  # noqa: E402

THUNDERGUARD_ROOT = PROJECT_ROOT / "20240320闪光活体归档/ThunderGuard"
TG_PROCESS_ROOT = THUNDERGUARD_ROOT / "tg_process"
FACE_ALIGN_ROOT = PROJECT_ROOT / "20240320闪光活体归档/FaceAlign"
FACE_DETECT_ROOT = FACE_ALIGN_ROOT / "face_detect"
PYFA_ROOT = FACE_ALIGN_ROOT / "pyfa"
DEFAULT_TG_MASK_PATH = PROJECT_ROOT / "20240320闪光活体归档/dataset/dataset/raw/ovalMask.jpg"
if str(TG_PROCESS_ROOT) not in sys.path:
    sys.path.insert(0, str(TG_PROCESS_ROOT))
if str(FACE_DETECT_ROOT) not in sys.path:
    sys.path.insert(0, str(FACE_DETECT_ROOT))
if str(PYFA_ROOT) not in sys.path:
    sys.path.insert(0, str(PYFA_ROOT))

from libtg import TG  # noqa: E402
from libfa.libfa import FaceAlign  # noqa: E402


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
DEFAULT_COLOR_TRIPLETS = [16717055, 1376020, 16716820]


def load_legacy_landmark_detector_class():
    module_path = FACE_DETECT_ROOT / "face_detect_with_landmark.py"
    spec = importlib.util.spec_from_file_location("legacy_face_detect_with_landmark", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed_to_load_spec:{module_path}")
    module = importlib.util.module_from_spec(spec)
    old_argv = sys.argv[:]
    try:
        sys.argv = [str(module_path)]
        spec.loader.exec_module(module)
    finally:
        sys.argv = old_argv
    return module.LandmarkFaceDetector


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict | list) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_tsv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export current flash-liveness video dataset into original ThunderGuard tg_export-style jpg/txt/_d.jpg samples."
    )
    parser.add_argument(
        "--input-root",
        default=str(PROJECT_ROOT / "dataset/flash_liveness_new_domain_video_protocol_v1v2"),
        help="Current dataset root containing train|val|test/live|spoof video+txt pairs.",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "dataset/tg_export_from_current_dataset"),
        help="Output root. Samples will be exported under split directories as flat tg_export-style files.",
    )
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="all")
    parser.add_argument("--limit", type=int, default=0, help="Optional sample limit for debugging.")
    parser.add_argument("--detector-model", default=None, help="Optional YOLOv7 face detector weights.")
    parser.add_argument("--detector-device", default=None)
    parser.add_argument("--detector-conf", type=float, default=0.5)
    parser.add_argument("--detector-iou", type=float, default=0.5)
    parser.add_argument(
        "--color-triplets",
        default="16717055,1376020,16716820",
        help="Packed RGB color triplets to export into final txt; defaults to ThunderGuard purple/green/red.",
    )
    parser.add_argument("--align-size", type=int, default=480, help="Aligned face size before TG normal-cue generation.")
    parser.add_argument(
        "--export-mode",
        choices=["face_preprocessor", "legacy_fast_align"],
        default="face_preprocessor",
        help="face_preprocessor uses the current crop/resize path; legacy_fast_align uses old eye-landmark + FaceAlign.fast_align.",
    )
    return parser.parse_args()


def parse_color_triplets(raw: str) -> list[int]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(int(item, 0))
    if len(values) != 3:
        raise ValueError("color triplets must contain exactly 3 packed colors")
    return values


def discover_samples(input_root: Path, split: str) -> list[tuple[str, str, str, int]]:
    rows: list[tuple[str, str, str, int]] = []
    splits = ("train", "val", "test") if split == "all" else (split,)
    for split_name in splits:
        for label_name, label_id in (("live", 1), ("spoof", 0)):
            label_dir = input_root / split_name / label_name
            if not label_dir.exists():
                continue
            for video_path in sorted(label_dir.iterdir()):
                if not video_path.is_file() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
                    continue
                txt_path = video_path.with_suffix(".txt")
                if not txt_path.exists():
                    continue
                rows.append((split_name, str(video_path), str(txt_path), label_id))
    return rows


def read_frame_color_list(txt_path: Path) -> list[int]:
    colors: list[int] = []
    with txt_path.open("r", encoding="utf-8") as file:
        for raw in file:
            line = raw.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",") if part.strip()]
            if not parts:
                continue
            colors.append(int(parts[-1]))
    return colors


def filter_frame_indices(frame_colors: list[int]) -> list[int]:
    valid = []
    for idx in range(len(frame_colors)):
        if idx < 2 or idx > len(frame_colors) - 3:
            continue
        cur = frame_colors[idx]
        if frame_colors[idx - 2] != cur or frame_colors[idx + 2] != cur:
            continue
        valid.append(idx)
    return valid


def choose_stable_indices(frame_colors: list[int], target_colors: list[int]) -> list[int]:
    valid_indices = set(filter_frame_indices(frame_colors))
    selected = []
    search_start = 0
    for color in target_colors:
        segment_indices = []
        in_segment = False
        for idx in range(search_start, len(frame_colors)):
            if frame_colors[idx] == color:
                in_segment = True
                if idx in valid_indices:
                    segment_indices.append(idx)
            elif in_segment:
                break
        if not segment_indices:
            raise RuntimeError(f"stable_segment_not_found_for_color:{color}")
        picked = segment_indices[len(segment_indices) // 2]
        selected.append(picked)
        search_start = picked + 1
    return selected


def load_selected_frames(video_path: Path, frame_indices: list[int]) -> list[np.ndarray]:
    target_set = set(frame_indices)
    captured: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed_to_open_video:{video_path}")
    frame_id = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id in target_set:
            captured[frame_id] = frame.copy()
            if len(captured) == len(target_set):
                break
        frame_id += 1
    cap.release()
    missing = [idx for idx in frame_indices if idx not in captured]
    if missing:
        raise RuntimeError(f"missing_frame_indices:{missing}")
    return [captured[idx] for idx in frame_indices]


def build_depth_tile(aligned_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    if gray.shape[0] != 480 or gray.shape[1] != 480:
        gray = cv2.resize(gray, (480, 480))
    offset = (480 - 288) // 2
    return gray[offset:offset + 288, offset:offset + 288]


def encode_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise RuntimeError(f"failed_to_encode:{path}")
    path.write_bytes(encoded.tobytes())


def export_one_sample(
    tg: TG,
    preprocessor: FacePreprocessor,
    legacy_face_detector,
    legacy_face_align: FaceAlign | None,
    split_name: str,
    video_path: Path,
    txt_path: Path,
    label_id: int,
    output_root: Path,
    target_colors: list[int],
    export_mode: str,
) -> dict:
    frame_colors = read_frame_color_list(txt_path)
    if len(frame_colors) < 5:
        raise RuntimeError("insufficient_frame_color_labels")

    picked_three = choose_stable_indices(frame_colors, target_colors)
    raw_frames = load_selected_frames(video_path, picked_three)
    if export_mode == "legacy_fast_align":
        if legacy_face_detector is None or legacy_face_align is None:
            raise RuntimeError("legacy_fast_align_components_not_initialized")
        aligned_three = []
        for frame in raw_frames:
            landmarks = legacy_face_detector.predict_landmark(frame)
            if landmarks is None:
                raise RuntimeError("legacy_landmark_detection_failed")
            left_eye_x = int(landmarks[2, 0])
            left_eye_y = int(landmarks[2, 1])
            right_eye_x = int(landmarks[3, 0])
            right_eye_y = int(landmarks[3, 1])
            aligned, _ = legacy_face_align.fast_align(frame, left_eye_x, left_eye_y, right_eye_x, right_eye_y)
            aligned_three.append(cv2.resize(aligned, (480, 480)))
    else:
        aligned_three = preprocessor.preprocess_frames(raw_frames, prefix=video_path.stem)
        if len(aligned_three) != 3:
            raise RuntimeError(f"aligned_frame_count_mismatch:{len(aligned_three)}")
        aligned_three = [cv2.resize(frame, (480, 480)) for frame in aligned_three]
    stack_frames = [aligned_three[0], aligned_three[0], aligned_three[1], aligned_three[2], aligned_three[2]]
    stack_colors = [target_colors[0], target_colors[0], target_colors[1], target_colors[2], target_colors[2]]
    stacked_img = np.concatenate(stack_frames, axis=0)
    normal_cue = tg.cal_normal_cues_map(stacked_img, stack_colors)
    if normal_cue is None:
        raise RuntimeError("tg_cal_normal_cues_map_failed")

    depth_stack = np.concatenate([build_depth_tile(frame) for frame in aligned_three], axis=0)

    split_dir = output_root / split_name
    ensure_dir(split_dir)
    stem = video_path.stem
    sample_jpg = split_dir / f"{stem}.jpg"
    sample_txt = split_dir / f"{stem}.txt"
    sample_depth = split_dir / f"{stem}_d.jpg"
    metadata_json = split_dir / f"{stem}_metadata.json"

    encode_image(sample_jpg, normal_cue)
    encode_image(sample_depth, depth_stack)
    sample_txt.write_text("\n".join(str(color) for color in target_colors) + "\n", encoding="utf-8")

    metadata = {
        "split": split_name,
        "label_id": label_id,
        "label_name": "live" if label_id == 1 else "spoof",
        "video_path": str(video_path),
        "video_txt_path": str(txt_path),
        "sample_jpg": str(sample_jpg),
        "sample_txt": str(sample_txt),
        "sample_depth": str(sample_depth),
        "chosen_frame_indices": picked_three,
        "chosen_colors_packed": target_colors,
        "stack_colors_packed": stack_colors,
        "normal_cue_shape": list(normal_cue.shape),
        "depth_shape": list(depth_stack.shape),
        "export_basis": [
            "selected 3 stable frames by ThunderGuard-style stable-color filtering",
            f"alignment path: {export_mode}",
            "duplicated edge colors to build 5-frame stack [c1,c1,c2,c3,c3]",
            "called TG.cal_normal_cues_map on the 5x480 stacked aligned faces",
            "wrote final ThunderGuard-style .jpg/.txt/_d.jpg outputs",
        ],
    }
    save_json(metadata_json, metadata)
    metadata["metadata_json"] = str(metadata_json)
    return metadata


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    target_colors = parse_color_triplets(args.color_triplets)
    samples = discover_samples(input_root, args.split)
    if args.limit > 0:
        samples = samples[:args.limit]

    ensure_dir(output_root)
    preprocessor = FacePreprocessor(
        detector_model_path=args.detector_model,
        detector_device=args.detector_device,
        target_size=(args.align_size, args.align_size),
        conf_threshold=args.detector_conf,
        iou_threshold=args.detector_iou,
    )
    legacy_face_detector = None
    legacy_face_align = None
    if args.export_mode == "legacy_fast_align":
        LandmarkFaceDetector = load_legacy_landmark_detector_class()
        legacy_face_detector = LandmarkFaceDetector(str(FACE_DETECT_ROOT / "face_tool/RBF_Final.pth"))
        legacy_face_align = FaceAlign(align_size=args.align_size, face_size=208, max_align_offset_pixel=4)
    tg = TG(str(DEFAULT_TG_MASK_PATH))

    rows: list[dict] = []
    for index, (split_name, video_path_str, txt_path_str, label_id) in enumerate(samples, start=1):
        video_path = Path(video_path_str)
        txt_path = Path(txt_path_str)
        print(f"[{index}/{len(samples)}] exporting {video_path.name}", flush=True)
        row = {
            "index": index,
            "split": split_name,
            "label_id": label_id,
            "label_name": "live" if label_id == 1 else "spoof",
            "video_path": str(video_path),
            "video_txt_path": str(txt_path),
            "status": "failed",
            "error": "",
            "sample_jpg": "",
            "sample_txt": "",
            "sample_depth": "",
            "metadata_json": "",
        }
        try:
            result = export_one_sample(
                tg=tg,
                preprocessor=preprocessor,
                legacy_face_detector=legacy_face_detector,
                legacy_face_align=legacy_face_align,
                split_name=split_name,
                video_path=video_path,
                txt_path=txt_path,
                label_id=label_id,
                output_root=output_root,
                target_colors=target_colors,
                export_mode=args.export_mode,
            )
            row.update(
                {
                    "status": "ok",
                    "sample_jpg": result["sample_jpg"],
                    "sample_txt": result["sample_txt"],
                    "sample_depth": result["sample_depth"],
                    "metadata_json": result["metadata_json"],
                }
            )
        except Exception as exc:
            row["error"] = repr(exc)
        rows.append(row)

    write_tsv(output_root / "manifest.tsv", rows)
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "split": args.split,
        "total_samples": len(rows),
        "ok_samples": sum(1 for row in rows if row["status"] == "ok"),
        "failed_samples": sum(1 for row in rows if row["status"] != "ok"),
        "target_colors": target_colors,
        "notes": [
            "This export matches ThunderGuard tg_export final format: .jpg/.txt/_d.jpg.",
            f"Alignment path: {args.export_mode}.",
            "It reuses TG.cal_normal_cues_map and writes 3 packed colors in the final txt.",
            "Depth export is a center-cropped grayscale placeholder derived from the 3 selected aligned frames.",
        ],
    }
    save_json(output_root / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
