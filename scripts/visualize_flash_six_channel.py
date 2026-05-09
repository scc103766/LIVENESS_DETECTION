from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flash_liveness_project_v2 import FacePreprocessor, ensure_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize how V2 builds RGB(3) + Diff(3) into a 6-channel tensor from video frames."
    )
    parser.add_argument("--video-path", required=True, help="Input video path.")
    parser.add_argument("--output-dir", required=True, help="Directory to save visualization images.")
    parser.add_argument("--frame-index", type=int, default=10, help="Center frame index i. Will use i-1 and i.")
    parser.add_argument("--target-size", type=int, default=224, help="Face crop output size.")
    parser.add_argument("--detector-model", default=None, help="Optional face detector model path.")
    parser.add_argument("--detector-device", default=None, help="Optional detector device, such as cuda:0.")
    parser.add_argument("--detector-conf", type=float, default=0.5)
    parser.add_argument("--detector-iou", type=float, default=0.5)
    return parser.parse_args()


def read_all_frames(video_path: Path) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame is not None and frame.size > 0:
            frames.append(frame.copy())
    cap.release()
    return frames


def to_uint8_vis(single_channel: np.ndarray) -> np.ndarray:
    channel = np.asarray(single_channel, dtype=np.float32)
    ch_min = float(channel.min())
    ch_max = float(channel.max())
    if ch_max - ch_min < 1e-8:
        scaled = np.zeros_like(channel, dtype=np.uint8)
    else:
        scaled = ((channel - ch_min) / (ch_max - ch_min) * 255.0).clip(0, 255).astype(np.uint8)
    return scaled


def save_image(path: Path, image: np.ndarray) -> None:
    ok = cv2.imwrite(str(path), image)
    if not ok:
        raise RuntimeError(f"failed_to_write_image:{path}")


def add_title(image: np.ndarray, title: str) -> np.ndarray:
    canvas = image.copy()
    cv2.putText(
        canvas,
        title,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )
    return canvas


def grid_2x3(images: list[np.ndarray], titles: list[str]) -> np.ndarray:
    titled = [add_title(img, title) for img, title in zip(images, titles)]
    row1 = np.concatenate(titled[:3], axis=1)
    row2 = np.concatenate(titled[3:], axis=1)
    return np.concatenate([row1, row2], axis=0)


def main() -> int:
    args = parse_args()
    video_path = Path(args.video_path).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    frames = read_all_frames(video_path)
    if len(frames) < 2:
        raise RuntimeError("video_has_less_than_2_frames")

    center_index = max(1, min(args.frame_index, len(frames) - 1))
    prev_index = center_index - 1

    preprocessor = FacePreprocessor(
        detector_model_path=args.detector_model,
        detector_device=args.detector_device,
        target_size=(args.target_size, args.target_size),
        conf_threshold=args.detector_conf,
        iou_threshold=args.detector_iou,
    )
    processed = preprocessor.preprocess_frames(
        [frames[prev_index], frames[center_index]],
        prefix=video_path.stem,
    )
    if len(processed) != 2:
        raise RuntimeError("face_preprocess_failed")

    prev_bgr, curr_bgr = processed
    prev_rgb = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    curr_rgb = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    diff = curr_rgb - prev_rgb
    six_channel = np.concatenate([curr_rgb, diff], axis=-1)

    curr_rgb_vis = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2RGB)
    rgb_channels = [curr_rgb_vis[:, :, i] for i in range(3)]
    rgb_vis = [cv2.cvtColor(ch, cv2.COLOR_GRAY2BGR) for ch in rgb_channels]

    diff_vis = [cv2.applyColorMap(to_uint8_vis(diff[:, :, i]), cv2.COLORMAP_JET) for i in range(3)]

    save_image(output_dir / "00_prev_raw_bgr.jpg", frames[prev_index])
    save_image(output_dir / "01_curr_raw_bgr.jpg", frames[center_index])
    save_image(output_dir / "02_prev_face_crop.jpg", prev_bgr)
    save_image(output_dir / "03_curr_face_crop.jpg", curr_bgr)
    save_image(output_dir / "04_curr_rgb_full.jpg", cv2.cvtColor(curr_rgb_vis, cv2.COLOR_RGB2BGR))
    save_image(output_dir / "05_rgb_r_channel.jpg", rgb_vis[0])
    save_image(output_dir / "06_rgb_g_channel.jpg", rgb_vis[1])
    save_image(output_dir / "07_rgb_b_channel.jpg", rgb_vis[2])
    save_image(output_dir / "08_diff_r_channel.jpg", diff_vis[0])
    save_image(output_dir / "09_diff_g_channel.jpg", diff_vis[1])
    save_image(output_dir / "10_diff_b_channel.jpg", diff_vis[2])

    rgb_grid = grid_2x3(
        images=rgb_vis + diff_vis,
        titles=[
            "RGB-R",
            "RGB-G",
            "RGB-B",
            "DIFF-R",
            "DIFF-G",
            "DIFF-B",
        ],
    )
    save_image(output_dir / "11_rgb_plus_diff_6ch_grid.jpg", rgb_grid)

    overlay = np.concatenate(
        [
            add_title(prev_bgr, f"Prev frame {prev_index}"),
            add_title(curr_bgr, f"Curr frame {center_index}"),
            add_title(cv2.cvtColor(curr_rgb_vis, cv2.COLOR_RGB2BGR), "Curr RGB"),
        ],
        axis=1,
    )
    save_image(output_dir / "12_pipeline_overview.jpg", overlay)

    np.save(output_dir / "six_channel_tensor.npy", six_channel.astype(np.float32))

    metadata = {
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "frame_count": len(frames),
        "prev_index": prev_index,
        "curr_index": center_index,
        "target_size": args.target_size,
        "six_channel_shape_hwc": list(six_channel.shape),
        "six_channel_layout": [
            "channel_0=curr_R",
            "channel_1=curr_G",
            "channel_2=curr_B",
            "channel_3=diff_R",
            "channel_4=diff_G",
            "channel_5=diff_B",
        ],
        "diff_value_range": {
            "min": float(diff.min()),
            "max": float(diff.max()),
            "mean": float(diff.mean()),
        },
        "detector_model": args.detector_model,
        "detector_used": bool(preprocessor.detector is not None),
        "detector_load_error": preprocessor.detector_load_error,
    }
    (output_dir / "visualization_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Saved visualization to: {output_dir}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
