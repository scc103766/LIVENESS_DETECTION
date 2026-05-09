from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np


COLOR_SEQUENCE_RGB = [
    (255, 20, 255),
    (20, 255, 20),
    (255, 20, 20),
]
NEUTRAL_SENTINEL_RGB = (-1, -1, -1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record a flash-liveness video while showing a fixed three-color fullscreen stimulus, "
            "and write a matching txt with frame-wise color labels."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory to save the recorded mp4 and txt files.",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default=None,
        help=(
            "Output file basename without extension. "
            "If omitted, use a timestamp-style name like 1710000000000_cam0_1_1_none_1."
        ),
    )
    parser.add_argument("--camera-id", type=int, default=0, help="OpenCV camera id.")
    parser.add_argument("--camera-width", type=int, default=1280, help="Requested camera width.")
    parser.add_argument("--camera-height", type=int, default=720, help="Requested camera height.")
    parser.add_argument("--camera-fps", type=float, default=30.0, help="Requested camera fps.")
    parser.add_argument(
        "--display-width",
        type=int,
        default=1920,
        help="Fullscreen flash stimulus width.",
    )
    parser.add_argument(
        "--display-height",
        type=int,
        default=1080,
        help="Fullscreen flash stimulus height.",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=1.0,
        help="Neutral/original-state warmup time before the first flash color.",
    )
    parser.add_argument(
        "--hold-seconds",
        type=float,
        default=0.35,
        help="How long to hold each flash color before switching.",
    )
    parser.add_argument(
        "--restore-seconds",
        type=float,
        default=0.15,
        help="Neutral/original-state recovery time after each flash color.",
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=4,
        help="How many full RGB cycles to record.",
    )
    parser.add_argument(
        "--tail-seconds",
        type=float,
        default=0.5,
        help="Neutral/original-state tail time after the final flash color.",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="flash_liveness_capture",
        help="OpenCV display window name.",
    )
    parser.add_argument(
        "--codec",
        type=str,
        default="mp4v",
        help="FourCC codec for mp4 writing, for example mp4v or avc1.",
    )
    parser.add_argument(
        "--show-preview",
        action="store_true",
        help="Also show a small camera preview window.",
    )
    return parser.parse_args()


def rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return rgb[2], rgb[1], rgb[0]


def rgb_to_packed_int(rgb: tuple[int, int, int]) -> int:
    r, g, b = rgb
    return (r << 16) + (g << 8) + b


def build_flash_frames(width: int, height: int) -> dict[tuple[int, int, int], np.ndarray]:
    frames: dict[tuple[int, int, int], np.ndarray] = {}
    for rgb in COLOR_SEQUENCE_RGB:
        bgr = rgb_to_bgr(rgb)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = bgr[0]
        frame[:, :, 1] = bgr[1]
        frame[:, :, 2] = bgr[2]
        frames[rgb] = frame
    # For offline protocol generation, the neutral sentinel means
    # "keep original / no extra flash". During live fullscreen collection
    # we still render a dark frame for this phase because the capture script
    # cannot preserve the scene underneath the fullscreen stimulus.
    frames[NEUTRAL_SENTINEL_RGB] = np.zeros((height, width, 3), dtype=np.uint8)
    return frames


def resolve_basename(camera_id: int, basename: str | None) -> str:
    if basename:
        return basename
    now_ms = int(time.time() * 1000)
    return f"{now_ms}_cam{camera_id}_1_1_none_1"


def build_timeline(
    warmup_seconds: float,
    hold_seconds: float,
    restore_seconds: float = 0.15,
    cycles: int = 4,
    tail_seconds: float = 0.5,
) -> list[tuple[float, float, tuple[int, int, int]]]:
    timeline: list[tuple[float, float, tuple[int, int, int]]] = []
    cursor = 0.0

    if warmup_seconds > 0:
        timeline.append((cursor, cursor + warmup_seconds, NEUTRAL_SENTINEL_RGB))
        cursor += warmup_seconds

    for _ in range(cycles):
        for rgb in COLOR_SEQUENCE_RGB:
            timeline.append((cursor, cursor + hold_seconds, rgb))
            cursor += hold_seconds
            if restore_seconds > 0:
                timeline.append((cursor, cursor + restore_seconds, NEUTRAL_SENTINEL_RGB))
                cursor += restore_seconds

    if tail_seconds > 0:
        timeline.append((cursor, cursor + tail_seconds, NEUTRAL_SENTINEL_RGB))
        cursor += tail_seconds

    return timeline


def build_repeated_timeline_for_duration(
    duration_seconds: float,
    warmup_seconds: float,
    hold_seconds: float,
    restore_seconds: float = 0.15,
    tail_seconds: float = 0.5,
    color_sequence: list[tuple[int, int, int]] | None = None,
) -> list[tuple[float, float, tuple[int, int, int]]]:
    """Build a collection-style timeline that fills an arbitrary duration.

    This mirrors the online collection protocol:
    - optional neutral/original-state warmup
    - repeated flash holds
    - optional neutral/original-state recovery after each flash
    - optional neutral/original-state tail

    The color section is repeated until the remaining available duration is filled.
    """
    if duration_seconds <= 0:
        return []
    if hold_seconds <= 0:
        raise ValueError("hold_seconds must be > 0")

    colors = color_sequence if color_sequence is not None else COLOR_SEQUENCE_RGB
    timeline: list[tuple[float, float, tuple[int, int, int]]] = []
    cursor = 0.0
    usable_warmup = max(min(warmup_seconds, duration_seconds), 0.0)
    if usable_warmup > 0:
        timeline.append((cursor, cursor + usable_warmup, NEUTRAL_SENTINEL_RGB))
        cursor += usable_warmup

    tail_start = max(cursor, duration_seconds - max(tail_seconds, 0.0))
    color_index = 0
    while cursor < tail_start - 1e-9:
        color = colors[color_index % len(colors)]
        next_cursor = min(cursor + hold_seconds, tail_start)
        timeline.append((cursor, next_cursor, color))
        cursor = next_cursor
        color_index += 1
        if restore_seconds > 0 and cursor < tail_start - 1e-9:
            restore_end = min(cursor + restore_seconds, tail_start)
            timeline.append((cursor, restore_end, NEUTRAL_SENTINEL_RGB))
            cursor = restore_end

    if cursor < duration_seconds:
        timeline.append((cursor, duration_seconds, NEUTRAL_SENTINEL_RGB))
    return timeline


def active_color_at(elapsed_seconds: float, timeline: list[tuple[float, float, tuple[int, int, int]]]) -> tuple[int, int, int]:
    for start_s, end_s, color in timeline:
        if start_s <= elapsed_seconds < end_s:
            return color
    return (-1, -1, -1)


def build_frame_color_labels(
    frame_count: int,
    fps: float,
    warmup_seconds: float,
    hold_seconds: float,
    restore_seconds: float = 0.15,
    tail_seconds: float = 0.5,
    color_sequence: list[tuple[int, int, int]] | None = None,
) -> list[int]:
    """Return packed RGB color labels for each frame using collection-style timing."""
    if frame_count <= 0:
        return []
    if fps <= 1e-6:
        raise ValueError("fps must be > 0")

    duration_seconds = frame_count / fps
    timeline = build_repeated_timeline_for_duration(
        duration_seconds=duration_seconds,
        warmup_seconds=warmup_seconds,
        hold_seconds=hold_seconds,
        restore_seconds=restore_seconds,
        tail_seconds=tail_seconds,
        color_sequence=color_sequence,
    )
    labels: list[int] = []
    for frame_index in range(frame_count):
        elapsed = frame_index / fps
        color = active_color_at(elapsed, timeline)
        labels.append(rgb_to_packed_int(color) if color != NEUTRAL_SENTINEL_RGB else 0)
    return labels


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {args.camera_id}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
    return cap


def open_writer(video_path: Path, width: int, height: int, fps: float, codec: str) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {video_path}")
    return writer


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    basename = resolve_basename(args.camera_id, args.basename)
    video_path = args.output_dir / f"{basename}.mp4"
    txt_path = args.output_dir / f"{basename}.txt"

    timeline = build_timeline(
        warmup_seconds=args.warmup_seconds,
        hold_seconds=args.hold_seconds,
        restore_seconds=args.restore_seconds,
        cycles=args.cycles,
        tail_seconds=args.tail_seconds,
    )
    total_duration = timeline[-1][1] if timeline else 0.0
    flash_frames = build_flash_frames(args.display_width, args.display_height)

    cap = open_camera(args)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.camera_width
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.camera_height
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    if actual_fps is None or actual_fps <= 1e-6:
        actual_fps = args.camera_fps

    writer = open_writer(video_path, actual_width, actual_height, actual_fps, args.codec)

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(args.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    if args.show_preview:
        cv2.namedWindow("camera_preview", cv2.WINDOW_NORMAL)

    print("Flash liveness collection is about to start.")
    print("Press 'q' to stop early.")
    print(f"video_path: {video_path}")
    print(f"txt_path: {txt_path}")
    print(f"actual_camera_size: {actual_width}x{actual_height}")
    print(f"actual_camera_fps: {actual_fps:.2f}")
    print(f"color_sequence_rgb: {COLOR_SEQUENCE_RGB}")
    print(f"timeline_seconds: {total_duration:.2f}")

    frame_index = 0
    start_time = time.perf_counter()
    txt_lines: list[str] = []

    try:
        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed >= total_duration:
                break

            current_color = active_color_at(elapsed, timeline)
            stimulus = flash_frames[current_color]
            cv2.imshow(args.window_name, stimulus)

            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("Camera frame read failed during collection.")

            writer.write(frame)

            packed_color = rgb_to_packed_int(current_color) if current_color != NEUTRAL_SENTINEL_RGB else 0
            txt_lines.append(f"{frame_index},{packed_color}")

            if args.show_preview:
                preview = frame.copy()
                label = f"frame={frame_index} color={packed_color}"
                cv2.putText(
                    preview,
                    label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                )
                cv2.imshow("camera_preview", preview)

            frame_index += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Collection stopped early by user.")
                break
    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    txt_path.write_text("\n".join(txt_lines) + ("\n" if txt_lines else ""), encoding="utf-8")

    print("Flash liveness collection finished.")
    print(f"saved_frames: {frame_index}")
    print(f"video_path: {video_path}")
    print(f"txt_path: {txt_path}")
    print("txt_format: frame_idx,color_int")
    print("packed_colors:")
    for rgb in COLOR_SEQUENCE_RGB:
        print(f"  rgb={rgb} packed={rgb_to_packed_int(rgb)}")


if __name__ == "__main__":
    main()
