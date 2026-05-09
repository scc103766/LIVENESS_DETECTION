from __future__ import annotations

import argparse
import random
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LEGACY_FAKE_ROOT = (
    PROJECT_ROOT / "20240320闪光活体归档/dataset/dataset/raw/fake"
)

WHITE_RGB = (255, 255, 255)
BLACK_RGB = (0, 0, 0)

# Reverse engineered from raw/fake txt files. These are the 12 middle flash
# colors seen between the initial white frame block and final black frame block.
LEGACY_COLOR_WHEEL_RGB = [
    (255, 20, 20),
    (255, 20, 128),
    (255, 20, 255),
    (255, 128, 20),
    (255, 255, 20),
    (128, 255, 20),
    (20, 255, 20),
    (20, 255, 128),
    (20, 255, 255),
    (20, 128, 255),
    (20, 20, 255),
    (128, 20, 255),
]

SEGMENT_FRAME_PROFILES = {
    # Most 2022 raw/fake samples are 84-89 frames at 30 fps.
    "short": [18, 18, 18, 18, 16],
    # A secondary group is around 130-138 frames at 30 fps.
    "long": [21, 30, 30, 30, 27],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record an old FaceCollect-style flash liveness AVI and matching "
            "frame-wise color txt, reconstructed from raw/fake dataset patterns."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--basename", type=str, default=None)
    parser.add_argument("--device-name", type=str, default="cam0")
    parser.add_argument("--subject", type=str, default="none")
    parser.add_argument(
        "--attack-label",
        type=str,
        default="5",
        help="Last filename field. raw/fake commonly uses 2/3/4/5.",
    )
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=1080)
    parser.add_argument("--camera-height", type=int, default=1920)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--display-width", type=int, default=1080)
    parser.add_argument("--display-height", type=int, default=1920)
    parser.add_argument("--codec", type=str, default="MJPG")
    parser.add_argument("--window-name", type=str, default="legacy_facecollect_flash")
    parser.add_argument("--show-preview", action="store_true")
    parser.add_argument("--no-fullscreen", action="store_true")
    parser.add_argument(
        "--rotate",
        choices=["none", "cw", "ccw", "180"],
        default="none",
        help="Rotate camera frames before writing.",
    )
    parser.add_argument(
        "--color-mode",
        choices=["empirical", "wheel", "explicit"],
        default="empirical",
        help=(
            "empirical samples a historical 3-color sequence from raw/fake txt; "
            "wheel samples 3 colors from the reconstructed 12-color wheel; "
            "explicit uses --colors."
        ),
    )
    parser.add_argument(
        "--legacy-data-root",
        type=Path,
        default=DEFAULT_LEGACY_FAKE_ROOT,
        help="Directory containing old .txt files for empirical sampling.",
    )
    parser.add_argument(
        "--colors",
        type=str,
        default=None,
        help=(
            "Three colors for explicit mode. Accepts packed ints like "
            "16717055,1376020,16716820 or RGB triplets like "
            "255,20,255;20,255,20;255,20,20."
        ),
    )
    parser.add_argument(
        "--segment-frames",
        type=str,
        default="auto",
        help=(
            "Five segment frame counts for white,c1,c2,c3,black. "
            "Use auto, short, long, or e.g. 18,18,18,18,16."
        ),
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return rgb[2], rgb[1], rgb[0]


def rgb_to_packed_int(rgb: tuple[int, int, int]) -> int:
    r, g, b = rgb
    return (r << 16) + (g << 8) + b


def packed_int_to_rgb(value: int) -> tuple[int, int, int]:
    return (value >> 16) & 255, (value >> 8) & 255, value & 255


def parse_color_spec(spec: str) -> list[tuple[int, int, int]]:
    spec = spec.strip()
    if ";" in spec:
        colors = []
        for chunk in spec.split(";"):
            parts = [int(part.strip()) for part in chunk.split(",")]
            if len(parts) != 3:
                raise ValueError(f"Invalid RGB color triplet: {chunk}")
            colors.append(tuple(parts))
    else:
        colors = [packed_int_to_rgb(int(part.strip())) for part in spec.split(",")]
    if len(colors) != 3:
        raise ValueError("--colors must contain exactly three colors")
    return colors


def parse_segment_frames(spec: str, rng: random.Random) -> list[int] | None:
    if spec == "auto":
        return None
    if spec in SEGMENT_FRAME_PROFILES:
        return list(SEGMENT_FRAME_PROFILES[spec])
    if spec == "mixed":
        return list(rng.choice(list(SEGMENT_FRAME_PROFILES.values())))
    frames = [int(part.strip()) for part in spec.split(",")]
    if len(frames) != 5 or any(frame <= 0 for frame in frames):
        raise ValueError("--segment-frames must contain five positive integers")
    return frames


def extract_segments_from_txt(path: Path) -> tuple[tuple[int, ...], tuple[int, ...]] | None:
    colors = []
    for line in path.read_text(errors="replace").splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",") if part.strip()]
        if len(parts) < 2:
            return None
        try:
            colors.append(int(parts[-1]))
        except ValueError:
            return None
    if not colors:
        return None

    segment_colors: list[int] = []
    segment_frames: list[int] = []
    start = 0
    for index in range(1, len(colors) + 1):
        if index == len(colors) or colors[index] != colors[start]:
            segment_colors.append(colors[start])
            segment_frames.append(index - start)
            start = index

    return tuple(segment_colors), tuple(segment_frames)


def load_empirical_protocols(root: Path) -> list[tuple[tuple[int, int, int], tuple[int, ...], str]]:
    protocols: list[tuple[tuple[int, int, int], tuple[int, ...], str]] = []
    if not root.exists():
        return protocols

    for txt_path in sorted(root.glob("*.txt")):
        parsed = extract_segments_from_txt(txt_path)
        if parsed is None:
            continue
        segment_colors, segment_frames = parsed
        if len(segment_colors) != 5:
            continue
        if segment_colors[0] != rgb_to_packed_int(WHITE_RGB) or segment_colors[-1] != 0:
            continue
        mid_colors = tuple(segment_colors[1:4])
        protocols.append((mid_colors, segment_frames, txt_path.name))
    return protocols


def choose_protocol(args: argparse.Namespace, rng: random.Random) -> tuple[list[tuple[int, int, int]], list[int], str]:
    requested_frames = parse_segment_frames(args.segment_frames, rng)

    if args.color_mode == "explicit":
        if not args.colors:
            raise ValueError("--colors is required when --color-mode explicit")
        colors = parse_color_spec(args.colors)
        frames = requested_frames or list(SEGMENT_FRAME_PROFILES["short"])
        return colors, frames, "explicit"

    if args.color_mode == "empirical":
        protocols = load_empirical_protocols(args.legacy_data_root)
        if protocols:
            mid_colors, empirical_frames, source = rng.choice(protocols)
            colors = [packed_int_to_rgb(value) for value in mid_colors]
            frames = requested_frames or list(empirical_frames)
            return colors, frames, f"empirical:{source}"

    colors = rng.sample(LEGACY_COLOR_WHEEL_RGB, 3)
    frames = requested_frames or list(SEGMENT_FRAME_PROFILES["short"])
    return colors, frames, "wheel"


def resolve_basename(args: argparse.Namespace) -> str:
    if args.basename:
        return args.basename
    now_ms = int(time.time() * 1000)
    return f"{now_ms}_{args.device_name}_1_1_{args.subject}_{args.attack_label}"


def build_stimulus_frames(
    colors: list[tuple[int, int, int]], width: int, height: int
) -> dict[tuple[int, int, int], np.ndarray]:
    frames = {}
    for rgb in [WHITE_RGB, *colors, BLACK_RGB]:
        bgr = rgb_to_bgr(rgb)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :, 0] = bgr[0]
        frame[:, :, 1] = bgr[1]
        frame[:, :, 2] = bgr[2]
        frames[rgb] = frame
    return frames


def rotate_frame(frame: np.ndarray, rotate: str) -> np.ndarray:
    if rotate == "cw":
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    if rotate == "ccw":
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotate == "180":
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {args.camera_id}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
    return cap


def open_writer(path: Path, width: int, height: int, fps: float, codec: str) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    return writer


def print_protocol(colors: list[tuple[int, int, int]], frames: list[int], source: str) -> None:
    segment_colors = [WHITE_RGB, *colors, BLACK_RGB]
    packed = [rgb_to_packed_int(rgb) for rgb in segment_colors]
    print(f"protocol_source: {source}")
    print(f"segment_frames: {frames}")
    print(f"segment_rgb: {segment_colors}")
    print(f"segment_packed: {packed}")
    print(f"total_frames: {sum(frames)}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    colors, segment_frames, source = choose_protocol(args, rng)
    print_protocol(colors, segment_frames, source)
    if args.dry_run:
        return

    global cv2, np
    import cv2
    import numpy as np

    basename = resolve_basename(args)
    video_path = args.output_dir / f"{basename}.avi"
    txt_path = args.output_dir / f"{basename}.txt"
    stimulus_frames = build_stimulus_frames(colors, args.display_width, args.display_height)

    cap = open_camera(args)
    actual_fps = cap.get(cv2.CAP_PROP_FPS) or args.camera_fps
    if actual_fps <= 1e-6:
        actual_fps = args.camera_fps
    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    if not args.no_fullscreen:
        cv2.setWindowProperty(args.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    if args.show_preview:
        cv2.namedWindow("legacy_facecollect_preview", cv2.WINDOW_NORMAL)

    print(f"video_path: {video_path}")
    print(f"txt_path: {txt_path}")
    print(f"actual_camera_fps: {actual_fps:.2f}")

    txt_lines: list[str] = []
    frame_index = 0
    frame_interval = 1.0 / actual_fps if actual_fps > 1e-6 else 0.0
    next_frame_time = time.perf_counter()
    all_segments = list(zip([WHITE_RGB, *colors, BLACK_RGB], segment_frames))
    writer: cv2.VideoWriter | None = None

    try:
        for rgb, count in all_segments:
            packed = rgb_to_packed_int(rgb)
            stimulus = stimulus_frames[rgb]
            for _ in range(count):
                cv2.imshow(args.window_name, stimulus)
                cv2.waitKey(1)

                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError("Camera frame read failed during collection")
                frame = rotate_frame(frame, args.rotate)
                if writer is None:
                    frame_height, frame_width = frame.shape[:2]
                    writer = open_writer(video_path, frame_width, frame_height, actual_fps, args.codec)
                    print(f"actual_camera_size: {frame_width}x{frame_height}")

                writer.write(frame)
                txt_lines.append(f"{frame_index},{packed}")

                if args.show_preview:
                    preview = frame.copy()
                    cv2.putText(
                        preview,
                        f"frame={frame_index} color={packed}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )
                    cv2.imshow("legacy_facecollect_preview", preview)

                frame_index += 1
                next_frame_time += frame_interval
                sleep_seconds = next_frame_time - time.perf_counter()
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    raise KeyboardInterrupt
    except KeyboardInterrupt:
        print("Collection stopped early by user.")
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

    txt_path.write_text("\n".join(txt_lines) + ("\n" if txt_lines else ""), encoding="utf-8")
    print("Legacy FaceCollect-style collection finished.")
    print(f"saved_frames: {frame_index}")
    print(f"video_path: {video_path}")
    print(f"txt_path: {txt_path}")


if __name__ == "__main__":
    main()
