from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# Keep this default aligned with thunderguard_facepreprocessor_service/app.py:
# --color-triplets 16717055,1376020,16716820
DEFAULT_FLASH_COLORS_PACKED = [16717055, 1376020, 16716820]
DEFAULT_WARMUP_COLOR_PACKED = 16777215
DEFAULT_TAIL_COLOR_PACKED = 0
DEFAULT_API_URL = "http://127.0.0.1:18121/predict"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Record a real screen-flash liveness video, write the matching frame/color txt, "
            "zip both files, and optionally upload the archive to the ThunderGuard API."
        )
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for mp4/txt/zip/result files.")
    parser.add_argument("--basename", type=str, default=None, help="Output basename without extension.")
    parser.add_argument("--camera-id", type=int, default=0)
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fps", type=float, default=30.0)
    parser.add_argument("--display-width", type=int, default=1920)
    parser.add_argument("--display-height", type=int, default=1080)
    parser.add_argument(
        "--flash-colors",
        default=",".join(str(value) for value in DEFAULT_FLASH_COLORS_PACKED),
        help=(
            "Three packed RGB colors used by the API, decimal or hex. "
            "Default: 16717055,1376020,16716820."
        ),
    )
    parser.add_argument("--warmup-color", default=str(DEFAULT_WARMUP_COLOR_PACKED), help="Packed RGB warmup color.")
    parser.add_argument("--tail-color", default=str(DEFAULT_TAIL_COLOR_PACKED), help="Packed RGB tail color.")
    parser.add_argument("--warmup-seconds", type=float, default=0.55, help="Warmup duration, about 17 frames at 30fps.")
    parser.add_argument("--hold-seconds", type=float, default=0.60, help="Each flash color duration, about 18 frames.")
    parser.add_argument("--restore-seconds", type=float, default=0.0, help="Optional recovery time after each flash.")
    parser.add_argument("--cycles", type=int, default=1, help="Number of full flash-color cycles.")
    parser.add_argument("--tail-seconds", type=float, default=0.50, help="Tail duration, about 15 frames at 30fps.")
    parser.add_argument("--window-name", type=str, default="flash_liveness_capture")
    parser.add_argument("--codec", type=str, default="mp4v")
    parser.add_argument("--show-preview", action="store_true")
    parser.add_argument(
        "--api-url",
        default=DEFAULT_API_URL,
        help="ThunderGuard /predict URL. Use --skip-upload to only record files.",
    )
    parser.add_argument("--upload-timeout", type=float, default=300.0)
    parser.add_argument("--skip-upload", action="store_true", help="Only record mp4/txt/zip; do not call the API.")
    parser.add_argument("--keep-window-on-error", action="store_true")
    return parser.parse_args()


def parse_packed_color(raw: str) -> int:
    value = int(raw.strip(), 0)
    if value < 0 or value > 0xFFFFFF:
        raise ValueError(f"color out of range: {raw}")
    return value


def parse_flash_colors(raw: str) -> list[int]:
    values = [parse_packed_color(item) for item in raw.split(",") if item.strip()]
    if len(values) != 3:
        raise ValueError("--flash-colors must contain exactly three packed RGB values")
    return values


def packed_to_rgb(color: int) -> tuple[int, int, int]:
    return ((color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF)


def rgb_to_bgr(rgb: tuple[int, int, int]) -> tuple[int, int, int]:
    return rgb[2], rgb[1], rgb[0]


def build_solid_frame(width: int, height: int, packed_color: int) -> np.ndarray:
    bgr = rgb_to_bgr(packed_to_rgb(packed_color))
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :, 0] = bgr[0]
    frame[:, :, 1] = bgr[1]
    frame[:, :, 2] = bgr[2]
    return frame


def resolve_basename(camera_id: int, basename: str | None) -> str:
    if basename:
        return basename
    return f"{int(time.time() * 1000)}_cam{camera_id}_1_1_none_1"


def build_timeline(
    flash_colors: list[int],
    warmup_color: int,
    tail_color: int,
    warmup_seconds: float,
    hold_seconds: float,
    restore_seconds: float,
    cycles: int,
    tail_seconds: float,
) -> list[tuple[float, float, int]]:
    if hold_seconds <= 0:
        raise ValueError("--hold-seconds must be > 0")
    if cycles <= 0:
        raise ValueError("--cycles must be > 0")

    timeline: list[tuple[float, float, int]] = []
    cursor = 0.0

    if warmup_seconds > 0:
        timeline.append((cursor, cursor + warmup_seconds, warmup_color))
        cursor += warmup_seconds

    for _ in range(cycles):
        for color in flash_colors:
            timeline.append((cursor, cursor + hold_seconds, color))
            cursor += hold_seconds
            if restore_seconds > 0:
                timeline.append((cursor, cursor + restore_seconds, tail_color))
                cursor += restore_seconds

    if tail_seconds > 0:
        timeline.append((cursor, cursor + tail_seconds, tail_color))

    return timeline


def active_color_at(elapsed_seconds: float, timeline: list[tuple[float, float, int]], fallback_color: int) -> int:
    for start_s, end_s, color in timeline:
        if start_s <= elapsed_seconds < end_s:
            return color
    return fallback_color


def open_camera(args: argparse.Namespace) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera {args.camera_id}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.camera_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.camera_height)
    cap.set(cv2.CAP_PROP_FPS, args.camera_fps)
    return cap


def open_writer(video_path: Path, width: int, height: int, fps: float, codec: str) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*codec), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {video_path}")
    return writer


def write_archive(zip_path: Path, video_path: Path, txt_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(video_path, arcname=video_path.name)
        archive.write(txt_path, arcname=txt_path.name)


def upload_archive(api_url: str, archive_path: Path, timeout: float) -> dict[str, Any]:
    boundary = f"----tg-capture-{uuid.uuid4().hex}"
    archive_bytes = archive_path.read_bytes()
    body = b"".join(
        [
            f"--{boundary}\r\n".encode("utf-8"),
            (
                f'Content-Disposition: form-data; name="file"; filename="{archive_path.name}"\r\n'
                "Content-Type: application/zip\r\n\r\n"
            ).encode("utf-8"),
            archive_bytes,
            f"\r\n--{boundary}--\r\n".encode("utf-8"),
        ]
    )
    request = urllib.request.Request(
        api_url,
        data=body,
        headers={
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_body = response.read()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"API upload failed: HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"API upload failed: {exc}") from exc

    try:
        return json.loads(response_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"API returned non-JSON response: {response_body[:500]!r}") from exc


def save_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    flash_colors = parse_flash_colors(args.flash_colors)
    warmup_color = parse_packed_color(args.warmup_color)
    tail_color = parse_packed_color(args.tail_color)

    basename = resolve_basename(args.camera_id, args.basename)
    video_path = args.output_dir / f"{basename}.mp4"
    txt_path = args.output_dir / f"{basename}.txt"
    zip_path = args.output_dir / f"{basename}.zip"
    capture_metadata_path = args.output_dir / f"{basename}_capture_metadata.json"
    response_path = args.output_dir / f"{basename}_api_response.json"

    timeline = build_timeline(
        flash_colors=flash_colors,
        warmup_color=warmup_color,
        tail_color=tail_color,
        warmup_seconds=args.warmup_seconds,
        hold_seconds=args.hold_seconds,
        restore_seconds=args.restore_seconds,
        cycles=args.cycles,
        tail_seconds=args.tail_seconds,
    )
    total_duration = timeline[-1][1] if timeline else 0.0
    flash_frames = {
        color: build_solid_frame(args.display_width, args.display_height, color)
        for color in sorted({warmup_color, tail_color, *flash_colors})
    }

    cap = open_camera(args)
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or args.camera_width
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or args.camera_height
    actual_fps = float(cap.get(cv2.CAP_PROP_FPS) or args.camera_fps)
    if actual_fps <= 1e-6:
        actual_fps = args.camera_fps
    writer = open_writer(video_path, actual_width, actual_height, actual_fps, args.codec)

    cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(args.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    if args.show_preview:
        cv2.namedWindow("camera_preview", cv2.WINDOW_NORMAL)

    print("Real flash liveness collection is about to start.")
    print("Keep the face centered and close enough for the screen flash to illuminate it.")
    print("Press 'q' to stop early.")
    print(f"video_path: {video_path}")
    print(f"txt_path: {txt_path}")
    print(f"api_url: {args.api_url if not args.skip_upload else '(upload skipped)'}")
    print(f"actual_camera_size: {actual_width}x{actual_height}")
    print(f"actual_camera_fps: {actual_fps:.2f}")
    print(f"flash_colors_packed: {flash_colors}")
    print(f"flash_colors_rgb: {[list(packed_to_rgb(color)) for color in flash_colors]}")
    print(f"timeline_seconds: {total_duration:.2f}")

    frame_index = 0
    start_time = time.perf_counter()
    txt_lines: list[str] = []
    stopped_early = False

    try:
        while True:
            elapsed = time.perf_counter() - start_time
            if elapsed >= total_duration:
                break

            current_color = active_color_at(elapsed, timeline, tail_color)
            cv2.imshow(args.window_name, flash_frames[current_color])

            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("Camera frame read failed during collection.")

            writer.write(frame)
            txt_lines.append(f"{frame_index},{current_color}")

            if args.show_preview:
                preview = frame.copy()
                label = f"frame={frame_index} color={current_color}"
                cv2.putText(preview, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("camera_preview", preview)

            frame_index += 1
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                stopped_early = True
                print("Collection stopped early by user.")
                break
    finally:
        cap.release()
        writer.release()
        if not args.keep_window_on_error:
            cv2.destroyAllWindows()

    txt_path.write_text("\n".join(txt_lines) + ("\n" if txt_lines else ""), encoding="utf-8")
    write_archive(zip_path, video_path, txt_path)

    color_counts: dict[str, int] = {}
    for line in txt_lines:
        color = line.split(",", 1)[1]
        color_counts[color] = color_counts.get(color, 0) + 1

    capture_metadata = {
        "basename": basename,
        "video_path": str(video_path),
        "txt_path": str(txt_path),
        "zip_path": str(zip_path),
        "frame_count": frame_index,
        "stopped_early": stopped_early,
        "actual_camera_width": actual_width,
        "actual_camera_height": actual_height,
        "actual_camera_fps": actual_fps,
        "flash_colors_packed": flash_colors,
        "flash_colors_rgb": [list(packed_to_rgb(color)) for color in flash_colors],
        "warmup_color_packed": warmup_color,
        "tail_color_packed": tail_color,
        "timeline": [
            {"start": round(start, 4), "end": round(end, 4), "color": color}
            for start, end, color in timeline
        ],
        "txt_color_counts": color_counts,
        "api_url": "" if args.skip_upload else args.api_url,
    }
    save_json(capture_metadata_path, capture_metadata)

    print("Collection finished.")
    print(f"saved_frames: {frame_index}")
    print(f"zip_path: {zip_path}")
    print(f"capture_metadata_json: {capture_metadata_path}")

    if args.skip_upload:
        return

    print("Uploading real capture archive to API...")
    response_payload = upload_archive(args.api_url, zip_path, args.upload_timeout)
    save_json(response_path, response_payload)
    print(f"api_response_json: {response_path}")
    print(json.dumps(response_payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
