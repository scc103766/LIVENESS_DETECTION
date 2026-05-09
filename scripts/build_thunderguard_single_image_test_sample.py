from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


DEFAULT_COLOR_TRIPLETS = [
    (255, 20, 255),
    (20, 255, 20),
    (255, 20, 20),
]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a ThunderGuard-compatible single-sample test package from one normal image. "
            "This is intended for pipeline debugging and inference smoke tests, not for real liveness validation."
        )
    )
    parser.add_argument("--input-image", required=True, help="Path to a normal input image.")
    parser.add_argument("--output-dir", required=True, help="Directory to write the generated sample files.")
    parser.add_argument(
        "--sample-name",
        default="single_image_debug_1",
        help="Output sample basename. Generated files will be <sample-name>.jpg/.txt and optional _d.jpg.",
    )
    parser.add_argument(
        "--color-triplets",
        default="255,20,255;20,255,20;255,20,20",
        help="Three RGB triplets separated by semicolons, for example '255,20,255;20,255,20;255,20,20'.",
    )
    parser.add_argument(
        "--create-depth-placeholder",
        action="store_true",
        help="Also create a placeholder _d.jpg so the output can mimic tg_export layout more closely.",
    )
    parser.add_argument(
        "--second-view-mode",
        default="blur",
        choices=["same", "blur", "brighten"],
        help="How to synthesize the second tile for each color block.",
    )
    return parser.parse_args()


def parse_color_triplets(raw_value: str) -> list[tuple[int, int, int]]:
    triplets: list[tuple[int, int, int]] = []
    for chunk in raw_value.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [part.strip() for part in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Invalid color triplet: {chunk}")
        rgb = tuple(int(part) for part in parts)
        if any(channel < 0 or channel > 255 for channel in rgb):
            raise ValueError(f"RGB values must be in [0, 255]: {chunk}")
        triplets.append(rgb)
    if len(triplets) != 3:
        raise ValueError("Exactly three color triplets are required.")
    return triplets


def rgb_to_packed_int(rgb: tuple[int, int, int]) -> int:
    r, g, b = rgb
    return (r << 16) + (g << 8) + b


def read_image(image_path: Path) -> np.ndarray:
    data = np.fromfile(str(image_path), dtype=np.uint8)
    if data.size == 0:
        raise ValueError(f"Failed to read image bytes: {image_path}")
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {image_path}")
    return image


def center_crop_and_resize(image: np.ndarray, size: int = 256) -> np.ndarray:
    height, width = image.shape[:2]
    side = min(height, width)
    top = max((height - side) // 2, 0)
    left = max((width - side) // 2, 0)
    cropped = image[top:top + side, left:left + side]
    if cropped.size == 0:
        cropped = image
    return cv2.resize(cropped, (size, size))


def tint_image(base_bgr: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    # ThunderGuard txt colors are stored in RGB order, while OpenCV images are BGR.
    bgr = np.asarray([rgb[2], rgb[1], rgb[0]], dtype=np.float32) / 255.0
    tinted = base_bgr.astype(np.float32) * bgr.reshape(1, 1, 3)
    return np.clip(tinted, 0, 255).astype(np.uint8)


def synthesize_second_view(image: np.ndarray, mode: str) -> np.ndarray:
    if mode == "same":
        return image.copy()
    if mode == "brighten":
        return np.clip(image.astype(np.float32) * 1.08 + 6.0, 0, 255).astype(np.uint8)
    return cv2.GaussianBlur(image, (5, 5), 0)


def pad_to_thunderguard_tile(image_256: np.ndarray) -> np.ndarray:
    tile = np.zeros((288, 288, 3), dtype=np.uint8)
    tile[16:272, 16:272, :] = image_256
    return tile


def build_stack_tiles(base_image: np.ndarray, color_triplets: list[tuple[int, int, int]], second_view_mode: str) -> np.ndarray:
    base_256 = center_crop_and_resize(base_image, size=256)
    stacked_tiles: list[np.ndarray] = []
    for rgb in color_triplets:
        primary = tint_image(base_256, rgb)
        secondary = synthesize_second_view(primary, second_view_mode)
        stacked_tiles.append(pad_to_thunderguard_tile(primary))
        stacked_tiles.append(pad_to_thunderguard_tile(secondary))
    return np.concatenate(stacked_tiles, axis=0)


def build_depth_placeholder(base_image: np.ndarray) -> np.ndarray:
    base_256 = center_crop_and_resize(base_image, size=256)
    gray = cv2.cvtColor(base_256, cv2.COLOR_BGR2GRAY)
    gray_3 = cv2.GaussianBlur(gray, (5, 5), 0)
    depth_tile = np.zeros((288, 288), dtype=np.uint8)
    depth_tile[16:272, 16:272] = gray_3
    return np.concatenate([depth_tile, depth_tile, depth_tile], axis=0)


def write_image(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(path.suffix, image)
    if not ok:
        raise ValueError(f"Failed to encode image for writing: {path}")
    path.write_bytes(encoded.tobytes())


def main() -> None:
    args = parse_args()
    input_image = Path(args.input_image)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if input_image.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported input image type: {input_image.suffix}")

    color_triplets = parse_color_triplets(args.color_triplets)
    base_image = read_image(input_image)

    stacked_image = build_stack_tiles(base_image, color_triplets, args.second_view_mode)
    packed_colors = [rgb_to_packed_int(rgb) for rgb in color_triplets]

    jpg_path = output_dir / f"{args.sample_name}.jpg"
    txt_path = output_dir / f"{args.sample_name}.txt"
    write_image(jpg_path, stacked_image)
    txt_path.write_text("\n".join(str(color) for color in packed_colors) + "\n", encoding="utf-8")

    depth_path = None
    if args.create_depth_placeholder:
        depth_path = output_dir / f"{args.sample_name}_d.jpg"
        write_image(depth_path, build_depth_placeholder(base_image))

    print("ThunderGuard synthetic sample generated")
    print(f"input_image: {input_image}")
    print(f"sample_jpg: {jpg_path}")
    print(f"sample_txt: {txt_path}")
    if depth_path is not None:
        print(f"sample_depth_placeholder: {depth_path}")
    print(f"color_triplets_rgb: {color_triplets}")
    print(f"color_triplets_packed: {packed_colors}")
    print(
        "note: this sample is synthesized from one normal image and is suitable for pipeline smoke tests only; "
        "it does not recreate real flash-liveness biological cues."
    )


if __name__ == "__main__":
    main()
