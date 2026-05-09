#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import shutil
from pathlib import Path

import cv2

from collect_flash_liveness_video import (
    COLOR_SEQUENCE_RGB,
    build_frame_color_labels,
    rgb_to_packed_int,
)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a non-destructive Flash Liveness V3 dataset with fixed-order "
            "collect_flash color txt files for every video."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/flash_liveness_asset_archive_by_type"),
        help="Input asset archive root containing manifest.tsv.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/flash_liveness_asset_archive_fixed_collect_protocol"
        ),
        help="Output derived dataset root. Existing content is kept unless --overwrite is set.",
    )
    parser.add_argument("--media-type", choices=["videos"], default="videos")
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--overwrite", action="store_true", help="Remove output-root before rebuilding.")
    parser.add_argument("--warmup-seconds", type=float, default=1.0)
    parser.add_argument("--hold-seconds", type=float, default=0.35)
    parser.add_argument("--restore-seconds", type=float, default=0.15)
    parser.add_argument("--tail-seconds", type=float, default=0.5)
    parser.add_argument("--fallback-fps", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=0, help="Optional smoke-test limit. 0 means all videos.")
    return parser.parse_args()


def ensure_output_root(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def video_metadata(video_path: Path, fallback_fps: float) -> tuple[int, float]:
    cap = cv2.VideoCapture(str(video_path))
    try:
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    if fps <= 1e-6:
        fps = fallback_fps
    return frame_count, fps


def write_fixed_color_txt(
    txt_path: Path,
    frame_count: int,
    fps: float,
    warmup_seconds: float,
    hold_seconds: float,
    restore_seconds: float,
    tail_seconds: float,
) -> None:
    labels = build_frame_color_labels(
        frame_count=frame_count,
        fps=fps,
        warmup_seconds=warmup_seconds,
        hold_seconds=hold_seconds,
        restore_seconds=restore_seconds,
        tail_seconds=tail_seconds,
        color_sequence=COLOR_SEQUENCE_RGB,
    )
    txt_path.write_text(
        "".join(f"{frame_idx},{color_value}\n" for frame_idx, color_value in enumerate(labels)),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = input_root / "manifest.tsv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.tsv not found: {manifest_path}")

    ensure_output_root(output_root, overwrite=args.overwrite)
    rows = read_manifest(manifest_path)

    output_manifest_rows: list[dict[str, object]] = []
    summary_by_category: dict[tuple[str, str], dict[str, int]] = {}
    skipped_rows: list[dict[str, object]] = []
    processed = 0

    for row in rows:
        if row.get("media_type") != args.media_type:
            continue
        label = row.get("label", "")
        if label not in {"live", "spoof"}:
            continue
        source_video = Path(row.get("archive_path", ""))
        if not source_video.exists() or source_video.suffix.lower() not in VIDEO_EXTENSIONS:
            skipped_rows.append(
                {
                    "archive_path": source_video,
                    "reason": "missing_or_unsupported_video",
                    "category": row.get("category", ""),
                    "label": label,
                }
            )
            continue
        if args.limit > 0 and processed >= args.limit:
            break

        category = row.get("category", "unknown") or "unknown"
        target_dir = output_root / "videos" / category
        target_dir.mkdir(parents=True, exist_ok=True)
        target_video = target_dir / source_video.name
        target_txt = target_video.with_suffix(".txt")

        frame_count, fps = video_metadata(source_video, args.fallback_fps)
        if frame_count <= 0:
            skipped_rows.append(
                {
                    "archive_path": source_video,
                    "reason": "no_frame_count",
                    "category": category,
                    "label": label,
                }
            )
            continue

        link_or_copy(source_video, target_video, args.link_mode)
        write_fixed_color_txt(
            target_txt,
            frame_count=frame_count,
            fps=fps,
            warmup_seconds=args.warmup_seconds,
            hold_seconds=args.hold_seconds,
            restore_seconds=args.restore_seconds,
            tail_seconds=args.tail_seconds,
        )

        note = row.get("note", "")
        fixed_note = (
            f"{note} | derived_fixed_collect_protocol: "
            f"warmup={args.warmup_seconds},hold={args.hold_seconds},restore={args.restore_seconds},"
            f"tail={args.tail_seconds},fps={fps:.6f}"
        ).strip()
        output_manifest_rows.append(
            {
                "media_type": "videos",
                "label": label,
                "category": category,
                "source_group": row.get("source_group", "unknown"),
                "archive_path": str(target_video),
                "source_path": row.get("source_path", str(source_video)),
                "note": fixed_note,
            }
        )
        key = (label, category)
        stats = summary_by_category.setdefault(key, {"count": 0, "frames": 0})
        stats["count"] += 1
        stats["frames"] += frame_count
        processed += 1

    write_tsv(
        output_root / "manifest.tsv",
        ["media_type", "label", "category", "source_group", "archive_path", "source_path", "note"],
        output_manifest_rows,
    )
    summary_rows = [
        {"label": label, "category": category, "count": stats["count"], "frames": stats["frames"]}
        for (label, category), stats in sorted(summary_by_category.items())
    ]
    write_tsv(output_root / "summary.tsv", ["label", "category", "count", "frames"], summary_rows)
    write_tsv(output_root / "skipped.tsv", ["archive_path", "reason", "category", "label"], skipped_rows)

    readme = [
        "# Flash Liveness Fixed Collect Protocol Dataset",
        "",
        "This is a derived dataset. It does not modify source videos or source random-color txt files.",
        "",
        f"- source root: `{input_root}`",
        f"- link mode: `{args.link_mode}`",
        f"- fixed color order RGB: `{COLOR_SEQUENCE_RGB}`",
        f"- packed colors: `{[rgb_to_packed_int(rgb) for rgb in COLOR_SEQUENCE_RGB]}`",
        f"- warmup_seconds: `{args.warmup_seconds}`",
        f"- hold_seconds: `{args.hold_seconds}`",
        f"- restore_seconds: `{args.restore_seconds}`",
        f"- tail_seconds: `{args.tail_seconds}`",
        f"- videos: `{len(output_manifest_rows)}`",
        f"- skipped: `{len(skipped_rows)}`",
        "",
        "Use with V3:",
        "",
        "```bash",
        "conda run -n anti-spoofing_scc_175 python flash_liveness_project_v3.py train \\",
        f"  --data-root {output_root} \\",
        "  --dataset-media videos \\",
        "  --require-color-txt",
        "```",
        "",
    ]
    (output_root / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"Built fixed protocol dataset: {output_root}")
    print(f"videos={len(output_manifest_rows)} skipped={len(skipped_rows)}")
    print(f"color_order_rgb={COLOR_SEQUENCE_RGB}")
    print(f"packed_colors={[rgb_to_packed_int(rgb) for rgb in COLOR_SEQUENCE_RGB]}")


if __name__ == "__main__":
    main()
