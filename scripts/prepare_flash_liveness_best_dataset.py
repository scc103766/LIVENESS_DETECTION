#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import shutil
from pathlib import Path

import cv2


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
NEUTRAL_COLOR_VALUE = 0

HARD_ATTACK_CATEGORIES = {
    "advanced_paper_attack",
    "cutout_attack",
    "latex_mask_attack",
    "mask_attack",
    "mask_fake_face_wig_hat_attack",
    "on_actor_paper_attack",
    "public_3d_attack",
    "silicone_mask_attack",
    "textile_3d_mask_attack",
    "three_d_head_model_attack",
    "three_d_head_model_attack_flash_archive",
    "three_d_paper_mask_attack",
    "wrapped_3d_paper_mask_attack",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a mixed flash-liveness training dataset from the typed asset archive. "
            "Real flash videos keep their frame-color txt files; non-flash public videos get neutral txt files."
        )
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/flash_liveness_asset_archive_by_type"),
        help="Typed asset archive root containing manifest.tsv.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/flash_liveness_best_protocol_v1"),
        help="Output dataset root with train/val/test/live|spoof directories.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--link-mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--hard-attack-repeat",
        type=int,
        default=8,
        help="Repeat hard spoof categories inside the train split to reduce silicone/mask under-sampling.",
    )
    parser.add_argument(
        "--neutral-for-missing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate color=0 txt for videos without real flash txt. Disable to keep only real flash-color samples.",
    )
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file, delimiter="\t"))


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def choose_split(token: str, train_ratio: float, val_ratio: float, seed: int) -> str:
    digest = hashlib.sha1(f"{seed}:{token}".encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def safe_stem(row: dict[str, str], repeat_idx: int = 0) -> str:
    source_path = row.get("source_path") or row.get("archive_path") or ""
    digest = hashlib.sha1(source_path.encode("utf-8")).hexdigest()[:12]
    original = Path(source_path).stem or Path(row.get("archive_path", "")).stem or "sample"
    original = original.replace(" ", "_").replace("/", "_")
    category = (row.get("category") or "unknown").replace(" ", "_").replace("/", "_")
    source_group = (row.get("source_group") or "unknown").replace(" ", "_").replace("/", "_")
    repeat_suffix = f"__r{repeat_idx:02d}" if repeat_idx else ""
    return f"{source_group}__{category}__{digest}{repeat_suffix}__{original}"


def ensure_output_root(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        for label in ("live", "spoof"):
            (path / split / label).mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def find_source_txt(video_path: Path) -> Path | None:
    candidates = [
        video_path.with_suffix(".txt"),
        video_path.parent / f"{video_path.stem}.txt",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def frame_count_for_video(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    return count


def write_neutral_txt(path: Path, frame_count: int) -> None:
    with path.open("w", encoding="utf-8") as file:
        for frame_idx in range(frame_count):
            file.write(f"{frame_idx},{NEUTRAL_COLOR_VALUE}\n")


def materialize_sample(
    *,
    row: dict[str, str],
    split: str,
    output_root: Path,
    link_mode: str,
    neutral_for_missing: bool,
    repeat_idx: int,
) -> tuple[dict[str, object] | None, dict[str, object] | None]:
    label = row.get("label", "")
    category = row.get("category", "")
    archive_path = Path(row.get("archive_path", ""))
    source_path = Path(row.get("source_path", "")) if row.get("source_path") else archive_path
    video_path = source_path if source_path.exists() else archive_path

    if label not in {"live", "spoof"}:
        return None, {"source_path": video_path, "reason": "unsupported_label", "label": label, "category": category}
    if not video_path.exists() or video_path.suffix.lower() not in VIDEO_EXTENSIONS:
        return None, {"source_path": video_path, "reason": "missing_or_unsupported_video", "label": label, "category": category}

    source_txt = find_source_txt(video_path)
    protocol = "source_flash_txt" if source_txt else "neutral_no_flash"
    if source_txt is None and not neutral_for_missing:
        return None, {"source_path": video_path, "reason": "missing_color_txt", "label": label, "category": category}

    target_stem = safe_stem(row, repeat_idx=repeat_idx)
    target_video = output_root / split / label / f"{target_stem}{video_path.suffix.lower()}"
    target_txt = target_video.with_suffix(".txt")

    link_or_copy(video_path.resolve(), target_video, link_mode)
    if source_txt is not None:
        link_or_copy(source_txt.resolve(), target_txt, link_mode)
    else:
        frame_count = frame_count_for_video(video_path)
        if frame_count <= 0:
            if target_video.exists() or target_video.is_symlink():
                target_video.unlink()
            return None, {"source_path": video_path, "reason": "no_frame_count_for_neutral_txt", "label": label, "category": category}
        write_neutral_txt(target_txt, frame_count)

    return (
        {
            "split": split,
            "label": label,
            "category": category,
            "source_group": row.get("source_group", ""),
            "protocol": protocol,
            "repeat_idx": repeat_idx,
            "media_path": str(target_video),
            "txt_path": str(target_txt),
            "source_path": str(video_path),
            "source_txt": str(source_txt or ""),
            "note": row.get("note", ""),
        },
        None,
    )


def should_repeat(row: dict[str, str], split: str, hard_attack_repeat: int) -> bool:
    return (
        split == "train"
        and row.get("label") == "spoof"
        and row.get("category") in HARD_ATTACK_CATEGORIES
        and hard_attack_repeat > 1
    )


def main() -> None:
    args = parse_args()
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
        raise ValueError("Require 0 < train_ratio and train_ratio + val_ratio < 1.")

    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    manifest_path = input_root / "manifest.tsv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.tsv not found: {manifest_path}")

    ensure_output_root(output_root, overwrite=args.overwrite)

    rows = [
        row
        for row in read_tsv(manifest_path)
        if row.get("media_type") == "videos" and row.get("label") in {"live", "spoof"}
    ]

    manifest_rows: list[dict[str, object]] = []
    skipped_rows: list[dict[str, object]] = []

    for row in rows:
        token = row.get("source_path") or row.get("archive_path") or repr(row)
        split = choose_split(token, args.train_ratio, args.val_ratio, args.seed)
        repeat_count = args.hard_attack_repeat if should_repeat(row, split, args.hard_attack_repeat) else 1
        for repeat_idx in range(repeat_count):
            record, skipped = materialize_sample(
                row=row,
                split=split,
                output_root=output_root,
                link_mode=args.link_mode,
                neutral_for_missing=args.neutral_for_missing,
                repeat_idx=repeat_idx,
            )
            if record is not None:
                manifest_rows.append(record)
            if skipped is not None:
                skipped["split"] = split
                skipped["repeat_idx"] = repeat_idx
                skipped_rows.append(skipped)

    summary: dict[tuple[str, str, str, str], int] = {}
    for row in manifest_rows:
        key = (
            str(row["split"]),
            str(row["label"]),
            str(row["category"]),
            str(row["protocol"]),
        )
        summary[key] = summary.get(key, 0) + 1

    summary_rows = [
        {"split": split, "label": label, "category": category, "protocol": protocol, "count": count}
        for (split, label, category, protocol), count in sorted(summary.items())
    ]

    write_tsv(
        output_root / "manifest.tsv",
        [
            "split",
            "label",
            "category",
            "source_group",
            "protocol",
            "repeat_idx",
            "media_path",
            "txt_path",
            "source_path",
            "source_txt",
            "note",
        ],
        manifest_rows,
    )
    write_tsv(output_root / "summary.tsv", ["split", "label", "category", "protocol", "count"], summary_rows)
    write_tsv(output_root / "skipped.tsv", ["split", "repeat_idx", "source_path", "reason", "label", "category"], skipped_rows)

    readme = [
        "# Flash Liveness Best Protocol V1",
        "",
        "This derived dataset is built for V3_1 mixed flash and hard-mask training.",
        "",
        f"- source archive: `{input_root}`",
        f"- link mode: `{args.link_mode}`",
        f"- train_ratio: `{args.train_ratio}`",
        f"- val_ratio: `{args.val_ratio}`",
        f"- hard_attack_repeat: `{args.hard_attack_repeat}`",
        f"- neutral_for_missing: `{args.neutral_for_missing}`",
        f"- materialized samples: `{len(manifest_rows)}`",
        f"- skipped samples: `{len(skipped_rows)}`",
        "",
        "Protocol policy:",
        "",
        "- Videos with a real adjacent `.txt` keep that frame-color sequence.",
        "- Videos without a real `.txt` receive a neutral `color=0` sequence, meaning no added flash stimulus.",
        "- Hard spoof categories are repeated only in the train split to reduce silicone/mask under-sampling.",
        "",
        "Recommended V3_1 training:",
        "",
        "```bash",
        "conda run -n anti-spoofing_scc_175 python flash_liveness_project_v3_1.py train \\",
        f"  --data-root {output_root} \\",
        "  --dataset-media videos \\",
        "  --require-color-txt \\",
        "  --missing-color-protocol neutral \\",
        "  --window-fusion quality_lower_trimmed_mean",
        "```",
        "",
    ]
    (output_root / "README.md").write_text("\n".join(readme), encoding="utf-8")

    print(f"Prepared best protocol dataset: {output_root}")
    print(f"materialized={len(manifest_rows)} skipped={len(skipped_rows)}")


if __name__ == "__main__":
    main()
