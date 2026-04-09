#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


@dataclass(frozen=True)
class SourceSpec:
    key: str
    label_name: str
    root: Path
    note: str


SOURCE_SPECS = [
    SourceSpec(
        key="archive_raw_real",
        label_name="live",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/dataset/dataset/raw/real"),
        note="归档中的真人原始视频，带逐帧颜色 txt",
    ),
    SourceSpec(
        key="archive_h5_vreal",
        label_name="live",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/dataset/dataset/h5_raw/vreal"),
        note="归档中的 h5 真人视频，带逐帧颜色 txt",
    ),
    SourceSpec(
        key="archive_raw_fake",
        label_name="spoof",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/dataset/dataset/raw/fake"),
        note="归档中的平面/屏幕/打印等攻击视频，带逐帧颜色 txt",
    ),
    SourceSpec(
        key="archive_raw_3dfake",
        label_name="spoof",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/dataset/dataset/raw/3dfake"),
        note="归档中的 3D 头模攻击视频，带逐帧颜色 txt",
    ),
    SourceSpec(
        key="history_flash_collect",
        label_name="spoof",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/炫彩闪烁活体/20230607_liveness_Detection"),
        note="历史补充攻击视频，目录内同名 txt 为逐帧颜色标注",
    ),
    SourceSpec(
        key="history_head_model",
        label_name="spoof",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/炫彩闪烁活体/20230828 头模 正常光"),
        note="历史补充头模攻击视频，目录内同名 txt 为逐帧颜色标注",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 flash liveness V2 数据集：视频与同名逐帧颜色 txt 成对整理。")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/flash_liveness_video_dataset_v2"),
    )
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink")
    return parser.parse_args()


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if mode == "symlink":
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def safe_name(source_key: str, src: Path) -> str:
    path_token = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:10]
    sanitized = src.name.replace(" ", "_")
    return f"{source_key}__{path_token}__{sanitized}"


def choose_split(token: str, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def iter_video_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def main() -> None:
    args = parse_args()
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio 必须小于 1")

    output_root = args.output_root.resolve()
    ensure_clean_dir(output_root)
    for split in ("train", "val", "test"):
        for label in ("live", "spoof"):
            (output_root / split / label).mkdir(parents=True, exist_ok=True)

    manifest_rows = ["split\tlabel_name\tsource_key\tvideo_target\ttxt_target\tvideo_source\ttxt_source\tnote"]
    summary_rows = ["source_key\tlabel_name\tvideos_seen\tpaired_kept\tmissing_txt\tnote"]
    missing_txt_rows = ["source_key\tvideo_source\texpected_txt"]
    readme_lines = [
        "# Flash Liveness Video Dataset V2",
        "",
        "该数据集用于 `flash_liveness_project_v2.py`，每个视频都会配套一个同名 `.txt` 颜色时序文件。",
        "",
        "目录结构:",
        "```",
        str(output_root),
        "├── train/live|spoof",
        "├── val/live|spoof",
        "├── test/live|spoof",
        "├── manifest.tsv",
        "├── source_summary.tsv",
        "└── missing_txt.tsv",
        "```",
        "",
        "颜色 txt 格式:",
        "- 每行一个 `frame_index,color_id`",
        "- 例如 `0,16777215`",
        "",
        "标签约定:",
        "- `live` -> 1",
        "- `spoof` -> 0",
        "",
    ]

    split_counts = {split: {"live": 0, "spoof": 0} for split in ("train", "val", "test")}

    for spec in SOURCE_SPECS:
        videos_seen = 0
        paired_kept = 0
        missing_txt = 0
        if not spec.root.exists():
            summary_rows.append(f"{spec.key}\t{spec.label_name}\t0\t0\t0\tmissing_root")
            continue

        for video_path in iter_video_files(spec.root):
            videos_seen += 1
            txt_path = video_path.with_suffix(".txt")
            if not txt_path.exists():
                missing_txt += 1
                missing_txt_rows.append(f"{spec.key}\t{video_path}\t{txt_path}")
                continue

            split = choose_split(f"{spec.key}:{video_path.name}:{video_path.stat().st_size}", args.train_ratio, args.val_ratio)
            target_video_name = safe_name(spec.key, video_path)
            target_txt_name = Path(target_video_name).with_suffix(".txt").name
            video_target = output_root / split / spec.label_name / target_video_name
            txt_target = output_root / split / spec.label_name / target_txt_name
            link_or_copy(video_path, video_target, args.link_mode)
            link_or_copy(txt_path, txt_target, args.link_mode)
            paired_kept += 1
            split_counts[split][spec.label_name] += 1
            manifest_rows.append(
                f"{split}\t{spec.label_name}\t{spec.key}\t{video_target}\t{txt_target}\t{video_path}\t{txt_path}\t{spec.note}"
            )

        summary_rows.append(f"{spec.key}\t{spec.label_name}\t{videos_seen}\t{paired_kept}\t{missing_txt}\t{spec.note}")

    readme_lines.extend(
        [
            "数据来源:",
            *[f"- `{spec.key}`: `{spec.root}` ({spec.note})" for spec in SOURCE_SPECS],
            "",
            "切分统计:",
            *[
                f"- `{split}`: live={split_counts[split]['live']}, spoof={split_counts[split]['spoof']}"
                for split in ("train", "val", "test")
            ],
            "",
            "说明:",
            "- 只有同时具备视频文件和同名 txt 的样本会纳入 V2 数据集。",
            "- 缺少 txt 的视频会记录在 `missing_txt.tsv`。",
            "",
        ]
    )

    (output_root / "manifest.tsv").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    (output_root / "source_summary.tsv").write_text("\n".join(summary_rows) + "\n", encoding="utf-8")
    (output_root / "missing_txt.tsv").write_text("\n".join(missing_txt_rows) + "\n", encoding="utf-8")
    (output_root / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(f"Prepared V2 dataset at: {output_root}")
    for split in ("train", "val", "test"):
        print(f"{split}: live={split_counts[split]['live']}, spoof={split_counts[split]['spoof']}")


if __name__ == "__main__":
    main()
