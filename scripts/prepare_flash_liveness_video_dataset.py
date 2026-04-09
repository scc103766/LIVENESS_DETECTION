#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


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
        root=Path(
            "/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/dataset/dataset/raw/real"
        ),
        note="归档中的真人原始视频",
    ),
    SourceSpec(
        key="archive_h5_vreal",
        label_name="live",
        root=Path(
            "/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/dataset/dataset/h5_raw/vreal"
        ),
        note="归档中的 h5 真人视频",
    ),
    SourceSpec(
        key="archive_raw_fake",
        label_name="spoof",
        root=Path(
            "/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/dataset/dataset/raw/fake"
        ),
        note="归档中的平面/屏幕/打印等攻击视频",
    ),
    SourceSpec(
        key="archive_raw_3dfake",
        label_name="spoof",
        root=Path(
            "/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/dataset/dataset/raw/3dfake"
        ),
        note="归档中的 3D 头模攻击视频",
    ),
    SourceSpec(
        key="history_flash_collect",
        label_name="spoof",
        root=Path(
            "/supercloud/llm-code/scc/scc/Liveness_Detection/炫彩闪烁活体/20230607_liveness_Detection"
        ),
        note="历史补充采集目录，文件名以 toutao/toumo 为主，视作攻击视频",
    ),
    SourceSpec(
        key="history_head_model",
        label_name="spoof",
        root=Path(
            "/supercloud/llm-code/scc/scc/Liveness_Detection/炫彩闪烁活体/20230828 头模 正常光"
        ),
        note="历史补充 3D 头模目录，视作攻击视频",
    ),
]


EXCLUDED_SOURCES = [
    (
        "/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/dataset/dataset/h5_raw/preal",
        "仅包含图片帧，不是视频文件，不能直接用于 flash_liveness_project.py",
    ),
    (
        "/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/dataset/dataset/h5_raw/pscreen",
        "仅包含图片帧，不是视频文件，不能直接用于 flash_liveness_project.py",
    ),
    (
        "/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/dataset/dataset/h5_raw/pprint",
        "仅包含图片帧，不是视频文件，不能直接用于 flash_liveness_project.py",
    ),
    (
        "/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/dataset/dataset/raw/new_real",
        "当前为空目录",
    ),
    (
        "/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/dataset/dataset/raw/pre_train_choose",
        "当前是中间筛选目录，目录层级复杂且缺少稳定标签映射，先不自动纳入",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="整理项目内可用视频，生成 flash_liveness_project.py 可直接训练的数据集目录。"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/flash_liveness_video_dataset_v1"),
        help="输出数据集目录，默认会创建 train/val/test 结构。",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.9,
        help="训练集比例，默认 0.9。",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.05,
        help="验证集比例，默认 0.05。",
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "copy"),
        default="symlink",
        help="默认使用软链接节省空间；如需实体文件可切换为 copy。",
    )
    return parser.parse_args()


def iter_video_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def stable_split_key(path: Path, source_key: str) -> str:
    return f"{source_key}:{path.name}:{path.stat().st_size}"


def choose_split(token: str, train_ratio: float, val_ratio: float) -> str:
    digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def safe_name(source_key: str, src: Path) -> str:
    sanitized = src.name.replace(" ", "_")
    path_token = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:10]
    return f"{source_key}__{path_token}__{sanitized}"


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if mode == "symlink":
        os.symlink(src, dst)
        return
    shutil.copy2(src, dst)


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if not (0 < args.train_ratio < 1):
        raise ValueError("--train-ratio 必须在 0 和 1 之间。")
    if not (0 <= args.val_ratio < 1):
        raise ValueError("--val-ratio 必须在 0 和 1 之间。")
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio 必须小于 1，剩余部分会分给 test。")

    output_root = args.output_root.resolve()
    ensure_clean_dir(output_root)
    for split in ("train", "val", "test"):
        for label in ("live", "spoof"):
            (output_root / split / label).mkdir(parents=True, exist_ok=True)

    manifest_rows = [
        "split\tlabel_name\tlabel_id\tsource_key\ttarget_path\tsource_path\tnote"
    ]
    source_summary_rows = ["source_key\tlabel_name\tfiles_seen\tfiles_kept\tnote"]
    duplicates_rows = ["dedupe_key\tkept_source\tkept_path\tskipped_source\tskipped_path"]

    dedupe_seen: dict[tuple[str, int], tuple[SourceSpec, Path]] = {}
    split_counts = {
        split: {"live": 0, "spoof": 0}
        for split in ("train", "val", "test")
    }
    label_to_id = {"live": 1, "spoof": 0}

    for spec in SOURCE_SPECS:
        if not spec.root.exists():
            source_summary_rows.append(f"{spec.key}\t{spec.label_name}\t0\t0\t缺失目录: {spec.note}")
            continue

        files_seen = 0
        files_kept = 0
        for src in iter_video_files(spec.root):
            files_seen += 1
            dedupe_key = (src.name, src.stat().st_size)
            kept = dedupe_seen.get(dedupe_key)
            if kept is not None:
                kept_spec, kept_path = kept
                duplicates_rows.append(
                    "\t".join(
                        [
                            f"{src.name}:{src.stat().st_size}",
                            kept_spec.key,
                            str(kept_path),
                            spec.key,
                            str(src),
                        ]
                    )
                )
                continue

            dedupe_seen[dedupe_key] = (spec, src)
            files_kept += 1
            split = choose_split(
                stable_split_key(src, spec.key),
                train_ratio=args.train_ratio,
                val_ratio=args.val_ratio,
            )
            target_name = safe_name(spec.key, src)
            dst = output_root / split / spec.label_name / target_name
            link_or_copy(src, dst, args.link_mode)
            split_counts[split][spec.label_name] += 1
            manifest_rows.append(
                "\t".join(
                    [
                        split,
                        spec.label_name,
                        str(label_to_id[spec.label_name]),
                        spec.key,
                        str(dst),
                        str(src),
                        spec.note,
                    ]
                )
            )

        source_summary_rows.append(
            f"{spec.key}\t{spec.label_name}\t{files_seen}\t{files_kept}\t{spec.note}"
        )

    readme = f"""# Flash Liveness Video Dataset v1

这个目录是为 [`flash_liveness_project.py`](/supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_project.py) 整理好的视频分类数据集。

## 目录结构

```
{output_root}
├── train/
│   ├── live/
│   └── spoof/
├── val/
│   ├── live/
│   └── spoof/
├── test/
│   ├── live/
│   └── spoof/
├── manifest.tsv
├── source_summary.tsv
├── duplicates_skipped.tsv
└── preparation_report.txt
```

## 标签约定

- `live` -> 1
- `spoof` -> 0

## 数据来源

{os.linesep.join([f"- `{spec.key}`: {spec.root} ({spec.note})" for spec in SOURCE_SPECS])}

## 明确排除的目录

{os.linesep.join([f"- `{path}`: {reason}" for path, reason in EXCLUDED_SOURCES])}

## 切分规则

- 使用 `sha1(source_key + filename + file_size)` 做稳定切分
- `train_ratio={args.train_ratio}`
- `val_ratio={args.val_ratio}`
- 剩余样本分到 `test`

## 调用方式

```bash
python /supercloud/llm-code/scc/scc/Liveness_Detection/scripts/prepare_flash_liveness_video_dataset.py
```

训练时可直接把 `data_root` 指向这个目录，例如：

```bash
python /supercloud/llm-code/scc/scc/Liveness_Detection/flash_liveness_project.py \\
  --data-root {output_root}
```
"""

    report_lines = [
        "Flash liveness 数据集整理报告",
        "",
        f"output_root: {output_root}",
        f"link_mode: {args.link_mode}",
        f"train_ratio: {args.train_ratio}",
        f"val_ratio: {args.val_ratio}",
        f"test_ratio: {1 - args.train_ratio - args.val_ratio}",
        "",
        "split 统计:",
    ]
    for split in ("train", "val", "test"):
        report_lines.append(
            f"- {split}: live={split_counts[split]['live']}, spoof={split_counts[split]['spoof']}"
        )
    report_lines.extend(
        [
            "",
            "说明:",
            "- 当前版本只纳入可直接供 flash_liveness_project.py 读取的视频文件。",
            "- 同名且文件大小相同的文件按重复样本处理，只保留首次出现的版本。",
            "- h5_raw 下的 preal/pscreen/pprint 目前是图片帧目录，因此没有自动纳入视频数据集。",
            "- 如需重新生成，可再次执行本脚本；它会重建整个输出目录。",
        ]
    )

    write_text(output_root / "README.md", readme)
    write_text(output_root / "manifest.tsv", "\n".join(manifest_rows) + "\n")
    write_text(output_root / "source_summary.tsv", "\n".join(source_summary_rows) + "\n")
    write_text(output_root / "duplicates_skipped.tsv", "\n".join(duplicates_rows) + "\n")
    write_text(output_root / "preparation_report.txt", "\n".join(report_lines) + "\n")

    print(f"Dataset prepared at: {output_root}")
    for split in ("train", "val", "test"):
        print(
            f"{split}: live={split_counts[split]['live']}, spoof={split_counts[split]['spoof']}"
        )


if __name__ == "__main__":
    main()
