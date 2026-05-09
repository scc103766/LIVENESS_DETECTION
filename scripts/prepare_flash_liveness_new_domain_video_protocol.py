#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from collect_flash_liveness_video import (
    build_frame_color_labels,
)


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


@dataclass(frozen=True)
class SourceSpec:
    key: str
    label_name: str
    root: Path
    note: str


SOURCE_SPECS = [
    SourceSpec(
        key="live_public_real",
        label_name="live",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/Public samples/1. Real"),
        note="新增公开样本中的真人自拍视频，和训练目标最接近。",
    ),
    SourceSpec(
        key="live_outdoor",
        label_name="live",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/Outside environment/Outside environment"),
        note="新增室外真人自拍视频。",
    ),
    SourceSpec(
        key="live_extra_angles",
        label_name="live",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/Extra shooting angles"),
        note="新增真人不同拍摄角度自拍视频。",
    ),
    SourceSpec(
        key="live_ibeta3_id3_real",
        label_name="live",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/ibeta 3 dataset sample/id_3/Real"),
        note="新增 iBeta Level 3 样本中显式标注为 Real 的真人视频。",
    ),
    SourceSpec(
        key="live_ibeta3_id4_real",
        label_name="live",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/ibeta 3 dataset sample/id_4/Real"),
        note="新增 iBeta Level 3 样本中显式标注为 Real 的真人视频。",
    ),
    SourceSpec(
        key="spoof_public_print_cut",
        label_name="spoof",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/Public samples/2. Print + Cut"),
        note="打印裁剪攻击视频。",
    ),
    SourceSpec(
        key="spoof_public_cylinder",
        label_name="spoof",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/Public samples/3. Сylinder_attack"),
        note="柱状纸面攻击视频。",
    ),
    SourceSpec(
        key="spoof_public_on_actor",
        label_name="spoof",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/Public samples/4. On actor"),
        note="佩戴/覆盖式物理呈现攻击视频。",
    ),
    SourceSpec(
        key="spoof_public_3d",
        label_name="spoof",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/Public samples/5. 3D_attack"),
        note="3D 攻击视频。",
    ),
    SourceSpec(
        key="spoof_public_pc_replay",
        label_name="spoof",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/Public samples/6. PC Replay attack"),
        note="PC 屏幕 replay 攻击视频。",
    ),
    SourceSpec(
        key="spoof_public_mobile_replay",
        label_name="spoof",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/Public samples/7. Mobile Replay attack"),
        note="手机 replay 攻击视频。",
    ),
    SourceSpec(
        key="spoof_replay_display",
        label_name="spoof",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/Replay_display_attacks/Screen"),
        note="显示器 replay 攻击视频。",
    ),
    SourceSpec(
        key="spoof_replay_mobile",
        label_name="spoof",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/Replay_mobile_attacks/Galaxy_a54-Pixel7"),
        note="手机屏幕 replay 攻击视频。",
    ),
    SourceSpec(
        key="spoof_3d_paper_mask",
        label_name="spoof",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/3D_paper_mask "),
        note="3D 纸面头模攻击视频。",
    ),
    SourceSpec(
        key="spoof_cutout",
        label_name="spoof",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/Cutout_attacks"),
        note="Cutout 攻击视频。",
    ),
    SourceSpec(
        key="spoof_latex_mask",
        label_name="spoof",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/Latex_mask"),
        note="乳胶面具攻击视频。",
    ),
    SourceSpec(
        key="spoof_silicone_mask",
        label_name="spoof",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/Silicone_mask"),
        note="硅胶面具攻击视频。",
    ),
    SourceSpec(
        key="spoof_silicone_public",
        label_name="spoof",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/Silicone Mask - 21 Public samples"),
        note="公开硅胶面具攻击视频。",
    ),
    SourceSpec(
        key="spoof_textile_mask",
        label_name="spoof",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/Textile 3D Face Mask Attack Sample"),
        note="纺织 3D 面具攻击视频。",
    ),
    SourceSpec(
        key="spoof_wrapped_3d_paper",
        label_name="spoof",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/Wrapped_3D_paper_mask"),
        note="包覆式 3D 纸面攻击视频。",
    ),
    SourceSpec(
        key="spoof_ibeta3_id3_mask",
        label_name="spoof",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/ibeta 3 dataset sample/id_3/Mask"),
        note="新增 iBeta Level 3 样本中显式标注为 Mask 的攻击视频。",
    ),
    SourceSpec(
        key="spoof_ibeta3_id4_mask",
        label_name="spoof",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/ibeta 3 dataset sample/id_4/Mask"),
        note="新增 iBeta Level 3 样本中显式标注为 Mask 的攻击视频。",
    ),
    SourceSpec(
        key="spoof_paper_ibeta2_advanced",
        label_name="spoof",
        root=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/Advanced Paper attacks - iBeta 2"),
        note="新增高级纸质攻击样本，全部按 spoof 使用。",
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将新增 dataset 整理成可同时用于 V1/V2 的纯视频 protocol 数据集，并为 V2 合成同名 txt。"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/flash_liveness_new_domain_video_protocol_v1v2"),
        help="输出目录。",
    )
    parser.add_argument("--train-ratio", type=float, default=0.7, help="train 比例。")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="val 比例。")
    parser.add_argument("--link-mode", choices=("symlink", "copy"), default="symlink", help="视频使用软链接或复制。")
    parser.add_argument("--flash-group-size", type=int, default=5, help="每多少帧切换一次颜色。")
    parser.add_argument(
        "--flash-colors",
        default="0xFF14FF,0x14FF14,0xFF1414",
        help="离线闪光视频与 txt 生成时使用的采集颜色序列。",
    )
    parser.add_argument("--warmup-seconds", type=float, default=1.0, help="复用采集脚本的原色视频预热时长。")
    parser.add_argument("--hold-seconds", type=float, default=0.35, help="复用采集脚本的单色保持时长。")
    parser.add_argument("--restore-seconds", type=float, default=0.15, help="每次闪光后恢复原色视频的时长。")
    parser.add_argument("--tail-seconds", type=float, default=0.5, help="复用采集脚本的原色视频收尾时长。")
    parser.add_argument("--flash-alpha", type=float, default=0.22, help="离线对原视频叠加闪光颜色的强度。")
    parser.add_argument(
        "--video-mode",
        choices=("flashed", "original"),
        default="flashed",
        help="flashed: 输出叠加闪光后的新视频并据此写 txt；original: 保留原视频，仅按采集时间线生成 txt。",
    )
    return parser.parse_args()


def ensure_clean_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def iter_video_files(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def parse_flash_colors(raw: str) -> tuple[int, ...]:
    colors = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            colors.append(int(item, 0))
    if not colors:
        raise ValueError("至少需要一个 flash color。")
    return tuple(colors)


def packed_int_to_rgb(value: int) -> tuple[int, int, int]:
    return ((value >> 16) & 0xFF, (value >> 8) & 0xFF, value & 0xFF)


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
    path_token = hashlib.sha1(str(src).encode("utf-8")).hexdigest()[:10]
    suffix = ".mp4"
    stem = src.stem.replace(" ", "_")
    return f"{source_key}__{path_token}__{stem}{suffix}"


def link_or_copy(src: Path, dst: Path, mode: str) -> None:
    if mode == "symlink":
        os.symlink(src, dst)
        return
    shutil.copy2(src, dst)


def count_video_frames(video_path: Path) -> int:
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total > 0:
        cap.release()
        return total

    total = 0
    while True:
        ok, _ = cap.read()
        if not ok:
            break
        total += 1
    cap.release()
    return total


def build_txt_lines_from_labels(frame_labels: list[int]) -> list[str]:
    lines = []
    for frame_index, color in enumerate(frame_labels):
        lines.append(f"{frame_index},{color}")
    return lines


def open_video_writer(video_path: Path, width: int, height: int, fps: float) -> cv2.VideoWriter:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed_to_open_video_writer:{video_path}")
    return writer


def render_flashed_video(
    src: Path,
    dst: Path,
    frame_labels: list[int],
    flash_alpha: float,
) -> None:
    cap = cv2.VideoCapture(str(src))
    if not cap.isOpened():
        raise RuntimeError(f"failed_to_open_video:{src}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError(f"invalid_video_size:{src}")
    if fps <= 1e-6:
        fps = 30.0

    writer = open_video_writer(dst, width, height, fps)
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            packed = frame_labels[frame_index] if frame_index < len(frame_labels) else 0
            if packed > 0:
                rgb = packed_int_to_rgb(packed)
                bgr = np.array([rgb[2], rgb[1], rgb[0]], dtype=np.float32).reshape(1, 1, 3)
                base = frame.astype(np.float32)
                frame = np.clip(base * (1.0 - flash_alpha) + bgr * flash_alpha, 0.0, 255.0).astype(np.uint8)
            writer.write(frame)
            frame_index += 1
    finally:
        cap.release()
        writer.release()

    if frame_index != len(frame_labels):
        raise RuntimeError(f"render_frame_mismatch:{src}:labels={len(frame_labels)} written={frame_index}")


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.train_ratio + args.val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio 必须小于 1。")

    flash_colors = parse_flash_colors(args.flash_colors)
    color_sequence_rgb = [packed_int_to_rgb(value) for value in flash_colors]
    output_root = args.output_root.resolve()
    ensure_clean_dir(output_root)
    for split in ("train", "val", "test"):
        for label in ("live", "spoof"):
            (output_root / split / label).mkdir(parents=True, exist_ok=True)

    manifest_rows = [
        "split\tlabel_name\tlabel_id\tsource_key\tvideo_target\ttxt_target\tvideo_source\tframe_count\tnote"
    ]
    summary_rows = ["source_key\tlabel_name\tvideos_seen\tvideos_kept\ttotal_frames\tnote"]
    split_counts = {split: {"live": 0, "spoof": 0} for split in ("train", "val", "test")}
    split_frames = {split: {"live": 0, "spoof": 0} for split in ("train", "val", "test")}
    label_to_id = {"live": 1, "spoof": 0}

    for spec in SOURCE_SPECS:
        videos_seen = 0
        videos_kept = 0
        total_frames = 0
        if not spec.root.exists():
            summary_rows.append(f"{spec.key}\t{spec.label_name}\t0\t0\t0\tmissing_root")
            continue

        for src in iter_video_files(spec.root):
            videos_seen += 1
            frame_count = count_video_frames(src)
            if frame_count <= 0:
                continue

            split = choose_split(stable_split_key(src, spec.key), args.train_ratio, args.val_ratio)
            target_video_name = safe_name(spec.key, src)
            target_txt_name = Path(target_video_name).with_suffix(".txt").name
            video_dst = output_root / split / spec.label_name / target_video_name
            txt_dst = output_root / split / spec.label_name / target_txt_name
            fps = 30.0
            cap = cv2.VideoCapture(str(src))
            if cap.isOpened():
                measured_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
                if measured_fps > 1e-6:
                    fps = measured_fps
            cap.release()
            frame_labels = build_frame_color_labels(
                frame_count=frame_count,
                fps=fps,
                warmup_seconds=args.warmup_seconds,
                hold_seconds=args.hold_seconds,
                restore_seconds=args.restore_seconds,
                tail_seconds=args.tail_seconds,
                color_sequence=color_sequence_rgb,
            )
            if args.video_mode == "flashed":
                render_flashed_video(
                    src=src,
                    dst=video_dst,
                    frame_labels=frame_labels,
                    flash_alpha=args.flash_alpha,
                )
            else:
                link_or_copy(src, video_dst, args.link_mode)
            write_text(txt_dst, "\n".join(build_txt_lines_from_labels(frame_labels)) + "\n")

            videos_kept += 1
            total_frames += frame_count
            split_counts[split][spec.label_name] += 1
            split_frames[split][spec.label_name] += frame_count
            manifest_rows.append(
                "\t".join(
                    [
                        split,
                        spec.label_name,
                        str(label_to_id[spec.label_name]),
                        spec.key,
                        str(video_dst),
                        str(txt_dst),
                        str(src),
                        str(frame_count),
                        spec.note,
                    ]
                )
            )

        summary_rows.append(
            f"{spec.key}\t{spec.label_name}\t{videos_seen}\t{videos_kept}\t{total_frames}\t{spec.note}"
        )

    readme_lines = [
        "# Flash Liveness New-Domain Video Protocol (V1 + V2)",
        "",
        "这个目录把 `dataset/` 下新增的纯视频数据整理成一套正式对比 protocol，目标是：",
        "- 让 V1 模型可以直接按 `train/val/test/live|spoof` 测试。",
        "- 让 V2 模型在尽量接近训练输入格式的条件下测试，因此为每个视频自动生成了同名逐帧颜色 `txt`。",
        "",
        "注意：",
        "- 这些新增数据原始并不带闪光时序 txt，因此这里的 txt 是合成协议，不等同于真实闪光采集。",
        "- 该目录现在默认使用 `collect_flash_liveness_video.py` 的采集时间线来离线构造闪光视频与 txt，不再按固定 group-size 直接循环写 txt。",
        "- 如果使用 `video-mode=flashed`，输出视频会对原始视频逐帧叠加采集风格的闪光颜色；这依然是离线近似，不是真实拍摄时的人脸光学响应。",
        "",
        "目录结构：",
        "```",
        str(output_root),
        "├── train/live|spoof",
        "├── val/live|spoof",
        "├── test/live|spoof",
        "├── manifest.tsv",
        "├── source_summary.tsv",
        "└── README.md",
        "```",
        "",
        "V2 合成 txt 规则：",
        f"- 颜色序列: {[int(x) for x in flash_colors]}",
        f"- warmup_seconds={args.warmup_seconds}",
        f"- hold_seconds={args.hold_seconds}",
        f"- restore_seconds={args.restore_seconds}",
        f"- tail_seconds={args.tail_seconds}",
        f"- video_mode={args.video_mode}",
        f"- flash_alpha={args.flash_alpha}",
        "- 每一帧按 `collect_flash_liveness_video.py` 的时间线映射到对应闪光颜色",
        "- txt 每行格式: `frame_index,color_int`",
        "",
        "来源与用途：",
    ]
    for spec in SOURCE_SPECS:
        readme_lines.append(f"- `{spec.key}` -> `{spec.label_name}`: `{spec.root}` ({spec.note})")

    readme_lines.extend(
        [
            "",
            "切分统计：",
        ]
    )
    for split in ("train", "val", "test"):
        readme_lines.append(
            f"- `{split}`: live={split_counts[split]['live']} ({split_frames[split]['live']} frames), "
            f"spoof={split_counts[split]['spoof']} ({split_frames[split]['spoof']} frames)"
        )

    write_text(output_root / "manifest.tsv", "\n".join(manifest_rows) + "\n")
    write_text(output_root / "source_summary.tsv", "\n".join(summary_rows) + "\n")
    write_text(output_root / "README.md", "\n".join(readme_lines) + "\n")

    print(f"Prepared protocol dataset at: {output_root}")
    for split in ("train", "val", "test"):
        print(
            f"{split}: "
            f"live={split_counts[split]['live']} ({split_frames[split]['live']} frames), "
            f"spoof={split_counts[split]['spoof']} ({split_frames[split]['spoof']} frames)"
        )


if __name__ == "__main__":
    main()
