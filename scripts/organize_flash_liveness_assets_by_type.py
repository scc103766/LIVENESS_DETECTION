#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class Asset:
    source_path: Path
    media_type: str
    category: str
    label: str
    source_group: str
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按活体/攻击类型归档炫彩活体视频和图片。")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset/flash_liveness_asset_archive_by_type"),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/dataset"),
    )
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档/dataset/dataset"),
    )
    parser.add_argument(
        "--history-root",
        type=Path,
        default=Path("/supercloud/llm-code/scc/scc/Liveness_Detection/炫彩闪烁活体"),
    )
    return parser.parse_args()


def safe_name(asset: Asset) -> str:
    digest = hashlib.sha1(str(asset.source_path).encode("utf-8")).hexdigest()[:12]
    stem = asset.source_path.stem.replace(" ", "_").replace("/", "_")
    return f"{asset.source_group}__{digest}__{stem}{asset.source_path.suffix.lower()}"


def rel_parts(path: Path, root: Path) -> list[str]:
    return [part.lower() for part in path.relative_to(root).parts]


def classify_dataset_asset(path: Path, root: Path) -> tuple[str, str, str, str] | None:
    parts = rel_parts(path, root)
    joined = "/".join(parts)

    generated_prefixes = (
        "flash_liveness_asset_archive_by_type",
        "flash_liveness_video_dataset_v1",
        "flash_liveness_video_dataset_v2",
        "flash_liveness_new_domain_video_protocol_v1v2",
        "tg_export_from_current_dataset",
        "tg_export_from_current_dataset_legacy_fastalign",
        "tg_export_from_current_dataset_smoke",
        "tg_export_from_current_dataset_legacy_fastalign_smoke",
        "faiss_all_combinations_results",
    )
    if parts and parts[0] in generated_prefixes:
        return None

    if path.name.lower() == "axon labs silicone mask sample.jpg":
        return "silicone_mask_attack", "spoof", "dataset_public", "硅胶面具攻击样例图片。"
    if "advanced paper attacks - ibeta 2" in joined:
        return "advanced_paper_attack", "spoof", "dataset_new", "高级纸质攻击样本。"
    if "ibeta 3 dataset sample" in joined:
        if "/real/" in f"/{joined}/":
            return "live_real", "live", "dataset_new", "iBeta Level 3 Real。"
        if "/mask/" in f"/{joined}/":
            return "mask_attack", "spoof", "dataset_new", "iBeta Level 3 Mask。"
        return "mask_fake_face_wig_hat_attack", "spoof", "dataset_new", "iBeta Level 3 戴面具、帽子和假头发的 fake 人脸视频。"
    if "public samples/1. real" in joined:
        return "live_real", "live", "dataset_public", "公开真人样本。"
    if "outside environment" in joined:
        return "live_real_outdoor", "live", "dataset_public", "室外真人样本。"
    if "extra shooting angles" in joined:
        return "live_real_extra_angles", "live", "dataset_public", "真人额外拍摄角度。"
    if "selfies" in joined:
        return "live_real_selfie_image", "live", "dataset_public", "真人自拍图片。"
    if "replay_display_attacks/real" in joined:
        return "live_real_replay_display_control", "live", "dataset_public", "屏幕重放数据中的真人对照。"
    if "replay_display_attacks/screen" in joined or "pc replay attack" in joined:
        return "replay_display_attack", "spoof", "dataset_public", "显示器/PC 屏幕重放攻击。"
    if "replay_mobile_attacks" in joined or "mobile replay attack" in joined:
        return "replay_mobile_attack", "spoof", "dataset_public", "手机屏幕重放攻击。"
    if "print + cut" in joined:
        return "print_cut_paper_attack", "spoof", "dataset_public", "打印裁剪纸质攻击。"
    if "сylinder_attack" in joined or "cylinder_attack" in joined:
        return "cylinder_paper_attack", "spoof", "dataset_public", "柱状纸质攻击。"
    if "on actor" in joined:
        return "on_actor_paper_attack", "spoof", "dataset_public", "贴附/佩戴在真人上的纸质攻击。"
    if "5. 3d_attack" in joined:
        return "public_3d_attack", "spoof", "dataset_public", "公开 3D 攻击。"
    if "3d_paper_mask" in joined:
        return "three_d_paper_mask_attack", "spoof", "dataset_public", "3D 纸面头模攻击。"
    if "wrapped_3d_paper_mask" in joined:
        return "wrapped_3d_paper_mask_attack", "spoof", "dataset_public", "包覆式 3D 纸面攻击。"
    if "cutout_attacks" in joined:
        return "cutout_attack", "spoof", "dataset_public", "Cutout 攻击。"
    if "latex_mask" in joined:
        return "latex_mask_attack", "spoof", "dataset_public", "乳胶面具攻击。"
    if "silicone mask - 21 public samples" in joined or "silicone_mask" in joined:
        return "silicone_mask_attack", "spoof", "dataset_public", "硅胶面具攻击。"
    if "textile 3d face mask attack sample" in joined:
        return "textile_3d_mask_attack", "spoof", "dataset_public", "纺织 3D 面具攻击。"
    return "unknown_review", "unknown", "dataset_public", "未匹配到明确类别，需人工确认。"


def classify_archive_asset(path: Path, root: Path) -> tuple[str, str, str, str] | None:
    parts = rel_parts(path, root)
    joined = "/".join(parts)
    if "h5_raw/vreal" in joined or "raw/real" in joined or "raw/new_real" in joined:
        return "live_real_flash_archive", "live", "archive_20240320", "归档真人炫彩活体。"
    if "h5_raw/preal" in joined:
        return "live_real_image_flash_archive", "live", "archive_20240320", "归档真人图片/预处理真人样本。"
    if "h5_raw/pprint" in joined:
        return "print_attack_flash_archive", "spoof", "archive_20240320", "归档打印攻击。"
    if "h5_raw/pscreen" in joined:
        return "screen_replay_attack_flash_archive", "spoof", "archive_20240320", "归档屏幕重放攻击。"
    if "raw/3dfake" in joined:
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return "unknown_review", "unknown", "archive_20240320", "圆形空白图，需人工确认用途。"
        return "three_d_head_model_attack_flash_archive", "spoof", "archive_20240320", "归档 3D 头模攻击。"
    if "raw/fake" in joined:
        if path.suffix.lower() in IMAGE_EXTENSIONS:
            return "unknown_review", "unknown", "archive_20240320", "圆形空白图，需人工确认用途。"
        return "flat_attack_flash_archive", "spoof", "archive_20240320", "归档平面/屏幕/打印等攻击。"
    if "raw/pre_train_choose" in joined:
        return "pretrain_chosen_photo_print_attack_archive", "spoof", "archive_20240320", "归档预训练筛选打印照片攻击图片。"
    if "raw/remove" in joined:
        return "live_real_lighting_pair_control_archive", "live", "archive_20240320", "归档真人对照试验视频，包含曝光明亮与正常光照成对样本。"
    return "unknown_review", "unknown", "archive_20240320", "归档中未匹配到明确类别。"


def classify_history_asset(path: Path, root: Path) -> tuple[str, str, str, str] | None:
    parts = rel_parts(path, root)
    joined = "/".join(parts)
    if "20230607_liveness_detection" in joined:
        if path.name == "1686033301460_zz_1_2_xx_3.avi":
            return "live_real_flash_archive", "live", "history_flash_liveness", "历史炫彩活体补充真人采集。"
        return "history_flash_collect_attack", "spoof", "history_flash_liveness", "历史炫彩活体补充攻击采集。"
    if "20230828 头模 正常光" in joined or "toumo" in joined:
        return "history_head_model_attack", "spoof", "history_flash_liveness", "历史头模正常光攻击采集。"
    return "unknown_review", "unknown", "history_flash_liveness", "历史炫彩活体目录中未匹配到明确类别。"


def iter_media_files(root: Path):
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in VIDEO_EXTENSIONS:
            yield path, "videos"
        elif suffix in IMAGE_EXTENSIONS:
            yield path, "images"


def video_fingerprint(path: Path) -> tuple[int, str]:
    stat = path.stat()
    digest = hashlib.sha1()
    digest.update(str(stat.st_size).encode("ascii"))
    with path.open("rb") as file:
        digest.update(file.read(1024 * 1024))
        if stat.st_size > 1024 * 1024:
            file.seek(max(stat.st_size - 1024 * 1024, 0))
            digest.update(file.read(1024 * 1024))
    return stat.st_size, digest.hexdigest()


def dedupe_video_assets(assets: list[Asset]) -> list[Asset]:
    priority = {
        "archive_20240320": 0,
        "dataset_public": 1,
        "dataset_new": 1,
        "history_flash_liveness": 2,
    }
    chosen: dict[tuple[int, str], Asset] = {}
    passthrough: list[Asset] = []
    for asset in assets:
        if asset.media_type != "videos":
            passthrough.append(asset)
            continue
        key = video_fingerprint(asset.source_path)
        existing = chosen.get(key)
        if existing is None:
            chosen[key] = asset
            continue
        old_priority = priority.get(existing.source_group, 99)
        new_priority = priority.get(asset.source_group, 99)
        if new_priority < old_priority:
            chosen[key] = asset
    return passthrough + list(chosen.values())


def collect_assets(dataset_root: Path, archive_root: Path, history_root: Path) -> list[Asset]:
    assets: list[Asset] = []
    for path, media_type in iter_media_files(dataset_root):
        classification = classify_dataset_asset(path, dataset_root)
        if classification is None:
            continue
        category, label, source_group, note = classification
        assets.append(Asset(path.resolve(), media_type, category, label, source_group, note))

    for path, media_type in iter_media_files(archive_root):
        category, label, source_group, note = classify_archive_asset(path, archive_root)
        assets.append(Asset(path.resolve(), media_type, category, label, source_group, note))

    if history_root.exists():
        for path, media_type in iter_media_files(history_root):
            category, label, source_group, note = classify_history_asset(path, history_root)
            assets.append(Asset(path.resolve(), media_type, category, label, source_group, note))

    return dedupe_video_assets(assets)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    assets = collect_assets(args.dataset_root.resolve(), args.archive_root.resolve(), args.history_root.resolve())
    manifest_rows = [
        "media_type\tlabel\tcategory\tsource_group\tarchive_path\tsource_path\tnote"
    ]
    summary: dict[tuple[str, str, str], int] = {}
    unknown_video_source_dirs: dict[str, int] = {}

    for asset in sorted(assets, key=lambda item: (item.media_type, item.category, str(item.source_path))):
        target_dir = output_root / asset.media_type / asset.category
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / safe_name(asset)
        os.symlink(asset.source_path, target_path)
        manifest_rows.append(
            "\t".join(
                [
                    asset.media_type,
                    asset.label,
                    asset.category,
                    asset.source_group,
                    str(target_path),
                    str(asset.source_path),
                    asset.note,
                ]
            )
        )
        key = (asset.media_type, asset.label, asset.category)
        summary[key] = summary.get(key, 0) + 1
        if asset.media_type == "videos" and asset.category == "unknown_review":
            parent = str(asset.source_path.parent)
            unknown_video_source_dirs[parent] = unknown_video_source_dirs.get(parent, 0) + 1

    summary_rows = ["media_type\tlabel\tcategory\tcount"]
    for (media_type, label, category), count in sorted(summary.items()):
        summary_rows.append(f"{media_type}\t{label}\t{category}\t{count}")

    readme_lines = [
        "# Flash Liveness Assets Archived By Type",
        "",
        "该目录按炫彩活体数据的语义类型归档视频和图片。",
        "",
        "说明:",
        "- 源文件保持原位，本目录中的文件均为软链接。",
        "- `videos/` 保存视频资产，`images/` 保存图片资产。",
        "- `manifest.tsv` 记录每个软链接对应的原始路径。",
        "- `summary.tsv` 记录按媒体类型、标签、类别聚合的数量。",
        "",
    ]
    (output_root / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    (output_root / "manifest.tsv").write_text("\n".join(manifest_rows) + "\n", encoding="utf-8")
    (output_root / "summary.tsv").write_text("\n".join(summary_rows) + "\n", encoding="utf-8")
    unknown_rows = ["count\tsource_dir"]
    for source_dir, count in sorted(unknown_video_source_dirs.items(), key=lambda item: (-item[1], item[0])):
        unknown_rows.append(f"{count}\t{source_dir}")
    (output_root / "unknown_video_source_dirs.tsv").write_text("\n".join(unknown_rows) + "\n", encoding="utf-8")

    print(f"Prepared archive: {output_root}")
    print(f"Total assets: {len(assets)}")
    print(f"Summary: {output_root / 'summary.tsv'}")
    print(f"Manifest: {output_root / 'manifest.tsv'}")


if __name__ == "__main__":
    main()
