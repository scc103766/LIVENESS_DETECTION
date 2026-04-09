#!/usr/bin/env python3
import argparse
import os
from pathlib import Path


ROOT = Path("/supercloud/llm-code/scc/scc/Liveness_Detection/20240320闪光活体归档")
SOURCE_ROOT = ROOT / "dataset" / "tg_export"
TARGET_ROOT = ROOT / "dataset" / "tg_export_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a small symlinked ThunderGuard smoke subset.")
    parser.add_argument("--target-root", default=str(TARGET_ROOT))
    parser.add_argument("--train-real", type=int, default=12)
    parser.add_argument("--train-print", type=int, default=6)
    parser.add_argument("--train-screen", type=int, default=6)
    parser.add_argument("--train-model", type=int, default=6)
    parser.add_argument("--test-real", type=int, default=6)
    parser.add_argument("--test-print", type=int, default=3)
    parser.add_argument("--test-screen", type=int, default=3)
    parser.add_argument("--test-model", type=int, default=3)
    return parser.parse_args()


def category_from_flag(flag: int) -> str | None:
    if flag == 1:
        return "real"
    if flag in {2, 3, 7}:
        return "print"
    if flag in {4, 5}:
        return "screen"
    if flag == 6:
        return "model"
    return None


def gather_basenames(split_dir: Path) -> dict[str, list[str]]:
    grouped = {"real": [], "print": [], "screen": [], "model": []}
    seen = set()
    for txt_path in sorted(split_dir.glob("*.txt")):
        stem = txt_path.stem
        parts = stem.split("_")
        if not parts:
            continue
        if stem.startswith("0_"):
            continue
        try:
            flag = int(parts[-1])
        except ValueError:
            continue
        category = category_from_flag(flag)
        if category is None:
            continue
        if stem in seen:
            continue
        jpg_path = split_dir / f"{stem}.jpg"
        depth_path = split_dir / f"{stem}_d.jpg"
        if not jpg_path.exists() or not depth_path.exists():
            continue
        grouped[category].append(stem)
        seen.add(stem)
    return grouped


def ensure_clean_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            raise RuntimeError(f"unexpected directory in smoke target: {child}")


def link_triplet(source_split: Path, target_split: Path, stem: str) -> None:
    for suffix in (".txt", ".jpg", "_d.jpg"):
        source = source_split / f"{stem}{suffix}"
        target = target_split / source.name
        os.symlink(source, target)


def build_split(
    split_name: str,
    source_split: Path,
    target_split: Path,
    limits: dict[str, int],
) -> dict[str, int]:
    ensure_clean_dir(target_split)
    grouped = gather_basenames(source_split)
    counts = {}
    for category, limit in limits.items():
        chosen = grouped[category][:limit]
        if len(chosen) < limit:
            raise RuntimeError(
                f"not enough samples for {split_name}/{category}: "
                f"need {limit}, found {len(chosen)}"
            )
        for stem in chosen:
            link_triplet(source_split, target_split, stem)
        counts[category] = len(chosen)
    return counts


def main() -> int:
    args = parse_args()
    target_root = Path(args.target_root)
    train_limits = {
        "real": args.train_real,
        "print": args.train_print,
        "screen": args.train_screen,
        "model": args.train_model,
    }
    test_limits = {
        "real": args.test_real,
        "print": args.test_print,
        "screen": args.test_screen,
        "model": args.test_model,
    }

    target_root.mkdir(parents=True, exist_ok=True)
    train_counts = build_split("train", SOURCE_ROOT / "train", target_root / "train", train_limits)
    test_counts = build_split("test", SOURCE_ROOT / "test", target_root / "test", test_limits)

    print("built smoke subset at", target_root)
    print("train counts", train_counts)
    print("test counts", test_counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
