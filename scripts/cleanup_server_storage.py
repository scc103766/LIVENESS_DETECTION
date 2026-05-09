from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path("/supercloud/llm-code/scc/scc/Liveness_Detection")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from server_storage_manager import (  # noqa: E402
    StorageRetentionConfig,
    StorageRetentionManager,
    iter_request_dirs,
    iter_video_files,
)


SERVICE_DEFAULTS = {
    "tg": Path("/raid/scc/data/TG_server_result"),
    "v3": Path("/raid/scc/data/liveness_v3_server_result"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backup conclusions and representative frames, then delete old server videos."
    )
    parser.add_argument("--service", choices=sorted(SERVICE_DEFAULTS), default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--backup-dir", type=Path, default=None)
    parser.add_argument("--max-videos", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=200)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir
    if args.service is not None:
        return SERVICE_DEFAULTS[args.service]
    raise SystemExit("either --service or --output-dir is required")


def main() -> None:
    args = parse_args()
    output_dir = resolve_output_dir(args).resolve()
    backup_dir = (
        args.backup_dir.resolve()
        if args.backup_dir is not None
        else (output_dir / "_retention_backups").resolve()
    )
    if args.dry_run:
        video_count = sum(len(iter_video_files(path)) for path in iter_request_dirs(output_dir, backup_dir))
        payload = {
            "dry_run": True,
            "output_dir": str(output_dir),
            "backup_dir": str(backup_dir),
            "video_count": video_count,
            "max_videos": args.max_videos,
            "would_cleanup": video_count > args.max_videos,
        }
    else:
        manager = StorageRetentionManager(
            StorageRetentionConfig(
                output_dir=output_dir,
                backup_dir=backup_dir,
                max_videos=args.max_videos,
                cleanup_batch_size=args.batch_size,
            )
        )
        payload = manager.enforce()
        payload["output_dir"] = str(output_dir)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
