from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
LARGE_MEDIA_SUFFIXES = VIDEO_SUFFIXES | {".zip"}
METADATA_SUFFIXES = {".json"}


@dataclass
class StorageRetentionConfig:
    output_dir: Path
    backup_dir: Path
    max_videos: int = 2000
    cleanup_batch_size: int = 200
    enabled: bool = True


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def save_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def iter_request_dirs(output_dir: Path, backup_dir: Path) -> list[Path]:
    if not output_dir.exists():
        return []
    request_dirs = []
    backup_resolved = backup_dir.resolve()
    for path in output_dir.iterdir():
        if not path.is_dir():
            continue
        if path.resolve() == backup_resolved:
            continue
        request_dirs.append(path)
    return request_dirs


def iter_video_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in VIDEO_SUFFIXES)


def iter_large_media_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in LARGE_MEDIA_SUFFIXES)


def iter_metadata_files(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(item for item in path.rglob("*") if item.is_file() and item.suffix.lower() in METADATA_SUFFIXES)


def request_sort_key(path: Path) -> tuple[float, str]:
    try:
        marker = path / "storage_cleanup_marker.json"
        if marker.exists():
            mtime = marker.stat().st_mtime
        else:
            mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return mtime, path.name


def safe_copy_metadata(src: Path, dst_root: Path, request_dir: Path) -> str:
    try:
        relative = src.relative_to(request_dir)
    except ValueError:
        relative = Path(src.name)
    dst = dst_root / "metadata" / relative
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return str(dst)


def extract_representative_frames(video_path: Path, output_dir: Path, prefix: str = "frame") -> list[str]:
    try:
        import cv2
    except Exception:
        return []

    ensure_dir(output_dir)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 1:
        indices = [
            0,
            max(0, min(total - 1, int(round((total - 1) * 0.25)))),
            max(0, min(total - 1, int(round((total - 1) * 0.50)))),
            max(0, min(total - 1, int(round((total - 1) * 0.75)))),
            total - 1,
        ]
    else:
        indices = [0, 0, 0, 0, 0]

    saved: list[str] = []
    for out_index, frame_index in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        out_path = output_dir / f"{prefix}_{out_index:02d}_src_{frame_index}.jpg"
        ok, encoded = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        if not ok:
            continue
        out_path.write_bytes(encoded.tobytes())
        saved.append(str(out_path))

    cap.release()
    return saved


def find_primary_video(request_dir: Path, metadata_payloads: list[dict[str, Any]]) -> Path | None:
    candidates: list[Path] = []
    metadata_keys = [
        "stored_video",
        "video_path",
        "generated_capture_video",
        "source_path",
        "source_video_path",
    ]
    for payload in metadata_payloads:
        for key in metadata_keys:
            value = payload.get(key)
            if isinstance(value, str) and value:
                candidates.append(Path(value))

    request_videos = iter_video_files(request_dir)
    candidates.extend(sorted(request_videos, key=lambda path: path.stat().st_size if path.exists() else 0, reverse=True))
    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except OSError:
            continue
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in VIDEO_SUFFIXES:
            if is_relative_to(candidate, request_dir):
                return candidate
    return request_videos[0] if request_videos else None


def summarize_conclusion(request_id: str, metadata_payloads: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for payload in metadata_payloads:
        merged.update(payload)
    keys = [
        "request_id",
        "result",
        "prediction_id",
        "score",
        "probability_live",
        "threshold",
        "color_validation_pass",
        "input_type",
        "uploaded_filename",
        "latency_ms",
        "num_frames",
        "num_frames_used",
        "num_windows",
        "metadata_json",
    ]
    conclusion = {key: merged.get(key) for key in keys if key in merged}
    conclusion.setdefault("request_id", request_id)
    return conclusion


class StorageRetentionManager:
    def __init__(self, config: StorageRetentionConfig) -> None:
        self.config = config
        ensure_dir(self.config.output_dir)
        ensure_dir(self.config.backup_dir)

    def media_video_count(self) -> int:
        total = 0
        for request_dir in iter_request_dirs(self.config.output_dir, self.config.backup_dir):
            total += len(iter_video_files(request_dir))
        return total

    def enforce(self, exclude_request_id: str | None = None) -> dict[str, Any]:
        if not self.config.enabled or self.config.max_videos <= 0:
            return {"enabled": self.config.enabled, "action": "disabled"}

        request_dirs = iter_request_dirs(self.config.output_dir, self.config.backup_dir)
        videos_by_request = {path: iter_video_files(path) for path in request_dirs}
        total_videos = sum(len(videos) for videos in videos_by_request.values())
        if total_videos <= self.config.max_videos:
            return {
                "enabled": True,
                "action": "none",
                "video_count": total_videos,
                "max_videos": self.config.max_videos,
            }

        removed_videos = 0
        cleaned_requests: list[dict[str, Any]] = []
        target_remove = total_videos - self.config.max_videos
        batch_limit = max(self.config.cleanup_batch_size, 1)

        for request_dir in sorted(request_dirs, key=request_sort_key):
            if exclude_request_id and request_dir.name == exclude_request_id:
                continue
            videos = videos_by_request.get(request_dir) or []
            if not videos:
                continue
            cleanup = self.backup_and_delete_media(request_dir)
            cleaned_requests.append(cleanup)
            removed_videos += int(cleanup.get("deleted_video_count", 0))
            if removed_videos >= target_remove or len(cleaned_requests) >= batch_limit:
                break

        return {
            "enabled": True,
            "action": "cleanup",
            "video_count_before": total_videos,
            "max_videos": self.config.max_videos,
            "removed_video_count": removed_videos,
            "cleaned_request_count": len(cleaned_requests),
            "backup_dir": str(self.config.backup_dir),
            "cleaned_requests": cleaned_requests,
        }

    def backup_and_delete_media(self, request_dir: Path) -> dict[str, Any]:
        request_id = request_dir.name
        backup_dir = self.config.backup_dir / request_id
        ensure_dir(backup_dir)

        metadata_files = iter_metadata_files(request_dir)
        metadata_payloads = [payload for path in metadata_files if (payload := load_json(path)) is not None]
        copied_metadata = [safe_copy_metadata(path, backup_dir, request_dir) for path in metadata_files]
        conclusion = summarize_conclusion(request_id, metadata_payloads)

        primary_video = find_primary_video(request_dir, metadata_payloads)
        frame_paths: list[str] = []
        if primary_video is not None:
            try:
                frame_paths = extract_representative_frames(primary_video, backup_dir / "representative_frames")
            except Exception:
                frame_paths = []

        media_files = iter_large_media_files(request_dir)
        deleted_media: list[str] = []
        deleted_video_count = 0
        for media_path in media_files:
            try:
                if media_path.suffix.lower() in VIDEO_SUFFIXES:
                    deleted_video_count += 1
                deleted_media.append(str(media_path))
                media_path.unlink()
            except OSError:
                continue

        summary = {
            "request_id": request_id,
            "backup_time_unix": time.time(),
            "request_dir": str(request_dir),
            "backup_dir": str(backup_dir),
            "conclusion": conclusion,
            "metadata_backups": copied_metadata,
            "primary_video": str(primary_video) if primary_video is not None else "",
            "representative_frames": frame_paths,
            "deleted_media": deleted_media,
            "deleted_video_count": deleted_video_count,
            "deleted_media_count": len(deleted_media),
        }
        save_json(backup_dir / "retention_backup_summary.json", summary)
        save_json(
            request_dir / "storage_cleanup_marker.json",
            {
                "backup_dir": str(backup_dir),
                "deleted_video_count": deleted_video_count,
                "deleted_media_count": len(deleted_media),
                "backup_time_unix": summary["backup_time_unix"],
            },
        )
        return summary
