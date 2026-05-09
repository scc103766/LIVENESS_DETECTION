from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
LABEL_NAME_TO_ID = {
    "live": 1,
    "real": 1,
    "bonafide": 1,
    "bona_fide": 1,
    "genuine": 1,
    "真人": 1,
    "spoof": 0,
    "fake": 0,
    "attack": 0,
    "imposter": 0,
    "print": 0,
    "replay": 0,
    "mask": 0,
    "head_model": 0,
    "headmodel": 0,
    "头模": 0,
    "攻击": 0,
}

KEY_MEANINGS = {
    "index": "样本序号。",
    "video_path": "被分析的视频路径。",
    "txt_path": "与视频同名的逐帧闪光颜色标注 txt 路径。",
    "label_id": "真值标签，1=live/真人，0=spoof/攻击。",
    "label_name": "真值标签名称，live 或 spoof。",
    "label_source": "真值标签来源，通常来自路径中的 live/spoof 目录名。",
    "decoded_frames": "OpenCV 成功解码读取到的帧数。",
    "kept_frames": "实际参与分析的帧数，受 --frame-stride 和 --max-frames 影响。",
    "txt_color_count": "txt 中读取到的颜色标注数量。",
    "has_txt": "是否找到同名 txt，1=找到，0=未找到。",
    "rppg_mean_g": "全视频人脸区域绿色通道均值。绿色通道与血红蛋白吸收和皮肤反射有关。",
    "rppg_std_g": "全视频绿色通道均值的时间标准差，表示绿色信号随时间波动强度。",
    "rppg_delta_g_abs_mean": "相邻帧绿色通道变化绝对值均值，表示绿色信号帧间变化幅度。",
    "rppg_fft_peak_score": "绿色通道时间序列频谱主峰占比，越高表示存在更集中的周期性波动。",
    "rppg_region_gap_mean": "脸颊区域和额头区域绿色通道差异均值，用于观察区域生理/反射差异。",
    "rppg_cheek_std_g": "脸颊区域绿色通道时间标准差。",
    "rppg_forehead_std_g": "额头区域绿色通道时间标准差。",
    "flash_unique_color_count": "视频中识别到的不同闪光颜色数量。",
    "flash_transition_ratio": "发生光色切换的帧比例。",
    "flash_transition_delta_abs_mean": "光色切换帧上的亮度变化绝对值均值，反映切换瞬间响应强度。",
    "flash_stable_delta_abs_mean": "非光色切换帧上的亮度变化绝对值均值，反映稳定阶段波动。",
    "flash_color_intensity_range": "不同闪光颜色分组后，人脸平均亮度均值的范围。",
    "flash_color_green_range": "不同闪光颜色分组后，人脸绿色通道均值的范围。",
    "flash_red_over_green_mean": "红色通道与绿色通道比值均值，用于描述颜色反射比例。",
    "flash_blue_over_green_mean": "蓝色通道与绿色通道比值均值，用于描述颜色反射比例。",
    "flash_chroma_range_mean": "每帧 RGB 最大通道和最小通道差值的均值，表示颜色饱和/色度变化。",
    "freq_high_energy_mean": "频域高频能量比例均值，可能反映屏幕像素纹、摩尔纹、边缘纹理。",
    "freq_high_energy_std": "频域高频能量比例的时间标准差。",
    "freq_mid_energy_mean": "频域中频能量比例均值。",
    "freq_mid_energy_std": "频域中频能量比例的时间标准差。",
    "freq_lap_var_mean": "Laplacian 方差均值，表示图像锐度和细节纹理强度。",
    "freq_lap_var_std": "Laplacian 方差的时间标准差。",
    "freq_row_periodicity_mean": "按行方向统计的周期性变化均值，用于观察横向扫描线/周期纹。",
    "freq_row_periodicity_std": "行方向周期性变化的时间标准差。",
    "freq_col_periodicity_mean": "按列方向统计的周期性变化均值，用于观察纵向像素纹/周期纹。",
    "freq_col_periodicity_std": "列方向周期性变化的时间标准差。",
    "status": "该视频分析状态，ok=成功，failed=失败。",
    "error": "失败原因；成功时为空。",
    "feature": "被统计区分度的特征名。",
    "live_mean": "该特征在 live 样本上的均值。",
    "spoof_mean": "该特征在 spoof 样本上的均值。",
    "live_std": "该特征在 live 样本上的标准差。",
    "spoof_std": "该特征在 spoof 样本上的标准差。",
    "cohen_d_live_minus_spoof": "Cohen's d 效应量，正值表示 live 更大，负值表示 spoof 更大。",
    "auc_live_high_raw": "假设特征越大越像 live 时的原始 AUC。",
    "separability_auc": "不关心方向后的可分性 AUC，越接近 1 说明越能区分 live/spoof。",
    "direction": "区分方向，live_higher 表示真人更大，live_lower 表示真人更小。",
    "best_balanced_accuracy": "只使用该单一特征和最佳阈值进行二分类时的平衡准确率。",
    "best_threshold": "该单一特征取得最佳平衡准确率时的阈值。",
}

HEAD_MODEL_TOKENS = ("toumo", "headmodel", "head_model", "head", "mask", "3dfake")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: dict | list) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_key_meanings(output_dir: Path) -> None:
    rows = [{"key": key, "meaning": meaning} for key, meaning in KEY_MEANINGS.items()]
    write_csv(output_dir / "physical_cues_key_meanings.csv", rows)
    save_json(output_dir / "physical_cues_key_meanings.json", KEY_MEANINGS)

    md_path = output_dir / "physical_cues_key_meanings.md"
    with md_path.open("w", encoding="utf-8") as file:
        file.write("# Physical Cues Key Meanings\n\n")
        file.write("该文件解释 `physical_cues_per_video.csv`、`physical_cues_feature_separation.csv` 和 `physical_cues_summary.json` 中各字段的含义。\n\n")
        file.write("| Key | Meaning |\n")
        file.write("| --- | --- |\n")
        for key, meaning in KEY_MEANINGS.items():
            file.write(f"| `{key}` | {meaning} |\n")


@contextmanager
def suppress_native_stderr(enabled: bool):
    if not enabled:
        yield
        return

    stderr_fd = sys.stderr.fileno()
    saved_fd = os.dup(stderr_fd)
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        os.dup2(devnull.fileno(), stderr_fd)
        try:
            yield
        finally:
            os.dup2(saved_fd, stderr_fd)
            os.close(saved_fd)


def infer_label_from_path(video_path: Path) -> tuple[int | None, str]:
    for parent in video_path.parents:
        normalized = parent.name.strip().lower().replace("-", "_").replace(" ", "_")
        if normalized in LABEL_NAME_TO_ID:
            return LABEL_NAME_TO_ID[normalized], f"dir:{parent.name}"
    return None, "unresolved"


def collect_videos(input_root: Path) -> list[Path]:
    if input_root.is_file() and input_root.suffix.lower() in VIDEO_EXTENSIONS:
        return [input_root]
    return sorted(
        path for path in input_root.rglob("*")
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )


def parse_color_txt(txt_path: Path | None) -> dict[int, int]:
    mapping: dict[int, int] = {}
    if txt_path is None or not txt_path.exists():
        return mapping

    with txt_path.open("r", encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line:
                continue
            try:
                if "," in line:
                    frame_str, color_str = line.split(",", 1)
                    mapping[int(frame_str)] = int(color_str)
                else:
                    mapping[len(mapping)] = int(line)
            except ValueError:
                continue
    return mapping


def color_int_to_rgb(color_value: int) -> tuple[int, int, int]:
    return (
        (int(color_value) >> 16) & 0xFF,
        (int(color_value) >> 8) & 0xFF,
        int(color_value) & 0xFF,
    )


def center_crop_and_resize(frame: np.ndarray, target_size: int) -> np.ndarray:
    if frame is None or frame.size == 0:
        return np.zeros((target_size, target_size, 3), dtype=np.uint8)
    height, width = frame.shape[:2]
    side = min(height, width)
    top = max((height - side) // 2, 0)
    left = max((width - side) // 2, 0)
    cropped = frame[top:top + side, left:left + side]
    if cropped.size == 0:
        cropped = frame
    return cv2.resize(cropped, (target_size, target_size))


def read_video_frames(
    video_path: Path,
    txt_path: Path | None,
    target_size: int,
    max_frames: int,
    frame_stride: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    cap = cv2.VideoCapture(str(video_path))
    color_map = parse_color_txt(txt_path)
    frames_rgb: list[np.ndarray] = []
    color_values: list[int] = []
    prev_color_value = 0
    decoded_count = 0
    kept_count = 0
    frame_idx = 0

    while True:
        success, frame_bgr = cap.read()
        if not success:
            break
        decoded_count += 1

        if frame_idx % frame_stride != 0:
            frame_idx += 1
            continue

        if frame_bgr is not None and frame_bgr.size > 0:
            face_bgr = center_crop_and_resize(frame_bgr, target_size)
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            frames_rgb.append(face_rgb.astype(np.float32) / 255.0)
            color_value = color_map.get(frame_idx, prev_color_value)
            color_values.append(int(color_value))
            prev_color_value = int(color_value)
            kept_count += 1
            if max_frames > 0 and kept_count >= max_frames:
                break

        frame_idx += 1

    cap.release()
    meta = {
        "decoded_frames": decoded_count,
        "kept_frames": kept_count,
        "txt_color_count": len(color_map),
        "has_txt": int(txt_path is not None and txt_path.exists()),
    }
    if not frames_rgb:
        return np.zeros((0, target_size, target_size, 3), dtype=np.float32), np.zeros((0,), dtype=np.int64), meta
    return np.stack(frames_rgb, axis=0), np.asarray(color_values, dtype=np.int64), meta


class RPPGFeatureExtractor:
    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def _fft_peak_score(self, signal: np.ndarray) -> float:
        if len(signal) < 8:
            return 0.0
        centered = signal - signal.mean()
        spectrum = np.abs(np.fft.rfft(centered))
        if len(spectrum) <= 2:
            return 0.0
        spectrum[0] = 0.0
        peak = float(spectrum.max())
        total = float(spectrum.sum()) + self.eps
        return peak / total

    def extract(self, frames_rgb: np.ndarray) -> dict[str, float]:
        if frames_rgb.size == 0:
            return {}

        height = frames_rgb.shape[1]
        mean_rgb = frames_rgb.mean(axis=(1, 2))
        mean_g = mean_rgb[:, 1]
        delta_g = np.zeros_like(mean_g)
        if len(mean_g) > 1:
            delta_g[1:] = mean_g[1:] - mean_g[:-1]

        forehead_g = frames_rgb[:, : height // 3, :, 1].mean(axis=(1, 2))
        cheek_g = frames_rgb[:, height // 2 :, :, 1].mean(axis=(1, 2))
        region_gap = np.abs(cheek_g - forehead_g)

        return {
            "rppg_mean_g": float(mean_g.mean()),
            "rppg_std_g": float(mean_g.std()),
            "rppg_delta_g_abs_mean": float(np.abs(delta_g).mean()),
            "rppg_fft_peak_score": float(self._fft_peak_score(mean_g)),
            "rppg_region_gap_mean": float(region_gap.mean()),
            "rppg_cheek_std_g": float(cheek_g.std()),
            "rppg_forehead_std_g": float(forehead_g.std()),
        }


class FlashResponseFeatureExtractor:
    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def extract(self, frames_rgb: np.ndarray, color_values: np.ndarray) -> dict[str, float]:
        if frames_rgb.size == 0:
            return {}

        rgb_mean = frames_rgb.mean(axis=(1, 2))
        intensity = rgb_mean.mean(axis=1)
        delta_intensity = np.zeros_like(intensity)
        if len(intensity) > 1:
            delta_intensity[1:] = intensity[1:] - intensity[:-1]

        transition = np.zeros_like(intensity)
        if len(color_values) > 1:
            transition[1:] = color_values[1:] != color_values[:-1]
        transition_delta = np.abs(delta_intensity[transition > 0])
        stable_delta = np.abs(delta_intensity[transition == 0])

        unique_colors = sorted(set(int(value) for value in color_values.tolist()))
        color_intensity_means = []
        color_green_means = []
        for color_value in unique_colors:
            mask = color_values == color_value
            if mask.any():
                color_intensity_means.append(float(intensity[mask].mean()))
                color_green_means.append(float(rgb_mean[mask, 1].mean()))

        red_over_green = rgb_mean[:, 0] / (rgb_mean[:, 1] + self.eps)
        blue_over_green = rgb_mean[:, 2] / (rgb_mean[:, 1] + self.eps)
        chroma_range = rgb_mean.max(axis=1) - rgb_mean.min(axis=1)

        return {
            "flash_unique_color_count": float(len(unique_colors)),
            "flash_transition_ratio": float(transition.mean()),
            "flash_transition_delta_abs_mean": float(transition_delta.mean()) if transition_delta.size else 0.0,
            "flash_stable_delta_abs_mean": float(stable_delta.mean()) if stable_delta.size else 0.0,
            "flash_color_intensity_range": float(np.ptp(color_intensity_means)) if color_intensity_means else 0.0,
            "flash_color_green_range": float(np.ptp(color_green_means)) if color_green_means else 0.0,
            "flash_red_over_green_mean": float(red_over_green.mean()),
            "flash_blue_over_green_mean": float(blue_over_green.mean()),
            "flash_chroma_range_mean": float(chroma_range.mean()),
        }


class FrequencyArtifactExtractor:
    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def _frame_features(self, frame_rgb: np.ndarray) -> dict[str, float]:
        gray = cv2.cvtColor((frame_rgb * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        gray_f = gray.astype(np.float32) / 255.0
        fft = np.fft.fftshift(np.fft.fft2(gray_f))
        mag = np.log1p(np.abs(fft))
        height, width = mag.shape
        cy, cx = height // 2, width // 2
        yy, xx = np.ogrid[:height, :width]
        radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        high_mask = radius > min(height, width) * 0.25
        mid_mask = (radius > min(height, width) * 0.10) & (radius <= min(height, width) * 0.25)
        total_energy = float(mag.mean()) + self.eps

        lap = cv2.Laplacian(gray_f, cv2.CV_32F)
        row_profile = gray_f.mean(axis=1)
        col_profile = gray_f.mean(axis=0)
        return {
            "high_energy": float(mag[high_mask].mean() / total_energy),
            "mid_energy": float(mag[mid_mask].mean() / total_energy),
            "lap_var": float(lap.var()),
            "row_periodicity": float(np.std(np.diff(row_profile))),
            "col_periodicity": float(np.std(np.diff(col_profile))),
        }

    def extract(self, frames_rgb: np.ndarray, max_frequency_frames: int) -> dict[str, float]:
        if frames_rgb.size == 0:
            return {}
        if max_frequency_frames > 0 and len(frames_rgb) > max_frequency_frames:
            indices = np.linspace(0, len(frames_rgb) - 1, max_frequency_frames).astype(np.int64)
            selected = frames_rgb[indices]
        else:
            selected = frames_rgb

        frame_rows = [self._frame_features(frame) for frame in selected]
        keys = frame_rows[0].keys()
        result = {}
        for key in keys:
            values = np.asarray([row[key] for row in frame_rows], dtype=np.float32)
            result[f"freq_{key}_mean"] = float(values.mean())
            result[f"freq_{key}_std"] = float(values.std())
        return result


class PhysicalCueAnalyzer:
    def __init__(self, max_frequency_frames: int) -> None:
        self.rppg = RPPGFeatureExtractor()
        self.flash = FlashResponseFeatureExtractor()
        self.frequency = FrequencyArtifactExtractor()
        self.max_frequency_frames = max_frequency_frames

    def extract(self, frames_rgb: np.ndarray, color_values: np.ndarray) -> dict[str, float]:
        features: dict[str, float] = {}
        features.update(self.rppg.extract(frames_rgb))
        features.update(self.flash.extract(frames_rgb, color_values))
        features.update(self.frequency.extract(frames_rgb, self.max_frequency_frames))
        return features


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = labels.astype(np.int32)
    scores = scores.astype(np.float64)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")

    wins = 0.0
    total = len(pos) * len(neg)
    for pos_score in pos:
        wins += float((pos_score > neg).sum())
        wins += 0.5 * float((pos_score == neg).sum())
    return float(wins / total)


def best_balanced_accuracy(labels: np.ndarray, scores: np.ndarray, live_high: bool) -> tuple[float, float]:
    labels = labels.astype(np.int32)
    scores = scores.astype(np.float64)
    candidates = sorted(set(scores.tolist()))
    if not candidates:
        return 0.0, 0.0

    best_score = -1.0
    best_threshold = candidates[0]
    for threshold in candidates:
        if live_high:
            preds = (scores >= threshold).astype(np.int32)
        else:
            preds = (scores <= threshold).astype(np.int32)
        tp = int(((preds == 1) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        positives = int((labels == 1).sum())
        negatives = int((labels == 0).sum())
        tpr = safe_divide(tp, positives)
        tnr = safe_divide(tn, negatives)
        balanced_acc = (tpr + tnr) / 2.0
        if balanced_acc > best_score:
            best_score = balanced_acc
            best_threshold = threshold
    return float(best_score), float(best_threshold)


def summarize_feature_separation(rows: list[dict]) -> list[dict]:
    labeled_rows = [row for row in rows if row.get("status") == "ok" and row.get("label_id") in (0, 1)]
    if not labeled_rows:
        return []

    labels = np.asarray([int(row["label_id"]) for row in labeled_rows], dtype=np.int32)
    skip_keys = {
        "index",
        "status",
        "error",
        "video_path",
        "txt_path",
        "label_id",
        "label_name",
        "label_source",
        "decoded_frames",
        "kept_frames",
        "txt_color_count",
        "has_txt",
    }
    feature_keys = [
        key for key, value in labeled_rows[0].items()
        if key not in skip_keys and isinstance(value, (int, float, np.integer, np.floating))
    ]

    summaries = []
    for key in feature_keys:
        values = np.asarray([float(row[key]) for row in labeled_rows], dtype=np.float64)
        if not np.isfinite(values).all() or np.all(values == values[0]):
            continue
        live_values = values[labels == 1]
        spoof_values = values[labels == 0]
        if len(live_values) == 0 or len(spoof_values) == 0:
            continue

        pooled_std = math.sqrt((float(live_values.var()) + float(spoof_values.var())) / 2.0)
        cohen_d = safe_divide(float(live_values.mean() - spoof_values.mean()), pooled_std)
        auc = compute_auc(labels, values)
        live_high = bool(auc >= 0.5)
        separability_auc = auc if live_high else 1.0 - auc
        balanced_acc, threshold = best_balanced_accuracy(labels, values, live_high=live_high)

        summaries.append(
            {
                "feature": key,
                "live_mean": round(float(live_values.mean()), 8),
                "spoof_mean": round(float(spoof_values.mean()), 8),
                "live_std": round(float(live_values.std()), 8),
                "spoof_std": round(float(spoof_values.std()), 8),
                "cohen_d_live_minus_spoof": round(float(cohen_d), 8),
                "auc_live_high_raw": round(float(auc), 8),
                "separability_auc": round(float(separability_auc), 8),
                "direction": "live_higher" if live_high else "live_lower",
                "best_balanced_accuracy": round(float(balanced_acc), 8),
                "best_threshold": round(float(threshold), 8),
            }
        )

    return sorted(summaries, key=lambda row: row["separability_auc"], reverse=True)


def analyze_video(
    index: int,
    video_path: Path,
    args: argparse.Namespace,
    analyzer: PhysicalCueAnalyzer,
) -> dict:
    txt_path = Path(args.txt_path).resolve() if args.txt_path and len(args.input_videos) == 1 else video_path.with_suffix(".txt")
    label_id, label_source = infer_label_from_path(video_path)

    base_row = {
        "index": index,
        "video_path": str(video_path),
        "txt_path": str(txt_path) if txt_path.exists() else "",
        "label_id": "" if label_id is None else int(label_id),
        "label_name": "" if label_id is None else ("live" if label_id == 1 else "spoof"),
        "label_source": label_source,
    }

    try:
        with suppress_native_stderr(not args.show_decoder_warnings):
            frames_rgb, color_values, meta = read_video_frames(
                video_path=video_path,
                txt_path=txt_path if txt_path.exists() else None,
                target_size=args.image_size,
                max_frames=args.max_frames,
                frame_stride=args.frame_stride,
            )
        if frames_rgb.size == 0:
            return {**base_row, **meta, "status": "failed", "error": "no_valid_frames"}

        features = analyzer.extract(frames_rgb, color_values)
        return {**base_row, **meta, **features, "status": "ok", "error": ""}
    except Exception as exc:
        return {**base_row, "status": "failed", "error": repr(exc)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze explicit physical cues for flash liveness V2 without changing the training model. "
            "Outputs per-video features and live/spoof separability metrics."
        )
    )
    parser.add_argument("--input-root", required=True, help="视频数据根目录，建议为 train/val/test/live|spoof 或 live|spoof 结构。")
    parser.add_argument("--output-dir", required=True, help="分析结果输出目录。")
    parser.add_argument("--txt-path", default=None, help="单视频分析时可显式指定 txt。")
    parser.add_argument("--image-size", type=int, default=224, help="中心裁剪后缩放尺寸。")
    parser.add_argument("--max-frames", type=int, default=0, help="最多分析多少帧，0 表示全帧。")
    parser.add_argument("--frame-stride", type=int, default=1, help="分析时每隔多少帧取一帧，1 表示不跳帧。")
    parser.add_argument("--max-videos", type=int, default=0, help="最多分析多少个视频，0 表示全部。")
    parser.add_argument("--samples-per-class", type=int, default=0, help="每类最多分析多少个视频，0 表示不限制。")
    parser.add_argument(
        "--smoke-live-headmodel-pair",
        action="store_true",
        help="只选择 1 个 live 真人视频和 1 个头模/3D/mask spoof 视频做 smoke test 对比。",
    )
    parser.add_argument("--max-frequency-frames", type=int, default=96, help="频域模块最多取多少帧计算，避免全帧 FFT 过慢。")
    parser.add_argument("--show-decoder-warnings", action="store_true", help="显示 OpenCV/FFmpeg 解码告警，默认隐藏。")
    return parser.parse_args()


def limit_samples(videos: list[Path], samples_per_class: int, max_videos: int) -> list[Path]:
    if samples_per_class > 0:
        grouped: dict[int | None, list[Path]] = {0: [], 1: [], None: []}
        for video in videos:
            label_id, _source = infer_label_from_path(video)
            grouped[label_id].append(video)
        limited = grouped[1][:samples_per_class] + grouped[0][:samples_per_class] + grouped[None][:samples_per_class]
        videos = sorted(limited)

    if max_videos > 0:
        videos = videos[:max_videos]
    return videos


def select_live_headmodel_pair(videos: list[Path]) -> list[Path]:
    live_candidates = []
    head_model_candidates = []
    spoof_fallback = []

    for video in videos:
        label_id, _source = infer_label_from_path(video)
        normalized_name = video.name.lower()
        if label_id == 1:
            live_candidates.append(video)
        elif label_id == 0:
            spoof_fallback.append(video)
            if any(token in normalized_name for token in HEAD_MODEL_TOKENS):
                head_model_candidates.append(video)

    if not live_candidates:
        raise ValueError("未找到 live 真人视频，无法执行 smoke-live-headmodel-pair。")
    if not head_model_candidates and not spoof_fallback:
        raise ValueError("未找到 spoof/头模攻击视频，无法执行 smoke-live-headmodel-pair。")

    live_video = sorted(live_candidates)[0]
    spoof_video = sorted(head_model_candidates or spoof_fallback)[0]
    return [live_video, spoof_video]


def main() -> int:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    videos = collect_videos(input_root)
    if args.smoke_live_headmodel_pair:
        videos = select_live_headmodel_pair(videos)
    else:
        videos = limit_samples(videos, samples_per_class=args.samples_per_class, max_videos=args.max_videos)
    args.input_videos = videos
    analyzer = PhysicalCueAnalyzer(max_frequency_frames=args.max_frequency_frames)

    rows = []
    for index, video_path in enumerate(videos, start=1):
        print(f"[{index}/{len(videos)}] analyzing {video_path}", flush=True)
        rows.append(analyze_video(index, video_path, args, analyzer))

    feature_rows = summarize_feature_separation(rows)
    ok_rows = [row for row in rows if row.get("status") == "ok"]
    labeled_ok_rows = [row for row in ok_rows if row.get("label_id") in (0, 1)]
    summary = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "total_videos": len(videos),
        "ok_videos": len(ok_rows),
        "failed_videos": len(videos) - len(ok_rows),
        "labeled_ok_videos": len(labeled_ok_rows),
        "live_count": sum(1 for row in labeled_ok_rows if row["label_id"] == 1),
        "spoof_count": sum(1 for row in labeled_ok_rows if row["label_id"] == 0),
        "top_features": feature_rows[:20],
        "notes": [
            "separability_auc 越接近 1，说明该物理特征越能区分 live/spoof。",
            "direction=live_higher 表示真人该特征更大；live_lower 表示真人该特征更小。",
            "该脚本只验证显式物理线索是否有区分度，不修改或调用 V2 训练模型。",
        ],
    }

    write_csv(output_dir / "physical_cues_per_video.csv", rows)
    write_csv(output_dir / "physical_cues_feature_separation.csv", feature_rows)
    write_key_meanings(output_dir)
    save_json(output_dir / "physical_cues_summary.json", summary)

    print(f"Saved per-video features: {output_dir / 'physical_cues_per_video.csv'}")
    print(f"Saved feature separation: {output_dir / 'physical_cues_feature_separation.csv'}")
    print(f"Saved key meanings: {output_dir / 'physical_cues_key_meanings.md'}")
    print(f"Saved summary: {output_dir / 'physical_cues_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
