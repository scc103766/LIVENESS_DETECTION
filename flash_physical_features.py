from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class PhysicalFeatureConfig:
    """Physical feature switches used by V2 flash-liveness experiments.

    The feature priorities come from:
    `flash_liveness_runs/physical_cue_analysis_v2_recommendation_report.md`.

    Priority order:
    1. Frequency/texture features.
    2. Flash-response features.
    3. Simple rPPG statistical features.
    4. Optional offline depth/normal features.
    """

    use_frequency: bool = True
    use_flash_response: bool = True
    use_rppg: bool = True
    use_depth_normal: bool = False
    eps: float = 1e-6


def _empty_feature(num_frames: int, dim: int) -> np.ndarray:
    return np.zeros((max(int(num_frames), 0), dim), dtype=np.float32)


def _as_float_rgb(frames_rgb: np.ndarray) -> np.ndarray:
    """Normalize RGB frame tensors to float32 in [0, 1].

    Expected input shape:
        T x H x W x 3
    """

    if frames_rgb.size == 0:
        return frames_rgb.astype(np.float32)
    frames = frames_rgb.astype(np.float32, copy=False)
    if frames.max(initial=0.0) > 1.5:
        frames = frames / 255.0
    return np.clip(frames, 0.0, 1.0)


def _match_length(array: np.ndarray, target_len: int) -> np.ndarray:
    """Resample the first dimension to match a video sequence length."""

    target_len = int(target_len)
    if target_len <= 0:
        return array[:0]
    if len(array) == target_len:
        return array
    if len(array) == 0:
        return np.zeros((target_len,) + array.shape[1:], dtype=np.float32)
    indices = np.linspace(0, len(array) - 1, target_len).astype(np.int64)
    return array[indices]


class FrequencyArtifactExtractor:
    """Frequency/texture cues for physical presentation attacks.

    Role in V2:
        This is the highest-priority physical module from the val/test analysis.
        It captures screen/print/head-model artifacts such as abnormal frequency
        energy, over-smoothing, sharpness changes, and row-wise periodic texture.

    Input:
        frames_rgb: T x H x W x 3, RGB, uint8 or float in [0, 1]

    Output from `per_frame`:
        T x 5 with columns:
        0. freq_high_energy
        1. freq_mid_energy
        2. freq_lap_var
        3. freq_row_periodicity
        4. freq_col_periodicity

    Recommended V2 use:
        feed the T x 5 tensor into `physical_proj`, then fuse with
        CNN frame features and color embeddings before the Transformer.
    """

    feature_names = (
        "freq_high_energy",
        "freq_mid_energy",
        "freq_lap_var",
        "freq_row_periodicity",
        "freq_col_periodicity",
    )

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def _extract_one(self, frame_rgb: np.ndarray) -> np.ndarray:
        frame_rgb = _as_float_rgb(frame_rgb)
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

        return np.asarray(
            [
                float(mag[high_mask].mean() / total_energy),
                float(mag[mid_mask].mean() / total_energy),
                float(lap.var()),
                float(np.std(np.diff(row_profile))),
                float(np.std(np.diff(col_profile))),
            ],
            dtype=np.float32,
        )

    def per_frame(self, frames_rgb: np.ndarray) -> np.ndarray:
        frames = _as_float_rgb(frames_rgb)
        if frames.size == 0:
            return _empty_feature(0, len(self.feature_names))
        return np.stack([self._extract_one(frame) for frame in frames], axis=0).astype(np.float32)

    def aggregate(self, frames_rgb: np.ndarray) -> dict[str, float]:
        per_frame = self.per_frame(frames_rgb)
        if per_frame.size == 0:
            return {f"{name}_mean": 0.0 for name in self.feature_names} | {
                f"{name}_std": 0.0 for name in self.feature_names
            }

        result: dict[str, float] = {}
        for index, name in enumerate(self.feature_names):
            values = per_frame[:, index]
            result[f"{name}_mean"] = float(values.mean())
            result[f"{name}_std"] = float(values.std())
        return result


class FlashResponseFeatureExtractor:
    """Flash color response and optical hysteresis cues.

    Role in V2:
        This module tells the model how the face responds to the known flash
        color sequence from txt labels. It is complementary to `color_tensor`:
        color_tensor says "what light was emitted"; this module says "how the
        observed face responded".

    Input:
        frames_rgb: T x H x W x 3
        color_values: T int RGB-packed colors, for example 16717055.

    Output from `per_frame`:
        T x 8 with columns:
        0. flash_r_mean
        1. flash_g_mean
        2. flash_b_mean
        3. flash_intensity
        4. flash_delta_intensity
        5. flash_transition
        6. flash_response_decay
        7. flash_chroma_ratio
    """

    feature_names = (
        "flash_r_mean",
        "flash_g_mean",
        "flash_b_mean",
        "flash_intensity",
        "flash_delta_intensity",
        "flash_transition",
        "flash_response_decay",
        "flash_chroma_ratio",
    )

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def per_frame(self, frames_rgb: np.ndarray, color_values: np.ndarray) -> np.ndarray:
        frames = _as_float_rgb(frames_rgb)
        if frames.size == 0:
            return _empty_feature(0, len(self.feature_names))

        color_values = np.asarray(color_values, dtype=np.int64)
        if len(color_values) != len(frames):
            color_values = _match_length(color_values, len(frames)).astype(np.int64)

        rgb_mean = frames.mean(axis=(1, 2))
        intensity = rgb_mean.mean(axis=1)
        delta_intensity = np.zeros_like(intensity)
        if len(intensity) > 1:
            delta_intensity[1:] = intensity[1:] - intensity[:-1]

        transition = np.zeros_like(intensity)
        if len(color_values) > 1:
            transition[1:] = color_values[1:] != color_values[:-1]

        # A simple hysteresis proxy: response is refreshed on color changes and
        # decays otherwise. Strong unstable decay was more common in attacks.
        response_decay = np.zeros_like(intensity)
        for idx in range(1, len(intensity)):
            if transition[idx] > 0:
                response_decay[idx] = abs(delta_intensity[idx])
            else:
                response_decay[idx] = response_decay[idx - 1] * 0.8

        red_over_green = rgb_mean[:, 0] / (rgb_mean[:, 1] + self.eps)
        blue_over_green = rgb_mean[:, 2] / (rgb_mean[:, 1] + self.eps)
        chroma_range = rgb_mean.max(axis=1) - rgb_mean.min(axis=1)
        chroma_ratio = red_over_green + blue_over_green + chroma_range

        return np.stack(
            [
                rgb_mean[:, 0],
                rgb_mean[:, 1],
                rgb_mean[:, 2],
                intensity,
                delta_intensity,
                transition,
                response_decay,
                chroma_ratio,
            ],
            axis=1,
        ).astype(np.float32)

    def aggregate(self, frames_rgb: np.ndarray, color_values: np.ndarray) -> dict[str, float]:
        frames = _as_float_rgb(frames_rgb)
        if frames.size == 0:
            return {
                "flash_unique_color_count": 0.0,
                "flash_transition_ratio": 0.0,
                "flash_transition_delta_abs_mean": 0.0,
                "flash_stable_delta_abs_mean": 0.0,
                "flash_color_intensity_range": 0.0,
                "flash_color_green_range": 0.0,
            }

        color_values = np.asarray(color_values, dtype=np.int64)
        if len(color_values) != len(frames):
            color_values = _match_length(color_values, len(frames)).astype(np.int64)

        per_frame = self.per_frame(frames, color_values)
        intensity = per_frame[:, 3]
        delta_abs = np.abs(per_frame[:, 4])
        transition = per_frame[:, 5] > 0
        rgb_mean = per_frame[:, :3]

        unique_colors = sorted(set(int(value) for value in color_values.tolist()))
        color_intensity_means = []
        color_green_means = []
        for color_value in unique_colors:
            mask = color_values == color_value
            if mask.any():
                color_intensity_means.append(float(intensity[mask].mean()))
                color_green_means.append(float(rgb_mean[mask, 1].mean()))

        return {
            "flash_unique_color_count": float(len(unique_colors)),
            "flash_transition_ratio": float(transition.mean()),
            "flash_transition_delta_abs_mean": float(delta_abs[transition].mean()) if transition.any() else 0.0,
            "flash_stable_delta_abs_mean": float(delta_abs[~transition].mean()) if (~transition).any() else 0.0,
            "flash_color_intensity_range": float(np.ptp(color_intensity_means)) if color_intensity_means else 0.0,
            "flash_color_green_range": float(np.ptp(color_green_means)) if color_green_means else 0.0,
        }


class RPPGFeatureExtractor:
    """Simple green-channel rPPG statistical cues.

    Role in V2:
        The val/test report showed rPPG-like statistics are useful but weaker
        than frequency features. In this dataset, attacks often have larger
        green-channel instability; therefore this module should be treated as
        auxiliary evidence, not a standalone heartbeat detector.

    Input:
        frames_rgb: T x H x W x 3

    Output from `per_frame`:
        T x 6 with columns:
        0. rppg_mean_g
        1. rppg_delta_g
        2. rppg_cheek_g
        3. rppg_forehead_g
        4. rppg_local_energy
        5. rppg_fft_peak_score
    """

    feature_names = (
        "rppg_mean_g",
        "rppg_delta_g",
        "rppg_cheek_g",
        "rppg_forehead_g",
        "rppg_local_energy",
        "rppg_fft_peak_score",
    )

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def _fft_peak_score(self, signal: np.ndarray) -> float:
        if len(signal) < 8:
            return 0.0
        centered = signal.astype(np.float32) - float(signal.mean())
        spectrum = np.abs(np.fft.rfft(centered))
        if len(spectrum) <= 2:
            return 0.0
        spectrum[0] = 0.0
        return float(spectrum.max() / (spectrum.sum() + self.eps))

    def per_frame(self, frames_rgb: np.ndarray) -> np.ndarray:
        frames = _as_float_rgb(frames_rgb)
        if frames.size == 0:
            return _empty_feature(0, len(self.feature_names))

        height = frames.shape[1]
        mean_g = frames[..., 1].mean(axis=(1, 2))
        delta_g = np.zeros_like(mean_g)
        if len(mean_g) > 1:
            delta_g[1:] = mean_g[1:] - mean_g[:-1]

        forehead_g = frames[:, : height // 3, :, 1].mean(axis=(1, 2))
        cheek_g = frames[:, height // 2 :, :, 1].mean(axis=(1, 2))

        local_energy = np.zeros_like(mean_g)
        for idx in range(len(mean_g)):
            left = max(idx - 2, 0)
            right = min(idx + 3, len(mean_g))
            local_energy[idx] = np.std(mean_g[left:right])

        fft_peak = self._fft_peak_score(mean_g)
        fft_peak_score = np.full_like(mean_g, fft_peak)

        return np.stack(
            [mean_g, delta_g, cheek_g, forehead_g, local_energy, fft_peak_score],
            axis=1,
        ).astype(np.float32)

    def aggregate(self, frames_rgb: np.ndarray) -> dict[str, float]:
        frames = _as_float_rgb(frames_rgb)
        if frames.size == 0:
            return {
                "rppg_mean_g": 0.0,
                "rppg_std_g": 0.0,
                "rppg_delta_g_abs_mean": 0.0,
                "rppg_fft_peak_score": 0.0,
                "rppg_cheek_std_g": 0.0,
                "rppg_forehead_std_g": 0.0,
            }

        per_frame = self.per_frame(frames)
        mean_g = per_frame[:, 0]
        delta_g = per_frame[:, 1]
        cheek_g = per_frame[:, 2]
        forehead_g = per_frame[:, 3]

        return {
            "rppg_mean_g": float(mean_g.mean()),
            "rppg_std_g": float(mean_g.std()),
            "rppg_delta_g_abs_mean": float(np.abs(delta_g).mean()),
            "rppg_fft_peak_score": float(per_frame[0, 5]) if len(per_frame) else 0.0,
            "rppg_cheek_std_g": float(cheek_g.std()),
            "rppg_forehead_std_g": float(forehead_g.std()),
        }


class DepthNormalFeatureLoader:
    """Optional offline depth/normal cues.

    Role in V2:
        Depth/normal was not part of the completed val/test cue analysis because
        no depth/normal files were generated. The loader is included so the next
        model version can consume precomputed geometry features without running
        a depth model inside DataLoader workers.

    Expected files:
        depth_path:  T x H x W npy
        normal_path: T x H x W x 3 npy

    Output:
        T x 6 with depth/normal summary features.
    """

    feature_names = (
        "depth_mean",
        "depth_std",
        "depth_range",
        "normal_strength",
        "planarity_score",
        "depth_cv",
    )

    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps

    def per_frame(self, depth_path: str | Path | None, normal_path: str | Path | None, target_len: int) -> np.ndarray:
        if not depth_path or not normal_path:
            return _empty_feature(target_len, len(self.feature_names))

        depth_file = Path(depth_path)
        normal_file = Path(normal_path)
        if not depth_file.exists() or not normal_file.exists():
            return _empty_feature(target_len, len(self.feature_names))

        depth = np.load(depth_file).astype(np.float32)
        normal = np.load(normal_file).astype(np.float32)
        depth = _match_length(depth, target_len)
        normal = _match_length(normal, target_len)

        depth_mean = depth.mean(axis=(1, 2))
        depth_std = depth.std(axis=(1, 2))
        depth_range = depth.max(axis=(1, 2)) - depth.min(axis=(1, 2))
        normal_mean = normal.mean(axis=(1, 2))
        normal_strength = np.linalg.norm(normal_mean, axis=1)
        planarity_score = 1.0 / (depth_std + self.eps)
        depth_cv = depth_std / (depth_mean + self.eps)

        return np.stack(
            [depth_mean, depth_std, depth_range, normal_strength, planarity_score, depth_cv],
            axis=1,
        ).astype(np.float32)


class PhysicalCueExtractor:
    """Unified physical-cue extractor for V2 flash-liveness training.

    Usage in `flash_liveness_project_v2.py`:
        physical_extractor = PhysicalCueExtractor()
        physical_np = physical_extractor.per_frame(frames, color_values)
        tensor_physical = torch.from_numpy(physical_np).float()

    The resulting tensor shape is:
        T x physical_dim

    Recommended fusion:
        physical_emb = physical_proj(tensor_physical)
        features = pos_encoder(cnn_features + color_emb + physical_emb)
    """

    def __init__(self, config: PhysicalFeatureConfig | None = None) -> None:
        self.config = config or PhysicalFeatureConfig()
        self.frequency = FrequencyArtifactExtractor(eps=self.config.eps)
        self.flash_response = FlashResponseFeatureExtractor(eps=self.config.eps)
        self.rppg = RPPGFeatureExtractor(eps=self.config.eps)
        self.depth_normal = DepthNormalFeatureLoader(eps=self.config.eps)

    @property
    def feature_names(self) -> tuple[str, ...]:
        names: list[str] = []
        if self.config.use_frequency:
            names.extend(FrequencyArtifactExtractor.feature_names)
        if self.config.use_flash_response:
            names.extend(FlashResponseFeatureExtractor.feature_names)
        if self.config.use_rppg:
            names.extend(RPPGFeatureExtractor.feature_names)
        if self.config.use_depth_normal:
            names.extend(DepthNormalFeatureLoader.feature_names)
        return tuple(names)

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def per_frame(
        self,
        frames_rgb: np.ndarray,
        color_values: np.ndarray | None = None,
        depth_path: str | Path | None = None,
        normal_path: str | Path | None = None,
    ) -> np.ndarray:
        frames = _as_float_rgb(frames_rgb)
        num_frames = len(frames)
        color_values = np.zeros((num_frames,), dtype=np.int64) if color_values is None else np.asarray(color_values)

        features: list[np.ndarray] = []
        if self.config.use_frequency:
            features.append(self.frequency.per_frame(frames))
        if self.config.use_flash_response:
            features.append(self.flash_response.per_frame(frames, color_values))
        if self.config.use_rppg:
            features.append(self.rppg.per_frame(frames))
        if self.config.use_depth_normal:
            features.append(self.depth_normal.per_frame(depth_path, normal_path, num_frames))

        if not features:
            return _empty_feature(num_frames, 0)
        return np.concatenate(features, axis=1).astype(np.float32)

    def aggregate(
        self,
        frames_rgb: np.ndarray,
        color_values: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Return video-level features for reports and sanity checks."""

        frames = _as_float_rgb(frames_rgb)
        num_frames = len(frames)
        color_values = np.zeros((num_frames,), dtype=np.int64) if color_values is None else np.asarray(color_values)

        result: dict[str, float] = {}
        if self.config.use_frequency:
            result.update(self.frequency.aggregate(frames))
        if self.config.use_flash_response:
            result.update(self.flash_response.aggregate(frames, color_values))
        if self.config.use_rppg:
            result.update(self.rppg.aggregate(frames))
        return result


__all__ = [
    "DepthNormalFeatureLoader",
    "FlashResponseFeatureExtractor",
    "FrequencyArtifactExtractor",
    "PhysicalCueExtractor",
    "PhysicalFeatureConfig",
    "RPPGFeatureExtractor",
]
