"""
FaceVital AI — Signal Quality Assessment
==========================================
Multi-factor signal quality scoring for rPPG signals.
"""

import numpy as np
from typing import Dict, Tuple

from backend.rppg.filters import bandpass_filter, compute_psd


def assess_signal_quality(
    rgb_signals: Dict[str, np.ndarray],
    fps: float,
    bvp_signal: np.ndarray = None,
    low_hz: float = 0.7,
    high_hz: float = 3.0,
) -> Tuple[float, Dict[str, float]]:
    """
    Compute a composite signal quality score from multiple factors.

    Parameters
    ----------
    rgb_signals : Dict of ROI name → ndarray shape (N, 3).
    fps : Frame rate.
    bvp_signal : Optional extracted BVP signal for periodicity assessment.
    low_hz, high_hz : Physiological frequency band.

    Returns
    -------
    quality : float in [0, 1].
    components : Dict of individual quality component scores.
    """
    components = {}

    # 1. Brightness quality — ROIs should have reasonable mean brightness
    brightness_scores = []
    for name, arr in rgb_signals.items():
        mean_brightness = arr.mean()
        # Optimal range: 50-200 (on 0-255 scale)
        if mean_brightness < 30:
            score = mean_brightness / 30.0
        elif mean_brightness > 240:
            score = (255 - mean_brightness) / 15.0
        elif 50 <= mean_brightness <= 200:
            score = 1.0
        else:
            score = 0.7
        brightness_scores.append(np.clip(score, 0, 1))
    components["brightness"] = float(np.mean(brightness_scores))

    # 2. Temporal stability — low motion = low variance in ROI means
    stability_scores = []
    for name, arr in rgb_signals.items():
        frame_means = arr.mean(axis=1)  # Mean brightness per frame
        cv = np.std(frame_means) / (np.mean(frame_means) + 1e-8)
        # Lower CV = more stable
        score = np.clip(1.0 - cv * 10, 0, 1)
        stability_scores.append(score)
    components["temporal_stability"] = float(np.mean(stability_scores))

    # 3. ROI consistency — all ROIs should be similar
    if len(rgb_signals) >= 2:
        roi_means = [arr.mean() for arr in rgb_signals.values()]
        consistency = 1.0 - (np.std(roi_means) / (np.mean(roi_means) + 1e-8))
        components["roi_consistency"] = float(np.clip(consistency, 0, 1))
    else:
        components["roi_consistency"] = 0.5

    # 4. Signal-to-noise ratio (SNR-like measure)
    snr_scores = []
    for name, arr in rgb_signals.items():
        green = arr[:, 1]  # Green channel
        if len(green) > 30:
            filtered = bandpass_filter(green, fps, low_hz, high_hz)
            signal_power = np.var(filtered)
            noise_power = np.var(green - filtered)
            if noise_power > 1e-8:
                snr = signal_power / noise_power
                score = np.clip(snr * 2, 0, 1)  # Scale to [0,1]
            else:
                score = 1.0
            snr_scores.append(score)
    if snr_scores:
        components["snr"] = float(np.mean(snr_scores))
    else:
        components["snr"] = 0.3

    # 5. Pulse periodicity (if BVP signal provided)
    if bvp_signal is not None and len(bvp_signal) > 60:
        freqs, psd = compute_psd(bvp_signal, fps)
        # Focus on physiological band
        mask = (freqs >= low_hz) & (freqs <= high_hz)
        if np.any(mask):
            band_psd = psd[mask]
            total_psd = np.sum(psd) + 1e-8
            band_power_ratio = np.sum(band_psd) / total_psd
            # Peak sharpness: ratio of peak to mean
            peak_ratio = np.max(band_psd) / (np.mean(band_psd) + 1e-8)
            periodicity = np.clip(band_power_ratio * 2, 0, 1) * np.clip(peak_ratio / 10, 0, 1)
            components["periodicity"] = float(np.clip(periodicity, 0, 1))
        else:
            components["periodicity"] = 0.2
    else:
        components["periodicity"] = 0.3

    # 6. Signal length quality — longer is better (up to a point)
    n_frames = min(arr.shape[0] for arr in rgb_signals.values())
    min_good_frames = fps * 6  # 6 seconds
    length_score = np.clip(n_frames / min_good_frames, 0, 1)
    components["signal_length"] = float(length_score)

    # Composite score — weighted combination
    weights = {
        "brightness": 0.15,
        "temporal_stability": 0.2,
        "roi_consistency": 0.1,
        "snr": 0.25,
        "periodicity": 0.2,
        "signal_length": 0.1,
    }

    quality = sum(
        components.get(k, 0.0) * w for k, w in weights.items()
    )
    quality = float(np.clip(quality, 0, 1))

    return quality, components
