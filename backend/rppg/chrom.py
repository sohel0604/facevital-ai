"""
FaceVital AI — CHROM rPPG Algorithm
=====================================
Chrominance-based remote PPG method.

Reference:
    de Haan, G., & Jeanne, V. (2013).
    Robust Pulse Rate From Chrominance-Based rPPG.
    IEEE Transactions on Biomedical Engineering, 60(10), 2878–2886.
"""

import numpy as np
from typing import Optional

from backend.rppg.filters import bandpass_filter, detrend_signal, normalize_signal


def chrom_rppg(
    rgb_signals: np.ndarray,
    fps: float,
    window_length: int = 45,
    low_hz: float = 0.7,
    high_hz: float = 3.0,
) -> np.ndarray:
    """
    Extract rPPG signal using the CHROM algorithm.

    Parameters
    ----------
    rgb_signals : ndarray of shape (N, 3)
        Temporal RGB signal (N frames, 3 channels: R, G, B).
    fps : float
        Frame rate.
    window_length : int
        Sliding window length in frames.
    low_hz : float
        Low cutoff for band-pass filter.
    high_hz : float
        High cutoff for band-pass filter.

    Returns
    -------
    bvp : ndarray of shape (N,)
        Blood Volume Pulse (BVP) signal.
    """
    n_frames = rgb_signals.shape[0]
    if n_frames < window_length:
        window_length = n_frames

    bvp = np.zeros(n_frames)

    for start in range(n_frames - window_length + 1):
        end = start + window_length
        window = rgb_signals[start:end, :]  # (W, 3)

        # Temporal normalization
        means = window.mean(axis=0)
        if np.any(means < 1e-6):
            continue
        normed = window / means  # (W, 3)

        # CHROM projections
        # X = 3R - 2G
        # Y = 1.5R + G - 1.5B
        x = 3.0 * normed[:, 0] - 2.0 * normed[:, 1]
        y = 1.5 * normed[:, 0] + normed[:, 1] - 1.5 * normed[:, 2]

        # Standard deviation ratio for combination
        std_x = np.std(x)
        std_y = np.std(y)

        if std_y < 1e-8:
            continue

        alpha = std_x / std_y
        pulse = x - alpha * y

        # Overlap-add
        pulse = pulse - np.mean(pulse)
        bvp[start:end] += pulse

    # Post-processing
    bvp = detrend_signal(bvp)
    bvp = bandpass_filter(bvp, fps, low_hz, high_hz)
    bvp = normalize_signal(bvp)

    return bvp
