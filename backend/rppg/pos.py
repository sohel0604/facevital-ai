"""
FaceVital AI — POS rPPG Algorithm
==================================
Plane-Orthogonal-to-Skin (POS) algorithm for remote photoplethysmography.

Reference:
    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
    Algorithmic Principles of Remote PPG.
    IEEE Transactions on Biomedical Engineering, 64(7), 1479–1491.
"""

import numpy as np
from typing import Optional

from backend.rppg.filters import bandpass_filter, detrend_signal, normalize_signal


def pos_rppg(
    rgb_signals: np.ndarray,
    fps: float,
    window_length: int = 45,
    low_hz: float = 0.7,
    high_hz: float = 3.0,
) -> np.ndarray:
    """
    Extract rPPG signal using the POS algorithm.

    Parameters
    ----------
    rgb_signals : ndarray of shape (N, 3)
        Temporal RGB signal (N frames, 3 channels: R, G, B).
        Each row is the spatial average RGB of the ROI for one frame.
    fps : float
        Frame rate.
    window_length : int
        Sliding window length in frames (default ~1.5s at 30 fps).
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

    # Adjust window length to at least cover one pulse cycle
    if window_length < int(fps * 1.0):
        window_length = min(n_frames, int(fps * 1.5))

    bvp = np.zeros(n_frames)

    for start in range(n_frames - window_length + 1):
        end = start + window_length
        window = rgb_signals[start:end, :]  # (W, 3)

        # Temporal normalization: divide by mean of each channel
        means = window.mean(axis=0)
        if np.any(means < 1e-6):
            continue
        normed = window / means  # (W, 3)

        # POS projection
        # S1 = G - B, S2 = G + B - 2R
        s1 = normed[:, 1] - normed[:, 2]          # G - B
        s2 = normed[:, 1] + normed[:, 2] - 2.0 * normed[:, 0]  # G + B - 2R

        # Standard deviation ratio
        std_s1 = np.std(s1)
        std_s2 = np.std(s2)

        if std_s2 < 1e-8:
            continue

        alpha = std_s1 / std_s2
        pulse = s1 + alpha * s2

        # Overlap-add
        pulse = pulse - np.mean(pulse)
        bvp[start:end] += pulse

    # Post-processing
    bvp = detrend_signal(bvp)
    bvp = bandpass_filter(bvp, fps, low_hz, high_hz)
    bvp = normalize_signal(bvp)

    return bvp
