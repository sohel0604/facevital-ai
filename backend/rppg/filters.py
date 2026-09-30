"""
FaceVital AI — Signal Processing Filters
==========================================
Band-pass, detrending, and normalization utilities for rPPG signals.
"""

import numpy as np
from scipy import signal as scipy_signal
from typing import Optional, Tuple


def bandpass_filter(
    sig: np.ndarray,
    fps: float,
    low_hz: float = 0.7,
    high_hz: float = 3.0,
    order: int = 4,
) -> np.ndarray:
    """
    Apply a Butterworth band-pass filter.

    Parameters
    ----------
    sig : 1-D array of the temporal signal.
    fps : Sampling rate in Hz.
    low_hz : Lower cutoff frequency (default 0.7 Hz ≈ 42 bpm).
    high_hz : Upper cutoff frequency (default 3.0 Hz ≈ 180 bpm).
    order : Filter order.

    Returns
    -------
    Filtered signal (same length).
    """
    nyquist = fps / 2.0
    if high_hz >= nyquist:
        high_hz = nyquist - 0.1
    if low_hz >= high_hz:
        return sig  # Cannot filter
    b, a = scipy_signal.butter(order, [low_hz / nyquist, high_hz / nyquist], btype="band")
    # Use filtfilt for zero-phase filtering
    if len(sig) < 3 * max(len(a), len(b)):
        return sig  # Too short to filter
    return scipy_signal.filtfilt(b, a, sig)


def detrend_signal(sig: np.ndarray, method: str = "linear") -> np.ndarray:
    """Remove linear or constant trend from the signal."""
    if method == "linear":
        return scipy_signal.detrend(sig, type="linear")
    elif method == "constant":
        return sig - np.mean(sig)
    return sig


def normalize_signal(sig: np.ndarray) -> np.ndarray:
    """Z-score normalization."""
    std = np.std(sig)
    if std < 1e-8:
        return sig - np.mean(sig)
    return (sig - np.mean(sig)) / std


def moving_average(sig: np.ndarray, window: int = 5) -> np.ndarray:
    """Simple moving average smoothing."""
    if window < 2 or len(sig) < window:
        return sig
    kernel = np.ones(window) / window
    return np.convolve(sig, kernel, mode="same")


def compute_psd(
    sig: np.ndarray,
    fps: float,
    nfft: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using Welch's method.

    Returns
    -------
    freqs, psd : Frequency axis and PSD values.
    """
    nperseg = min(len(sig), nfft)
    freqs, psd = scipy_signal.welch(sig, fs=fps, nperseg=nperseg, nfft=nfft)
    return freqs, psd
