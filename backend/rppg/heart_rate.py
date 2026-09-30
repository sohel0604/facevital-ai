"""
FaceVital AI — Heart Rate Estimation
======================================
Compute heart rate from BVP/rPPG signal using FFT peak detection.
"""

import numpy as np
from typing import Dict, Optional, Tuple

from backend.rppg.filters import bandpass_filter, compute_psd


def estimate_heart_rate(
    bvp_signal: np.ndarray,
    fps: float,
    min_hr: float = 40.0,
    max_hr: float = 180.0,
    nfft: int = 2048,
    method: str = "fft",
) -> Dict:
    """
    Estimate heart rate from a BVP signal.

    Parameters
    ----------
    bvp_signal : 1-D array of BVP values.
    fps : Sampling rate in Hz.
    min_hr : Minimum plausible HR in bpm.
    max_hr : Maximum plausible HR in bpm.
    nfft : FFT resolution.
    method : "fft" (frequency domain) or "peak" (time domain).

    Returns
    -------
    Dict with keys: hr_bpm, confidence, method, frequency_hz.
    """
    if len(bvp_signal) < int(fps * 2):
        return {
            "hr_bpm": None,
            "confidence": 0.0,
            "method": method,
            "frequency_hz": None,
            "error": "Signal too short for HR estimation",
        }

    min_hz = min_hr / 60.0
    max_hz = max_hr / 60.0

    if method == "fft":
        return _hr_from_fft(bvp_signal, fps, min_hz, max_hz, nfft)
    elif method == "peak":
        return _hr_from_peaks(bvp_signal, fps, min_hz, max_hz)
    else:
        return _hr_from_fft(bvp_signal, fps, min_hz, max_hz, nfft)


def _hr_from_fft(
    bvp: np.ndarray,
    fps: float,
    min_hz: float,
    max_hz: float,
    nfft: int,
) -> Dict:
    """Frequency-domain HR estimation using Welch PSD."""
    freqs, psd = compute_psd(bvp, fps, nfft=nfft)

    # Restrict to physiological range
    mask = (freqs >= min_hz) & (freqs <= max_hz)
    if not np.any(mask):
        return {
            "hr_bpm": None,
            "confidence": 0.0,
            "method": "fft",
            "frequency_hz": None,
            "error": "No power in physiological band",
        }

    band_freqs = freqs[mask]
    band_psd = psd[mask]

    # Peak frequency
    peak_idx = np.argmax(band_psd)
    peak_freq = band_freqs[peak_idx]
    hr_bpm = peak_freq * 60.0

    # Confidence: peak sharpness relative to band power
    peak_power = band_psd[peak_idx]
    mean_power = np.mean(band_psd) + 1e-10
    total_power = np.sum(psd) + 1e-10
    band_power = np.sum(band_psd)

    # Sharpness ratio
    sharpness = peak_power / mean_power
    # Band concentration
    concentration = band_power / total_power

    confidence = float(np.clip(
        0.3 * np.clip(sharpness / 5.0, 0, 1) +
        0.4 * np.clip(concentration, 0, 1) +
        0.3 * np.clip(len(bvp) / (fps * 10), 0, 1),  # Signal length factor
        0, 1,
    ))

    return {
        "hr_bpm": float(round(hr_bpm, 1)),
        "confidence": round(confidence, 3),
        "method": "fft",
        "frequency_hz": float(round(peak_freq, 4)),
    }


def _hr_from_peaks(
    bvp: np.ndarray,
    fps: float,
    min_hz: float,
    max_hz: float,
) -> Dict:
    """Time-domain HR estimation using peak detection."""
    from scipy.signal import find_peaks

    min_distance = int(fps / max_hz)
    max_distance = int(fps / min_hz)

    peaks, properties = find_peaks(
        bvp,
        distance=max(1, min_distance),
        prominence=0.1 * np.std(bvp),
    )

    if len(peaks) < 2:
        return {
            "hr_bpm": None,
            "confidence": 0.0,
            "method": "peak",
            "frequency_hz": None,
            "error": "Insufficient peaks detected",
        }

    # Inter-beat intervals
    ibis = np.diff(peaks) / fps  # in seconds
    # Filter out physiologically implausible intervals
    valid_mask = (ibis >= 1.0 / max_hz) & (ibis <= 1.0 / min_hz)
    valid_ibis = ibis[valid_mask]

    if len(valid_ibis) < 1:
        return {
            "hr_bpm": None,
            "confidence": 0.0,
            "method": "peak",
            "frequency_hz": None,
            "error": "No valid inter-beat intervals",
        }

    mean_ibi = np.mean(valid_ibis)
    hr_bpm = 60.0 / mean_ibi
    freq_hz = 1.0 / mean_ibi

    # Confidence based on IBI regularity
    ibi_cv = np.std(valid_ibis) / (mean_ibi + 1e-8)
    regularity = np.clip(1.0 - ibi_cv * 5, 0, 1)
    peak_ratio = len(valid_ibis) / (len(ibis) + 1e-8)

    confidence = float(np.clip(
        0.5 * regularity + 0.3 * peak_ratio +
        0.2 * np.clip(len(valid_ibis) / 5, 0, 1),
        0, 1,
    ))

    return {
        "hr_bpm": float(round(hr_bpm, 1)),
        "confidence": round(confidence, 3),
        "method": "peak",
        "frequency_hz": float(round(freq_hz, 4)),
    }


def compute_hr_windowed(
    bvp_signal: np.ndarray,
    fps: float,
    window_seconds: float = 10.0,
    overlap: float = 0.5,
    min_hr: float = 40.0,
    max_hr: float = 180.0,
) -> Dict:
    """
    Compute HR over sliding windows and return the smoothed estimate.

    Returns
    -------
    Dict with: hr_bpm, confidence, window_hrs, timestamp.
    """
    window_frames = int(window_seconds * fps)
    step_frames = int(window_frames * (1.0 - overlap))
    if step_frames < 1:
        step_frames = 1

    n_frames = len(bvp_signal)
    if n_frames < window_frames:
        # Use the full signal
        return estimate_heart_rate(bvp_signal, fps, min_hr, max_hr)

    window_hrs = []
    window_confs = []

    for start in range(0, n_frames - window_frames + 1, step_frames):
        end = start + window_frames
        segment = bvp_signal[start:end]
        result = estimate_heart_rate(segment, fps, min_hr, max_hr)
        if result["hr_bpm"] is not None:
            window_hrs.append(result["hr_bpm"])
            window_confs.append(result["confidence"])

    if not window_hrs:
        return {
            "hr_bpm": None,
            "confidence": 0.0,
            "method": "windowed_fft",
            "window_count": 0,
            "error": "No valid windows",
        }

    # Weighted average by confidence
    weights = np.array(window_confs)
    weight_sum = weights.sum()
    if weight_sum < 1e-8:
        hr_avg = np.mean(window_hrs)
        conf_avg = 0.0
    else:
        hr_avg = np.average(window_hrs, weights=weights)
        conf_avg = np.mean(window_confs)

    return {
        "hr_bpm": float(round(hr_avg, 1)),
        "confidence": float(round(conf_avg, 3)),
        "method": "windowed_fft",
        "window_count": len(window_hrs),
        "window_hrs": [round(h, 1) for h in window_hrs],
    }
