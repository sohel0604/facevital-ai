"""
FaceVital AI — rPPG Pipeline Tests
====================================
Unit tests for the rPPG signal processing pipeline.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from backend.rppg.filters import (
    bandpass_filter,
    detrend_signal,
    normalize_signal,
    moving_average,
    compute_psd,
)
from backend.rppg.pos import pos_rppg
from backend.rppg.chrom import chrom_rppg
from backend.rppg.roi import parse_roi_signals, combine_roi_signals
from backend.rppg.heart_rate import estimate_heart_rate, compute_hr_windowed


# ---------------------------------------------------------------------------
# Synthetic test signal helpers
# ---------------------------------------------------------------------------

def generate_synthetic_ppg(
    fps: float = 30.0,
    duration: float = 10.0,
    hr_bpm: float = 72.0,
    noise_level: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Generate a synthetic PPG-like signal at a known heart rate."""
    rng = np.random.RandomState(seed)
    n = int(fps * duration)
    t = np.arange(n) / fps

    freq = hr_bpm / 60.0
    # Primary pulse wave
    signal = np.sin(2 * np.pi * freq * t)
    # Add harmonics
    signal += 0.3 * np.sin(2 * np.pi * 2 * freq * t)
    # Add noise
    signal += noise_level * rng.randn(n)

    return signal


def generate_synthetic_rgb(
    fps: float = 30.0,
    duration: float = 10.0,
    hr_bpm: float = 72.0,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic RGB ROI signals with embedded PPG."""
    rng = np.random.RandomState(seed)
    n = int(fps * duration)
    t = np.arange(n) / fps

    freq = hr_bpm / 60.0

    # Base skin color (approximately)
    r_base, g_base, b_base = 180.0, 140.0, 120.0

    # PPG modulation (primarily in green channel)
    ppg = 0.5 * np.sin(2 * np.pi * freq * t) + 0.15 * np.sin(2 * np.pi * 2 * freq * t)

    r = r_base + 0.3 * ppg + 0.2 * rng.randn(n)
    g = g_base + 1.0 * ppg + 0.2 * rng.randn(n)
    b = b_base + 0.2 * ppg + 0.2 * rng.randn(n)

    return np.stack([r, g, b], axis=1)


# ---------------------------------------------------------------------------
# Filter tests
# ---------------------------------------------------------------------------

class TestFilters:
    def test_bandpass_filter_preserves_signal_in_band(self):
        fps = 30.0
        n = 300
        t = np.arange(n) / fps
        # 1 Hz signal (60 bpm) — should pass through 0.7-3.0 Hz band
        sig = np.sin(2 * np.pi * 1.0 * t)
        filtered = bandpass_filter(sig, fps, low_hz=0.7, high_hz=3.0)
        # Signal should still have significant power
        assert np.std(filtered) > 0.3

    def test_bandpass_filter_removes_dc(self):
        fps = 30.0
        sig = np.ones(300) * 100  # DC signal
        filtered = bandpass_filter(sig, fps, low_hz=0.7, high_hz=3.0)
        assert np.abs(np.mean(filtered)) < 1.0

    def test_detrend_linear(self):
        sig = np.arange(100, dtype=float) + 50
        detrended = detrend_signal(sig, method="linear")
        assert abs(np.mean(detrended)) < 1e-10

    def test_normalize_signal(self):
        sig = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
        normed = normalize_signal(sig)
        assert abs(np.mean(normed)) < 1e-10
        assert abs(np.std(normed) - 1.0) < 1e-10

    def test_moving_average(self):
        sig = np.array([0, 0, 10, 0, 0], dtype=float)
        smoothed = moving_average(sig, window=3)
        assert len(smoothed) == len(sig)

    def test_compute_psd(self):
        fps = 30.0
        n = 300
        t = np.arange(n) / fps
        sig = np.sin(2 * np.pi * 1.5 * t)
        freqs, psd = compute_psd(sig, fps)
        # Peak should be near 1.5 Hz
        peak_freq = freqs[np.argmax(psd)]
        assert abs(peak_freq - 1.5) < 0.5

    def test_short_signal_filter(self):
        """Short signals should not crash."""
        sig = np.array([1.0, 2.0, 3.0])
        result = bandpass_filter(sig, 30.0)
        assert len(result) == 3


# ---------------------------------------------------------------------------
# rPPG algorithm tests
# ---------------------------------------------------------------------------

class TestPOS:
    def test_pos_output_shape(self):
        rgb = generate_synthetic_rgb(fps=30.0, duration=10.0)
        bvp = pos_rppg(rgb, fps=30.0)
        assert bvp.shape == (rgb.shape[0],)

    def test_pos_produces_nonzero_signal(self):
        rgb = generate_synthetic_rgb(fps=30.0, duration=10.0, hr_bpm=72.0)
        bvp = pos_rppg(rgb, fps=30.0)
        assert np.std(bvp) > 0.01

    def test_pos_with_short_signal(self):
        rgb = generate_synthetic_rgb(fps=30.0, duration=2.0)
        bvp = pos_rppg(rgb, fps=30.0)
        assert len(bvp) == rgb.shape[0]


class TestCHROM:
    def test_chrom_output_shape(self):
        rgb = generate_synthetic_rgb(fps=30.0, duration=10.0)
        bvp = chrom_rppg(rgb, fps=30.0)
        assert bvp.shape == (rgb.shape[0],)

    def test_chrom_produces_nonzero_signal(self):
        rgb = generate_synthetic_rgb(fps=30.0, duration=10.0, hr_bpm=72.0)
        bvp = chrom_rppg(rgb, fps=30.0)
        assert np.std(bvp) > 0.01


# ---------------------------------------------------------------------------
# Heart rate estimation tests
# ---------------------------------------------------------------------------

class TestHeartRate:
    def test_hr_from_synthetic_signal(self):
        """HR estimated from a clean synthetic signal should be close to ground truth."""
        target_hr = 72.0
        bvp = generate_synthetic_ppg(fps=30.0, duration=15.0, hr_bpm=target_hr, noise_level=0.05)
        result = estimate_heart_rate(bvp, fps=30.0)
        assert result["hr_bpm"] is not None
        # Should be within 5 bpm of the target
        assert abs(result["hr_bpm"] - target_hr) < 5.0

    def test_hr_confidence_range(self):
        bvp = generate_synthetic_ppg(fps=30.0, duration=10.0)
        result = estimate_heart_rate(bvp, fps=30.0)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_hr_too_short(self):
        bvp = np.array([0.1, 0.2, 0.3])
        result = estimate_heart_rate(bvp, fps=30.0)
        assert result["hr_bpm"] is None
        assert result["confidence"] == 0.0

    def test_hr_peak_method(self):
        target_hr = 80.0
        bvp = generate_synthetic_ppg(fps=30.0, duration=15.0, hr_bpm=target_hr, noise_level=0.05)
        result = estimate_heart_rate(bvp, fps=30.0, method="peak")
        if result["hr_bpm"] is not None:
            assert abs(result["hr_bpm"] - target_hr) < 10.0

    def test_windowed_hr(self):
        target_hr = 72.0
        bvp = generate_synthetic_ppg(fps=30.0, duration=30.0, hr_bpm=target_hr, noise_level=0.05)
        result = compute_hr_windowed(bvp, fps=30.0, window_seconds=10.0)
        assert result["hr_bpm"] is not None
        assert result.get("window_count", 0) > 1


# ---------------------------------------------------------------------------
# ROI tests
# ---------------------------------------------------------------------------

class TestROI:
    def test_parse_roi_signals(self):
        roi_dict = {
            "forehead": [[100, 120, 110]] * 60,
            "left_cheek": [[95, 115, 105]] * 60,
        }
        parsed = parse_roi_signals(roi_dict)
        assert "forehead" in parsed
        assert parsed["forehead"].shape == (60, 3)

    def test_parse_invalid_shape(self):
        roi_dict = {"forehead": [[100, 120]] * 60}  # Missing B channel
        with pytest.raises(ValueError):
            parse_roi_signals(roi_dict)

    def test_combine_rois_mean(self):
        parsed = {
            "a": np.ones((60, 3)) * 100,
            "b": np.ones((60, 3)) * 200,
        }
        combined = combine_roi_signals(parsed, method="mean")
        assert combined.shape == (60, 3)
        np.testing.assert_allclose(combined, 150.0)


# ---------------------------------------------------------------------------
# Integration: full rPPG pipeline
# ---------------------------------------------------------------------------

class TestFullPipeline:
    def test_end_to_end_pos(self):
        """Full pipeline: RGB → POS → HR."""
        target_hr = 72.0
        rgb = generate_synthetic_rgb(fps=30.0, duration=15.0, hr_bpm=target_hr, seed=42)
        bvp = pos_rppg(rgb, fps=30.0)
        result = estimate_heart_rate(bvp, fps=30.0)
        assert result["hr_bpm"] is not None
        # With synthetic data, should be reasonably close
        assert abs(result["hr_bpm"] - target_hr) < 10.0

    def test_end_to_end_chrom(self):
        """Full pipeline: RGB → CHROM → HR."""
        target_hr = 80.0
        rgb = generate_synthetic_rgb(fps=30.0, duration=15.0, hr_bpm=target_hr, seed=123)
        bvp = chrom_rppg(rgb, fps=30.0)
        result = estimate_heart_rate(bvp, fps=30.0)
        assert result["hr_bpm"] is not None
