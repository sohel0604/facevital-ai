"""
FaceVital AI — Signal Quality Tests
=====================================
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from backend.rppg.signal_quality import assess_signal_quality


class TestSignalQuality:
    def test_good_signal_high_quality(self):
        """Well-lit, stable signal should get high quality."""
        n = 300
        rng = np.random.RandomState(42)
        t = np.arange(n) / 30.0
        ppg = np.sin(2 * np.pi * 1.2 * t)

        rgb_signals = {
            "forehead": np.stack([
                130 + 0.3 * ppg + 0.1 * rng.randn(n),
                140 + ppg + 0.1 * rng.randn(n),
                120 + 0.2 * ppg + 0.1 * rng.randn(n),
            ], axis=1),
            "left_cheek": np.stack([
                128 + 0.3 * ppg + 0.1 * rng.randn(n),
                138 + ppg + 0.1 * rng.randn(n),
                118 + 0.2 * ppg + 0.1 * rng.randn(n),
            ], axis=1),
        }

        quality, components = assess_signal_quality(rgb_signals, fps=30.0, bvp_signal=ppg)
        assert 0.0 <= quality <= 1.0
        assert quality > 0.3  # Should be decent quality
        assert "brightness" in components
        assert "temporal_stability" in components

    def test_dark_signal_low_quality(self):
        """Very dark signal should get lower brightness quality."""
        n = 300
        rgb_signals = {
            "forehead": np.ones((n, 3)) * 10,  # Very dark
        }
        quality, components = assess_signal_quality(rgb_signals, fps=30.0)
        assert components["brightness"] < 0.5

    def test_noisy_signal_lower_quality(self):
        """Very noisy signal should get lower quality."""
        n = 300
        rng = np.random.RandomState(42)
        rgb_signals = {
            "forehead": 140 + 50 * rng.randn(n, 3),  # Very noisy
        }
        quality, components = assess_signal_quality(rgb_signals, fps=30.0)
        assert quality < 0.8  # Should be penalized

    def test_quality_in_range(self):
        """Quality should always be in [0, 1]."""
        rng = np.random.RandomState(42)
        for _ in range(10):
            n = rng.randint(60, 600)
            signals = {"roi": rng.rand(n, 3) * 255}
            quality, _ = assess_signal_quality(signals, fps=30.0)
            assert 0.0 <= quality <= 1.0
