"""
FaceVital AI — ROI Signal Processing
======================================
Utilities for combining and preparing multi-ROI temporal signals.
"""

import numpy as np
from typing import Dict, List, Optional


def parse_roi_signals(roi_dict: Dict[str, List[List[float]]]) -> Dict[str, np.ndarray]:
    """
    Convert raw ROI signal lists to numpy arrays.

    Parameters
    ----------
    roi_dict : Dict mapping ROI name to list of [R, G, B] per frame.

    Returns
    -------
    Dict mapping ROI name to ndarray of shape (N, 3).
    """
    result = {}
    for name, frames in roi_dict.items():
        arr = np.array(frames, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(
                f"ROI '{name}' must have shape (N, 3), got {arr.shape}"
            )
        result[name] = arr
    return result


def combine_roi_signals(
    roi_arrays: Dict[str, np.ndarray],
    method: str = "mean",
) -> np.ndarray:
    """
    Combine multiple ROI signals into a single temporal signal.

    Parameters
    ----------
    roi_arrays : Dict of ROI name → ndarray of shape (N, 3).
    method : "mean" averages all ROIs; "weighted" uses inverse-variance weighting.

    Returns
    -------
    Combined signal of shape (N, 3).
    """
    arrays = list(roi_arrays.values())
    if not arrays:
        raise ValueError("No ROI signals provided")

    # Ensure all same length — truncate to shortest
    min_len = min(a.shape[0] for a in arrays)
    arrays = [a[:min_len, :] for a in arrays]

    if method == "mean":
        return np.mean(arrays, axis=0)
    elif method == "weighted":
        # Inverse-variance weighting per channel
        stds = [np.std(a, axis=0) for a in arrays]
        weights = []
        for s in stds:
            w = np.where(s > 1e-8, 1.0 / s, 1.0)
            weights.append(w)
        weight_sum = np.sum(weights, axis=0)
        combined = np.zeros_like(arrays[0])
        for a, w in zip(arrays, weights):
            combined += a * (w / weight_sum)
        return combined
    else:
        return np.mean(arrays, axis=0)


def extract_green_channel(rgb_signal: np.ndarray) -> np.ndarray:
    """Extract the green channel, which carries the strongest PPG signal."""
    return rgb_signal[:, 1]


def compute_roi_statistics(roi_arrays: Dict[str, np.ndarray]) -> Dict[str, Dict]:
    """Compute basic statistics for each ROI for quality assessment."""
    stats = {}
    for name, arr in roi_arrays.items():
        stats[name] = {
            "mean_rgb": arr.mean(axis=0).tolist(),
            "std_rgb": arr.std(axis=0).tolist(),
            "min_rgb": arr.min(axis=0).tolist(),
            "max_rgb": arr.max(axis=0).tolist(),
            "n_frames": arr.shape[0],
        }
    return stats
