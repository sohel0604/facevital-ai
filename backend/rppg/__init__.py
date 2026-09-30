"""
FaceVital AI — rPPG Core Processing Package
"""

from backend.rppg.pos import pos_rppg
from backend.rppg.chrom import chrom_rppg
from backend.rppg.heart_rate import estimate_heart_rate
from backend.rppg.signal_quality import assess_signal_quality
from backend.rppg.filters import bandpass_filter, detrend_signal
from backend.rppg.roi import parse_roi_signals, combine_roi_signals

# Friendly aliases
pos = pos_rppg
chrom = chrom_rppg
extract_heart_rate = estimate_heart_rate
compute_signal_quality = assess_signal_quality
combine_rois = combine_roi_signals

__all__ = [
    "pos",
    "chrom",
    "pos_rppg",
    "chrom_rppg",
    "estimate_heart_rate",
    "extract_heart_rate",
    "assess_signal_quality",
    "compute_signal_quality",
    "bandpass_filter",
    "detrend_signal",
    "parse_roi_signals",
    "combine_roi_signals",
    "combine_rois",
]
