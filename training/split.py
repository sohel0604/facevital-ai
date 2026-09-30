"""
FaceVital AI — Subject-Wise Data Splitting
============================================
CRITICAL: All splits are SUBJECT-WISE to prevent data leakage.
Overlapping temporal windows from the same subject NEVER appear
in different splits.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


def subject_wise_split(
    subject_ids: List[str],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[str]]:
    """
    Split subjects into train/val/test sets.

    IMPORTANT: This ensures that NO subject appears in more than one split.
    All windows/samples from a given subject go entirely into one split.

    Parameters
    ----------
    subject_ids : List of unique subject identifiers.
    train_ratio, val_ratio, test_ratio : Split proportions (must sum to ~1.0).
    seed : Random seed for reproducibility.

    Returns
    -------
    Dict with keys "train", "val", "test", each mapping to a list of subject IDs.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}"

    unique_subjects = sorted(set(subject_ids))
    n = len(unique_subjects)

    rng = np.random.RandomState(seed)
    indices = rng.permutation(n)

    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # n_test gets the remainder

    train_subjects = [unique_subjects[i] for i in indices[:n_train]]
    val_subjects = [unique_subjects[i] for i in indices[n_train:n_train + n_val]]
    test_subjects = [unique_subjects[i] for i in indices[n_train + n_val:]]

    # Verify no overlap
    assert len(set(train_subjects) & set(val_subjects)) == 0, "Train/val overlap!"
    assert len(set(train_subjects) & set(test_subjects)) == 0, "Train/test overlap!"
    assert len(set(val_subjects) & set(test_subjects)) == 0, "Val/test overlap!"

    return {
        "train": train_subjects,
        "val": val_subjects,
        "test": test_subjects,
    }


def get_samples_for_split(
    all_samples: List[Dict],
    split_subjects: Dict[str, List[str]],
    subject_key: str = "subject_id",
) -> Dict[str, List[Dict]]:
    """
    Assign samples to splits based on their subject.

    Parameters
    ----------
    all_samples : List of sample dicts, each containing subject_key.
    split_subjects : Output from subject_wise_split.
    subject_key : Key in sample dict that identifies the subject.

    Returns
    -------
    Dict with keys "train", "val", "test" → list of samples.
    """
    subject_to_split = {}
    for split_name, subjects in split_subjects.items():
        for s in subjects:
            subject_to_split[s] = split_name

    result = {"train": [], "val": [], "test": []}
    unassigned = 0

    for sample in all_samples:
        sid = sample.get(subject_key)
        split = subject_to_split.get(sid)
        if split:
            result[split].append(sample)
        else:
            unassigned += 1

    if unassigned > 0:
        import warnings
        warnings.warn(f"{unassigned} samples had unknown subject IDs and were excluded.")

    return result


def print_split_summary(
    split_subjects: Dict[str, List[str]],
    split_samples: Optional[Dict[str, List]] = None,
) -> str:
    """Generate a human-readable split summary."""
    lines = [
        "=" * 60,
        "SUBJECT-WISE SPLIT SUMMARY",
        "=" * 60,
        f"  Train subjects: {len(split_subjects['train'])}",
        f"  Val subjects:   {len(split_subjects['val'])}",
        f"  Test subjects:  {len(split_subjects['test'])}",
        f"  Total subjects: {sum(len(v) for v in split_subjects.values())}",
    ]

    if split_samples:
        lines.extend([
            "",
            f"  Train samples:  {len(split_samples['train'])}",
            f"  Val samples:    {len(split_samples['val'])}",
            f"  Test samples:   {len(split_samples['test'])}",
            f"  Total samples:  {sum(len(v) for v in split_samples.values())}",
        ])

    lines.append("=" * 60)
    lines.append("NOTE: train subjects ∩ val subjects ∩ test subjects = ∅")
    lines.append("=" * 60)

    summary = "\n".join(lines)
    print(summary)
    return summary
