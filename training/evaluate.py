"""
FaceVital AI — Evaluation Module
===================================
Compute metrics for HR and biomarker predictions.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def compute_regression_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Compute standard regression metrics.

    Returns: MAE, RMSE, R2, Pearson correlation.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    n = len(y_true)
    if n == 0:
        return {"mae": None, "rmse": None, "r2": None, "pearson_r": None, "n": 0}

    errors = y_pred - y_true
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 1e-8 else None

    if np.std(y_true) > 1e-8 and np.std(y_pred) > 1e-8:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = None

    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4) if r2 is not None else None,
        "pearson_r": round(pearson_r, 4) if pearson_r is not None else None,
        "n": n,
        "mean_error": round(float(np.mean(errors)), 4),
        "std_error": round(float(np.std(errors)), 4),
    }


def bland_altman_analysis(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Bland-Altman analysis: mean difference and limits of agreement.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    diff = y_pred - y_true
    mean_vals = (y_pred + y_true) / 2.0

    mean_diff = float(np.mean(diff))
    std_diff = float(np.std(diff, ddof=1)) if len(diff) > 1 else 0.0

    return {
        "mean_difference": round(mean_diff, 4),
        "std_difference": round(std_diff, 4),
        "upper_loa": round(mean_diff + 1.96 * std_diff, 4),
        "lower_loa": round(mean_diff - 1.96 * std_diff, 4),
        "n": len(y_true),
    }


def generate_evaluation_report(
    metrics: Dict[str, Dict],
    report_dir: str = "./reports",
    prefix: str = "",
) -> str:
    """
    Save evaluation metrics to JSON and generate markdown report.
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Save individual metric files
    for target, metric in metrics.items():
        path = report_dir / f"{prefix}{target}_metrics.json"
        with open(path, "w") as f:
            json.dump(metric, f, indent=2)

    # Generate markdown report
    report_lines = [
        "# FaceVital AI — Evaluation Report",
        "",
        "**IMPORTANT**: This report documents the evaluation framework.",
        "Actual metrics require training on labeled datasets.",
        "",
    ]

    for target, metric in metrics.items():
        report_lines.extend([
            f"## {target.upper()}",
            "",
        ])
        if metric.get("status") == "NOT_RUN":
            report_lines.append(f"**Status**: NOT RUN — {metric.get('reason', 'No data')}")
        else:
            report_lines.append(f"| Metric | Value |")
            report_lines.append(f"|--------|-------|")
            for k, v in metric.items():
                if k != "status":
                    report_lines.append(f"| {k} | {v} |")
        report_lines.append("")

    report_path = report_dir / f"{prefix}evaluation_report.md"
    report_text = "\n".join(report_lines)
    with open(report_path, "w") as f:
        f.write(report_text)

    return str(report_path)
