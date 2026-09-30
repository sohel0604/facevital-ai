#!/usr/bin/env python3
"""
FaceVital AI — Research Model Training Script
==============================================
Trains the MultiTaskBiomarkerModel (Temporal 1D-CNN) on synthetic calibration
datasets and saves the model checkpoint to models/biomarker_model.pth.
"""

import os
import sys
from pathlib import Path
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.ml.biomarker_model import MultiTaskBiomarkerModel


def generate_synthetic_dataset(n_samples: int = 500, n_frames: int = 300):
    """
    Generate synthetic dataset matching physiological relationships:
    - SBP: 90 - 150 mmHg
    - DBP: 60 - 95 mmHg
    - Glucose: 70 - 140 mg/dL
    - Cholesterol: 150 - 240 mg/dL
    """
    np.random.seed(42)
    torch.manual_seed(42)

    X_list = []
    y_sbp_list = []
    y_dbp_list = []
    y_glu_list = []
    y_chol_list = []

    for _ in range(n_samples):
        # Base physiological parameters
        sbp = np.random.uniform(105, 140)
        dbp = sbp * np.random.uniform(0.62, 0.68)
        glucose = np.random.uniform(80, 130)
        cholesterol = np.random.uniform(160, 220)

        # Pulse wave properties correlated with arterial stiffness
        hr_bpm = np.random.uniform(55, 95)
        pulse_freq = hr_bpm / 60.0
        t = np.linspace(0, 10, n_frames, endpoint=False)

        # Higher BP -> faster pulse wave transit & sharper systolic peak
        stiffness_factor = (sbp - 100) / 40.0
        wave = (
            np.sin(2 * np.pi * pulse_freq * t)
            + 0.3 * stiffness_factor * np.sin(4 * np.pi * pulse_freq * t)
        )

        # 9 channels: 3 ROIs x 3 RGB
        channels = []
        for r in range(3):
            roi_offset = 140.0 + r * 5.0
            r_chan = roi_offset + wave * 5.0 + np.random.normal(0, 0.5, n_frames)
            g_chan = roi_offset - 20.0 + wave * 12.0 + np.random.normal(0, 0.5, n_frames)
            b_chan = roi_offset - 30.0 + wave * 4.0 + np.random.normal(0, 0.5, n_frames)
            channels.extend([r_chan, g_chan, b_chan])

        x = np.array(channels, dtype=np.float32)  # shape (9, n_frames)
        # Normalize per sample
        x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)

        X_list.append(x)
        y_sbp_list.append(sbp)
        y_dbp_list.append(dbp)
        y_glu_list.append(glucose)
        y_chol_list.append(cholesterol)

    X_tensor = torch.tensor(np.array(X_list), dtype=torch.float32)
    y_sbp = torch.tensor(y_sbp_list, dtype=torch.float32)
    y_dbp = torch.tensor(y_dbp_list, dtype=torch.float32)
    y_glu = torch.tensor(y_glu_list, dtype=torch.float32)
    y_chol = torch.tensor(y_chol_list, dtype=torch.float32)

    return TensorDataset(X_tensor, y_sbp, y_dbp, y_glu, y_chol)


def train_model(epochs: int = 25, batch_size: int = 32):
    print("=" * 60)
    print("FaceVital AI — Training Research Multi-Task Biomarker Model")
    print("=" * 60)

    dataset = generate_synthetic_dataset(n_samples=400, n_frames=300)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = MultiTaskBiomarkerModel(
        n_roi_channels=9,
        hidden_channels=64,
        n_temporal_blocks=4,
        embedding_dim=128,
    )

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    print(f"\nTraining on {train_size} samples, validating on {val_size} samples...")
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X, y_sbp, y_dbp, y_glu, y_chol in train_loader:
            optimizer.zero_grad()
            preds = model(X)
            loss_sbp = criterion(preds["sbp"], y_sbp)
            loss_dbp = criterion(preds["dbp"], y_dbp)
            loss_glu = criterion(preds["glucose"], y_glu)
            loss_chol = criterion(preds["cholesterol"], y_chol)

            # Combined multi-task loss with balanced weighting
            total_loss = loss_sbp + loss_dbp + 0.5 * loss_glu + 0.5 * loss_chol
            total_loss.backward()
            optimizer.step()
            train_loss += total_loss.item() * len(X)

        train_loss /= len(train_set)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y_sbp, y_dbp, y_glu, y_chol in val_loader:
                preds = model(X)
                l = (
                    criterion(preds["sbp"], y_sbp)
                    + criterion(preds["dbp"], y_dbp)
                    + 0.5 * criterion(preds["glucose"], y_glu)
                    + 0.5 * criterion(preds["cholesterol"], y_chol)
                )
                val_loss += l.item() * len(X)
        val_loss /= len(val_set)

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f}")

    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f}s.")

    # Save model weights and scaler params in models/weights
    weights_dir = PROJECT_ROOT / "models" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = weights_dir / "biomarker_model.pt"

    torch.save(model.state_dict(), checkpoint_path)

    # Save scaler params
    scaler_params = {
        "means": [140.0] * 9,
        "stds": [10.0] * 9,
        "sbp_mean": 120.0,
        "sbp_std": 15.0,
        "dbp_mean": 80.0,
        "dbp_std": 10.0,
        "glucose_mean": 100.0,
        "glucose_std": 20.0,
        "cholesterol_mean": 190.0,
        "cholesterol_std": 30.0,
    }
    with open(weights_dir / "scaler_params.json", "w") as f:
        import json
        json.dump(scaler_params, f, indent=2)

    print(f"Saved model checkpoint to: {checkpoint_path}")
    print(f"Saved scaler params to: {weights_dir / 'scaler_params.json'}")
    print("=" * 60)


if __name__ == "__main__":
    train_model(epochs=15, batch_size=32)
