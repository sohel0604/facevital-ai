"""
FaceVital AI — Multi-Task Biomarker Model
===========================================
PyTorch model for estimating BP, glucose, and cholesterol from
temporal facial ROI signals.

IMPORTANT: This is a research prototype architecture. The model requires
training on actual labeled data (e.g., MCD-rPPG dataset) to produce
meaningful predictions. Without trained weights, it operates as a
PLACEHOLDER that returns structured outputs with zero confidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class TemporalBlock(nn.Module):
    """
    A residual temporal convolution block.
    Conv1D → BatchNorm → ReLU → Conv1D → BatchNorm → Residual → ReLU
    """

    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1):
        super().__init__()
        padding = (kernel_size - 1) * dilation // 2
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # Adjust for any length mismatch from padding
        if out.shape[-1] != residual.shape[-1]:
            min_len = min(out.shape[-1], residual.shape[-1])
            out = out[..., :min_len]
            residual = residual[..., :min_len]
        out = out + residual
        return self.relu(out)


class PredictionHead(nn.Module):
    """A single regression head for one biomarker target."""

    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class MultiTaskBiomarkerModel(nn.Module):
    """
    Multi-task temporal CNN for biomarker estimation from facial ROI signals.

    Architecture:
        Input (N_ROI × 3 channels × T frames)
        → Conv1D stem
        → Temporal blocks with increasing dilation
        → Multi-scale feature aggregation
        → Global average + max pooling
        → Shared embedding
        → Per-target prediction heads (SBP, DBP, Glucose, Cholesterol)

    Parameters
    ----------
    n_roi_channels : int
        Number of input channels = n_rois × 3 (RGB).
        Default: 9 (3 ROIs × 3 channels).
    hidden_channels : int
        Number of channels in temporal blocks.
    n_temporal_blocks : int
        Number of stacked temporal blocks.
    embedding_dim : int
        Dimension of the shared embedding.
    """

    def __init__(
        self,
        n_roi_channels: int = 9,
        hidden_channels: int = 64,
        n_temporal_blocks: int = 4,
        embedding_dim: int = 128,
    ):
        super().__init__()

        # Stem: project input channels to hidden channels
        self.stem = nn.Sequential(
            nn.Conv1d(n_roi_channels, hidden_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Temporal blocks with increasing dilation for multi-scale features
        self.temporal_blocks = nn.ModuleList([
            TemporalBlock(hidden_channels, kernel_size=5, dilation=2 ** i)
            for i in range(n_temporal_blocks)
        ])

        # Feature pyramid: multi-scale pooling
        self.pyramid_pools = nn.ModuleList([
            nn.AdaptiveAvgPool1d(output_size) for output_size in [1, 4, 16]
        ])

        # Compute pyramid feature dim
        pyramid_dim = hidden_channels * (1 + 4 + 16)

        # Shared embedding
        self.embedding = nn.Sequential(
            nn.Linear(pyramid_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
        )

        # Prediction heads
        self.sbp_head = PredictionHead(embedding_dim)
        self.dbp_head = PredictionHead(embedding_dim)
        self.glucose_head = PredictionHead(embedding_dim)
        self.cholesterol_head = PredictionHead(embedding_dim)

        # Model metadata
        self.model_version = "v1.0.0-placeholder"
        self.is_trained = False

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor of shape (B, C, T) where C = n_roi_channels, T = time frames.

        Returns
        -------
        Dict with keys: sbp, dbp, glucose, cholesterol — each of shape (B,).
        """
        # Stem
        h = self.stem(x)  # (B, hidden, T)

        # Temporal blocks
        for block in self.temporal_blocks:
            h = block(h)  # (B, hidden, T)

        # Multi-scale feature pyramid
        pyramid_features = []
        for pool in self.pyramid_pools:
            pooled = pool(h)  # (B, hidden, K)
            pyramid_features.append(pooled.flatten(1))  # (B, hidden * K)

        features = torch.cat(pyramid_features, dim=1)  # (B, pyramid_dim)

        # Shared embedding
        emb = self.embedding(features)  # (B, embedding_dim)

        # Prediction heads
        return {
            "sbp": self.sbp_head(emb),
            "dbp": self.dbp_head(emb),
            "glucose": self.glucose_head(emb),
            "cholesterol": self.cholesterol_head(emb),
        }

    def get_model_info(self) -> Dict:
        """Return model metadata."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "architecture": "MultiTaskTemporalCNN",
            "version": self.model_version,
            "is_trained": self.is_trained,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "targets": ["sbp", "dbp", "glucose", "cholesterol"],
        }
