"""
FaceVital AI — Training Configuration
=======================================
"""

# Training configuration — loaded from config.yaml or used as defaults.
DEFAULT_CONFIG = {
    "seed": 42,
    "batch_size": 32,
    "learning_rate": 1e-3,
    "weight_decay": 1e-4,
    "epochs": 100,
    "early_stopping_patience": 15,
    "scheduler": "cosine",

    # Model
    "n_roi_channels": 9,
    "hidden_channels": 64,
    "n_temporal_blocks": 4,
    "embedding_dim": 128,

    # Data
    "window_seconds": 10.0,
    "fps": 30,
    "overlap": 0.5,
    "min_signal_length": 150,  # frames

    # rPPG
    "rppg_low_hz": 0.7,
    "rppg_high_hz": 3.0,

    # Targets
    "targets": ["sbp", "dbp", "glucose", "cholesterol"],

    # Splits
    "train_ratio": 0.7,
    "val_ratio": 0.15,
    "test_ratio": 0.15,

    # Paths
    "data_dir": "./datasets",
    "output_dir": "./models/weights",
    "report_dir": "./reports",
}
