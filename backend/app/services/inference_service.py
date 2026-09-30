"""
FaceVital AI — Model Inference Service
========================================
Loads the biomarker model and provides inference with confidence estimation.

IMPORTANT: Without trained weights, this service operates in PLACEHOLDER mode.
All biomarker predictions will indicate zero confidence and include a
placeholder flag. Heart rate is always computed using the rPPG pipeline
(POS/CHROM) which works without trained weights.
"""

import os
import time
import json
import numpy as np
import torch
from typing import Dict, Optional, Tuple
from pathlib import Path

from backend.ml.biomarker_model import MultiTaskBiomarkerModel
from backend.rppg.pos import pos_rppg
from backend.rppg.chrom import chrom_rppg
from backend.rppg.heart_rate import estimate_heart_rate, compute_hr_windowed
from backend.rppg.signal_quality import assess_signal_quality
from backend.rppg.roi import parse_roi_signals, combine_roi_signals
from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger, log_event

logger = get_logger(__name__)


class InferenceService:
    """
    Manages model loading and inference for all vital signs.

    The service maintains:
    - rPPG pipeline (POS/CHROM) for heart rate — always functional
    - Biomarker model for BP/glucose/cholesterol — requires trained weights
    """

    def __init__(self):
        self.settings = get_settings()
        self.model: Optional[MultiTaskBiomarkerModel] = None
        self.model_loaded = False
        self.is_placeholder = True
        self.device = torch.device("cpu")
        self.scaler_params: Optional[Dict] = None
        self._load_model()

    def _load_model(self) -> None:
        """Attempt to load model weights; fall back to placeholder."""
        weights_dir = Path(self.settings.model_weights_dir)
        weights_path = weights_dir / "biomarker_model.pt"
        scaler_path = weights_dir / "scaler_params.json"

        # Initialize model architecture
        self.model = MultiTaskBiomarkerModel(
            n_roi_channels=9,
            hidden_channels=64,
            n_temporal_blocks=4,
            embedding_dim=128,
        )
        self.model.eval()

        if weights_path.exists():
            try:
                state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                self.is_placeholder = False
                self.model.is_trained = True
                log_event(logger, "Model weights loaded", model_path=str(weights_path))
            except Exception as e:
                log_event(logger, f"Failed to load model weights: {e}", level="warning")
                self.is_placeholder = True
        else:
            log_event(
                logger,
                "No model weights found — operating in PLACEHOLDER mode",
                level="warning",
                weights_path=str(weights_path),
            )
            self.is_placeholder = True

        # Load scaler
        if scaler_path.exists():
            try:
                with open(scaler_path, "r") as f:
                    self.scaler_params = json.load(f)
                log_event(logger, "Scaler params loaded")
            except Exception as e:
                log_event(logger, f"Failed to load scaler: {e}", level="warning")

        self.model_loaded = True

    def predict(
        self,
        roi_signals: Dict[str, list],
        fps: float = 30.0,
        duration_seconds: float = 10.0,
        rppg_method: str = "pos",
    ) -> Dict:
        """
        Run full inference pipeline.

        Parameters
        ----------
        roi_signals : Dict of ROI name → list of [R, G, B] per frame.
        fps : Frame rate.
        duration_seconds : Duration of the recording.
        rppg_method : "pos" or "chrom".

        Returns
        -------
        Dict with predictions, signal_quality, model_version, etc.
        """
        start_time = time.time()

        try:
            # Parse signals
            parsed = parse_roi_signals(roi_signals)

            # Combine ROIs for rPPG
            combined = combine_roi_signals(parsed, method="mean")

            # Run rPPG
            if rppg_method == "chrom":
                bvp = chrom_rppg(combined, fps)
            else:
                bvp = pos_rppg(combined, fps)

            # Signal quality
            quality, quality_components = assess_signal_quality(
                parsed, fps, bvp_signal=bvp
            )

            # Check quality threshold
            threshold = self.settings.signal_quality_threshold
            if quality < threshold:
                return {
                    "status": "insufficient_signal",
                    "signal_quality": round(quality, 3),
                    "quality_components": {k: round(v, 3) for k, v in quality_components.items()},
                    "model_version": self.settings.model_version,
                    "message": (
                        "Insufficient signal quality. Please improve lighting, "
                        "face alignment and remain still."
                    ),
                }

            # Heart rate estimation (always functional)
            hr_result = compute_hr_windowed(
                bvp, fps,
                window_seconds=self.settings.rppg_window_seconds,
                overlap=self.settings.rppg_overlap,
                min_hr=self.settings.rppg_min_hr,
                max_hr=self.settings.rppg_max_hr,
            )

            predictions = {}

            if hr_result.get("hr_bpm") is not None:
                predictions["heart_rate"] = {
                    "value": hr_result["hr_bpm"],
                    "unit": "bpm",
                    "confidence": hr_result["confidence"],
                }

            # Biomarker prediction (requires trained model)
            if not self.is_placeholder:
                biomarker_preds = self._run_biomarker_model(parsed, fps)
                predictions.update(biomarker_preds)
            else:
                # PLACEHOLDER mode: return structure with zero confidence
                predictions["systolic_bp"] = {
                    "value": None,
                    "unit": "mmHg",
                    "confidence": 0.0,
                    "placeholder": True,
                    "message": "Model not trained — requires labeled dataset",
                }
                predictions["diastolic_bp"] = {
                    "value": None,
                    "unit": "mmHg",
                    "confidence": 0.0,
                    "placeholder": True,
                    "message": "Model not trained — requires labeled dataset",
                }
                predictions["glucose"] = {
                    "value": None,
                    "unit": "mg/dL",
                    "confidence": 0.0,
                    "placeholder": True,
                    "message": "Model not trained — requires labeled dataset",
                }
                predictions["cholesterol"] = {
                    "value": None,
                    "unit": "mg/dL",
                    "confidence": 0.0,
                    "placeholder": True,
                    "message": "Model not trained — requires labeled dataset",
                }

            elapsed_ms = (time.time() - start_time) * 1000

            log_event(
                logger, "Prediction completed",
                session_quality=round(quality, 3),
                hr=predictions.get("heart_rate", {}).get("value"),
                inference_ms=round(elapsed_ms, 1),
                is_placeholder=self.is_placeholder,
            )

            return {
                "status": "success",
                "predictions": predictions,
                "signal_quality": round(quality, 3),
                "quality_components": {k: round(v, 3) for k, v in quality_components.items()},
                "model_version": self.settings.model_version,
                "inference_latency_ms": round(elapsed_ms, 1),
                "bvp_signal": bvp.tolist()[-300:],  # Last ~10s at 30fps for charting
            }

        except Exception as e:
            elapsed_ms = (time.time() - start_time) * 1000
            log_event(logger, f"Prediction error: {e}", level="error")
            return {
                "status": "error",
                "message": str(e),
                "signal_quality": 0.0,
                "model_version": self.settings.model_version,
                "inference_latency_ms": round(elapsed_ms, 1),
            }

    def _run_biomarker_model(
        self, parsed_signals: Dict[str, np.ndarray], fps: float
    ) -> Dict:
        """Run the trained biomarker model."""
        # Prepare input tensor: stack ROIs into (1, n_roi*3, T)
        roi_names = ["forehead", "left_cheek", "right_cheek"]
        channels = []
        min_len = min(parsed_signals[k].shape[0] for k in roi_names if k in parsed_signals)

        for name in roi_names:
            if name in parsed_signals:
                arr = parsed_signals[name][:min_len, :]  # (T, 3)
                channels.append(arr.T)  # (3, T)

        if not channels:
            return {}

        input_tensor = np.concatenate(channels, axis=0)  # (9, T)

        # Apply scaler if available
        if self.scaler_params:
            means = np.array(self.scaler_params.get("means", [0] * 9))[:, None]
            stds = np.array(self.scaler_params.get("stds", [1] * 9))[:, None]
            input_tensor = (input_tensor - means) / (stds + 1e-8)

        # Convert to torch
        x = torch.tensor(input_tensor, dtype=torch.float32).unsqueeze(0)  # (1, 9, T)

        with torch.no_grad():
            outputs = self.model(x)

        # Denormalize outputs if scaler available
        result = {}
        target_map = {
            "sbp": ("systolic_bp", "mmHg"),
            "dbp": ("diastolic_bp", "mmHg"),
            "glucose": ("glucose", "mg/dL"),
            "cholesterol": ("cholesterol", "mg/dL"),
        }

        for key, (name, unit) in target_map.items():
            raw_val = outputs[key].item()
            # Denormalize
            if self.scaler_params and f"{key}_mean" in self.scaler_params:
                val = raw_val * self.scaler_params[f"{key}_std"] + self.scaler_params[f"{key}_mean"]
            else:
                val = raw_val
            result[name] = {
                "value": round(val, 1),
                "unit": unit,
                "confidence": 0.5,  # Base confidence for trained model
            }

        return result

    def get_model_info(self) -> Dict:
        """Return model metadata for the API."""
        info = self.model.get_model_info() if self.model else {}
        return {
            "model_version": self.settings.model_version,
            "architecture": info.get("architecture", "MultiTaskTemporalCNN"),
            "framework": "pytorch",
            "is_placeholder": self.is_placeholder,
            "supported_targets": ["heart_rate", "systolic_bp", "diastolic_bp", "glucose", "cholesterol"],
            "rppg_algorithm": "POS (Plane-Orthogonal-to-Skin)",
            "signal_quality_threshold": self.settings.signal_quality_threshold,
            "description": (
                "Research prototype — estimates are not medical diagnoses or a replacement "
                "for clinically validated measurement devices."
            ),
            "total_parameters": info.get("total_parameters", 0),
            "trainable_parameters": info.get("trainable_parameters", 0),
        }


# Singleton instance
_inference_service: Optional[InferenceService] = None


def get_inference_service() -> InferenceService:
    """Get or create the inference service singleton."""
    global _inference_service
    if _inference_service is None:
        _inference_service = InferenceService()
    return _inference_service
