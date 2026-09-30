"""
FaceVital AI — Pydantic Schemas
================================
Request/response models for the API.
"""

from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Shared / nested
# ---------------------------------------------------------------------------

class VitalEstimate(BaseModel):
    """A single vital-sign estimate with unit and confidence."""
    value: float
    unit: str
    confidence: float = Field(ge=0.0, le=1.0)


class PredictionResult(BaseModel):
    """All vital-sign predictions."""
    heart_rate: Optional[VitalEstimate] = None
    systolic_bp: Optional[VitalEstimate] = None
    diastolic_bp: Optional[VitalEstimate] = None
    glucose: Optional[VitalEstimate] = None
    cholesterol: Optional[VitalEstimate] = None


# ---------------------------------------------------------------------------
# Session
# ---------------------------------------------------------------------------

class SessionStartRequest(BaseModel):
    """POST /api/v1/session/start"""
    device_metadata: Optional[Dict[str, Any]] = None


class SessionStartResponse(BaseModel):
    """Response after starting a new session."""
    session_id: str
    status: str = "active"
    started_at: datetime
    model_version: str


class SessionInfoResponse(BaseModel):
    """GET /api/v1/session/{session_id}"""
    session_id: str
    status: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    signal_quality_avg: Optional[float] = None
    model_version: Optional[str] = None
    measurement_count: int = 0


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class ROISignals(BaseModel):
    """Temporal RGB signals from facial ROIs."""
    forehead: List[List[float]] = Field(
        ..., description="List of [R, G, B] values per frame for forehead ROI"
    )
    left_cheek: List[List[float]] = Field(
        ..., description="List of [R, G, B] values per frame for left cheek ROI"
    )
    right_cheek: List[List[float]] = Field(
        ..., description="List of [R, G, B] values per frame for right cheek ROI"
    )

    @field_validator("forehead", "left_cheek", "right_cheek")
    @classmethod
    def validate_signal_length(cls, v: List[List[float]]) -> List[List[float]]:
        if len(v) < 30:  # At least ~1 second at 30 fps
            raise ValueError("Signal must contain at least 30 frames")
        if len(v) > 9000:  # Max ~5 minutes at 30 fps
            raise ValueError("Signal too long (max 9000 frames)")
        return v


class PredictionRequest(BaseModel):
    """POST /api/v1/predict"""
    session_id: str
    fps: float = Field(ge=10, le=120, default=30.0)
    duration_seconds: float = Field(ge=1.0, le=300.0)
    roi_signals: ROISignals


class PredictionResponse(BaseModel):
    """Successful prediction response."""
    status: str  # "success" or "insufficient_signal"
    predictions: Optional[PredictionResult] = None
    signal_quality: float = Field(ge=0.0, le=1.0)
    model_version: str
    inference_latency_ms: Optional[float] = None
    message: Optional[str] = None


# ---------------------------------------------------------------------------
# Signal quality
# ---------------------------------------------------------------------------

class SignalQualityRequest(BaseModel):
    """POST /api/v1/signal-quality"""
    roi_signals: ROISignals
    fps: float = 30.0


class SignalQualityResponse(BaseModel):
    """Signal quality assessment response."""
    signal_quality: float = Field(ge=0.0, le=1.0)
    components: Dict[str, float] = Field(
        default_factory=dict,
        description="Individual quality component scores",
    )
    is_sufficient: bool
    message: str


# ---------------------------------------------------------------------------
# Model info
# ---------------------------------------------------------------------------

class ModelInfoResponse(BaseModel):
    """GET /api/v1/model-info"""
    model_version: str
    architecture: str
    framework: str
    is_placeholder: bool
    supported_targets: List[str]
    rppg_algorithm: str
    signal_quality_threshold: float
    description: str


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    """GET /api/v1/health"""
    status: str
    version: str
    database: str
    model_loaded: bool
    uptime_seconds: float
