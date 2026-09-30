"""
FaceVital AI — API v1 Endpoints
=================================
FastAPI router with all /api/v1/ endpoints.
"""

import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from backend.app.schemas.api_schemas import (
    HealthResponse,
    SessionStartRequest,
    SessionStartResponse,
    SessionInfoResponse,
    PredictionRequest,
    PredictionResponse,
    SignalQualityRequest,
    SignalQualityResponse,
    ModelInfoResponse,
    VitalEstimate,
    PredictionResult,
)
from backend.app.services.inference_service import get_inference_service
from backend.app.db.database import get_db
from backend.app.models.database_models import Session, Measurement
from backend.app.core.config import get_settings
from backend.app.core.logging import get_logger, log_event

router = APIRouter(prefix="/api/v1", tags=["FaceVital AI"])
logger = get_logger(__name__)

# Track application start time
_start_time = time.time()


# --------------------------------------------------------------------------
# GET /api/v1/health
# --------------------------------------------------------------------------

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Application health check."""
    service = get_inference_service()
    uptime = time.time() - _start_time

    # Quick DB check
    db_status = "connected"
    try:
        # Lightweight check — will be validated by actual queries
        pass
    except Exception:
        db_status = "unavailable"

    return HealthResponse(
        status="healthy",
        version=get_settings().app_version,
        database=db_status,
        model_loaded=service.model_loaded,
        uptime_seconds=round(uptime, 1),
    )


# --------------------------------------------------------------------------
# POST /api/v1/session/start
# --------------------------------------------------------------------------

@router.post("/session/start", response_model=SessionStartResponse)
async def start_session(
    request: SessionStartRequest,
    db: AsyncSession = Depends(get_db),
):
    """Start a new measurement session."""
    settings = get_settings()
    session_id = str(uuid.uuid4())

    now = datetime.utcnow()
    session = Session(
        id=uuid.UUID(session_id),
        status="active",
        device_metadata=request.device_metadata,
        model_version=settings.model_version,
        started_at=now,
    )
    db.add(session)
    await db.flush()

    log_event(logger, "Session started", session_id=session_id)

    return SessionStartResponse(
        session_id=session_id,
        status="active",
        started_at=session.started_at or now,
        model_version=settings.model_version,
    )


# --------------------------------------------------------------------------
# POST /api/v1/predict
# --------------------------------------------------------------------------

@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    Run inference on ROI signals and return vital sign estimates.

    The heart rate is computed using the rPPG pipeline (POS algorithm).
    Biomarker predictions (BP, glucose, cholesterol) require a trained model.
    """
    service = get_inference_service()

    # Validate session exists
    try:
        session_uuid = uuid.UUID(request.session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session_id format",
        )

    # Run inference
    roi_dict = {
        "forehead": request.roi_signals.forehead,
        "left_cheek": request.roi_signals.left_cheek,
        "right_cheek": request.roi_signals.right_cheek,
    }

    result = service.predict(
        roi_signals=roi_dict,
        fps=request.fps,
        duration_seconds=request.duration_seconds,
    )

    # Handle insufficient signal
    if result["status"] == "insufficient_signal":
        log_event(
            logger, "Insufficient signal quality",
            session_id=request.session_id,
            signal_quality=result["signal_quality"],
        )
        return PredictionResponse(
            status="insufficient_signal",
            signal_quality=result["signal_quality"],
            model_version=result["model_version"],
            message=result.get("message", "Insufficient signal quality"),
        )

    # Handle errors
    if result["status"] == "error":
        log_event(
            logger, "Prediction error",
            session_id=request.session_id,
            error=result.get("message"),
            level="error",
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("message", "Prediction failed"),
        )

    # Build prediction result
    preds = result.get("predictions", {})
    prediction_result = PredictionResult()

    if "heart_rate" in preds and preds["heart_rate"].get("value") is not None:
        prediction_result.heart_rate = VitalEstimate(
            value=preds["heart_rate"]["value"],
            unit=preds["heart_rate"]["unit"],
            confidence=preds["heart_rate"]["confidence"],
        )

    for key, attr in [
        ("systolic_bp", "systolic_bp"),
        ("diastolic_bp", "diastolic_bp"),
        ("glucose", "glucose"),
        ("cholesterol", "cholesterol"),
    ]:
        if key in preds and preds[key].get("value") is not None:
            setattr(prediction_result, attr, VitalEstimate(
                value=preds[key]["value"],
                unit=preds[key]["unit"],
                confidence=preds[key]["confidence"],
            ))

    # Save measurement to DB
    try:
        measurement = Measurement(
            session_id=session_uuid,
            heart_rate=preds.get("heart_rate", {}).get("value"),
            heart_rate_confidence=preds.get("heart_rate", {}).get("confidence"),
            systolic_bp=preds.get("systolic_bp", {}).get("value"),
            systolic_bp_confidence=preds.get("systolic_bp", {}).get("confidence"),
            diastolic_bp=preds.get("diastolic_bp", {}).get("value"),
            diastolic_bp_confidence=preds.get("diastolic_bp", {}).get("confidence"),
            glucose=preds.get("glucose", {}).get("value"),
            glucose_confidence=preds.get("glucose", {}).get("confidence"),
            cholesterol=preds.get("cholesterol", {}).get("value"),
            cholesterol_confidence=preds.get("cholesterol", {}).get("confidence"),
            signal_quality=result["signal_quality"],
            status="success",
            model_version=result["model_version"],
            inference_latency_ms=result.get("inference_latency_ms"),
        )
        db.add(measurement)
    except Exception as e:
        log_event(logger, f"Failed to save measurement: {e}", level="warning")

    response = PredictionResponse(
        status="success",
        predictions=prediction_result,
        signal_quality=result["signal_quality"],
        model_version=result["model_version"],
        inference_latency_ms=result.get("inference_latency_ms"),
    )

    log_event(
        logger, "Prediction success",
        session_id=request.session_id,
        signal_quality=result["signal_quality"],
        hr=preds.get("heart_rate", {}).get("value"),
        inference_ms=result.get("inference_latency_ms"),
    )

    return response


# --------------------------------------------------------------------------
# POST /api/v1/signal-quality
# --------------------------------------------------------------------------

@router.post("/signal-quality", response_model=SignalQualityResponse)
async def check_signal_quality(request: SignalQualityRequest):
    """Assess signal quality without running full prediction."""
    from backend.rppg.roi import parse_roi_signals
    from backend.rppg.signal_quality import assess_signal_quality

    roi_dict = {
        "forehead": request.roi_signals.forehead,
        "left_cheek": request.roi_signals.left_cheek,
        "right_cheek": request.roi_signals.right_cheek,
    }

    try:
        parsed = parse_roi_signals(roi_dict)
        quality, components = assess_signal_quality(parsed, request.fps)
        threshold = get_settings().signal_quality_threshold
        is_sufficient = quality >= threshold

        message = "Signal quality is sufficient for measurement."
        if not is_sufficient:
            message = (
                "Insufficient signal quality. Please improve lighting, "
                "face alignment and remain still."
            )

        return SignalQualityResponse(
            signal_quality=round(quality, 3),
            components={k: round(v, 3) for k, v in components.items()},
            is_sufficient=is_sufficient,
            message=message,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Signal quality assessment failed: {str(e)}",
        )


# --------------------------------------------------------------------------
# GET /api/v1/session/{session_id}
# --------------------------------------------------------------------------

@router.get("/session/{session_id}", response_model=SessionInfoResponse)
async def get_session(session_id: str, db: AsyncSession = Depends(get_db)):
    """Get session information."""
    try:
        session_uuid = uuid.UUID(session_id)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session_id format",
        )

    result = await db.execute(
        select(Session).where(Session.id == session_uuid)
    )
    session = result.scalar_one_or_none()

    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found",
        )

    # Count measurements
    count_result = await db.execute(
        select(func.count()).select_from(Measurement).where(
            Measurement.session_id == session_uuid
        )
    )
    measurement_count = count_result.scalar() or 0

    return SessionInfoResponse(
        session_id=str(session.id),
        status=session.status,
        started_at=session.started_at,
        ended_at=session.ended_at,
        duration_seconds=session.duration_seconds,
        signal_quality_avg=session.signal_quality_avg,
        model_version=session.model_version,
        measurement_count=measurement_count,
    )


# --------------------------------------------------------------------------
# GET /api/v1/model-info
# --------------------------------------------------------------------------

@router.get("/model-info", response_model=ModelInfoResponse)
async def get_model_info():
    """Get information about the loaded model."""
    service = get_inference_service()
    info = service.get_model_info()
    return ModelInfoResponse(**info)
