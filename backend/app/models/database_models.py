"""
FaceVital AI — ORM Models
==========================
SQLAlchemy models for users, sessions, measurements, and model versions.
"""

import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    Column, String, Float, Integer, DateTime, ForeignKey,
    JSON, Boolean, Text, Index,
)
from sqlalchemy.types import TypeDecorator, CHAR
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import relationship

from backend.app.db.database import Base


class GUID(TypeDecorator):
    """Platform-independent GUID type.
    Uses PostgreSQL's UUID type, otherwise uses CHAR(36), storing as stringified hex values.
    """
    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(PG_UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == "postgresql":
            return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))
        else:
            if not isinstance(value, uuid.UUID):
                return str(uuid.UUID(str(value)))
            else:
                return str(value)

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        else:
            if not isinstance(value, uuid.UUID):
                value = uuid.UUID(value)
            return value


class User(Base):
    """Application user (for JWT-ready auth)."""
    __tablename__ = "users"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(255), unique=True, nullable=True)
    hashed_password = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    sessions = relationship("Session", back_populates="user", cascade="all, delete-orphan")


class Session(Base):
    """A measurement session."""
    __tablename__ = "sessions"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=True)
    status = Column(String(50), default="active")  # active, completed, failed
    started_at = Column(DateTime, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Float, nullable=True)
    device_metadata = Column(JSON, nullable=True)  # browser, resolution, fps, etc.
    signal_quality_avg = Column(Float, nullable=True)
    model_version = Column(String(50), nullable=True)

    user = relationship("User", back_populates="sessions")
    measurements = relationship("Measurement", back_populates="session", cascade="all, delete-orphan")

    __table_args__ = (
        Index("ix_sessions_started_at", "started_at"),
    )


class Measurement(Base):
    """A single prediction result within a session."""
    __tablename__ = "measurements"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    session_id = Column(GUID(), ForeignKey("sessions.id"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Predictions
    heart_rate = Column(Float, nullable=True)
    heart_rate_confidence = Column(Float, nullable=True)
    systolic_bp = Column(Float, nullable=True)
    systolic_bp_confidence = Column(Float, nullable=True)
    diastolic_bp = Column(Float, nullable=True)
    diastolic_bp_confidence = Column(Float, nullable=True)
    glucose = Column(Float, nullable=True)
    glucose_confidence = Column(Float, nullable=True)
    cholesterol = Column(Float, nullable=True)
    cholesterol_confidence = Column(Float, nullable=True)

    # Quality metrics
    signal_quality = Column(Float, nullable=True)
    status = Column(String(50), default="success")  # success, insufficient_signal, error

    # Model info
    model_version = Column(String(50), nullable=True)
    inference_latency_ms = Column(Float, nullable=True)

    session = relationship("Session", back_populates="measurements")

    __table_args__ = (
        Index("ix_measurements_session_id", "session_id"),
        Index("ix_measurements_timestamp", "timestamp"),
    )


class ModelVersion(Base):
    """Tracks deployed model versions and their evaluation metrics."""
    __tablename__ = "model_versions"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    version = Column(String(50), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Model metadata
    architecture = Column(String(100), nullable=True)
    framework = Column(String(50), default="pytorch")
    weights_path = Column(String(500), nullable=True)
    scaler_path = Column(String(500), nullable=True)
    config = Column(JSON, nullable=True)

    # Evaluation metrics
    hr_mae = Column(Float, nullable=True)
    hr_rmse = Column(Float, nullable=True)
    sbp_mae = Column(Float, nullable=True)
    dbp_mae = Column(Float, nullable=True)
    glucose_mae = Column(Float, nullable=True)
    cholesterol_mae = Column(Float, nullable=True)
    evaluation_metrics = Column(JSON, nullable=True)  # Full metrics JSON
