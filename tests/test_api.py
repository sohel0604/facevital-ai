"""
FaceVital AI — API Tests
==========================
Tests for FastAPI endpoints using httpx test client.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from unittest.mock import patch, AsyncMock, MagicMock

from fastapi.testclient import TestClient


# Mock the database dependency before importing the app
def get_mock_db():
    """Create a mock async database session."""
    mock = MagicMock()
    mock.add = MagicMock()
    mock.flush = AsyncMock()
    mock.commit = AsyncMock()
    mock.rollback = AsyncMock()
    mock.close = AsyncMock()
    return mock


@pytest.fixture
def client():
    """Create a test client with mocked DB."""
    # Patch DB initialization to avoid real DB connection
    with patch("backend.app.db.database.init_db", new_callable=AsyncMock):
        with patch("backend.app.db.database.close_db", new_callable=AsyncMock):
            from backend.app.main import app
            from backend.app.db.database import get_db

            async def override_db():
                mock = get_mock_db()
                yield mock

            app.dependency_overrides[get_db] = override_db
            with TestClient(app) as c:
                yield c
            app.dependency_overrides.clear()


class TestHealthEndpoint:
    def test_health_check(self, client):
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "version" in data

    def test_health_has_model_status(self, client):
        response = client.get("/api/v1/health")
        data = response.json()
        assert "model_loaded" in data


class TestModelInfo:
    def test_model_info(self, client):
        response = client.get("/api/v1/model-info")
        assert response.status_code == 200
        data = response.json()
        assert "model_version" in data
        assert "is_placeholder" in data
        assert "supported_targets" in data
        assert len(data["supported_targets"]) == 5


class TestPredict:
    def _make_predict_payload(self, n_frames: int = 300, fps: float = 30.0):
        """Generate a valid prediction request payload."""
        rng = np.random.RandomState(42)
        t = np.arange(n_frames) / fps
        ppg = np.sin(2 * np.pi * 1.2 * t)

        def make_roi():
            return [
                [
                    float(140 + 0.3 * ppg[i] + 0.1 * rng.randn()),
                    float(150 + ppg[i] + 0.1 * rng.randn()),
                    float(130 + 0.2 * ppg[i] + 0.1 * rng.randn()),
                ]
                for i in range(n_frames)
            ]

        return {
            "session_id": "00000000-0000-0000-0000-000000000001",
            "fps": fps,
            "duration_seconds": n_frames / fps,
            "roi_signals": {
                "forehead": make_roi(),
                "left_cheek": make_roi(),
                "right_cheek": make_roi(),
            },
        }

    def test_predict_success(self, client):
        payload = self._make_predict_payload(n_frames=300)
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ("success", "insufficient_signal")
        assert "signal_quality" in data

    def test_predict_short_signal(self, client):
        """Signal too short should fail validation."""
        payload = self._make_predict_payload(n_frames=10)
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 422  # Validation error

    def test_predict_invalid_session(self, client):
        payload = self._make_predict_payload()
        payload["session_id"] = "not-a-uuid"
        response = client.post("/api/v1/predict", json=payload)
        assert response.status_code == 400


class TestSignalQuality:
    def test_signal_quality_endpoint(self, client):
        rng = np.random.RandomState(42)
        n = 150
        t = np.arange(n) / 30.0
        ppg = np.sin(2 * np.pi * 1.2 * t)

        def make_roi():
            return [
                [float(140 + rng.randn()), float(150 + ppg[i]), float(130 + rng.randn())]
                for i in range(n)
            ]

        payload = {
            "roi_signals": {
                "forehead": make_roi(),
                "left_cheek": make_roi(),
                "right_cheek": make_roi(),
            },
            "fps": 30.0,
        }

        response = client.post("/api/v1/signal-quality", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert 0.0 <= data["signal_quality"] <= 1.0
        assert "is_sufficient" in data
        assert "components" in data


class TestSession:
    def test_start_session(self, client):
        response = client.post("/api/v1/session/start", json={})
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert data["status"] == "active"
