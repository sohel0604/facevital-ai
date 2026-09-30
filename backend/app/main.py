"""
FaceVital AI — FastAPI Application Entry Point
================================================
Main application factory with all middleware, routes, and lifecycle hooks.

RESEARCH PROTOTYPE — Estimates are NOT medical diagnoses or a replacement
for clinically validated measurement devices.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.core.config import get_settings
from backend.app.core.logging import setup_logging, get_logger, log_event
from backend.app.api.v1.endpoints import router as v1_router
from backend.app.middleware.middleware import (
    RequestLoggingMiddleware,
    RequestSizeLimitMiddleware,
)

settings = get_settings()
setup_logging(level=settings.log_level, structured=settings.enable_structured_logging)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: startup and shutdown."""
    log_event(logger, "FaceVital AI starting", version=settings.app_version)

    # Initialize DB tables (dev convenience — use Alembic in prod)
    try:
        from backend.app.db.database import init_db
        await init_db()
        log_event(logger, "Database initialized")
    except Exception as e:
        log_event(logger, f"Database init skipped: {e}", level="warning")

    # Pre-load inference service
    try:
        from backend.app.services.inference_service import get_inference_service
        service = get_inference_service()
        log_event(
            logger, "Inference service ready",
            is_placeholder=service.is_placeholder,
            model_version=settings.model_version,
        )
    except Exception as e:
        log_event(logger, f"Inference service init failed: {e}", level="error")

    yield  # Application runs

    # Shutdown
    try:
        from backend.app.db.database import close_db
        await close_db()
    except Exception:
        pass
    log_event(logger, "FaceVital AI shutdown complete")


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="FaceVital AI",
        description=(
            "Research-grade non-contact facial health monitoring system. "
            "**Research prototype — estimates are NOT medical diagnoses.**"
        ),
        version=settings.app_version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Custom middleware
    app.add_middleware(RequestLoggingMiddleware)
    app.add_middleware(RequestSizeLimitMiddleware, max_body_bytes=10 * 1024 * 1024)

    # Root welcome route
    @app.get("/")
    async def root():
        return {
            "name": "FaceVital AI API",
            "version": settings.app_version,
            "status": "online",
            "docs_url": "/docs",
            "health_url": "/api/v1/health",
            "disclaimer": "Research Prototype — Not for Clinical Diagnostic Use",
        }

    # Routes
    app.include_router(v1_router)

    return app


app = create_app()
