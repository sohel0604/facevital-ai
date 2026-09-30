"""
FaceVital AI — Middleware
==========================
CORS, rate limiting, request validation.
"""

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
import time

from backend.app.core.logging import get_logger, log_event

logger = get_logger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Log request method, path, status, and latency."""

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed = (time.time() - start) * 1000

        log_event(
            logger,
            f"{request.method} {request.url.path}",
            status_code=response.status_code,
            latency_ms=round(elapsed, 1),
            client=request.client.host if request.client else "unknown",
        )

        # Add timing header
        response.headers["X-Response-Time-Ms"] = str(round(elapsed, 1))
        return response


class RequestSizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject request bodies larger than max_body_bytes."""

    def __init__(self, app, max_body_bytes: int = 10 * 1024 * 1024):
        super().__init__(app)
        self.max_body_bytes = max_body_bytes

    async def dispatch(self, request: Request, call_next):
        content_length = request.headers.get("content-length")
        if content_length and int(content_length) > self.max_body_bytes:
            return Response(
                content='{"detail": "Request body too large"}',
                status_code=413,
                media_type="application/json",
            )
        return await call_next(request)
