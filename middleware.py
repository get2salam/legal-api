"""
Middleware for request logging, correlation IDs, and response timing.
"""

import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("legal_api")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    Add correlation ID to every request and log request/response details.

    - Injects ``X-Request-ID`` header (or reuses one from the client).
    - Logs method, path, status, and duration in a structured format.
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        # Use client-provided ID or generate one
        request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        start = time.perf_counter()

        response: Response = await call_next(request)

        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms}ms"

        logger.info(
            "request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "query": str(request.url.query),
                "status": response.status_code,
                "duration_ms": duration_ms,
                "client": request.client.host if request.client else "unknown",
            },
        )

        return response


def setup_logging(level: str = "INFO") -> None:
    """Configure structured JSON-style logging."""
    fmt = (
        "%(asctime)s | %(levelname)-5s | %(name)s | "
        "%(message)s | %(request_id)s | %(method)s %(path)s "
        "-> %(status)s (%(duration_ms)sms)"
    )

    class _Filter(logging.Filter):
        """Ensure extra fields exist even for non-request logs."""

        def filter(self, record: logging.LogRecord) -> bool:
            for key in ("request_id", "method", "path", "query", "status", "duration_ms", "client"):
                if not hasattr(record, key):
                    setattr(record, key, "-")
            return True

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(fmt))
    handler.addFilter(_Filter())

    root = logging.getLogger("legal_api")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    root.addHandler(handler)
    root.propagate = False
