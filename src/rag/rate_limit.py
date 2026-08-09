import time
from collections import defaultdict, deque
from threading import Lock

from fastapi import Request

from .config import settings
from .contracts import ErrorDetails, ErrorResponse


class InProcessRateLimiter:
    def __init__(self):
        self._requests = defaultdict(deque)
        self._lock = Lock()

    def allow(self, client_id: str) -> bool:
        now = time.monotonic()
        cutoff = now - settings.rate_limit_window_seconds
        with self._lock:
            bucket = self._requests[client_id]
            while bucket and bucket[0] <= cutoff:
                bucket.popleft()
            if len(bucket) >= settings.rate_limit_requests:
                return False
            bucket.append(now)
            return True


limiter = InProcessRateLimiter()


def enforce_rate_limit(request: Request) -> None:
    client_id = request.headers.get("X-API-Key") or (request.client.host if request.client else "unknown")
    if not limiter.allow(client_id):
        request_id = getattr(request.state, "request_id", "unknown")
        payload = ErrorResponse(
            error=ErrorDetails(
                code="RATE_LIMIT_EXCEEDED",
                message="Too many requests",
                request_id=request_id,
            )
        )
        from fastapi import HTTPException
        raise HTTPException(status_code=429, detail=payload.model_dump())
