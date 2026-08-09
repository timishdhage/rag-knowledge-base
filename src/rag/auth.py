import hmac
from uuid import uuid4

from fastapi import Header, HTTPException, Request, status

from .config import settings
from .contracts import ErrorDetails, ErrorResponse


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None),
) -> None:
    """Require an API key when API_AUTH_KEY is configured."""
    if settings.api_auth_key is None:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key, settings.api_auth_key):
        request_id = getattr(request.state, "request_id", str(uuid4()))
        payload = ErrorResponse(
            error=ErrorDetails(
                code="UNAUTHORIZED",
                message="Invalid or missing API key",
                request_id=request_id,
            )
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=payload.model_dump(),
            headers={"WWW-Authenticate": "ApiKey"},
        )
