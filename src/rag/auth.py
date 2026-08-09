import hmac

from fastapi import Header, HTTPException, status

from .config import settings


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """Require an API key when API_AUTH_KEY is configured."""
    if settings.api_auth_key is None:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key, settings.api_auth_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )
