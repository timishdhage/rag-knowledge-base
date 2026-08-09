from __future__ import annotations

import hmac
import json
import os
from functools import lru_cache
from typing import Any
from urllib.request import urlopen

from fastapi import Header, HTTPException, Request, status
from jose import JWTError, jwt

from .config import settings
from .contracts import ErrorDetails, ErrorResponse
from .ownership import owner_id_from_claims


@lru_cache(maxsize=1)
def cognito_settings() -> tuple[str | None, str | None]:
    region = os.getenv("AWS_REGION")
    pool_id = os.getenv("COGNITO_USER_POOL_ID")
    client_id = os.getenv("COGNITO_APP_CLIENT_ID")
    issuer = os.getenv("COGNITO_ISSUER") or (f"https://cognito-idp.{region}.amazonaws.com/{pool_id}" if region and pool_id else None)
    return issuer, client_id


@lru_cache(maxsize=1)
def cognito_jwks(issuer: str) -> dict[str, Any]:
    with urlopen(f"{issuer}/.well-known/jwks.json", timeout=5) as response:
        return json.load(response)


def _unauthorized(request: Request, message: str) -> HTTPException:
    request_id = getattr(request.state, "request_id", "unknown")
    return HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail={"code": "UNAUTHORIZED", "message": message, "request_id": request_id}, headers={"WWW-Authenticate": "Bearer"})


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    if settings.api_auth_key is None:
        return
    if x_api_key is None or not hmac.compare_digest(x_api_key, settings.api_auth_key):
        request_id = getattr(request.state, "request_id", "unknown")
        payload = ErrorResponse(error=ErrorDetails(code="UNAUTHORIZED", message="Invalid or missing API key", request_id=request_id))
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=payload.model_dump(), headers={"WWW-Authenticate": "ApiKey"})


def require_cognito_identity(request: Request, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    issuer, client_id = cognito_settings()
    if not issuer or not client_id:
        raise _unauthorized(request, "Cognito is not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise _unauthorized(request, "Missing bearer token")
    token = authorization.removeprefix("Bearer ").strip()
    try:
        header = jwt.get_unverified_header(token)
        key = next(item for item in cognito_jwks(issuer)["keys"] if item["kid"] == header["kid"])
        claims = jwt.decode(token, key, algorithms=["RS256"], issuer=issuer, options={"verify_aud": False})
        token_use = claims.get("token_use")
        if token_use not in {"access", "id"}:
            raise _unauthorized(request, "Invalid token use")
        audience = claims.get("client_id") if token_use == "access" else claims.get("aud")
        if audience != client_id:
            raise _unauthorized(request, "Invalid token audience")
        return claims
    except (JWTError, KeyError, StopIteration, OSError, ValueError) as exc:
        raise _unauthorized(request, "Invalid bearer token") from exc


def owner_from_identity(claims: dict[str, Any]) -> str:
    return owner_id_from_claims(claims)
