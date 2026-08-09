from __future__ import annotations


def owner_id_from_claims(claims: dict[str, object], requested_owner_id: str | None = None) -> str:
    if requested_owner_id is not None:
        raise PermissionError("owner_id is controlled by authentication")
    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject.strip():
        raise PermissionError("authenticated subject is required")
    return subject.strip()
