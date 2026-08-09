from __future__ import annotations

import pytest

from rag.ownership import owner_id_from_claims


def test_owner_id_comes_from_authenticated_subject() -> None:
    assert owner_id_from_claims({"sub": "user-123"}) == "user-123"


def test_missing_subject_is_rejected() -> None:
    with pytest.raises(PermissionError, match="subject"):
        owner_id_from_claims({})


def test_client_cannot_override_owner() -> None:
    with pytest.raises(PermissionError, match="authentication"):
        owner_id_from_claims({"sub": "user-123"}, requested_owner_id="user-456")
