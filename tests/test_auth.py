from __future__ import annotations

import pytest

from rag.auth import owner_from_identity


def test_verified_identity_maps_to_owner() -> None:
    assert owner_from_identity({"sub": "user-123"}) == "user-123"


def test_identity_without_subject_is_rejected() -> None:
    with pytest.raises(PermissionError, match="subject"):
        owner_from_identity({})
