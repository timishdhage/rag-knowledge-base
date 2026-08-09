from __future__ import annotations

import pytest

from rag.vectorstore import owner_where


def test_owner_filter_is_exact() -> None:
    assert owner_where("user-123") == {"owner_id": "user-123"}


def test_owner_filter_rejects_blank_ids() -> None:
    with pytest.raises(ValueError, match="owner_id"):
        owner_where("  ")
