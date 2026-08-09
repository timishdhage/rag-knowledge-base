from __future__ import annotations

import pytest

from rag.storage import select_storage_backend


def test_local_backend_is_default() -> None:
    settings = select_storage_backend()
    assert settings.backend == "local"
    assert settings.database_url is None


def test_postgres_backend_requires_database_url() -> None:
    with pytest.raises(ValueError, match="RAG_DATABASE_URL"):
        select_storage_backend("postgres")


def test_unknown_backend_is_rejected() -> None:
    with pytest.raises(ValueError, match="RAG_STORAGE_BACKEND"):
        select_storage_backend("sqlite")


def test_postgres_backend_accepts_database_url() -> None:
    settings = select_storage_backend("postgres", "postgresql://localhost/rag")
    assert settings.backend == "postgres"
    assert settings.database_url == "postgresql://localhost/rag"
