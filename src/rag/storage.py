from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

StorageBackend = Literal["local", "postgres"]


class DocumentStore(Protocol):
    def healthcheck(self) -> bool:
        ...


@dataclass(frozen=True)
class StorageSettings:
    backend: StorageBackend = "local"
    database_url: str | None = None

    @classmethod
    def from_values(cls, backend: str = "local", database_url: str | None = None) -> "StorageSettings":
        normalized = backend.strip().lower()
        if normalized not in {"local", "postgres"}:
            raise ValueError("RAG_STORAGE_BACKEND must be 'local' or 'postgres'")
        if normalized == "postgres" and not database_url:
            raise ValueError("RAG_DATABASE_URL is required when RAG_STORAGE_BACKEND=postgres")
        return cls(backend=normalized, database_url=database_url)


def select_storage_backend(backend: str = "local", database_url: str | None = None) -> StorageSettings:
    return StorageSettings.from_values(backend, database_url)
