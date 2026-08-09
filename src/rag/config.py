import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class Settings(BaseModel):
    chroma_path: str = Field(default_factory=lambda: os.getenv("CHROMA_PATH", "./chroma"))
    embedding_model: str = Field(default_factory=lambda: os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"))
    generation_model: str = Field(default_factory=lambda: os.getenv("GENERATION_MODEL", "gpt-4o-mini"))
    openai_api_key: str | None = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))
    top_k_dense: int = 10
    top_k_sparse: int = 10
    top_k_final: int = 5

    @property
    def chroma_directory(self) -> Path:
        return Path(self.chroma_path)


settings = Settings()
