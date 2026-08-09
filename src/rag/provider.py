from typing import Protocol

from openai import OpenAI

from .config import settings


class ModelGateway(Protocol):
    def generate(self, prompt: str) -> str:
        """Generate text from a prompt."""


class OpenAIModelGateway:
    def __init__(self):
        self._client: OpenAI | None = None

    def _get_client(self) -> OpenAI:
        if self._client is None:
            if not settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is required for answer generation")
            self._client = OpenAI(api_key=settings.openai_api_key)
        return self._client

    def generate(self, prompt: str) -> str:
        response = self._get_client().responses.create(
            model=settings.generation_model,
            input=prompt,
        )
        return getattr(response, "output_text", str(response))
