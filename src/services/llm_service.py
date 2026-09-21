"""LLM service for CMN-C1-009 — wraps the shared LLM client.

The main slot calls ``ClassificationLLMService.generate(prompt) -> str``. The
service adapts the canonical ``shared.services.llm.base_llm.BaseLLM.complete(
messages) -> dict`` interface to the single-prompt / single-string shape the
classification prompt uses.

v1: the shared client is a placeholder that raises NotImplementedError until an
API key is provisioned (S-3 secrets via bound_secrets at the entry point).
"""

from __future__ import annotations

from typing import Any, Optional

from shared.services.llm.base_llm import BaseLLM


class ClassificationLLMService:
    """Adapter exposing ``generate(prompt) -> str`` over a shared BaseLLM client."""

    def __init__(self, llm: Optional[BaseLLM] = None, config: Optional[dict[str, Any]] = None):
        self._llm = llm
        self._config = config or {}

    def _client(self) -> BaseLLM:
        if self._llm is not None:
            return self._llm
        try:
            from shared.services.llm.azure_openai_client import AzureOpenAIClient

            self._llm = AzureOpenAIClient(self._config)
            return self._llm
        except (ImportError, ValueError) as exc:
            raise NotImplementedError(f"ClassificationLLMService: LLM client not yet wired ({exc})") from exc

    def generate(self, prompt: str) -> str:
        """Send a single user prompt and return the assistant text."""
        response = self._client().complete([{"role": "user", "content": prompt}])
        return str(response.get("content", "")) if isinstance(response, dict) else str(response)
