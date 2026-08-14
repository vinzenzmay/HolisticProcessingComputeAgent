"""OpenAI-compatible embeddings client (§5.6.2).

Embeddings come from the backend's ``/v1/embeddings`` endpoint (a small vLLM
sidecar serving e.g. all-MiniLM-L6-v2), keeping HPCA itself free of local ML
dependencies.
"""

from __future__ import annotations

import httpx

BATCH_SIZE = 64

# What an OpenAI-compatible server says when the input overruns the model's
# window. Matched on text because the status code alone does not distinguish
# it from a malformed request, and only this case is worth re-trying smaller.
_TOO_LONG_MARKERS = (
    "maximum context length",
    "longer than the maximum",
    "reduce the length",
    "too long",
)


def _is_too_long(detail: str) -> bool:
    lowered = detail.lower()
    return any(marker in lowered for marker in _TOO_LONG_MARKERS)


class EmbeddingError(Exception):
    """Any failure talking to the embedding backend."""


class InputTooLong(EmbeddingError):
    """One of the texts is longer than the model's context window.

    Told apart from the rest because it is the one embedding failure a caller
    can do something about: the text can be split and tried again, whereas a
    refused connection or a 500 only gets worse for being retried. See
    ``hpca.rag.embed_fitting``.
    """


class EmbeddingClient:
    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str | None = None,
        timeout_s: int = 60,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self.model = model
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/",
            headers=headers,
            timeout=httpx.Timeout(timeout_s, connect=10),
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), BATCH_SIZE):
            batch = texts[start : start + BATCH_SIZE]
            try:
                response = await self._client.post(
                    "embeddings", json={"model": self.model, "input": batch}
                )
            except httpx.HTTPError as e:
                raise EmbeddingError(f"Embedding request failed: {e}") from e
            if response.status_code != 200:
                detail = response.text[:300]
                message = (
                    f"Embedding request failed ({response.status_code}): {detail}"
                )
                if response.status_code == 400 and _is_too_long(detail):
                    raise InputTooLong(message)
                raise EmbeddingError(message)
            data = sorted(response.json()["data"], key=lambda d: d["index"])
            vectors.extend(entry["embedding"] for entry in data)
        return vectors
