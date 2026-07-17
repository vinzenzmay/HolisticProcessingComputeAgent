"""OpenAI-compatible embeddings client (§5.6.2).

Embeddings come from the backend's ``/v1/embeddings`` endpoint (a small vLLM
sidecar serving e.g. all-MiniLM-L6-v2), keeping HPCA itself free of local ML
dependencies.
"""

from __future__ import annotations

import httpx

BATCH_SIZE = 64


class EmbeddingError(Exception):
    """Any failure talking to the embedding backend."""


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
                raise EmbeddingError(
                    f"Embedding request failed ({response.status_code}): "
                    f"{response.text[:300]}"
                )
            data = sorted(response.json()["data"], key=lambda d: d["index"])
            vectors.extend(entry["embedding"] for entry in data)
        return vectors
