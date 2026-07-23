"""Shared plumbing for live-backend integration tests.

The served model is *discovered* from /v1/models rather than pinned, so
swapping the model on the backend (35B → 27B → …) requires no test changes.
Override with $HPCA_TEST_LLM_URL / $HPCA_TEST_LLM_MODEL.
"""

import os

import httpx
import pytest

LIVE_URL = os.environ.get("HPCA_TEST_LLM_URL", "http://localhost:20001/v1")


def _discover_model() -> str | None:
    override = os.environ.get("HPCA_TEST_LLM_MODEL")
    if override:
        return override
    try:
        data = httpx.get(f"{LIVE_URL}/models", timeout=3).json().get("data", [])
    except Exception:
        return None
    return data[0]["id"] if data else None


LIVE_MODEL = _discover_model()

integration = pytest.mark.skipif(
    LIVE_MODEL is None, reason=f"LLM backend at {LIVE_URL} not reachable"
)
