"""Shared plumbing for live-backend integration tests.

The served model is *discovered* from /v1/models rather than pinned, so
swapping the model on the backend (35B → 27B → …) requires no test changes.
Override with $HPCA_TEST_LLM_URL / $HPCA_TEST_LLM_MODEL.
"""

import os

import httpx
import pytest

LIVE_URL = os.environ.get("HPCA_TEST_LLM_URL", "http://localhost:20001/v1")
LIVE_KEY = os.environ.get("HPCA_TEST_LLM_KEY")


def _discover_model() -> str | None:
    override = os.environ.get("HPCA_TEST_LLM_MODEL")
    if override:
        return override
    headers = {"Authorization": f"Bearer {LIVE_KEY}"} if LIVE_KEY else {}
    try:
        data = (
            httpx.get(f"{LIVE_URL}/models", timeout=3, headers=headers)
            .json()
            .get("data", [])
        )
    except Exception:
        return None
    return data[0]["id"] if data else None


LIVE_MODEL = _discover_model()

_skip_unless_live = pytest.mark.skipif(
    LIVE_MODEL is None, reason=f"LLM backend at {LIVE_URL} not reachable"
)


def integration(obj):
    """Mark a live-backend test: tagged ``integration`` (deselected by default,
    run with ``pytest -m integration``) and skipped when no backend answers."""
    return pytest.mark.integration(_skip_unless_live(obj))
