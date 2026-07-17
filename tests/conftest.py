"""Test-wide isolation of anything the app writes outside its app dir."""

import pytest

import hpca.logs


@pytest.fixture(autouse=True)
def isolate_session_logs(monkeypatch, tmp_path, request):
    """Session logs default to ./hpca-logs, i.e. wherever hpca was started —
    which under pytest is the checkout. Redirect them per test so a suite run
    never litters the repo (or reads another test's transcripts). Tests marked
    ``real_log_default`` opt out: they assert what ships.
    """
    if "real_log_default" in request.keywords:
        return
    monkeypatch.setattr(hpca.logs, "DEFAULT_LOG_DIR", str(tmp_path / "hpca-logs"))
