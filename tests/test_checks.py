"""Tests for hpca.checks: the deterministic syntax gate (§5.2)."""

import pytest

from hpca.checks import syntax_check


def write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return path


class TestBash:
    async def test_valid(self, tmp_path):
        path = write(tmp_path, "ok.sh", "echo hello\nls -l\n")
        result = await syntax_check("bash", path)
        assert result.ok
        assert result.checker == "bash -n"

    async def test_invalid(self, tmp_path):
        path = write(tmp_path, "bad.sh", "if [ 1 -eq 1 ]; then\necho unclosed\n")
        result = await syntax_check("bash", path)
        assert not result.ok
        assert "syntax error" in result.errors.lower()


class TestPython:
    async def test_valid(self, tmp_path):
        path = write(tmp_path, "ok.py", "import sys\nprint(sys.argv)\n")
        result = await syntax_check("python", path)
        assert result.ok

    async def test_invalid(self, tmp_path):
        path = write(tmp_path, "bad.py", "def broken(:\n    pass\n")
        result = await syntax_check("python", path)
        assert not result.ok
        assert "SyntaxError" in result.errors


class TestUnavailableChecker:
    async def test_missing_binary_reports_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path))  # nothing on PATH
        path = write(tmp_path, "x.sh", "echo hi\n")
        result = await syntax_check("bash", path)
        assert result.ok  # absence of a checker must not block (§5.2 spirit)
        assert result.skipped
        assert "not found" in result.errors


class TestUnknownKind:
    async def test_rejected(self, tmp_path):
        path = write(tmp_path, "x.txt", "hi")
        with pytest.raises(ValueError, match="kind"):
            await syntax_check("perl", path)
