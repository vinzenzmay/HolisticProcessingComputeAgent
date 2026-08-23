"""Tests for hpca.checks: the deterministic syntax gate (§5.2)."""

from hpca.checks import syntax_check


def write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content)
    return path


class TestBash:
    async def test_valid(self, tmp_path):
        path = write(tmp_path, "ok.sh", "echo hello\nls -l\n")
        result = await syntax_check(path)
        assert result.ok
        assert result.checker == "bash -n"

    async def test_invalid(self, tmp_path):
        path = write(tmp_path, "bad.sh", "if [ 1 -eq 1 ]; then\necho unclosed\n")
        result = await syntax_check(path)
        assert not result.ok
        assert "syntax error" in result.errors.lower()


class TestUnavailableChecker:
    async def test_missing_binary_reports_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path))  # nothing on PATH
        path = write(tmp_path, "x.sh", "echo hi\n")
        result = await syntax_check(path)
        assert result.ok  # absence of a checker must not block (§5.2 spirit)
        assert result.skipped
        assert "not found" in result.errors
