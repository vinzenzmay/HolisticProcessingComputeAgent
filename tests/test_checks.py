"""Tests for hpca.checks: deterministic syntax/dry-run gate (§5.2)."""

import shutil

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


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="Rscript not installed")
class TestR:
    async def test_valid(self, tmp_path):
        path = write(tmp_path, "ok.R", "x <- c(1, 2, 3)\nmean(x)\n")
        result = await syntax_check("R", path)
        assert result.ok

    async def test_invalid(self, tmp_path):
        path = write(tmp_path, "bad.R", "x <- c(1, 2,\n")
        result = await syntax_check("R", path)
        assert not result.ok


@pytest.mark.skipif(
    shutil.which("snakemake") is None, reason="snakemake not installed"
)
class TestSnakemake:
    async def test_valid(self, tmp_path):
        path = write(
            tmp_path,
            "Snakefile",
            'rule all:\n    input: []\n',
        )
        result = await syntax_check("snakemake", path)
        assert result.ok

    async def test_invalid(self, tmp_path):
        path = write(tmp_path, "Snakefile", "rule broken\n    output: x\n")
        result = await syntax_check("snakemake", path)
        assert not result.ok


class TestUnavailableChecker:
    async def test_missing_binary_reports_skipped(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path))  # nothing on PATH
        path = write(tmp_path, "x.R", "x <- 1\n")
        result = await syntax_check("R", path)
        assert result.ok  # absence of a checker must not block (§5.2 spirit)
        assert result.skipped
        assert "not found" in result.errors


class TestUnknownKind:
    async def test_rejected(self, tmp_path):
        path = write(tmp_path, "x.txt", "hi")
        with pytest.raises(ValueError, match="kind"):
            await syntax_check("perl", path)
