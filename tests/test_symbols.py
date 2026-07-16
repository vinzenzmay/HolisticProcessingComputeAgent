"""Tests for hpca.symbols: the exact-match symbol table (§5.6.1)."""

import pytest

from hpca.db import connect, init_db
from hpca.symbols import (
    SymbolIndex,
    index_python_source,
    parse_manpage_flags,
    parse_python_module,
)

PY_MODULE = '''\
"""Example module."""

CONSTANT = 1


def align(reads, reference, *, threads=1, min_quality=20):
    """Align reads to a reference."""


def _private_helper(x):
    pass


class BamFile:
    """A BAM file wrapper."""

    def subset(self, region, output=None):
        """Subset by region."""

    def _internal(self):
        pass
'''

MAN_PAGE = """\
SAMTOOLS(1)                                                        SAMTOOLS(1)

NAME
       samtools view - views and converts SAM/BAM/CRAM files

SYNOPSIS
       samtools view [options] in.sam|in.bam

OPTIONS
       -b      Output in the BAM format.

       -o FILE
              Output to FILE [stdout].

       --threads INT
              Number of additional threads to use [0].

       -q INT Skip alignments with MAPQ smaller than INT [0].

SEE ALSO
       samtools(1)
"""


class TestParsePythonModule:
    def test_functions_with_params(self):
        symbols = parse_python_module(PY_MODULE, module="tools.align", source="x.py")
        align = next(s for s in symbols if s.name == "align")
        assert align.kind == "function"
        assert align.parent == "tools.align"
        assert set(align.params) == {"reads", "reference", "threads", "min_quality"}
        assert "align(reads, reference" in align.signature
        assert align.doc == "Align reads to a reference."

    def test_classes_and_methods(self):
        symbols = parse_python_module(PY_MODULE, module="tools.align", source="x.py")
        names = {(s.kind, s.name) for s in symbols}
        assert ("class", "BamFile") in names
        method = next(s for s in symbols if s.name == "subset")
        assert method.kind == "method"
        assert method.parent == "tools.align.BamFile"
        assert "region" in method.params

    def test_private_symbols_skipped(self):
        symbols = parse_python_module(PY_MODULE, module="m", source="x.py")
        names = {s.name for s in symbols}
        assert "_private_helper" not in names
        assert "_internal" not in names

    def test_syntax_error_returns_empty(self):
        assert parse_python_module("def broken(:", module="m", source="x.py") == []


class TestParseManpageFlags:
    def test_flags_extracted_with_descriptions(self):
        symbols = parse_manpage_flags(MAN_PAGE, command="samtools-view")
        by_name = {s.name: s for s in symbols}
        assert set(by_name) == {"-b", "-o", "--threads", "-q"}
        assert by_name["-b"].kind == "cli-flag"
        assert by_name["-b"].parent == "samtools-view"
        assert "BAM format" in by_name["-b"].doc

    def test_see_also_section_not_scanned(self):
        symbols = parse_manpage_flags(MAN_PAGE, command="samtools-view")
        assert all(s.name.startswith("-") for s in symbols)


class TestSymbolIndex:
    @pytest.fixture
    def index(self, tmp_path):
        conn = connect(tmp_path / "hpca.db")
        init_db(conn)
        idx = SymbolIndex(conn)
        yield idx
        conn.close()

    def test_add_and_lookup(self, index):
        symbols = parse_python_module(PY_MODULE, module="m", source="x.py")
        index.add(symbols)
        found = index.lookup("align")
        assert len(found) == 1
        assert found[0].signature.startswith("align(")

    def test_lookup_unknown_returns_empty(self, index):
        assert index.lookup("frobnicate") == []

    def test_flags_for_command(self, index):
        index.add(parse_manpage_flags(MAN_PAGE, command="samtools-view"))
        flags = index.flags_for("samtools-view")
        assert "--threads" in flags and "-b" in flags

    def test_kwargs_for_function(self, index):
        index.add(parse_python_module(PY_MODULE, module="m", source="x.py"))
        assert "threads" in index.kwargs_for("align")
        assert index.kwargs_for("nonexistent") is None

    def test_has_command(self, index):
        index.add(parse_manpage_flags(MAN_PAGE, command="samtools-view"))
        assert index.has_command("samtools-view")
        assert not index.has_command("bcftools")

    def test_reindex_replaces_source(self, index):
        index.add(parse_python_module(PY_MODULE, module="m", source="x.py"))
        index.clear_source("x.py")
        assert index.lookup("align") == []

    def test_count(self, index):
        assert index.count() == 0
        index.add(parse_manpage_flags(MAN_PAGE, command="samtools-view"))
        assert index.count() == 4


class TestIndexPythonSource:
    def test_walks_tree_and_indexes(self, tmp_path):
        pkg = tmp_path / "mypkg"
        (pkg / "sub").mkdir(parents=True)
        (pkg / "core.py").write_text("def entry(a, b=1):\n    pass\n")
        (pkg / "sub" / "util.py").write_text("def helper(x):\n    pass\n")
        (pkg / "broken.py").write_text("def broken(:\n")

        conn = connect(tmp_path / "hpca.db")
        init_db(conn)
        index = SymbolIndex(conn)
        count = index_python_source(index, pkg)
        assert count == 2
        assert index.lookup("entry")[0].parent == "mypkg.core"
        assert index.lookup("helper")[0].parent == "mypkg.sub.util"
        conn.close()
