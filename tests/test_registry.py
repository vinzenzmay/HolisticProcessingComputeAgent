"""Tests for hpca.registry: the path registry (§4.3).

The model never reproduces literal paths; tools accept registry keys and the
middleware resolves them. Unknown-key errors are fed back to the model, so
their message must name the available keys.
"""

import pytest

from hpca.db import connect, init_db
from hpca.registry import PathRegistry, RegistryError, UnknownKeyError


@pytest.fixture
def registry(tmp_path):
    conn = connect(tmp_path / "hpca.db")
    init_db(conn)
    yield PathRegistry(conn, profile="default", session_id="sess-1")
    conn.close()


class TestRegisterResolve:
    def test_roundtrip(self, registry, tmp_path):
        registry.register("input_bam", tmp_path / "a.bam")
        assert registry.resolve("input_bam") == tmp_path / "a.bam"

    def test_unknown_key_error_names_available_keys(self, registry, tmp_path):
        registry.register("input_bam", tmp_path / "a.bam")
        registry.register("ref_fasta", tmp_path / "ref.fa")
        with pytest.raises(UnknownKeyError) as exc:
            registry.resolve("input_bm")
        message = str(exc.value)
        assert "input_bm" in message
        assert "input_bam" in message
        assert "ref_fasta" in message

    def test_relative_path_rejected(self, registry):
        with pytest.raises(RegistryError, match="absolute"):
            registry.register("x", "relative/path.txt")

    def test_duplicate_key_different_path_rejected(self, registry, tmp_path):
        registry.register("x", tmp_path / "a.txt")
        with pytest.raises(RegistryError, match="already registered"):
            registry.register("x", tmp_path / "b.txt")

    def test_duplicate_key_same_path_is_idempotent(self, registry, tmp_path):
        registry.register("x", tmp_path / "a.txt")
        registry.register("x", tmp_path / "a.txt")  # no error
        assert registry.resolve("x") == tmp_path / "a.txt"

    def test_invalid_key_rejected(self, registry, tmp_path):
        with pytest.raises(RegistryError, match="key"):
            registry.register("has spaces!", tmp_path / "a.txt")

    def test_list_all(self, registry, tmp_path):
        registry.register("a", tmp_path / "a.txt")
        registry.register("b", tmp_path / "b.txt")
        assert registry.list() == {
            "a": tmp_path / "a.txt",
            "b": tmp_path / "b.txt",
        }


class TestAutoRegister:
    def test_generates_key_from_hint(self, registry, tmp_path):
        key = registry.register_auto(tmp_path / "sample1.bam", hint="output")
        assert key == "output"
        assert registry.resolve(key) == tmp_path / "sample1.bam"

    def test_same_path_returns_existing_key(self, registry, tmp_path):
        first = registry.register_auto(tmp_path / "x.log", hint="log")
        second = registry.register_auto(tmp_path / "x.log", hint="somethingelse")
        assert first == second

    def test_key_collision_gets_numeric_suffix(self, registry, tmp_path):
        k1 = registry.register_auto(tmp_path / "a.log", hint="log")
        k2 = registry.register_auto(tmp_path / "b.log", hint="log")
        k3 = registry.register_auto(tmp_path / "c.log", hint="log")
        assert k1 == "log"
        assert k2 == "log_2"
        assert k3 == "log_3"

    def test_hint_defaults_to_filename(self, registry, tmp_path):
        key = registry.register_auto(tmp_path / "counts matrix.tsv")
        assert key == "counts_matrix.tsv"

    def test_hint_is_slugified(self, registry, tmp_path):
        key = registry.register_auto(tmp_path / "x", hint="My Fancy Output!!")
        assert key == "my_fancy_output"


class TestScoping:
    def test_sessions_are_isolated(self, tmp_path):
        conn = connect(tmp_path / "hpca.db")
        init_db(conn)
        r1 = PathRegistry(conn, profile="default", session_id="s1")
        r2 = PathRegistry(conn, profile="default", session_id="s2")
        r1.register("x", tmp_path / "a.txt")
        with pytest.raises(UnknownKeyError):
            r2.resolve("x")
        conn.close()


class TestRemoveReassign:
    def test_remove(self, registry, tmp_path):
        registry.register("x", tmp_path / "a.txt")
        registry.remove("x")
        assert "x" not in registry

    def test_remove_unknown_raises(self, registry):
        with pytest.raises(UnknownKeyError):
            registry.remove("ghost")

    def test_reassign(self, registry, tmp_path):
        registry.register("x", tmp_path / "a.txt")
        registry.reassign("x", tmp_path / "b.txt")
        assert registry.resolve("x") == tmp_path / "b.txt"

    def test_reassign_unknown_raises(self, registry, tmp_path):
        with pytest.raises(UnknownKeyError):
            registry.reassign("ghost", tmp_path / "b.txt")

    def test_reassign_relative_rejected(self, registry, tmp_path):
        registry.register("x", tmp_path / "a.txt")
        with pytest.raises(RegistryError, match="absolute"):
            registry.reassign("x", "rel/path.txt")
