"""Tests for the RAG-scope retrieval index (redesign Phase 5)."""

from datetime import date, timedelta

import pytest

from hpca.db import connect, init_db
from hpca.memory_index import MemoryIndex, demote
from hpca.profiles import MemoryScope, Profile

SP = MemoryScope.SYSTEM_PROMPT
RAG = MemoryScope.RAG


@pytest.fixture
def index(tmp_path):
    conn = connect(tmp_path / "hpca.db")
    init_db(conn)
    yield MemoryIndex(conn)
    conn.close()


def profile_with(*entries, name="default"):
    """entries: (text, scope[, backend, created, kind])"""
    profile = Profile(name=name)
    for entry in entries:
        text, scope = entry[0], entry[1]
        memory = profile.add_memory(text, scope=scope)
        if len(entry) > 2:
            memory.backend = entry[2]
        if len(entry) > 3:
            memory.created = entry[3]
        if len(entry) > 4:
            memory.kind = entry[4]
    return profile


class TestIndexing:
    def test_only_rag_indexed(self, index):
        profile = profile_with(
            ("Cluster is cubi.", SP),
            ("User prefers R.", SP),
            ("Snakemake dry-runs fail with site profiles.", RAG),
        )
        assert index.reindex(profile) == 1
        hits = index.search("snakemake", profile="default")
        assert len(hits) == 1

    def test_reindex_replaces_not_appends(self, index):
        index.reindex(profile_with(("bwa mem needs a prebuilt index.", RAG)))
        index.reindex(profile_with(("bwa mem needs a prebuilt index.", RAG)))
        assert len(index.search("bwa", profile="default")) == 1

    def test_removed_memory_disappears(self, index):
        index.reindex(profile_with(("bwa mem needs an index.", RAG)))
        index.reindex(profile_with())
        assert index.search("bwa", profile="default") == []

    def test_profiles_isolated(self, index):
        index.reindex(profile_with(("deepvariant needs a GPU.", RAG), name="genetics"))
        index.reindex(profile_with(("drain the node first.", RAG), name="hpc-admin"))
        assert len(index.search("deepvariant", profile="genetics")) == 1
        assert index.search("deepvariant", profile="hpc-admin") == []

    def test_forget_profile(self, index):
        index.reindex(profile_with(("deepvariant needs a GPU.", RAG), name="genetics"))
        index.forget_profile("genetics")
        assert index.search("deepvariant", profile="genetics") == []


class TestSearch:
    def test_matches_any_term(self, index):
        index.reindex(
            profile_with(("Snakemake dry-runs fail with site profiles.", RAG))
        )
        assert index.search("my snakemake workflow broke", profile="default")

    def test_no_match(self, index):
        index.reindex(profile_with(("Snakemake dry-runs fail.", RAG)))
        assert index.search("kubernetes ingress", profile="default") == []

    def test_short_words_ignored(self, index):
        """Two-letter words would match nearly everything."""
        index.reindex(profile_with(("Snakemake dry-runs fail.", RAG)))
        assert index.search("is it ok", profile="default") == []

    def test_limit_respected(self, index):
        index.reindex(
            profile_with(*[(f"bam handling note {i}", RAG) for i in range(6)])
        )
        assert len(index.search("bam", profile="default", limit=2)) == 2

    def test_active_backend_ranks_higher(self, index):
        recent = date.today().isoformat()
        index.reindex(
            profile_with(
                ("bam indexing is slow here", RAG, "old-model", recent),
                ("bam indexing is slow here too", RAG, "qwen3-35b", recent),
            )
        )
        hits = index.search(
            "bam indexing", profile="default", active_backend="qwen3-35b"
        )
        assert hits[0].backend == "qwen3-35b"

    def test_recent_memory_ranks_higher(self, index):
        old = (date.today() - timedelta(days=400)).isoformat()
        recent = date.today().isoformat()
        index.reindex(
            profile_with(
                ("bam indexing is slow here", RAG, "", old),
                ("bam indexing is slow here too", RAG, "", recent),
            )
        )
        hits = index.search("bam indexing", profile="default")
        assert hits[0].created == recent

    def test_malformed_date_does_not_crash(self, index):
        index.reindex(profile_with(("bam note", RAG, "", "not-a-date")))
        assert len(index.search("bam", profile="default")) == 1


class TestBuiltinMemories:
    """Shipped memories are searchable from any profile, without reindexing."""

    def test_terminal_help_recalled_on_a_fresh_profile(self, index):
        hits = index.search("how do I copy text to the clipboard", profile="default")
        assert any("OSC 52" in h.text for h in hits)

    def test_recalled_from_an_unrelated_profile(self, index):
        # A profile that was never reindexed still gets the built-in note.
        hits = index.search("paste not working in the terminal", profile="genetics")
        assert any("GNOME Terminal" in h.text for h in hits)

    def test_gnome_and_multiplexer_keywords_match(self, index):
        assert index.search("tmux clipboard gnome", profile="default")
        assert index.search("screen copy ubuntu terminal", profile="default")

    def test_irrelevant_query_does_not_recall_builtin(self, index):
        assert index.search("deepvariant joint genotyping", profile="default") == []

    def test_builtin_survives_profile_reindex(self, index):
        index.reindex(profile_with(("Snakemake dry-runs fail.", 3)))
        hits = index.search("copy paste clipboard terminal", profile="default")
        assert any("wl-clipboard" in h.text for h in hits)


class TestDemote:
    def test_moves_to_rag(self):
        profile = profile_with(("situational thing", SP))
        assert demote(profile, profile.memories) == 1
        assert profile.memories[0].scope is RAG

    def test_already_rag_is_a_noop(self):
        profile = profile_with(("situational thing", RAG))
        assert demote(profile, profile.memories) == 0

    def test_survives_save_load(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HPCA_HOME", str(tmp_path))
        profile = profile_with(("situational thing", SP))
        demote(profile, profile.memories)
        profile.save()
        loaded = Profile.load("default")
        assert loaded.memories[0].scope is RAG
        # and it is out of the injected scope entirely
        assert loaded.scope_text(MemoryScope.SYSTEM_PROMPT) == ""
