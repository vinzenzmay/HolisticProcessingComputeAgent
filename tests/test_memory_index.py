"""Tests for the tier-3 retrieval index (redesign Phase 5)."""

from datetime import date, timedelta

import pytest

from hpca.db import connect, init_db
from hpca.memory_index import MemoryIndex, demote
from hpca.profiles import Profile


@pytest.fixture
def index(tmp_path):
    conn = connect(tmp_path / "hpca.db")
    init_db(conn)
    yield MemoryIndex(conn)
    conn.close()


def profile_with(*entries, name="default"):
    """entries: (text, tier[, backend, created, kind])"""
    profile = Profile(name=name)
    for entry in entries:
        text, tier = entry[0], entry[1]
        memory = profile.add_memory(text, tier=tier)
        if len(entry) > 2:
            memory.backend = entry[2]
        if len(entry) > 3:
            memory.created = entry[3]
        if len(entry) > 4:
            memory.kind = entry[4]
    return profile


class TestIndexing:
    def test_only_tier3_indexed(self, index):
        profile = profile_with(
            ("Cluster is cubi.", 1),
            ("User prefers R.", 2),
            ("Snakemake dry-runs fail with site profiles.", 3),
        )
        assert index.reindex(profile) == 1
        hits = index.search("snakemake", profile="default")
        assert len(hits) == 1

    def test_reindex_replaces_not_appends(self, index):
        index.reindex(profile_with(("bwa mem needs a prebuilt index.", 3)))
        index.reindex(profile_with(("bwa mem needs a prebuilt index.", 3)))
        assert len(index.search("bwa", profile="default")) == 1

    def test_removed_memory_disappears(self, index):
        index.reindex(profile_with(("bwa mem needs an index.", 3)))
        index.reindex(profile_with())
        assert index.search("bwa", profile="default") == []

    def test_profiles_isolated(self, index):
        index.reindex(profile_with(("deepvariant needs a GPU.", 3), name="genetics"))
        index.reindex(profile_with(("drain the node first.", 3), name="hpc-admin"))
        assert len(index.search("deepvariant", profile="genetics")) == 1
        assert index.search("deepvariant", profile="hpc-admin") == []

    def test_forget_profile(self, index):
        index.reindex(profile_with(("deepvariant needs a GPU.", 3), name="genetics"))
        index.forget_profile("genetics")
        assert index.search("deepvariant", profile="genetics") == []


class TestSearch:
    def test_matches_any_term(self, index):
        index.reindex(
            profile_with(("Snakemake dry-runs fail with site profiles.", 3))
        )
        assert index.search("my snakemake workflow broke", profile="default")

    def test_no_match(self, index):
        index.reindex(profile_with(("Snakemake dry-runs fail.", 3)))
        assert index.search("kubernetes ingress", profile="default") == []

    def test_short_words_ignored(self, index):
        """Two-letter words would match nearly everything."""
        index.reindex(profile_with(("Snakemake dry-runs fail.", 3)))
        assert index.search("is it ok", profile="default") == []

    def test_limit_respected(self, index):
        index.reindex(
            profile_with(*[(f"bam handling note {i}", 3) for i in range(6)])
        )
        assert len(index.search("bam", profile="default", limit=2)) == 2

    def test_active_backend_ranks_higher(self, index):
        recent = date.today().isoformat()
        index.reindex(
            profile_with(
                ("bam indexing is slow here", 3, "old-model", recent),
                ("bam indexing is slow here too", 3, "qwen3-35b", recent),
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
                ("bam indexing is slow here", 3, "", old),
                ("bam indexing is slow here too", 3, "", recent),
            )
        )
        hits = index.search("bam indexing", profile="default")
        assert hits[0].created == recent

    def test_malformed_date_does_not_crash(self, index):
        index.reindex(profile_with(("bam note", 3, "", "not-a-date")))
        assert len(index.search("bam", profile="default")) == 1


class TestDemote:
    def test_moves_to_tier3(self):
        profile = profile_with(("situational thing", 2))
        assert demote(profile, profile.memories) == 1
        assert profile.memories[0].tier == 3

    def test_already_tier3_is_a_noop(self):
        profile = profile_with(("situational thing", 3))
        assert demote(profile, profile.memories) == 0

    def test_survives_save_load(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HPCA_HOME", str(tmp_path))
        profile = profile_with(("situational thing", 2))
        demote(profile, profile.memories)
        profile.save()
        loaded = Profile.load("default")
        assert loaded.memories[0].tier == 3
        # and it is out of the injected tiers entirely
        assert loaded.tier_text(2) == ""
