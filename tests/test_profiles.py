"""Tests for hpca.profiles: two-tier markdown memory store (§6)."""

import pytest

from hpca.profiles import Memory, Profile, estimate_tokens, profiles_dir


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


class TestNewProfile:
    def test_load_missing_creates_empty(self, hpca_home):
        profile = Profile.load("default")
        assert profile.name == "default"
        assert profile.memories == []
        assert profile.problems == []

    def test_save_creates_file(self, hpca_home):
        Profile.load("default").save()
        assert (profiles_dir() / "default.md").exists()

    def test_list_profiles(self, hpca_home):
        Profile.load("alpha").save()
        Profile.load("beta").save()
        assert Profile.list_profiles() == ["alpha", "beta"]


class TestRoundTrip:
    def test_memories_survive_save_load(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory(
            "Cluster is called cubi, scheduler is Slurm 25.05.",
            tier=1, backend="qwen3-6b", kind="fact",
        )
        profile.add_memory(
            "STAR alignments need at least 40G of memory here.",
            tier=2, backend="qwen3-6b", kind="learning",
        )
        profile.save()
        loaded = Profile.load("default")
        assert len(loaded.memories) == 2
        tier1 = [m for m in loaded.memories if m.tier == 1]
        assert tier1[0].text == "Cluster is called cubi, scheduler is Slurm 25.05."
        assert tier1[0].backend == "qwen3-6b"
        assert tier1[0].kind == "fact"
        tier2 = [m for m in loaded.memories if m.tier == 2]
        assert "STAR" in tier2[0].text

    def test_front_matter_preserved(self, hpca_home):
        profile = Profile.load("default")
        profile.default_backend = "qwen3-6b"
        profile.save()
        loaded = Profile.load("default")
        assert loaded.default_backend == "qwen3-6b"
        assert loaded.created  # stamped on creation

    def test_multiline_memory_text(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("line one\nline two\nline three", tier=2)
        profile.save()
        loaded = Profile.load("default")
        assert loaded.memories[0].text == "line one\nline two\nline three"


class TestLenientParsing:
    def write(self, hpca_home, content):
        profiles_dir().mkdir(parents=True, exist_ok=True)
        (profiles_dir() / "edited.md").write_text(content)
        return Profile.load("edited")

    def test_hand_written_block_without_metadata(self, hpca_home):
        profile = self.write(
            hpca_home,
            "---\nname: edited\n---\n\n## [tier1]\n\n"
            "The site firewall blocks outgoing traffic.\n",
        )
        assert len(profile.memories) == 1
        assert profile.memories[0].tier == 1
        assert profile.memories[0].backend == ""
        assert "firewall" in profile.memories[0].text

    def test_moving_block_between_headings_changes_tier(self, hpca_home):
        profile = self.write(
            hpca_home,
            "---\nname: edited\n---\n\n## [tier1]\n\n## [tier2]\n\n"
            "<!-- backend: q, created: 2026-01-01, kind: fact -->\nmoved here\n",
        )
        assert profile.memories[0].tier == 2

    def test_missing_front_matter_reports_problem_but_parses(self, hpca_home):
        profile = self.write(
            hpca_home, "## [tier2]\n\nremember this\n"
        )
        assert profile.problems  # reported
        assert profile.memories[0].text == "remember this"

    def test_unknown_heading_reports_problem(self, hpca_home):
        profile = self.write(
            hpca_home,
            "---\nname: edited\n---\n\n## random notes\n\nnot a memory\n\n"
            "## [tier2]\n\na real memory\n",
        )
        assert any("random notes" in p for p in profile.problems)
        assert [m.text for m in profile.memories] == ["a real memory"]

    def test_tier3_accepted_in_format(self, hpca_home):
        # deferred tier (§6.1): parsed, stored, not injected anywhere yet
        profile = self.write(
            hpca_home, "---\nname: edited\n---\n\n## [tier3]\n\nfuture memory\n"
        )
        assert profile.memories[0].tier == 3

    def test_empty_file(self, hpca_home):
        profile = self.write(hpca_home, "")
        assert profile.memories == []

    def test_malformed_metadata_comment_kept_as_text_problem(self, hpca_home):
        profile = self.write(
            hpca_home,
            "---\nname: edited\n---\n\n## [tier2]\n\n"
            "<!-- backend qwen no colons here -->\nsome memory text\n",
        )
        assert profile.memories[0].text == "some memory text"


class TestTierAccess:
    def test_tier_text_joins_memories(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("fact one", tier=1)
        profile.add_memory("fact two", tier=1)
        profile.add_memory("tier2 thing", tier=2)
        text = profile.tier_text(1)
        assert "fact one" in text and "fact two" in text
        assert "tier2 thing" not in text

    def test_tier_tokens_positive_and_scales(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("word " * 400, tier=2)
        assert profile.tier_tokens(2) > 100
        assert profile.tier_tokens(1) == 0

    def test_over_cap_tiers(self, hpca_home):
        profile = Profile.load("default")
        profile.add_memory("x" * 4000, tier=1)  # ~1000 tokens
        assert profile.over_cap_tiers(tier1_cap=300, tier2_cap=800) == [1]


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0

    def test_reasonable_scale(self):
        text = "the quick brown fox jumps over the lazy dog " * 50
        tokens = estimate_tokens(text)
        assert 200 < tokens < 1200
