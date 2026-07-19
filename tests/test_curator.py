"""Tests for the curator: ageing tier-3 memories out (redesign Phase 6)."""

from datetime import date, datetime, timedelta, timezone

import pytest

from hpca import curator
from hpca.profiles import Profile


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


def days_ago(n):
    return (date.today() - timedelta(days=n)).isoformat()


def profile_with(*entries, name="default"):
    """entries: (text, tier, age_days[, kind])"""
    profile = Profile(name=name)
    for entry in entries:
        memory = profile.add_memory(entry[0], tier=entry[1])
        memory.created = days_ago(entry[2])
        if len(entry) > 3:
            memory.kind = entry[3]
    return profile


class TestCurate:
    def test_old_tier3_is_archived(self, hpca_home):
        profile = profile_with(("an old struggle note", 3, 120))
        report = curator.curate(profile)
        assert len(report.archived) == 1
        assert profile.memories == []

    def test_middle_aged_is_only_flagged_stale(self, hpca_home):
        profile = profile_with(("a note", 3, 45))
        report = curator.curate(profile)
        assert report.archived == []
        assert len(report.stale) == 1
        assert len(profile.memories) == 1  # still there

    def test_recent_is_left_alone(self, hpca_home):
        profile = profile_with(("a note", 3, 5))
        report = curator.curate(profile)
        assert report.archived == [] and report.stale == []

    def test_injected_tiers_are_never_aged(self, hpca_home):
        """Tiers 1 and 2 are small, curated, and in front of the user
        already; tier 3 is the one that grows unattended."""
        profile = profile_with(
            ("an ancient site fact", 1, 900), ("an ancient preference", 2, 900)
        )
        report = curator.curate(profile)
        assert report.archived == []
        assert len(profile.memories) == 2

    def test_pinned_entries_survive(self, hpca_home):
        profile = profile_with(("keep me forever", 3, 900, "pinned"))
        report = curator.curate(profile)
        assert report.archived == []
        assert len(profile.memories) == 1

    def test_undated_entries_are_left_alone(self, hpca_home):
        """A hand-written entry with no date must not be aged out on a
        guess."""
        profile = Profile(name="default")
        memory = profile.add_memory("hand written", tier=3)
        memory.created = ""
        assert curator.curate(profile).archived == []
        assert len(profile.memories) == 1

    def test_thresholds_configurable(self, hpca_home):
        profile = profile_with(("a note", 3, 10))
        report = curator.curate(profile, stale_days=5, archive_days=8)
        assert len(report.archived) == 1


class TestArchive:
    def test_archive_file_written_and_recoverable(self, hpca_home):
        profile = profile_with(("an old struggle note", 3, 120))
        profile.save()
        reports = curator.run(["default"])
        assert "default" in reports
        archive = curator.archive_path("default")
        assert archive.exists()
        text = archive.read_text()
        assert "an old struggle note" in text
        assert "archived:" in text  # when, so it can be judged later
        # and it is out of the live profile
        assert Profile.load("default").memories == []

    def test_archive_appends_across_runs(self, hpca_home):
        profile = profile_with(("first old note", 3, 120))
        profile.save()
        curator.run(["default"])
        curator.save_state({})  # force the next run to be due
        profile = Profile.load("default")
        memory = profile.add_memory("second old note", tier=3)
        memory.created = days_ago(200)
        profile.save()
        curator.run(["default"])
        text = curator.archive_path("default").read_text()
        assert "first old note" in text and "second old note" in text

    def test_nothing_to_do_leaves_no_archive(self, hpca_home):
        profile_with(("a fresh note", 3, 1)).save()
        curator.run(["default"])
        assert not curator.archive_path("default").exists()


class TestDue:
    def test_due_when_never_run(self, hpca_home):
        assert curator.due(state={})

    def test_not_due_right_after_a_run(self, hpca_home):
        now = datetime.now(timezone.utc)
        state = {"last_run": now.isoformat()}
        assert not curator.due(state=state, now=now, interval_days=7)

    def test_due_after_the_interval(self, hpca_home):
        now = datetime.now(timezone.utc)
        state = {"last_run": (now - timedelta(days=8)).isoformat()}
        assert curator.due(state=state, now=now, interval_days=7)

    def test_unreadable_timestamp_does_not_run(self, hpca_home):
        """Failing open would mean curating on every idle tick."""
        assert not curator.due(state={"last_run": "not-a-date"})

    def test_run_records_the_timestamp(self, hpca_home):
        curator.run(["default"])
        assert curator.load_state().get("last_run")
        assert not curator.due()


class TestMultipleProfiles:
    def test_each_profile_curated_separately(self, hpca_home):
        profile_with(("genetics old note", 3, 120), name="genetics").save()
        profile_with(("admin fresh note", 3, 2), name="hpc-admin").save()
        reports = curator.run(["genetics", "hpc-admin"])
        assert "genetics" in reports
        assert "hpc-admin" not in reports
        assert curator.archive_path("genetics").exists()
        assert not curator.archive_path("hpc-admin").exists()


class TestArchiveMetadata:
    def test_tier_recorded_for_restoration(self, hpca_home):
        """Restoring means moving the block back under the right heading, and
        the parser accepts it anywhere — so the tier has to be written down."""
        profile_with(("an old note", 3, 120)).save()
        curator.run(["default"])
        text = curator.archive_path("default").read_text()
        assert "tier: 3" in text
        assert "tierN" in text  # the how-to-restore header
