"""Tests for hpca.looplag: the event-loop lag probe (specs-core-process.md §8).

The probe is the instrument the core-subprocess refactor is judged with, so
these tests are about it being *trustworthy*, not about it being fast: a block
must show up at roughly its real size, the ring must stay bounded inside a
long-lived TUI, and the percentiles must be exactly what the samples say.

Timing assertions are one-sided on purpose. The suite runs in parallel on a
busy box, so "at least this much lag was seen" is safe while "no more than
this much" is a flake waiting to happen. Anything needing an exact number
feeds samples in through ``record()`` instead of waiting for the clock.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time

import pytest

from hpca.looplag import (
    DEFAULT_CAPACITY,
    DEFAULT_INTERVAL_S,
    LoopLagProbe,
    Spike,
)

# Short enough that the timing tests finish in a few hundredths of a second,
# long enough to be well clear of asyncio's own timer resolution.
TICK = 0.005

# The deliberate stall. Ten times the tick, so the drift it produces cannot be
# confused with ordinary scheduling noise even on a loaded machine.
BLOCK = 0.05


async def wait_for_samples(
    probe: LoopLagProbe, want: int, timeout: float = 5.0
) -> tuple[float, ...]:
    """Yield until the probe has ``want`` samples. Returns them.

    Polling with a deadline rather than sleeping for a fixed span: on an idle
    box this returns in milliseconds, and on a box that is swapping it still
    passes instead of timing out at some hand-picked constant.
    """
    deadline = time.perf_counter() + timeout
    while len(probe.samples) < want and time.perf_counter() < deadline:
        await asyncio.sleep(TICK)
    assert len(probe.samples) >= want, "the probe stopped sampling"
    return probe.samples


class TestSampling:
    async def test_a_quiet_loop_drifts_only_a_little(self):
        probe = LoopLagProbe(interval_s=TICK)
        probe.start()
        await wait_for_samples(probe, 5)
        await probe.stop()

        s = probe.summary()
        assert s["count"] >= 5
        assert all(x >= 0.0 for x in probe.samples)
        # A median of 50ms on a loop doing nothing would mean the machine is
        # too busy for any timing test in the suite to pass.
        assert s["p50"] < 0.05

    async def test_a_synchronous_block_is_seen_at_its_real_size(self):
        probe = LoopLagProbe(interval_s=TICK)
        probe.start()
        await wait_for_samples(probe, 2)
        before = len(probe.samples)

        time.sleep(BLOCK)  # the thing being measured: sync work on the loop

        await wait_for_samples(probe, before + 1)
        await probe.stop()

        # The tick the block swallowed was scheduled up to one interval before
        # the block began, so this is the strongest bound that is honest.
        assert probe.summary()["max"] >= BLOCK - TICK

    async def test_a_block_is_one_sample_not_a_smear(self):
        # Measuring drift against the previous *actual* run, or catching the
        # tick grid up afterwards, would turn one stall into ten decaying ones.
        probe = LoopLagProbe(interval_s=TICK)
        probe.start()
        await wait_for_samples(probe, 2)

        time.sleep(BLOCK)

        await wait_for_samples(probe, len(probe.samples) + 3)
        await probe.stop()

        # A smear would produce BLOCK/TICK ≈ 10 large samples with decaying
        # sizes; one unrelated hiccup on a loaded box produces two. The gap
        # between 2 and 10 is what makes this bound safe to assert.
        big = [x for x in probe.samples if x > BLOCK / 2]
        assert 1 <= len(big) <= 2, f"the block landed in {len(big)} samples"

    async def test_a_disabled_probe_records_nothing(self, tmp_path):
        probe = LoopLagProbe(interval_s=TICK, enabled=False)
        probe.start()
        await asyncio.sleep(5 * TICK)
        await probe.stop()

        assert not probe.running
        assert probe.samples == ()
        assert probe.summary()["count"] == 0
        # Left in a shutdown path permanently, it must also cost no file.
        report = tmp_path / "looplag.log"
        assert probe.write_report(report) is False
        assert not report.exists()

    def test_an_early_wake_is_not_negative_lag(self):
        # A loop may fire a timer up to its clock resolution early; recording
        # that as negative would quietly flatter the mean.
        probe = LoopLagProbe()
        probe.record(-0.004)
        assert probe.samples == (0.0,)


class TestLifecycle:
    async def test_start_is_idempotent_and_stop_leaks_no_task(self):
        outside = asyncio.all_tasks()
        probe = LoopLagProbe(interval_s=TICK)
        probe.start()
        probe.start()

        assert probe.running
        assert len(asyncio.all_tasks() - outside) == 1

        await probe.stop()
        await probe.stop()

        assert not probe.running
        # all_tasks() only lists tasks that are not done, so an empty
        # difference is the whole "no leaked task" claim.
        assert asyncio.all_tasks() - outside == set()

    async def test_stop_without_start_is_a_no_op(self):
        await LoopLagProbe().stop()

    async def test_it_can_be_restarted(self):
        probe = LoopLagProbe(interval_s=TICK)
        probe.start()
        await wait_for_samples(probe, 2)
        await probe.stop()
        paused = probe.elapsed

        probe.start()
        await wait_for_samples(probe, 4)
        await probe.stop()

        # Elapsed accumulates across cycles; the gap between them does not.
        assert probe.elapsed > paused


class TestRing:
    def test_the_ring_is_bounded(self):
        probe = LoopLagProbe(capacity=10)
        for i in range(50):
            probe.record(i / 1000)

        assert len(probe.samples) == 10
        assert probe.samples[0] == pytest.approx(0.040)
        assert probe.samples[-1] == pytest.approx(0.049)
        # The lifetime count still knows how many there really were.
        assert probe.summary()["total"] == 50

    def test_the_default_ring_holds_about_an_hour(self):
        assert DEFAULT_CAPACITY * DEFAULT_INTERVAL_S == pytest.approx(3600, rel=0.05)


class TestSummary:
    def test_percentiles_on_a_known_set(self):
        probe = LoopLagProbe()
        for i in range(1, 101):  # 1ms .. 100ms
            probe.record(i / 1000)

        s = probe.summary()
        assert s["count"] == 100
        assert s["mean"] == pytest.approx(0.0505)
        assert s["p50"] == pytest.approx(0.050)
        assert s["p90"] == pytest.approx(0.090)
        assert s["p99"] == pytest.approx(0.099)
        assert s["max"] == pytest.approx(0.100)

    def test_percentiles_do_not_interpolate(self):
        # A reported p99 has to be a delay that really happened, or the number
        # cannot be quoted as one.
        probe = LoopLagProbe()
        for value in (0.001, 0.002, 0.500):
            probe.record(value)

        s = probe.summary()
        assert s["p50"] == 0.002
        assert s["p99"] == 0.500

    def test_an_empty_probe_summarises_to_zeroes(self):
        s = LoopLagProbe().summary()
        assert s["count"] == 0
        assert s["mean"] == 0.0
        assert s["p99"] == 0.0
        assert s["max"] == 0.0
        assert s["worst"] == []

    def test_the_summary_is_json_ready(self):
        # It is meant to be loggable and shippable over the core protocol, so
        # the worst list holds dicts rather than dataclasses.
        probe = LoopLagProbe(worst_n=2, label=lambda: "read_file")
        probe.record(0.4)
        json.dumps(probe.summary())


class TestWorst:
    def test_it_keeps_the_biggest_and_caps_the_list(self):
        probe = LoopLagProbe(worst_n=3)
        for value in (0.01, 0.9, 0.02, 0.5, 0.03, 0.7):
            probe.record(value)

        worst = probe.summary()["worst"]
        assert [w["drift"] for w in worst] == [0.9, 0.7, 0.5]

    def test_a_spike_carries_a_wall_clock_stamp(self):
        started = time.time()
        probe = LoopLagProbe(worst_n=1)
        probe.record(0.3)

        spike = probe.summary()["worst"][0]
        assert started <= spike["at"] <= time.time()

    def test_the_label_hook_says_what_was_running(self):
        running = {"tool": "read_file"}
        probe = LoopLagProbe(
            worst_n=2, label=lambda: running["tool"], label_threshold_s=0.05
        )
        probe.record(0.2)
        running["tool"] = "index_docs"
        probe.record(0.9)

        worst = probe.summary()["worst"]
        assert [(w["drift"], w["label"]) for w in worst] == [
            (0.9, "index_docs"),
            (0.2, "read_file"),
        ]

    def test_the_label_hook_is_not_called_below_the_threshold(self):
        calls = []

        def label():
            calls.append(1)
            return "tool"

        probe = LoopLagProbe(worst_n=5, label=label, label_threshold_s=0.05)
        probe.record(0.001)
        probe.record(0.060)

        assert len(calls) == 1
        assert [w["label"] for w in probe.summary()["worst"]] == ["tool", ""]

    def test_a_broken_label_hook_does_not_break_the_probe(self):
        def label():
            raise RuntimeError("the app is mid-teardown")

        probe = LoopLagProbe(worst_n=1, label=label, label_threshold_s=0.0)
        probe.record(0.3)

        spike = probe.summary()["worst"][0]
        assert spike["drift"] == 0.3
        assert spike["label"] == ""

    def test_worst_survives_the_ring_falling_off_the_end(self):
        # The percentiles describe the window; the worst list describes the
        # run, which is what "did it ever hang" is asking.
        probe = LoopLagProbe(capacity=3, worst_n=1)
        probe.record(2.0)
        for _ in range(10):
            probe.record(0.001)

        assert probe.summary()["count"] == 3
        assert probe.summary()["worst"][0]["drift"] == 2.0


class TestReport:
    def test_it_creates_the_file_and_then_appends(self, tmp_path):
        path = tmp_path / "reports" / "looplag.log"  # a dir that does not exist
        probe = LoopLagProbe(worst_n=1)
        probe.record(0.25)

        assert probe.write_report(path, note="baseline") is True
        first = path.read_text()
        assert "=== looplag" in first
        assert "baseline" in first
        assert "samples 1 of 1" in first

        probe.record(0.75)
        assert probe.write_report(path, note="after") is True
        second = path.read_text()

        assert second.startswith(first)  # the baseline block is still there
        assert second.count("=== looplag") == 2
        assert "after" in second

    def test_the_block_carries_the_numbers_and_the_worst(self, tmp_path):
        path = tmp_path / "looplag.log"
        probe = LoopLagProbe(worst_n=2, label=lambda: "index_docs")
        probe.record(0.812)

        probe.write_report(path)
        text = path.read_text()

        assert "812.0ms" in text
        assert "index_docs" in text
        assert "p99" in text

    def test_an_unwritable_path_is_reported_not_raised(self, tmp_path):
        blocker = tmp_path / "not-a-dir"
        blocker.write_text("")
        probe = LoopLagProbe()
        probe.record(0.1)

        assert probe.write_report(blocker / "sub" / "looplag.log") is False

    def test_a_probe_with_no_samples_still_leaves_a_block(self, tmp_path):
        # "The probe was on and saw nothing" is a result, and telling it apart
        # from "nobody ran the probe" is the point of writing it down.
        path = tmp_path / "looplag.log"
        assert LoopLagProbe().write_report(path) is True
        assert "no samples recorded" in path.read_text()


def test_spike_is_a_plain_frozen_record():
    spike = Spike(drift=0.5, at=1.0, label="read_file")
    assert spike.as_dict() == {"drift": 0.5, "at": 1.0, "label": "read_file"}
    with pytest.raises(dataclasses.FrozenInstanceError):
        spike.drift = 0.1  # type: ignore[misc]
