"""Tests for the context meter (rendering and thresholds)."""

from hpca.tui.context_bar import (
    BAR_CELLS,
    ContextBar,
    render_bar,
    severity,
)


class TestRenderBar:
    def test_shows_used_window_and_percentage(self):
        text = render_bar(8000, 32000)
        assert "8,000 / 32,000" in text
        assert "(25%)" in text

    def test_bar_length_is_constant(self):
        for used in (0, 1, 16_000, 32_000):
            bar = render_bar(used, 32_000).split("[")[1].split("]")[0]
            assert len(bar) == BAR_CELLS

    def test_fill_tracks_the_fraction(self):
        def filled(text):
            return text.split("[")[1].split("]")[0].count("█")

        assert filled(render_bar(0, 32_000)) == 0
        assert filled(render_bar(16_000, 32_000)) == BAR_CELLS // 2
        assert filled(render_bar(32_000, 32_000)) == BAR_CELLS

    def test_overflow_does_not_exceed_the_bar(self):
        """A prompt over the window must not render a longer bar than the
        widget has room for."""
        text = render_bar(99_000, 32_000)
        assert text.split("[")[1].split("]")[0].count("█") == BAR_CELLS
        assert "(100%)" in text

    def test_unknown_window_says_so_rather_than_guessing(self):
        text = render_bar(8000, None)
        assert "window unknown" in text
        assert "8,000" in text

    def test_estimates_are_marked(self):
        assert "~8,000" in render_bar(8000, 32_000, estimated=True)
        assert "~" not in render_bar(8000, 32_000)


class TestSeverity:
    def test_thresholds(self):
        assert severity(1_000, 32_000) == "ok"
        assert severity(22_000, 32_000) == "ok"       # 69%
        assert severity(22_400, 32_000) == "warn"     # 70%, compaction nears
        assert severity(29_000, 32_000) == "danger"   # 91%

    def test_unknown_window(self):
        assert severity(1000, None) == "unknown"

    def test_small_window_reaches_danger_quickly(self):
        """The case this exists for: on 32k a few tool results fill it."""
        assert severity(29_500, 32_768) == "danger"
        # the same prompt is unremarkable on a large model
        assert severity(29_500, 192_000) == "ok"


class TestWidgetState:
    def test_no_reply_yet(self):
        bar = ContextBar()
        bar.set_window(32_000)
        assert "no reply yet" in bar.text
        assert "32,000" in bar.text

    def test_measured_value_replaces_the_placeholder(self):
        bar = ContextBar()
        bar.set_window(32_000)
        bar.set_used(8_000)
        assert "8,000 / 32,000" in bar.text

    def test_measured_supersedes_an_estimate(self):
        bar = ContextBar()
        bar.set_window(32_000)
        bar.set_estimate(9_000)
        assert "~9,000" in bar.text
        bar.set_used(8_123)
        rendered = bar.text
        assert "8,123" in rendered
        assert "~" not in rendered

    def test_reset_clears_to_no_reply(self):
        bar = ContextBar()
        bar.set_window(32_000)
        bar.set_used(8_000)
        bar.reset()
        assert "no reply yet" in bar.text

    def test_window_can_arrive_after_the_usage(self):
        """Discovery is async: a reply can land before the probe answers."""
        bar = ContextBar()
        bar.set_used(8_000)
        assert "window unknown" in bar.text
        bar.set_window(32_000)
        assert "8,000 / 32,000" in bar.text
