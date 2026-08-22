"""The context meter's two pure functions: the bar and the thresholds.

Written against `hpca/tui/context_bar.py` and repointed at `hpca/ui/meter.py`,
which is where `render_bar` and `severity` now live — they were *copied* out of
the Textual module rather than shared with it (see `meter`'s docstring), so
this file would have become a collection error the day `tui/` went.

What went with the widget is the widget's own state — measured versus
estimated, the speed, the thinking level. That is `state.Context` now, and it
is asserted claim for claim in
`tests/test_ui_turn.py::TestTheContextMeterState`. The two assertions with no
successor there — an overflowing bar staying inside its cells, and a small
window reaching danger where a large one does not — are why the rest of this
file stayed.
"""

from hpca.ui.meter import BAR_CELLS, render_bar, severity


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
