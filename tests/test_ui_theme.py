"""The palette, and the two properties a themeable one has to have.

That a colour named in the settings file reaches the escape sequence a row is
drawn with, and that changing it *while the app runs* reaches rows that were
already on screen. The second is the whole reason `ui.theme` is a module with a
lookup in it rather than a handful of constants, so most of what is here is
about the moment a save lands.
"""

import re

import pytest

from hpca.ui import rain, theme
from hpca.ui.ansi import RESET, pulse


@pytest.fixture(autouse=True)
def built_in_palette():
    """Every test starts and ends on the built-in palette.

    The palette is module state — one terminal per process is the argument —
    so a test that swapped it and walked away would be the next test's
    surprise.
    """
    theme.reset()
    yield
    theme.reset()


class TestWritingAColour:
    def test_an_xterm_index_is_the_256_colour_form(self):
        assert theme.sgr("215") == "\x1b[38;5;215m"

    def test_a_hex_triple_is_the_truecolor_form(self):
        assert theme.sgr("#ffaf5f") == "\x1b[38;2;255;175;95m"

    def test_a_background_is_the_same_colour_on_the_other_layer(self):
        assert theme.sgr("23", background=True) == "\x1b[48;5;23m"
        assert theme.sgr("#005f5f", background=True) == "\x1b[48;2;0;95;95m"

    @pytest.mark.parametrize("bad", ["", "256", "nope", "#ff88", "0x10", " 12"])
    def test_and_what_is_not_a_colour_is_not_written_at_all(self, bad):
        assert not theme.valid(bad)
        with pytest.raises(ValueError):
            theme.sgr(bad)


class TestReadingAColourBack:
    """`rgb` is what the ramp interpolates in, so it has to be right about
    all three regions of the 256-colour space, not only the cube."""

    def test_the_cube(self):
        assert theme.rgb("73") == (95, 175, 175)

    def test_the_grey_ramp(self):
        assert theme.rgb("232") == (8, 8, 8)
        assert theme.rgb("255") == (238, 238, 238)

    def test_and_a_hex_triple_is_already_the_answer(self):
        assert theme.rgb("#88ccff") == (136, 204, 255)


class TestTheRamp:
    def test_it_starts_and_ends_exactly_where_it_was_asked_to(self):
        walk = theme.ramp("255", "73", 7)
        assert (walk[0], walk[-1]) == ("255", "73")

    def test_an_index_walk_stays_in_the_cube(self):
        # Interpolated in level space rather than RGB: the cube's levels are
        # unevenly spaced, so an even RGB walk quantised back rounds the three
        # channels at different points and drops a grey step into the middle.
        walk = theme.ramp("255", "73", 7)
        assert all(x.isdigit() for x in walk)
        assert len(set(walk)) == len(walk), "and never repeats a colour"

    def test_a_hex_end_makes_the_whole_walk_truecolor(self):
        # It can: truecolor has every colour between, so there is nothing to
        # round to and the more faithful path is the available one.
        walk = theme.ramp("#000000", "#ffffff", 5)
        assert walk == ("#000000", "#404040", "#808080", "#bfbfbf", "#ffffff")

    def test_the_pulse_walks_between_the_two_colours_it_is_named_for(self):
        assert theme.pulse[0] == theme.agent
        assert theme.pulse[-1] == theme.chrome


class TestSwappingThePalette:
    def test_a_named_colour_takes_effect(self):
        theme.apply(chrome="#88ccff")
        assert theme.chrome == "\x1b[38;2;136;204;255m"

    def test_and_the_ones_not_named_keep_the_built_in(self):
        before = theme.user
        theme.apply(chrome="12")
        assert theme.user == before

    def test_the_pulse_is_rebuilt_with_it(self):
        # Derived, not stored: the ramp's ends *are* two palette entries, so a
        # theme that moves them and leaves the breath behind would have the
        # decision prompt fading towards a colour no longer on the screen.
        theme.apply(chrome="#ff0000")
        assert theme.pulse[-1] == "\x1b[38;2;255;0;0m"

    def test_a_colour_it_cannot_draw_costs_only_that_colour(self):
        # The settings model refuses these where the user can read the
        # refusal. This is the second line, in the process holding a terminal
        # in raw mode, and there a palette is not worth a lost screen.
        good = theme.user
        theme.apply(user="nonsense", chrome="12")
        assert theme.user == good, "the bad one fell back"
        assert theme.chrome == "\x1b[38;5;12m", "the good one still applied"

    def test_an_empty_trail_falls_back_rather_than_drawing_nothing(self):
        before = theme.spinner
        theme.apply(spinner=[])
        assert theme.spinner == before

    def test_and_reset_puts_the_built_in_one_back(self):
        theme.apply(chrome="9")
        theme.reset()
        assert theme.chrome == "\x1b[38;5;73m"


class TestWhatTheSwapReaches:
    """The point of the module: things that read a colour must read it *late*.

    Each of these is a place that used to hold a finished escape sequence in a
    module-level constant, and each of them would have gone on drawing the old
    palette after a save.
    """

    def test_the_spinner_trail(self):
        theme.apply(spinner=["9", "1"])
        assert rain.spinner_trail() == (RESET + "\x1b[38;5;9m", RESET + "\x1b[38;5;1m")

    def test_a_shorter_trail_fades_over_fewer_cells(self):
        # A ramp of one is a spinner with no fade in it, which the palette is
        # allowed to ask for. Not a spinner with holes: the cells the trail
        # does not reach take the ramp's dimmest end, which here is its only
        # end, so all four are that one colour (`rain.spinner`).
        theme.apply(spinner=["9"])
        glyphs, styles = rain.spinner(0)
        assert len(glyphs) == rain.SPINNER_WIDTH, "still four cells wide"
        assert set(styles) == {RESET + "\x1b[38;5;9m"}

    def test_the_rain_itself(self):
        theme.apply(ok="#ff0000")
        assert "38;2;255;0;0" in "".join(rain.rain(20, 6, 0.0))

    def test_the_decision_prompt_s_breath(self):
        theme.apply(agent="#ff0000", chrome="#ff0000")
        # Both ends the same colour: whatever instant is sampled, that is the
        # colour, which is what makes this readable without pinning a phase.
        assert "38;2;255;0;0" in pulse(0.3)

    def test_and_the_mode_line(self):
        from hpca.ui.state import mode_colour

        theme.apply(ok="#00ff00")
        assert mode_colour("auto") == "\x1b[38;2;0;255;0m"
        assert mode_colour("not-a-mode") == theme.faint


class TestTheFlash:
    def test_it_is_a_background_and_not_a_foreground(self):
        # The only background colour this UI draws. Everything else has always
        # been foreground over whatever ground the terminal paints.
        assert re.match(r"\x1b\[48;", theme.flash)

    def test_the_hold_comes_with_the_palette(self):
        theme.apply(flash_hold=0.25)
        assert theme.flash_hold == 0.25

    def test_and_a_nonsense_hold_falls_back(self):
        theme.apply(flash_hold=-1)
        assert theme.flash_hold == theme.FLASH_HOLD
