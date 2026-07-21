"""A toast must never take the app down, whatever its text.

Toasts carry dynamic strings — LLM output, exception messages, memory
excerpts. When Textual renders a notification as content *markup*, text like
``$(mk`` or ``[foo]`` raises ``MarkupError`` deep inside the compositor and the
whole app crashes on the next layout. ``HpcaApp.notify`` defaults ``markup`` to
False so such text is shown literally instead.
"""

import pytest

from hpca.llm import ChatResponse
from hpca.tui.app import HpcaApp

# Fragments that are valid-ish Textual markup and used to crash the toast: the
# real dump was a truncated LLM payload containing ``$(mk`` and ``[...]``.
MARKUP_TRAPS = (
    'Structured output truncated (model looped?): {"action": "tool_call $(mk',
    "unmatched [bracket and $(expr",
    "[bold]never rendered as style[/bold]",
)


class FakeLLM:
    async def chat(self, messages, *, json_schema=None, **kwargs):
        return ChatResponse(content="")

    async def supports_constrained_decoding(self):
        return True


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("message", MARKUP_TRAPS)
async def test_markup_like_toast_does_not_crash(hpca_home, message):
    app = HpcaApp(llm=FakeLLM())
    async with app.run_test(size=(120, 30)) as pilot:
        app.notify(message, severity="error")
        await pilot.pause()
        await pilot.pause()
        assert app.is_running


async def test_caller_can_still_opt_into_markup(hpca_home):
    app = HpcaApp(llm=FakeLLM())
    async with app.run_test(size=(120, 30)) as pilot:
        app.notify("[bold]styled[/bold]", markup=True)
        await pilot.pause()
        assert app.is_running
