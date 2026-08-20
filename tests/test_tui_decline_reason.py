"""Declining with a reason (§5.3): "n" on a gated call opens a box for why,
and whatever is typed there rides back to the model with the refusal.

The point is the corrected script. A bare refusal tells the model only to
stop; "wrong partition — use gpu" tells it what to fix, so the next thing it
puts up for approval is worth reading. Enter sends either way — an empty box
is the plain refusal it has always been — and escape stays the way out
without explaining.
"""

import json

import pytest
from pydantic import BaseModel, Field
from textual.widgets import Static

from hpca.agent.tools import Tool, ToolRegistry
from hpca.llm import ChatResponse
from hpca.tui.app import ChatInput, DecisionBar, HpcaApp, ReasonInput


def is_title_request(json_schema):
    return bool(json_schema) and "title" in (json_schema.get("properties") or {})


TITLE_REPLY = json.dumps({"title": "a test session"})


def tool_json(tool, **arguments):
    return json.dumps({"action": "tool_call", "tool": tool, "arguments": arguments})


def respond_json(text="done"):
    return json.dumps({"action": "respond", "response": text})


class DeleteParams(BaseModel):
    target: str = Field(description="What to delete")


async def delete_handler(args, ctx):
    return f"deleted {args.target}"


def destructive_tools():
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="delete",
            description="Delete something",
            params=DeleteParams,
            handler=delete_handler,
            destructive=True,
        )
    )
    return registry


class RecordingLLM:
    """Keeps every message list it was handed, so a test can check what the
    model actually saw rather than only what the chat log shows."""

    def __init__(self, outputs):
        self._outputs = list(outputs)
        self.calls = []

    async def chat(self, messages, *, json_schema=None, **kwargs):
        if is_title_request(json_schema):
            return ChatResponse(content=TITLE_REPLY)
        self.calls.append(messages)
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True


@pytest.fixture
def hpca_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


def decision_bar(app):
    return app.query_one("#decision-bar", DecisionBar)


def reason_box(app):
    found = app.query("#decision-reason")
    return found.first(ReasonInput) if found else None


def chat_texts(app):
    return app.chat_log_texts()


def stage(app, session=None):
    session = session or app.active_session
    return app._pending_decision[session.session_id].get("stage")


def bar_texts(app):
    """Every line the inline prompt is showing."""
    return [str(child.render()) for child in decision_bar(app).query(Static)]


async def submit_chat(app, pilot, text):
    if app.active_session is None:
        await app.start_new_session()
    chat_input = app.query_one("#chat-input", ChatInput)
    chat_input.focus()
    chat_input.text = text
    await pilot.press("enter")
    await app.workers.wait_for_complete()
    await pilot.pause()


async def park_on_a_decision(app, pilot):
    """Send a message whose tool call parks the turn on the approval prompt."""
    await submit_chat(app, pilot, "delete results")
    assert decision_bar(app).display, "expected the turn to park on the prompt"


class TestOpeningTheBox:
    async def test_n_opens_a_box_and_answers_nothing_yet(self, hpca_home):
        app = HpcaApp(
            llm=RecordingLLM([tool_json("delete", target="results/")]),
            tools=destructive_tools(),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await park_on_a_decision(app, pilot)
            await pilot.press("n")
            await pilot.pause()

            box = reason_box(app)
            assert box is not None, "declining must offer somewhere to say why"
            assert app.focused is box, "the box is where the next keystroke goes"
            # The refusal is not sent until the box is answered: the turn is
            # still parked and the model has not been called again.
            assert stage(app) == "reason"
            assert not any("DENIED" in t for t in chat_texts(app))
            assert len(app._llm.calls) == 1

    async def test_the_call_stays_on_screen_while_the_reason_is_typed(
        self, hpca_home
    ):
        app = HpcaApp(
            llm=RecordingLLM([tool_json("delete", target="results/")]),
            tools=destructive_tools(),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await park_on_a_decision(app, pilot)
            await pilot.press("n")
            await pilot.pause()
            # What is being declined has to stay readable — the reason is
            # written *about* it. What that is, is the call: this tool has no
            # description of its own to give, so the arguments are it.
            joined = "\n".join(bar_texts(app))
            assert "target: results/" in joined

    async def test_the_box_takes_the_keys_the_prompt_used(self, hpca_home):
        app = HpcaApp(
            llm=RecordingLLM([tool_json("delete", target="results/")]),
            tools=destructive_tools(),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await park_on_a_decision(app, pilot)
            await pilot.press("n")
            await pilot.pause()
            # "y" and "n" are letters again once the box is open, not a second
            # answer to a question already answered.
            await pilot.press("n", "o", "y")
            await pilot.pause()
            assert reason_box(app).text == "noy"
            assert stage(app) == "reason"


class TestSending:
    async def test_enter_sends_the_typed_reason_to_the_model(self, hpca_home):
        app = HpcaApp(
            llm=RecordingLLM(
                [tool_json("delete", target="results/"), respond_json("understood")]
            ),
            tools=destructive_tools(),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await park_on_a_decision(app, pilot)
            await pilot.press("n")
            await pilot.pause()
            reason_box(app).text = "results/ is the wrong path, use scratch/"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            # It reached the model, not just the screen.
            second_call = app._llm.calls[1]
            assert any(
                "results/ is the wrong path, use scratch/" in m["content"]
                for m in second_call
            )
            # and the tool did not run
            assert not any("deleted results/" in t for t in chat_texts(app))
            # the decision is answered and the prompt gone
            assert app.active_session.session_id not in app._pending_decision
            assert not decision_bar(app).display

    async def test_an_empty_box_is_the_plain_refusal(self, hpca_home):
        app = HpcaApp(
            llm=RecordingLLM(
                [tool_json("delete", target="results/"), respond_json("understood")]
            ),
            tools=destructive_tools(),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await park_on_a_decision(app, pilot)
            await pilot.press("n")
            await pilot.pause()
            await pilot.press("enter")  # nothing typed: refuse without a reason
            await app.workers.wait_for_complete()
            await pilot.pause()

            texts = chat_texts(app)
            assert any("DENIED" in t for t in texts)
            assert not any("deleted results/" in t for t in texts)
            assert app.active_session.session_id not in app._pending_decision

    async def test_escape_refuses_without_asking_why(self, hpca_home):
        app = HpcaApp(
            llm=RecordingLLM(
                [tool_json("delete", target="results/"), respond_json("understood")]
            ),
            tools=destructive_tools(),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await park_on_a_decision(app, pilot)
            # escape is the way out of the prompt, so it must not open a box
            # the user then has to get out of as well.
            await pilot.press("escape")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert reason_box(app) is None
            assert any("DENIED" in t for t in chat_texts(app))
            assert app.active_session.session_id not in app._pending_decision

    async def test_escape_out_of_the_box_still_refuses(self, hpca_home):
        app = HpcaApp(
            llm=RecordingLLM(
                [tool_json("delete", target="results/"), respond_json("understood")]
            ),
            tools=destructive_tools(),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await park_on_a_decision(app, pilot)
            await pilot.press("n")
            await pilot.pause()
            reason_box(app).text = "changed my mind about explaining"
            await pilot.press("escape")
            await app.workers.wait_for_complete()
            await pilot.pause()
            # The answer was already "no" when the box opened; leaving it
            # sends that, without the half-written reason.
            texts = chat_texts(app)
            assert any("DENIED" in t for t in texts)
            assert not any("changed my mind" in t for t in texts)
            assert app.active_session.session_id not in app._pending_decision

    async def test_approving_never_asks_why(self, hpca_home):
        app = HpcaApp(
            llm=RecordingLLM(
                [tool_json("delete", target="results/"), respond_json("it is gone")]
            ),
            tools=destructive_tools(),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await park_on_a_decision(app, pilot)
            await pilot.press("y")
            await app.workers.wait_for_complete()
            await pilot.pause()
            assert reason_box(app) is None
            assert any("deleted results/" in t for t in chat_texts(app))


class TestManualSkip:
    async def test_a_skipped_script_carries_the_reason(self, hpca_home):
        app = HpcaApp(  # manual mode gates run_bash on the script itself
            llm=RecordingLLM(
                [
                    tool_json("run_bash", content_lines=["squeue -u me"]),
                    respond_json("understood"),
                ]
            )
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await submit_chat(app, pilot, "how is my job doing")
            assert decision_bar(app).display
            await pilot.press("n")
            await pilot.pause()
            reason_box(app).text = "that only shows mine, ask for the whole partition"
            await pilot.press("enter")
            await app.workers.wait_for_complete()
            await pilot.pause()

            texts = chat_texts(app)
            assert any("SKIPPED" in t for t in texts)
            assert any("ask for the whole partition" in t for t in texts)


class TestSessionSwitching:
    async def test_a_half_written_reason_waits_in_its_own_session(self, hpca_home):
        app = HpcaApp(
            llm=RecordingLLM(
                [tool_json("delete", target="results/"), respond_json("understood")]
            ),
            tools=destructive_tools(),
        )
        async with app.run_test(size=(120, 40)) as pilot:
            await park_on_a_decision(app, pilot)
            parked = app.active_session
            await pilot.press("n")
            await pilot.pause()
            reason_box(app).text = "wrong path"

            await app.start_new_session()  # leave mid-sentence
            await pilot.pause()
            assert reason_box(app) is None  # nothing of it in the new session

            await app.open_session(parked)
            await pilot.pause()
            # back where it was left: still the box, still the words
            box = reason_box(app)
            assert box is not None
            assert box.text == "wrong path"
            assert stage(app, parked) == "reason"
