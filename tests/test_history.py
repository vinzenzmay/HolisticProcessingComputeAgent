"""Tests for hpca.agent.history: the two messages one tool call adds (§4.3).

The model has to see its own call in the shape it was trained on — an
assistant message carrying the call, then the result answering it — without
paying for the payload a second time.
"""

import json

from pydantic import BaseModel, Field

from hpca.agent.history import (
    MAX_LIST_ITEMS,
    MAX_STRING_CHARS,
    elide_arguments,
    is_tool_call_message,
    result_message,
    tool_exchange,
)
from hpca.agent.middleware import decision_schema
from hpca.agent.tools import Tool, ToolRegistry


class TestExchangeShape:
    def test_two_messages_call_then_result(self):
        messages = tool_exchange("read_file", {"registry_key": "cohort"}, "12 lines")
        assert [m["role"] for m in messages] == ["assistant", "user"]

    def test_the_assistant_message_is_the_decision_envelope(self):
        messages = tool_exchange("read_file", {"registry_key": "cohort"}, "ok")
        assert json.loads(messages[0]["content"]) == {
            "action": "tool_call",
            "tool": "read_file",
            "arguments": {"registry_key": "cohort"},
        }

    def test_the_envelope_matches_what_the_schema_constrains(self):
        # The point of the assistant copy is that it is what the model itself
        # would have written; if the envelope drifts from decision_schema the
        # history teaches the model a format it is not allowed to emit.
        class EchoParams(BaseModel):
            text: str = Field(description="Text to echo")

        async def echo(args, ctx):
            return "ok"

        tools = ToolRegistry()
        tools.register(
            Tool(name="echo", description="Echo", params=EchoParams, handler=echo)
        )
        branch = next(
            b
            for b in decision_schema(tools)["anyOf"]
            if b["properties"]["action"]["const"] == "tool_call"
        )
        envelope = json.loads(tool_exchange("echo", {"text": "hi"}, "ok")[0]["content"])
        assert set(envelope) == set(branch["required"])
        assert envelope["tool"] == branch["properties"]["tool"]["const"]

    def test_the_result_message_is_unchanged_from_before(self):
        # Byte-identical to what the graph has always written: the model has
        # learned this prefix, and the transcript keys tool steps off it.
        messages = tool_exchange("run_bash", {}, "exit 0\nhello")
        assert messages[1] == {
            "role": "user",
            "content": "[tool result] run_bash: exit 0\nhello",
        }

    def test_non_ascii_survives_the_serialization(self):
        messages = tool_exchange("echo", {"text": "µ-Ansatz — 5 °C"}, "ok")
        assert "µ-Ansatz — 5 °C" in messages[0]["content"]

    def test_the_call_is_recognisable_as_a_call(self):
        call, result = tool_exchange("echo", {"text": "hi"}, "ok")
        assert is_tool_call_message(call)
        assert not is_tool_call_message(result)

    def test_a_real_answer_is_not_a_call(self):
        assert not is_tool_call_message(
            {"role": "assistant", "content": 'The file starts {"action": "tool_call"}'}
        )
        assert not is_tool_call_message({"role": "assistant", "content": "Four BAMs."})


class TestElision:
    """A create_file call carries the whole file in ``content_lines``. Echoing
    it verbatim would store the file twice in the window on exactly the turns
    that are already tight, so the assistant copy keeps the shape, not the
    payload."""

    def test_a_long_list_keeps_a_head_and_says_what_was_dropped(self):
        lines = [f"line {i}" for i in range(100)]
        elided = elide_arguments({"path": "notes.md", "content_lines": lines})
        assert elided["path"] == "notes.md"  # the target is never touched
        assert elided["content_lines"][:3] == ["line 0", "line 1", "line 2"]
        assert elided["content_lines"][3] == "... 97 more lines elided ..."
        assert len(elided["content_lines"]) == 4

    def test_a_list_at_the_limit_is_left_whole(self):
        lines = [f"line {i}" for i in range(MAX_LIST_ITEMS)]
        assert elide_arguments({"content_lines": lines})["content_lines"] == lines

    def test_one_element_over_the_limit_elides(self):
        lines = [f"line {i}" for i in range(MAX_LIST_ITEMS + 1)]
        elided = elide_arguments({"content_lines": lines})["content_lines"]
        assert elided[-1] == f"... {MAX_LIST_ITEMS + 1 - 3} more lines elided ..."

    def test_a_long_string_is_truncated_and_says_so(self):
        text = "x" * (MAX_STRING_CHARS + 50)
        elided = elide_arguments({"script": text})["script"]
        assert elided.startswith("x" * MAX_STRING_CHARS)
        assert elided.endswith("... 50 more chars elided ...")

    def test_a_short_string_is_left_alone(self):
        assert elide_arguments({"path": "/data/x.bam"}) == {"path": "/data/x.bam"}

    def test_other_scalars_pass_through(self):
        arguments = {"timeout_s": 60, "recursive": True, "start": None}
        assert elide_arguments(arguments) == arguments

    def test_a_payload_nested_one_level_down_is_elided_too(self):
        arguments = {"edit": {"new_lines": [f"l{i}" for i in range(40)]}}
        elided = elide_arguments(arguments)["edit"]["new_lines"]
        assert elided[3] == "... 37 more lines elided ..."

    def test_a_long_line_inside_a_kept_list_is_still_cut(self):
        arguments = {"content_lines": ["y" * (MAX_STRING_CHARS + 10), "short"]}
        elided = elide_arguments(arguments)["content_lines"]
        assert elided[0].endswith("... 10 more chars elided ...")
        assert elided[1] == "short"

    def test_the_originals_are_not_mutated(self):
        # The same dict is the record the chat shows (AgentState.calls); the
        # user's copy has to keep every line.
        lines = [f"line {i}" for i in range(100)]
        arguments = {"content_lines": lines}
        elide_arguments(arguments)
        assert arguments["content_lines"] == lines
        assert len(lines) == 100

    def test_the_elision_shows_up_in_the_message(self):
        call = tool_exchange(
            "create_file", {"content_lines": [f"l{i}" for i in range(500)]}, "written"
        )[0]
        assert "more lines elided" in call["content"]
        assert len(call["content"]) < 400


class TestNativeProtocol:
    """The same pair, encoded for the backend's own tool channel."""

    def test_the_call_rides_tool_calls_and_the_result_the_tool_role(self):
        call, result = tool_exchange(
            "read_file", {"registry_key": "cohort"}, "12 lines", call_id="call_1"
        )
        assert call["role"] == "assistant"
        assert call["tool_calls"][0]["function"]["name"] == "read_file"
        assert json.loads(call["tool_calls"][0]["function"]["arguments"]) == {
            "registry_key": "cohort"
        }
        assert result["role"] == "tool"
        assert result["content"] == "[tool result] read_file: 12 lines"

    def test_the_id_ties_the_result_to_its_call(self):
        call, result = tool_exchange("echo", {"text": "hi"}, "ok", call_id="abc")
        assert call["tool_calls"][0]["id"] == "abc" == result["tool_call_id"]

    def test_payloads_are_elided_on_this_channel_too(self):
        # Not a protocol matter: the reason is that a create_file payload would
        # otherwise sit in the window twice, whichever encoding carries it.
        call = tool_exchange(
            "create_file",
            {"content_lines": [f"l{i}" for i in range(500)]},
            "written",
            call_id="c1",
        )[0]
        assert "more lines elided" in call["tool_calls"][0]["function"]["arguments"]

    def test_a_native_call_is_recognised_as_a_call(self):
        call, result = tool_exchange("echo", {"text": "hi"}, "ok", call_id="c1")
        assert is_tool_call_message(call)  # or the transcript renders it as a reply
        assert not is_tool_call_message(result)

    def test_content_is_passed_through_untouched(self):
        # What answers a call is also a denial or a tool error; relabelling one
        # as a result is how a refused call reads as a successful one.
        denial = result_message("delete_file", "[denied] the user said no", "c1")
        assert denial["content"] == "[denied] the user said no"
        assert denial["role"] == "tool"
        assert result_message("echo", "[tool error] boom")["role"] == "user"
