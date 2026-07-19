"""Tests for fenced recall injection (redesign Phase 1)."""

from hpca.agent.memory_context import (
    FENCE_CLOSE,
    FENCE_HEADER,
    FENCE_OPEN,
    build_memory_context,
    compose_api_content,
    note_line,
)
from hpca.llm import wire_messages
from hpca.profiles import Memory


def struggle(text, **kwargs):
    return Memory(text=text, tier=2, kind="struggle", **kwargs)


class TestNoteLine:
    def test_keywords_line_stripped_and_provenance_shown(self):
        memory = struggle(
            "Snakemake dry-runs fail here.\nkeywords: snakemake, dry-run",
            created="2026-06-02",
            backend="qwen3-6b",
        )
        line = note_line(memory)
        assert line == (
            "Past struggle (2026-06-02, backend qwen3-6b): "
            "Snakemake dry-runs fail here."
        )

    def test_without_metadata(self):
        assert note_line(struggle("It went badly.")) == "Past struggle: It went badly."


class TestBuildMemoryContext:
    def test_fenced_block(self):
        block = build_memory_context(["one", "two"])
        assert block.startswith(FENCE_OPEN)
        assert block.endswith(FENCE_CLOSE)
        assert FENCE_HEADER in block
        assert "NOT new user input" in block
        assert "one" in block and "two" in block

    def test_note_cap(self):
        block = build_memory_context(["one", "two", "three"], max_notes=2)
        assert "three" not in block

    def test_char_cap(self):
        block = build_memory_context(["a" * 100, "b" * 100], max_chars=150)
        assert "a" * 100 in block
        assert "b" not in block

    def test_empty_when_nothing_fits(self):
        assert build_memory_context([]) == ""
        assert build_memory_context(["a" * 100], max_chars=10) == ""


class TestWireSidecar:
    def test_api_content_substituted_and_stripped(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "clean", "api_content": "clean\n\nfenced"},
            {"role": "assistant", "content": "reply"},
        ]
        wired = wire_messages(messages)
        assert wired[1] == {"role": "user", "content": "clean\n\nfenced"}
        assert all("api_content" not in m for m in wired)
        # the original transcript is untouched
        assert messages[1]["content"] == "clean"

    def test_plain_messages_pass_through(self):
        messages = [{"role": "user", "content": "hi"}]
        assert wire_messages(messages) == [{"role": "user", "content": "hi"}]

    def test_compose_api_content(self):
        assert compose_api_content("hi", "<memory-context>x</memory-context>") == (
            "hi\n\n<memory-context>x</memory-context>"
        )
