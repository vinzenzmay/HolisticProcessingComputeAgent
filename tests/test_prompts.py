"""Tests for hpca.agent.prompts: prompt assembly with memories (§4.3, §6.1)."""

from datetime import datetime

from hpca.agent.prompts import environment_facts, orchestrator_system_prompt


MEMORY_LABEL = "Memory (site facts, preferences, learnings)"


class TestOrchestratorPrompt:
    def test_memories_injected_under_the_merged_label(self):
        prompt = orchestrator_system_prompt(
            system_prompt_memories="Cluster is cubi.\n\nUser prefers minimap2 over bwa."
        )
        assert f"{MEMORY_LABEL}:\nCluster is cubi." in prompt
        assert "User prefers minimap2 over bwa." in prompt

    def test_empty_memories_add_no_section(self):
        prompt = orchestrator_system_prompt()
        assert MEMORY_LABEL not in prompt

    def test_volatile_date_stays_out_of_the_prefix(self):
        # The date/time lives at the tail (api_content), never the system
        # prompt: putting it here invalidated the backend's prefix KV cache
        # every minute. The prompt must be a stable, cacheable prefix.
        prompt = orchestrator_system_prompt()
        assert f"{datetime.now():%Y-%m-%d}" not in prompt
        assert "Current date and time" not in prompt

    def test_prefix_is_identical_across_calls(self):
        # Same inputs must yield byte-identical output so the prefix caches.
        args = dict(system_prompt_memories="Cluster is cubi.\n\nUser prefers R.")
        assert orchestrator_system_prompt(**args) == orchestrator_system_prompt(**args)

    def test_environment_facts_contains_time(self):
        assert f"{datetime.now():%Y-%m-%d}" in environment_facts()

    def test_usage_meter_shown_when_given(self):
        prompt = orchestrator_system_prompt(
            system_prompt_memories="Cluster is cubi.",
            memory_meter="58% — 1392/2400 tokens",
        )
        assert (
            f"{MEMORY_LABEL} [58% — 1392/2400 tokens]:\nCluster is cubi." in prompt
        )

    def test_meter_without_memory_text_adds_no_section(self):
        prompt = orchestrator_system_prompt(memory_meter="0% — 0/2400 tokens")
        assert MEMORY_LABEL not in prompt

    def test_session_search_guidance_only_when_tool_present(self):
        assert "session_search" not in orchestrator_system_prompt()
        with_tool = orchestrator_system_prompt(session_search=True)
        assert "session_search" in with_tool
        # the source-first limit: recall is what was said, not what now is
        assert "current state of files" in with_tool
