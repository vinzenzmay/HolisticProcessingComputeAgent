"""Tests for hpca.agent.prompts: prompt assembly with memories (§4.3, §6.1)."""

from datetime import datetime

from hpca.agent.prompts import environment_facts, orchestrator_system_prompt


class TestOrchestratorPrompt:
    def test_tiers_injected_with_labels(self):
        prompt = orchestrator_system_prompt(
            tier1="Cluster is cubi.", tier2="User prefers R over Python."
        )
        assert "Standing site notes:\nCluster is cubi." in prompt
        assert "User prefers R over Python." in prompt

    def test_empty_tiers_add_no_sections(self):
        prompt = orchestrator_system_prompt()
        assert "Standing site notes" not in prompt
        assert "Learnings" not in prompt

    def test_volatile_date_stays_out_of_the_prefix(self):
        # The date/time lives at the tail (api_content), never the system
        # prompt: putting it here invalidated the backend's prefix KV cache
        # every minute. The prompt must be a stable, cacheable prefix.
        prompt = orchestrator_system_prompt()
        assert f"{datetime.now():%Y-%m-%d}" not in prompt
        assert "Current date and time" not in prompt

    def test_prefix_is_identical_across_calls(self):
        # Same inputs must yield byte-identical output so the prefix caches.
        args = dict(tier1="Cluster is cubi.", tier2="User prefers R.")
        assert orchestrator_system_prompt(**args) == orchestrator_system_prompt(**args)

    def test_environment_facts_contains_time(self):
        assert f"{datetime.now():%Y-%m-%d}" in environment_facts()

    def test_usage_meters_shown_when_given(self):
        prompt = orchestrator_system_prompt(
            tier1="Cluster is cubi.",
            tier2="User prefers R.",
            tier1_meter="10% — 16/1200 chars",
            tier2_meter="1% — 14/3200 chars",
        )
        assert "Standing site notes [10% — 16/1200 chars]:\nCluster is cubi." in prompt
        assert (
            "Learnings and preferences from earlier sessions "
            "[1% — 14/3200 chars]:\nUser prefers R." in prompt
        )

    def test_meter_without_tier_text_adds_no_section(self):
        prompt = orchestrator_system_prompt(tier1_meter="0% — 0/1200 chars")
        assert "Standing site notes" not in prompt

    def test_session_search_guidance_only_when_tool_present(self):
        assert "session_search" not in orchestrator_system_prompt()
        with_tool = orchestrator_system_prompt(session_search=True)
        assert "session_search" in with_tool
        # the source-first limit: recall is what was said, not what now is
        assert "current state of files" in with_tool
