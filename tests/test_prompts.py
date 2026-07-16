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

    def test_dynamic_date_rendered_per_call(self):
        prompt = orchestrator_system_prompt()
        assert f"{datetime.now():%Y-%m-%d}" in prompt

    def test_environment_facts_contains_time(self):
        assert f"{datetime.now():%Y-%m-%d}" in environment_facts()
