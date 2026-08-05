"""Tests for hpca.autoconnect: the pre-agent connection decision (spec §4.4/§4.6)."""

from hpca.autoconnect import (
    AutoConnectPlan,
    is_connectable,
    offcluster_help,
    plan_auto_connect,
)
from hpca.cluster_endpoints import ClusterEndpoints
from hpca.discover import DiscoveredBackend


def llm(model, port, *, needs_key=False, api_key=None):
    return DiscoveredBackend(
        base_url=f"http://172.16.0.9:{port}/v1",
        model=model,
        max_model_len=131072,
        needs_key=needs_key,
        api_key=api_key,
    )


class TestIsConnectable:
    def test_open_endpoint(self):
        assert is_connectable(llm("m", 20001)) is True

    def test_locked_without_key(self):
        assert is_connectable(llm("m", 20001, needs_key=True)) is False

    def test_unlocked_by_pool_key(self):
        assert is_connectable(llm("m", 20001, needs_key=True, api_key="k")) is True


class TestPlanAutoConnect:
    def test_no_endpoints(self):
        plan = plan_auto_connect(ClusterEndpoints([], None))
        assert plan.connect is None
        assert plan.choices == []
        assert plan.embedding_base_url is None

    def test_single_llm_auto_connects(self):
        b = llm("Qwen/Qwen3.6-35B-A3B-FP8", 20001)
        plan = plan_auto_connect(ClusterEndpoints([b], None))
        assert plan.connect is b
        assert plan.choices == [b]

    def test_many_llms_needs_pick(self):
        a, b = llm("A", 20001), llm("B", 20003)
        plan = plan_auto_connect(ClusterEndpoints([a, b], None))
        assert plan.connect is None  # ambiguous -> user picks
        assert plan.choices == [a, b]

    def test_single_locked_llm_does_not_auto_connect(self):
        b = llm("A", 20001, needs_key=True)  # no key -> not connectable
        plan = plan_auto_connect(ClusterEndpoints([b], None))
        assert plan.connect is None
        assert plan.choices == [b]  # still surfaced, for the add-a-key flow

    def test_single_connectable_among_locked_auto_connects(self):
        open_b = llm("open", 20001)
        locked = llm("locked", 20003, needs_key=True)
        plan = plan_auto_connect(ClusterEndpoints([open_b, locked], None))
        assert plan.connect is open_b  # exactly one *connectable*

    def test_preferred_model_wins_even_with_many(self):
        a = llm("Qwen/Qwen3.6-35B-A3B-FP8", 20001)
        b = llm("Qwen/Qwen3.6-27B-FP8", 20003)
        plan = plan_auto_connect(
            ClusterEndpoints([a, b], None), preferred_models=["27B"]
        )
        assert plan.connect is b

    def test_preferred_order_respected(self):
        a = llm("gemma-4-31B", 20004)
        b = llm("Qwen3.6-27B", 20003)
        plan = plan_auto_connect(
            ClusterEndpoints([a, b], None), preferred_models=["nomatch", "27B", "gemma"]
        )
        assert plan.connect is b  # first pattern that matches anything

    def test_preferred_is_case_insensitive(self):
        a = llm("Qwen3.6-27B-FP8", 20003)
        plan = plan_auto_connect(
            ClusterEndpoints([a], None), preferred_models=["qwen3.6-27b"]
        )
        assert plan.connect is a

    def test_preferred_skips_locked(self):
        locked = llm("27B", 20003, needs_key=True)
        plan = plan_auto_connect(
            ClusterEndpoints([locked], None), preferred_models=["27B"]
        )
        assert plan.connect is None  # matches the pattern but isn't connectable

    def test_embedding_url_surfaced(self):
        emb = DiscoveredBackend(
            base_url="http://172.16.0.9:20000/v1",
            model="sentence-transformers/all-MiniLM-L6-v2",
        )
        plan = plan_auto_connect(ClusterEndpoints([], emb))
        assert plan.embedding_base_url == "http://172.16.0.9:20000/v1"


class TestPlanNotice:
    """The user-facing line for found-but-not-connected outcomes. A lone locked
    endpoint used to fail in complete silence (the bug that hid a stale pool
    key); every no-connect-but-found plan must now carry a notice."""

    def test_none_when_connected(self):
        plan = plan_auto_connect(ClusterEndpoints([llm("A", 20001)], None))
        assert plan.connect is not None
        assert plan.notice is None

    def test_none_when_nothing_found(self):
        plan = plan_auto_connect(ClusterEndpoints([], None))
        assert plan.notice is None

    def test_single_locked_llm_notice_names_model_and_key(self):
        b = llm("Qwen/Qwen3.6-35B-A3B-FP8", 20001, needs_key=True)
        plan = plan_auto_connect(ClusterEndpoints([b], None))
        assert plan.connect is None
        assert "Qwen/Qwen3.6-35B-A3B-FP8" in plan.notice
        assert "API key" in plan.notice
        # (m), not ctrl+l: the endpoint was never added to the catalog, and
        # ctrl+l only lists the catalog — it would be a dead end.
        assert "(m)" in plan.notice
        assert "ctrl+l" not in plan.notice

    def test_many_llms_notice_offers_picker(self):
        a, b = llm("A", 20001), llm("B", 20003)
        plan = plan_auto_connect(ClusterEndpoints([a, b], None))
        assert plan.notice == "2 cluster LLMs discovered — press (m) to pick one"

    def test_preferred_miss_with_single_locked_still_notices(self):
        locked = llm("27B", 20003, needs_key=True)
        plan = plan_auto_connect(
            ClusterEndpoints([locked], None), preferred_models=["27B"]
        )
        assert plan.connect is None
        assert "API key" in plan.notice


class TestOffclusterHelp:
    def test_includes_login_target_and_ports(self):
        msg = offcluster_help(
            "mayv_c@hpc-login-2.cubi.bihealth.org", "/shared/hpca_connections"
        )
        assert "mayv_c@hpc-login-2.cubi.bihealth.org" in msg
        assert "20000" in msg  # the embeddings forward
        assert "ssh" in msg and "-L" in msg

    def test_shows_the_configured_endpoints_dir(self):
        # The dir is configurable, so the "list running endpoints" step must
        # name the one this app actually reads, not a hard-coded path.
        msg = offcluster_help("host", "/data/cephfs-1/work/groups/cubi/tools/eps")
        assert "/data/cephfs-1/work/groups/cubi/tools/eps/*.json" in msg

    def test_bare_host_when_no_user(self):
        msg = offcluster_help(
            "hpc-login-2.cubi.bihealth.org", "/shared/hpca_connections"
        )
        assert "hpc-login-2.cubi.bihealth.org" in msg
