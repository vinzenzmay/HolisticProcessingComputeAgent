"""Tests for hpca.config: settings defaults, persistence, and lenient loading."""

import json

import pytest

from hpca.config import (LLMBackend, Settings, SettingsError, app_dir,
                         settings_path)


class TestAppDir:
    def test_defaults_to_home_dotdir(self, monkeypatch):
        monkeypatch.delenv("HPCA_HOME", raising=False)
        monkeypatch.setenv("HOME", "/home/someone")
        assert str(app_dir()) == "/home/someone/.HolisticProcessingComputeAgent"

    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HPCA_HOME", str(tmp_path / "custom"))
        assert app_dir() == tmp_path / "custom"

    def test_settings_path_inside_app_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HPCA_HOME", str(tmp_path))
        assert settings_path() == tmp_path / "settings.json"


class TestDefaults:
    def test_llm_defaults(self):
        s = Settings()
        assert s.llm.base_url == "http://localhost:8000/v1"
        assert s.llm.api_key is None
        assert s.llm.model == "qwen3-6b"
        assert s.llm.constrained_decoding == "auto"
        assert s.llm.max_retries == 3
        assert s.llm.request_timeout_s == 120
        # thinking is slow and not generally better: opt in, per backend
        assert s.llm.enable_thinking is False
        # -1 = no cap by default; the user may set a positive bound
        assert s.llm.max_tool_rounds == -1

    def test_cluster_defaults(self):
        s = Settings()
        assert s.cluster.submit_host is None
        assert s.cluster.job_poll_seconds == 30

    def test_safety_defaults(self):
        s = Settings()
        assert s.safety.backup_limit_gb == 1
        assert s.safety.trash_ttl_days == 7

    def test_memory_defaults(self):
        s = Settings()
        assert s.memory.system_prompt_token_cap == 2400

    def test_clipboard_defaults(self):
        s = Settings()
        assert s.clipboard.mode == "auto"
        assert s.clipboard.command is None
        assert s.clipboard.osc52_limit_kb == 74

    def test_editor_and_rag_defaults(self):
        s = Settings()
        assert s.editor is None
        assert s.rag.embedding == "sentence-transformers/all-MiniLM-L6-v2"
        assert s.rag.embedding_base_url == "http://localhost:20000/v1"


class TestLoad:
    def test_missing_file_returns_defaults(self, tmp_path):
        s = Settings.load(tmp_path / "nope.json")
        assert s == Settings()

    def test_roundtrip(self, tmp_path):
        path = tmp_path / "settings.json"
        s = Settings()
        s.llm.model = "Qwen/Qwen3.6-35B-A3B-FP8"
        s.llm.base_url = "http://localhost:20001/v1"
        s.cluster.submit_host = "login01"
        s.save(path)
        loaded = Settings.load(path)
        assert loaded == s

    def test_partial_file_merges_defaults(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"llm": {"model": "other"}}))
        s = Settings.load(path)
        assert s.llm.model == "other"
        # untouched values keep defaults
        assert s.llm.max_retries == 3
        assert s.clipboard.mode == "auto"

    def test_unknown_keys_ignored(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"llm": {"model": "x", "bogus": 1}, "novel": {}}))
        s = Settings.load(path)
        assert s.llm.model == "x"

    def test_invalid_json_raises_settings_error(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text("{not json")
        with pytest.raises(SettingsError) as exc:
            Settings.load(path)
        assert "settings.json" in str(exc.value) or str(path) in str(exc.value)

    def test_invalid_value_raises_settings_error(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"clipboard": {"mode": "telepathy"}}))
        with pytest.raises(SettingsError):
            Settings.load(path)

    def test_load_default_location(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HPCA_HOME", str(tmp_path))
        (tmp_path / "settings.json").write_text(json.dumps({"editor": "vim"}))
        s = Settings.load()
        assert s.editor == "vim"


class TestSave:
    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "settings.json"
        Settings().save(path)
        assert path.exists()

    def test_saved_file_is_valid_pretty_json(self, tmp_path):
        path = tmp_path / "settings.json"
        Settings().save(path)
        text = path.read_text()
        data = json.loads(text)
        assert data["llm"]["model"] == "qwen3-6b"
        assert "\n" in text  # pretty-printed for hand editing

    def test_save_default_location(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HPCA_HOME", str(tmp_path))
        Settings().save()
        assert (tmp_path / "settings.json").exists()


class TestBackendCatalog:
    def test_default_empty(self):
        assert Settings().backends == []

    def test_roundtrip(self, tmp_path):
        path = tmp_path / "settings.json"
        s = Settings()
        s.backends = [
            LLMBackend(
                model="Qwen/Qwen3.6-27B-FP8",
                base_url="http://localhost:20001/v1",
                max_model_len=192000,
            ),
            LLMBackend(
                model="other",
                base_url="http://localhost:8000/v1",
                api_key="sekrit",
            ),
        ]
        s.save(path)
        loaded = Settings.load(path)
        assert len(loaded.backends) == 2
        assert loaded.backends[0].max_model_len == 192000
        assert loaded.backends[1].api_key == "sekrit"

    def test_activate_backend_updates_llm_section(self):
        s = Settings()
        backend = LLMBackend(
            model="m", base_url="http://localhost:9/v1", api_key="k"
        )
        s.activate_backend(backend)
        assert s.llm.model == "m"
        assert s.llm.base_url == "http://localhost:9/v1"
        assert s.llm.api_key == "k"
        # client behavior settings are not clobbered
        assert s.llm.max_retries == 3

    def test_backends_default_to_not_thinking(self):
        assert LLMBackend(model="m", base_url="http://localhost:9/v1") \
            .enable_thinking is False

    def test_activating_carries_the_backends_thinking_mode(self):
        s = Settings()
        thinker = LLMBackend(
            model="reasoner", base_url="http://localhost:9/v1", enable_thinking=True
        )
        plain = LLMBackend(model="plain", base_url="http://localhost:8/v1")
        s.activate_backend(thinker)
        assert s.llm.enable_thinking is True
        s.activate_backend(plain)  # the next backend does not inherit it
        assert s.llm.enable_thinking is False

    def test_set_thinking_on_the_active_backend_applies_at_once(self):
        s = Settings()
        backend = LLMBackend(model="m", base_url="http://localhost:9/v1")
        s.backends = [backend]
        s.activate_backend(backend)
        s.set_thinking(backend, True)
        assert backend.enable_thinking is True
        assert s.llm.enable_thinking is True
        s.set_thinking(backend, False)
        assert s.llm.enable_thinking is False

    def test_set_thinking_on_an_inactive_backend_leaves_the_client_alone(self):
        s = Settings()
        other = LLMBackend(model="other", base_url="http://localhost:8/v1")
        s.set_thinking(other, True)
        assert other.enable_thinking is True
        assert s.llm.enable_thinking is False  # not the one in use

    def test_thinking_mode_roundtrips(self, tmp_path):
        s = Settings()
        s.backends = [
            LLMBackend(
                model="m", base_url="http://localhost:9/v1", enable_thinking=True
            )
        ]
        s.save(tmp_path / "settings.json")
        loaded = Settings.load(tmp_path / "settings.json")
        assert loaded.backends[0].enable_thinking is True

    def test_is_active(self):
        s = Settings()
        backend = LLMBackend(model="m", base_url="http://localhost:9/v1")
        assert not s.is_active(backend)
        s.activate_backend(backend)
        assert s.is_active(backend)


class TestKnownLLMPorts:
    def test_default_empty(self):
        assert Settings().known_llm_ports == []

    def test_remember_collects_ports_and_reports_change(self):
        s = Settings()
        assert s.remember_llm_ports(
            ["http://localhost:20001/v1", "http://127.0.0.1:20000/v1"]
        )
        assert s.known_llm_ports == [20001, 20000]

    def test_remember_is_idempotent(self):
        s = Settings()
        s.remember_llm_ports(["http://localhost:20001/v1"])
        assert not s.remember_llm_ports(["http://localhost:20001/v1"])
        assert s.known_llm_ports == [20001]

    def test_new_port_learned_alongside_known_one(self):
        s = Settings()
        s.remember_llm_ports(["http://localhost:20001/v1"])
        assert s.remember_llm_ports(
            ["http://localhost:20001/v1", "http://localhost:8000/v1"]
        )
        assert s.known_llm_ports == [20001, 8000]

    def test_urls_without_a_port_ignored(self):
        s = Settings()
        assert not s.remember_llm_ports(["http://example.invalid/v1"])
        assert s.known_llm_ports == []

    def test_roundtrip(self, tmp_path):
        s = Settings()
        s.remember_llm_ports(["http://localhost:20001/v1"])
        s.save(tmp_path / "settings.json")
        assert Settings.load(tmp_path / "settings.json").known_llm_ports == [20001]


class TestLLMKeyPool:
    def test_default_empty(self):
        assert Settings().llm_api_keys == []

    def test_remember_appends_new_key_and_reports_change(self):
        s = Settings()
        assert s.remember_llm_key("sekrit")
        assert s.llm_api_keys == ["sekrit"]

    def test_remember_duplicate_is_noop(self):
        s = Settings()
        s.remember_llm_key("sekrit")
        assert not s.remember_llm_key("sekrit")
        assert s.llm_api_keys == ["sekrit"]

    def test_remember_none_is_noop(self):
        s = Settings()
        assert not s.remember_llm_key(None)
        assert s.llm_api_keys == []

    def test_remember_empty_string_is_noop(self):
        s = Settings()
        assert not s.remember_llm_key("")
        assert s.llm_api_keys == []

    def test_roundtrip(self, tmp_path):
        s = Settings()
        s.remember_llm_key("key-a")
        s.remember_llm_key("key-b")
        s.save(tmp_path / "settings.json")
        loaded = Settings.load(tmp_path / "settings.json")
        assert loaded.llm_api_keys == ["key-a", "key-b"]


class TestMemoryScopeBudget:
    """Redesign §6.4: one token budget on the injected system-prompt scope."""

    def test_system_prompt_token_cap_default(self):
        assert Settings().memory.system_prompt_token_cap == 2400

    def test_explicit_token_cap_wins(self):
        s = Settings()
        s.memory.system_prompt_token_cap = 2000
        assert s.memory.system_prompt_token_cap == 2000

    def test_token_cap_roundtrips(self, tmp_path):
        s = Settings()
        s.memory.system_prompt_token_cap = 3600
        s.save(tmp_path / "settings.json")
        loaded = Settings.load(tmp_path / "settings.json")
        assert loaded.memory.system_prompt_token_cap == 3600

    def test_cross_profile_search_off_by_default(self):
        assert Settings().memory.cross_profile_search is False

    def test_rag_and_curator_defaults(self):
        memory = Settings().memory
        assert memory.curator_interval_days == 7
        assert memory.curator_stale_days == 30
        assert memory.curator_archive_days == 90
        assert memory.rag_prefetch_chars == 800
        assert memory.rag_prefetch_count == 3
        assert memory.propose_new_skills is True

    def test_endpoints_defaults(self):
        endpoints = Settings().endpoints
        # The shared group dir, so every HPCA on the cluster sees every
        # endpoint without per-user configuration.
        assert (
            endpoints.endpoints_dir
            == "/data/cephfs-1/work/groups/cubi/tools/hpca_connections"
        )
        assert endpoints.preferred_models == []
        assert endpoints.login_host == "hpc-login-2.cubi.bihealth.org"
        assert endpoints.login_user is None


class TestEndpointsSettings:
    def test_dir_path_expands_user(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/someone")
        s = Settings()
        s.endpoints.endpoints_dir = "~/.hpca/endpoints"
        assert str(s.endpoints.dir_path()) == "/home/someone/.hpca/endpoints"

    def test_default_dir_path_is_the_shared_group_dir(self):
        assert str(Settings().endpoints.dir_path()) == (
            "/data/cephfs-1/work/groups/cubi/tools/hpca_connections"
        )

    def test_dir_path_honors_override(self, tmp_path):
        s = Settings()
        s.endpoints.endpoints_dir = str(tmp_path / "eps")
        assert s.endpoints.dir_path() == tmp_path / "eps"

    def test_login_target_without_user(self):
        s = Settings()
        assert s.endpoints.login_target() == "hpc-login-2.cubi.bihealth.org"

    def test_login_target_with_user(self):
        s = Settings()
        s.endpoints.login_user = "mayv_c"
        assert s.endpoints.login_target() == "mayv_c@hpc-login-2.cubi.bihealth.org"

    def test_preferred_models_roundtrips(self, tmp_path):
        s = Settings()
        s.endpoints.preferred_models = ["35B", "27B"]
        s.save(tmp_path / "settings.json")
        loaded = Settings.load(tmp_path / "settings.json")
        assert loaded.endpoints.preferred_models == ["35B", "27B"]

    def test_unknown_keys_ignored(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"endpoints": {"endpoints_dir": "/x", "bogus": 1}}))
        loaded = Settings.load(path)
        assert loaded.endpoints.endpoints_dir == "/x"


class TestDatabaseSettings:
    def test_local_cache_is_on_by_default(self):
        # NFS homes are the norm on the cluster this targets; a local disk
        # only pays a couple of small copies at start and exit.
        assert Settings().database.local_cache is True

    def test_local_dir_defaults_to_the_node_temp_dir(self):
        assert Settings().database.local_dir is None

    def test_sync_interval_bounds_how_much_a_hard_kill_can_lose(self):
        assert Settings().database.sync_interval_s == 60

    def test_roundtrips(self, tmp_path):
        s = Settings()
        s.database.local_cache = False
        s.database.local_dir = "/scratch/local"
        s.database.sync_interval_s = 15
        s.save(tmp_path / "settings.json")
        loaded = Settings.load(tmp_path / "settings.json")
        assert loaded.database.local_cache is False
        assert loaded.database.local_dir == "/scratch/local"
        assert loaded.database.sync_interval_s == 15

    def test_missing_section_falls_back_to_defaults(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"llm": {"model": "m"}}))
        assert Settings.load(path).database.local_cache is True

    def test_unknown_keys_ignored(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"database": {"sync_interval_s": 5, "bogus": 1}}))
        assert Settings.load(path).database.sync_interval_s == 5

    def test_negative_sync_interval_is_rejected(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"database": {"sync_interval_s": -1}}))
        with pytest.raises(SettingsError):
            Settings.load(path)
