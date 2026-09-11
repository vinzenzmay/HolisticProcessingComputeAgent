"""Tests for hpca.config: settings defaults, persistence, and lenient loading."""

import json

import pytest
from pydantic import ValidationError

from hpca.config import (LLMBackend, LLMSettings, Settings, SettingsError,
                         app_dir, llm_settings_for, settings_path)


NAME = ".HolisticProcessingComputeAgent"


@pytest.fixture
def home(monkeypatch, tmp_path):
    """A home directory of our own, with no $HPCA_HOME over it.

    `app_dir` reads the real filesystem to decide — whether ``~/work`` is
    there, whether an app dir is already in one place or the other — so these
    tests need a home they can arrange, not a string.
    """
    monkeypatch.delenv("HPCA_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


class TestAppDir:
    def test_defaults_to_home_dotdir(self, home):
        # A laptop: no ~/work, so nothing changes about where things go.
        assert app_dir() == home / NAME

    def test_prefers_work_when_it_exists(self, home):
        # A cluster node: ~/work is the fast filesystem, and this is the whole
        # point of the rule.
        (home / "work").mkdir()
        assert app_dir() == home / "work" / NAME

    def test_work_wins_on_a_first_start(self, home):
        # Nothing exists yet either side, so the rule alone decides — and the
        # directory it names is what the first run creates.
        (home / "work").mkdir()
        assert not app_dir().exists()
        assert app_dir() == home / "work" / NAME

    def test_an_existing_home_dir_is_not_abandoned(self, home):
        # ~/work appearing under a user who has been running out of ~ must not
        # silently strand their databases. Moving the directory is what moves
        # the app.
        (home / "work").mkdir()
        (home / NAME).mkdir()
        assert app_dir() == home / NAME

    def test_moving_the_directory_moves_the_app(self, home):
        # The other half of the same story: once it is over there, it is over
        # there, even though the one in ~ has not been cleaned up.
        (home / "work").mkdir()
        (home / NAME).mkdir()
        (home / "work" / NAME).mkdir()
        assert app_dir() == home / "work" / NAME

    def test_env_override(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HPCA_HOME", str(tmp_path / "custom"))
        assert app_dir() == tmp_path / "custom"

    def test_env_override_beats_work(self, home, monkeypatch):
        (home / "work").mkdir()
        monkeypatch.setenv("HPCA_HOME", str(home / "custom"))
        assert app_dir() == home / "custom"

    def test_settings_path_inside_app_dir(self, monkeypatch, tmp_path):
        monkeypatch.setenv("HPCA_HOME", str(tmp_path))
        assert settings_path() == tmp_path / "settings.json"


class TestTheConfiguredAppDir:
    """``app_dir`` in settings.json, which sends the data somewhere else."""

    def _write(self, root, value):
        root.mkdir(parents=True, exist_ok=True)
        (root / "settings.json").write_text(json.dumps({"app_dir": value}))

    def test_settings_redirect_the_data(self, home, tmp_path):
        self._write(home / NAME, str(tmp_path / "elsewhere"))
        assert app_dir() == tmp_path / "elsewhere"

    def test_read_from_the_directory_the_rule_found(self, home, tmp_path):
        (home / "work").mkdir()
        self._write(home / "work" / NAME, str(tmp_path / "elsewhere"))
        assert app_dir() == tmp_path / "elsewhere"

    def test_a_tilde_is_expanded(self, home):
        self._write(home / NAME, "~/somewhere")
        assert app_dir() == home / "somewhere"

    def test_redirects_do_not_chain(self, home, tmp_path):
        # One hop. The file at the far end is the settings the app runs on,
        # and its own app_dir key says nothing about where it lives.
        self._write(home / NAME, str(tmp_path / "one"))
        self._write(tmp_path / "one", str(tmp_path / "two"))
        assert app_dir() == tmp_path / "one"

    @pytest.mark.parametrize("value", [None, "", "   ", 7, []])
    def test_no_answer_leaves_the_directory_alone(self, home, value):
        self._write(home / NAME, value)
        assert app_dir() == home / NAME

    def test_a_malformed_file_does_not_cost_the_data_directory(self, home):
        (home / NAME).mkdir()
        (home / NAME / "settings.json").write_text("{not json")
        assert app_dir() == home / NAME

    def test_the_env_override_is_not_redirectable(self, monkeypatch, tmp_path):
        # $HPCA_HOME that a file inside the directory it names could overrule
        # would be no override at all — and the whole suite runs behind it.
        monkeypatch.setenv("HPCA_HOME", str(tmp_path))
        self._write(tmp_path, str(tmp_path / "elsewhere"))
        assert app_dir() == tmp_path

    def test_saving_keeps_the_redirect(self, monkeypatch, tmp_path):
        # It is a modelled field precisely so that save() cannot drop it: an
        # unmodelled key would survive hand-editing right up until the
        # settings editor next wrote the file.
        monkeypatch.setenv("HPCA_HOME", str(tmp_path))
        settings_path().parent.mkdir(parents=True, exist_ok=True)
        settings_path().write_text(json.dumps({"app_dir": "/data/hpca"}))
        Settings.load().save()
        assert json.loads(settings_path().read_text())["app_dir"] == "/data/hpca"

    def test_unset_by_default(self):
        assert Settings().app_dir is None


class TestDefaults:
    def test_llm_defaults(self):
        s = Settings()
        assert s.llm.base_url == "http://localhost:8000/v1"
        assert s.llm.api_key is None
        assert s.llm.model == "qwen3-6b"
        assert s.llm.constrained_decoding == "auto"
        assert s.llm.max_retries == 3
        assert s.llm.request_timeout_s == 120
        # the client-level fallback stays off; sessions reason at xhigh
        assert s.llm.enable_thinking is False
        assert s.agent.default_thinking == "xhigh"
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

    def test_a_backend_entrys_old_thinking_flag_is_dropped_not_rejected(self):
        # Every settings file written before v0.22.0 has this key on its
        # catalog entries. Loading one must not fail, and must not resurrect
        # the per-backend switch either: thinking is a session level now.
        backend = LLMBackend.model_validate(
            {
                "model": "m",
                "base_url": "http://localhost:9/v1",
                "enable_thinking": True,
            }
        )
        assert not hasattr(backend, "enable_thinking")

    def test_activating_a_backend_leaves_thinking_alone(self):
        # It used to be carried across from the entry. The client-level flag is
        # only a fallback now, and the session's level is what decides.
        s = Settings()
        s.llm.enable_thinking = True
        s.activate_backend(
            LLMBackend(model="plain", base_url="http://localhost:8/v1")
        )
        assert s.llm.enable_thinking is True

    def test_backends_default_to_the_global_tool_protocol(self):
        # None, not "native": an entry written before v0.22.0 has no opinion,
        # and must keep following whatever llm.tool_protocol says.
        assert LLMBackend(model="m", base_url="http://localhost:9/v1") \
            .tool_protocol is None

    def test_activating_carries_a_backends_tool_protocol_override(self):
        s = Settings()
        assert s.llm.tool_protocol == "envelope"
        native_server = LLMBackend(
            model="new", base_url="http://localhost:9/v1", tool_protocol="native"
        )
        s.activate_backend(native_server)
        assert s.llm.tool_protocol == "native"

    def test_activating_a_backend_without_an_override_leaves_the_protocol(self):
        # The override is one-way: an entry that says nothing must not reset a
        # protocol the user chose globally, in either direction.
        s = Settings()
        s.llm.tool_protocol = "native"
        s.activate_backend(LLMBackend(model="m", base_url="http://localhost:9/v1"))
        assert s.llm.tool_protocol == "native"

    def test_a_backends_extra_fields_reach_the_active_client(self):
        # The measured case: Ollama drops `chat_template_kwargs`, so thinking
        # only goes off if `reasoning_effort: "none"` is sent — and the
        # bootstrap client is built from `llm`, not from the catalog entry.
        s = Settings()
        s.activate_backend(
            LLMBackend(
                model="m",
                base_url="http://localhost:9/v1",
                extra_body={"reasoning_effort": "none"},
            )
        )
        assert s.llm.extra_body == {"reasoning_effort": "none"}

    def test_an_entrys_extra_fields_are_merged_over_the_base_not_swapped_in(self):
        # The base carries what every backend here needs; an entry states only
        # its difference.
        base = LLMSettings(extra_body={"seed": 1, "reasoning_effort": "low"})
        entry = LLMBackend(
            model="m",
            base_url="http://localhost:8/v1",
            extra_body={"reasoning_effort": "none"},
        )
        assert llm_settings_for(entry, base).extra_body == {
            "seed": 1,
            "reasoning_effort": "none",
        }

    def test_extra_fields_may_not_take_over_the_request(self, tmp_path):
        # Refused at load, not at request time: `model` set from a settings
        # file would send every turn to a model nobody chose, and the only
        # symptom would be answers from the wrong place.
        for key, value in (("model", "other"), ("messages", []), ("stream", True)):
            with pytest.raises(ValidationError):
                LLMBackend(
                    model="m", base_url="http://x/v1", extra_body={key: value}
                )
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"llm": {"extra_body": {"tools": []}}}))
        with pytest.raises(SettingsError):
            Settings.load(path)

    def test_per_session_settings_do_not_carry_a_thinking_flag(self):
        # One client is shared by every session on a backend
        # (core.backends.client_for_backend), so a thinking flag baked into its
        # settings would be one session's choice imposed on the others.
        base = LLMSettings()
        base.enable_thinking = True
        derived = llm_settings_for(
            LLMBackend(model="m", base_url="http://localhost:8/v1"), base
        )
        assert derived.enable_thinking is True  # copied from base, not the entry

    def test_per_session_settings_take_the_backends_protocol(self):
        base = LLMSettings()
        old_server = LLMBackend(
            model="old", base_url="http://localhost:9/v1", tool_protocol="envelope"
        )
        assert llm_settings_for(old_server, base).tool_protocol == "envelope"
        plain = LLMBackend(model="m", base_url="http://localhost:8/v1")
        assert llm_settings_for(plain, base).tool_protocol == base.tool_protocol

    def test_a_settings_file_with_the_old_backend_flag_still_loads(self, tmp_path):
        # The whole-file version of the check above: an in-the-wild settings.json
        # keeps working, minus the key.
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({
            "backends": [{
                "model": "m",
                "base_url": "http://localhost:9/v1",
                "enable_thinking": True,
            }]
        }))
        loaded = Settings.load(path)
        assert loaded.backends[0].model == "m"
        assert "enable_thinking" not in loaded.model_dump()["backends"][0]

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


class TestWatchSettings:
    """`watches.peek_chars` — how much tail the Enter peek is worth.

    A setting rather than a constant because both ends of the range are real:
    a tail read over a tunnel wants to stay small, and a traceback wants to
    arrive with the line that names the exception still on it.
    """

    def test_the_peek_ships_a_couple_of_screenfuls_by_default(self):
        assert Settings().watches.peek_chars == 2000

    def test_the_module_default_is_the_settings_default(self):
        # One number, not two: `watches.PEEK_CHARS` is the fallback for a
        # caller with no settings in hand, and it is taken off the model so
        # that editing one of them cannot leave the other behind.
        from hpca.watches import PEEK_CHARS

        assert PEEK_CHARS == Settings().watches.peek_chars

    def test_a_configured_size_is_what_the_file_says(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"watches": {"peek_chars": 40}}))
        assert Settings.load(path).watches.peek_chars == 40

    def test_a_missing_section_falls_back_to_the_default(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"llm": {"model": "m"}}))
        assert Settings.load(path).watches.peek_chars == 2000

    def test_a_peek_of_nothing_is_refused(self, tmp_path):
        # Zero would answer every box with an empty string, which reads as the
        # log being empty — a different fact from one nobody asked to see.
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"watches": {"peek_chars": 0}}))
        with pytest.raises(SettingsError):
            Settings.load(path)


class TestDisplaySettings:
    """The section nothing in the core reads: it is what the *front-end* draws
    with, and it reaches the UI over the wire (`protocol.DisplaySettings`)
    because rule 2 of §4.2 keeps the settings file out of its reach."""

    def test_the_stamp_is_on_by_default(self):
        assert Settings().display.chat_stamps is True

    def test_the_decision_breathes_once_a_second_by_default(self):
        assert Settings().display.decision_pulse_seconds == 1.0

    def test_the_section_roundtrips(self, tmp_path):
        s = Settings()
        s.display.chat_stamps = False
        s.display.decision_pulse_seconds = 3.5
        s.save(tmp_path / "settings.json")
        loaded = Settings.load(tmp_path / "settings.json")
        assert loaded.display.chat_stamps is False
        assert loaded.display.decision_pulse_seconds == 3.5

    def test_a_missing_section_falls_back_to_the_defaults(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"llm": {"model": "m"}}))
        loaded = Settings.load(path)
        assert loaded.display.chat_stamps is True
        assert loaded.display.decision_pulse_seconds == 1.0

    def test_a_period_of_zero_is_refused(self, tmp_path):
        # It divides by zero in the sweep. Refused here, where the answer is
        # one line beside the editor that typed it, rather than in a repaint
        # loop with the terminal in raw mode.
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"display": {"decision_pulse_seconds": 0}}))
        with pytest.raises(SettingsError):
            Settings.load(path)

    def test_a_negative_period_is_refused_too(self, tmp_path):
        # It would run the sweep backwards, which is not a thing anyone means.
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"display": {"decision_pulse_seconds": -2}}))
        with pytest.raises(SettingsError):
            Settings.load(path)

    def test_a_chat_row_leaves_one_blank_line_under_itself(self):
        assert Settings().display.spacer_lines == 1

    def test_and_the_spacing_can_be_turned_off(self, tmp_path):
        # Zero is a real answer and not a grudging one: on a short terminal
        # every blank is a line of conversation that is not on screen.
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"display": {"spacer_lines": 0}}))
        assert Settings.load(path).display.spacer_lines == 0

    def test_or_opened_up(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"display": {"spacer_lines": 2}}))
        assert Settings.load(path).display.spacer_lines == 2

    def test_but_not_negative(self, tmp_path):
        # A count a repaint loop builds a list from.
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"display": {"spacer_lines": -1}}))
        with pytest.raises(SettingsError):
            Settings.load(path)

    def test_and_not_a_pane_made_of_gap(self, tmp_path):
        # Past a handful the chat is one message a screen, which is not a
        # spacious log — it is a broken one.
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"display": {"spacer_lines": 800}}))
        with pytest.raises(SettingsError):
            Settings.load(path)


class TestThePalette:
    """The colours, which a user is meant to be able to replace outright."""

    def test_the_defaults_are_the_dark_palette(self):
        palette = Settings().display.palette
        assert (palette.chrome, palette.warn) == ("73", "172")
        assert palette.spinner == ["73", "66", "23", "236"]

    def test_the_greys_are_readable_on_a_dark_terminal(self):
        # A turn's working — the bulkiest thing in the log — is drawn entirely
        # in `faint`, and the grey it used to be (240, #585858) sat under 3:1
        # against a dark ground: legible on the screen it was picked on, murky
        # on the next one. Tertiary means quieter than the prose, not harder
        # to read than it, and `muted` stays the brighter of the two.
        palette = Settings().display.palette
        assert (palette.muted, palette.faint) == ("250", "245")
        assert int(palette.muted) > int(palette.faint)

    def test_an_xterm_index_and_a_hex_triple_are_both_colours(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(
            json.dumps({"display": {"palette": {"ok": "12", "danger": "#ff8800"}}})
        )
        loaded = Settings.load(path)
        assert (loaded.display.palette.ok, loaded.display.palette.danger) == (
            "12",
            "#ff8800",
        )

    def test_naming_one_colour_keeps_the_rest(self, tmp_path):
        # The whole offer: a theme is an edit to the palette, not a
        # replacement of it, so a file naming one colour still draws.
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"display": {"palette": {"user": "33"}}}))
        loaded = Settings.load(path)
        assert loaded.display.palette.user == "33"
        assert loaded.display.palette.agent == "255"

    @pytest.mark.parametrize("bad", ["nope", "256", "#ff88", "#gggggg", "-1", ""])
    def test_what_is_not_a_colour_is_refused(self, tmp_path, bad):
        # Refused at load, where the answer is a line beside the field that
        # named it. A malformed escape sequence reaching a terminal in raw
        # mode does not report a bad setting so much as stop being a terminal.
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"display": {"palette": {"ok": bad}}}))
        with pytest.raises(SettingsError):
            Settings.load(path)

    def test_a_spinner_with_no_cells_is_refused(self, tmp_path):
        # An empty trail is a spinner that draws nothing, which is a working
        # row that has stopped saying anything is happening.
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"display": {"palette": {"spinner": []}}}))
        with pytest.raises(SettingsError):
            Settings.load(path)

    def test_a_shorter_trail_is_allowed(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"display": {"palette": {"spinner": ["9"]}}}))
        assert Settings.load(path).display.palette.spinner == ["9"]

    def test_the_flash_can_be_turned_off(self, tmp_path):
        # Zero is off, which is why the floor is ge and not gt: a user who
        # finds the wash startling should be able to say so.
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"display": {"focus_flash_seconds": 0}}))
        assert Settings.load(path).display.focus_flash_seconds == 0

    def test_but_not_held_forever(self, tmp_path):
        path = tmp_path / "settings.json"
        path.write_text(json.dumps({"display": {"focus_flash_seconds": 5}}))
        with pytest.raises(SettingsError):
            Settings.load(path)

    def test_a_palette_survives_a_round_trip(self, tmp_path):
        path = tmp_path / "settings.json"
        settings = Settings()
        settings.display.palette.chrome = "#88ccff"
        settings.save(path)
        assert Settings.load(path).display.palette.chrome == "#88ccff"
