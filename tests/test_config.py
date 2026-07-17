"""Tests for hpca.config: settings defaults, persistence, and lenient loading."""

import json

import pytest

from hpca.config import LLMBackend, Settings, SettingsError, app_dir, settings_path


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
        assert s.memory.tier1_token_cap == 300
        assert s.memory.tier2_token_cap == 800

    def test_clipboard_defaults(self):
        s = Settings()
        assert s.clipboard.mode == "auto"
        assert s.clipboard.command is None
        assert s.clipboard.osc52_limit_kb == 74

    def test_editor_and_rag_defaults(self):
        s = Settings()
        assert s.editor is None
        assert s.rag.store == "sqlite-vec"
        assert s.rag.embedding == "sentence-transformers/all-MiniLM-L6-v2"
        assert s.rag.embedding_base_url == "http://localhost:51943/v1"


class TestLoad:
    def test_missing_file_returns_defaults(self, tmp_path):
        s = Settings.load(tmp_path / "nope.json")
        assert s == Settings()

    def test_roundtrip(self, tmp_path):
        path = tmp_path / "settings.json"
        s = Settings()
        s.llm.model = "Qwen/Qwen3.6-35B-A3B-FP8"
        s.llm.base_url = "http://localhost:51941/v1"
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
                base_url="http://localhost:51941/v1",
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

    def test_is_active(self):
        s = Settings()
        backend = LLMBackend(model="m", base_url="http://localhost:9/v1")
        assert not s.is_active(backend)
        s.activate_backend(backend)
        assert s.is_active(backend)
