"""Tests for hpca.logs: plain-text session transcripts and sub-agent capture."""

from datetime import datetime
from pathlib import Path

import pytest

import hpca.logs

from hpca.config import Settings
from hpca.llm import ChatResponse
from hpca.logs import (
    LoggedLLM,
    SessionLog,
    log_dir,
    log_path,
    open_log,
    render_messages,
)
from hpca.sessions import Session

SESSION = Session(
    session_id="a1b2c3d4-0000-0000-0000-000000000000",
    profile="default",
    title="which BAMs",
    created_at="2026-07-17T09:12:03.123456+00:00",
    checkpoint_ref="a1b2c3d4-0000-0000-0000-000000000000",
)


def at(second: int):
    return lambda: datetime(2026, 7, 17, 9, 12, second)


class TestPaths:
    @pytest.mark.real_log_default
    def test_defaults_to_hpca_logs_in_the_working_directory(self):
        directory = log_dir(Settings())
        assert directory == Path("hpca-logs")
        assert not directory.is_absolute()  # i.e. where hpca was started

    def test_no_dir_setting_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setattr(hpca.logs, "DEFAULT_LOG_DIR", "/somewhere/else")
        assert log_dir(Settings()) == Path("/somewhere/else")

    def test_dir_setting_overrides(self):
        settings = Settings()
        settings.logging.dir = "/data/transcripts"
        assert str(log_dir(settings)) == "/data/transcripts"

    def test_filename_is_local_start_time_and_session(self, tmp_path):
        # created_at is UTC; the name must agree with the local stamps inside
        local = (
            datetime.fromisoformat(SESSION.created_at)
            .astimezone()
            .strftime("%Y-%m-%dT%H-%M-%S")
        )
        path = log_path(tmp_path, SESSION)
        assert path.name == f"{local}--a1b2c3d4.log"
        assert path.parent == tmp_path

    def test_unparsable_start_time_still_yields_a_name(self, tmp_path):
        odd = Session(**{**SESSION.__dict__, "created_at": "not-a-date"})
        assert log_path(tmp_path, odd).name.endswith("--a1b2c3d4.log")

    def test_filename_survives_a_rename(self, tmp_path):
        renamed = Session(**{**SESSION.__dict__, "title": "something else"})
        assert log_path(tmp_path, renamed) == log_path(tmp_path, SESSION)


class TestOpenLog:
    def test_disabled_returns_nothing(self, tmp_path):
        settings = Settings()
        settings.logging.enabled = False
        assert open_log(settings, SESSION) is None

    def test_enabled_returns_a_log_at_the_configured_dir(self, tmp_path):
        settings = Settings()
        settings.logging.dir = str(tmp_path)
        log = open_log(settings, SESSION)
        assert log is not None
        assert log.path.parent == tmp_path


class TestSessionLog:
    def test_writes_timestamped_blocks_in_order(self, tmp_path):
        log = SessionLog(tmp_path / "s.log", now=at(3))
        log.write("user", "which BAMs are in the cohort?")
        log._now = at(9)
        log.write("agent", "Four BAMs match.")
        assert log.path.read_text() == (
            "[2026-07-17 09:12:03] user\n"
            "which BAMs are in the cohort?\n"
            "\n"
            "[2026-07-17 09:12:09] agent\n"
            "Four BAMs match.\n"
            "\n"
        )

    def test_appends_across_instances(self, tmp_path):
        SessionLog(tmp_path / "s.log", now=at(3)).write("user", "first")
        SessionLog(tmp_path / "s.log", now=at(4)).write("user", "second")
        assert "first" in (tmp_path / "s.log").read_text()
        assert "second" in (tmp_path / "s.log").read_text()

    def test_creates_missing_directories(self, tmp_path):
        log = SessionLog(tmp_path / "deep" / "nested" / "s.log", now=at(3))
        log.write("user", "hi")
        assert log.path.exists()

    def test_multiline_text_kept_verbatim(self, tmp_path):
        log = SessionLog(tmp_path / "s.log", now=at(3))
        log.write("thinking", "— reasoning —\nline one\nline two")
        assert "line one\nline two" in log.path.read_text()

    def test_a_broken_log_never_breaks_the_session(self, tmp_path):
        blocked = tmp_path / "file"
        blocked.write_text("not a directory")
        log = SessionLog(blocked / "s.log", now=at(3))
        log.write("user", "hi")  # must not raise


class TestRenderMessages:
    def test_roles_and_content(self):
        rendered = render_messages(
            [
                {"role": "system", "content": "you are a doc reader"},
                {"role": "user", "content": "what is a BAM?"},
            ]
        )
        assert rendered == "system: you are a doc reader\nuser: what is a BAM?"


class FakeLLM:
    def __init__(self, reply="binary alignment map"):
        self.reply = reply
        self.calls = []

    async def chat(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        return ChatResponse(content=self.reply)

    async def supports_constrained_decoding(self):
        return True


class TestLoggedLLM:
    async def test_query_and_reply_land_in_the_log(self, tmp_path):
        log = SessionLog(tmp_path / "s.log", now=at(3))
        llm = LoggedLLM(FakeLLM(), log)
        response = await llm.chat([{"role": "user", "content": "what is a BAM?"}])
        assert response.content == "binary alignment map"
        text = log.path.read_text()
        assert "subagent query" in text
        assert "user: what is a BAM?" in text
        assert "subagent reply" in text
        assert "binary alignment map" in text

    async def test_label_names_the_calling_tool(self, tmp_path):
        log = SessionLog(tmp_path / "s.log", now=at(3))
        llm = LoggedLLM(FakeLLM(), log, label=lambda: "subagent:ask_docs")
        await llm.chat([{"role": "user", "content": "q"}])
        assert "subagent:ask_docs query" in log.path.read_text()
        assert "subagent:ask_docs reply" in log.path.read_text()

    async def test_arguments_pass_through_untouched(self, tmp_path):
        inner = FakeLLM()
        llm = LoggedLLM(inner, SessionLog(tmp_path / "s.log"))
        await llm.chat([{"role": "user", "content": "q"}], json_schema={"a": 1})
        assert inner.calls[0][1] == {"json_schema": {"a": 1}}

    async def test_other_client_methods_still_reachable(self, tmp_path):
        llm = LoggedLLM(FakeLLM(), SessionLog(tmp_path / "s.log"))
        assert await llm.supports_constrained_decoding() is True
