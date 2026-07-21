"""Tests for hpca.clipboard (§3.4): OSC 52 tiers, multiplexer handling, fallbacks."""

import base64

import pytest

from hpca.clipboard import (
    ClipboardManager,
    detect_multiplexer,
    is_ssh,
    local_clipboard_argv,
    osc52_sequence,
    screen_dcs_chunks,
    tmux_passthrough,
)
from hpca.config import ClipboardSettings

B64_HELLO = base64.b64encode(b"hello").decode()


class TestDetectMultiplexer:
    def test_tmux(self):
        assert detect_multiplexer({"TMUX": "/tmp/tmux-1000/default,123,0"}) == "tmux"

    def test_screen(self):
        assert detect_multiplexer({"STY": "12345.pts-0.host"}) == "screen"

    def test_zellij(self):
        assert detect_multiplexer({"ZELLIJ": "0"}) == "zellij"

    def test_none(self):
        assert detect_multiplexer({"TERM": "xterm-256color"}) is None

    def test_tmux_wins_inside_screen_env(self):
        # $STY may leak through nested sessions; $TMUX is the more specific signal
        assert detect_multiplexer({"TMUX": "x", "STY": "y"}) == "tmux"


class TestSequenceBuilding:
    def test_plain_osc52(self):
        assert osc52_sequence("hello") == f"\x1b]52;c;{B64_HELLO}\x07"

    def test_tmux_passthrough_wraps_and_doubles_escapes(self):
        inner = osc52_sequence("hello")
        wrapped = tmux_passthrough(inner)
        assert wrapped.startswith("\x1bPtmux;")
        assert wrapped.endswith("\x1b\\")
        body = wrapped[len("\x1bPtmux;") : -len("\x1b\\")]
        assert body == inner.replace("\x1b", "\x1b\x1b")

    def test_screen_chunks_reassemble_to_original(self):
        inner = osc52_sequence("x" * 5000)
        chunks = screen_dcs_chunks(inner)
        assert len(chunks) > 1
        reassembled = ""
        for chunk in chunks:
            assert chunk.startswith("\x1bP")
            assert chunk.endswith("\x1b\\")
            reassembled += chunk[2:-2]
        assert reassembled == inner

    def test_screen_chunk_payloads_bounded(self):
        chunks = screen_dcs_chunks(osc52_sequence("x" * 5000))
        assert all(len(c) <= 768 + 4 for c in chunks)


class FakeIO:
    """Records emissions and commands; scriptable command results."""

    def __init__(self, responses=None, emit_fails=False, run_fails=False):
        self.emitted: list[str] = []
        self.commands: list[tuple[list[str], str | None]] = []
        self.responses = responses or {}
        self.emit_fails = emit_fails
        self.run_fails = run_fails

    def emit(self, seq: str) -> None:
        if self.emit_fails:
            raise OSError("no tty")
        self.emitted.append(seq)

    def run(self, argv: list[str], stdin: str | None = None):
        self.commands.append((argv, stdin))
        if self.run_fails:
            return (False, "")
        return self.responses.get(tuple(argv), (True, ""))


def fake_which(*present):
    """A shutil.which stand-in: names in ``present`` resolve, others don't."""
    have = set(present)
    return lambda name: f"/usr/bin/{name}" if name in have else None


def make_manager(io: FakeIO, env: dict, tmp_path, which=None, **settings_kwargs):
    settings = ClipboardSettings(**settings_kwargs)
    return ClipboardManager(
        settings,
        emit=io.emit,
        run=io.run,
        env=env,
        fallback_dir=tmp_path,
        which=which or fake_which(),  # nothing installed unless a test says so
    )


class TestCopyAuto:
    def test_bare_terminal_emits_plain_osc52(self, tmp_path):
        io = FakeIO()
        mgr = make_manager(io, {}, tmp_path)
        result = mgr.copy("hello")
        assert io.emitted == [osc52_sequence("hello")]
        assert result.ok
        assert "osc52" in result.methods

    def test_tmux_native_when_set_clipboard_on(self, tmp_path):
        io = FakeIO(
            responses={("tmux", "show", "-gv", "set-clipboard"): (True, "on\n")}
        )
        mgr = make_manager(io, {"TMUX": "x"}, tmp_path)
        result = mgr.copy("hello")
        assert osc52_sequence("hello") in io.emitted
        # buffer fallback also written, via stdin (no arg-length limit)
        assert (["tmux", "load-buffer", "-"], "hello") in io.commands
        assert "tmux-buffer" in result.methods

    def test_tmux_passthrough_when_set_clipboard_off(self, tmp_path):
        io = FakeIO(
            responses={("tmux", "show", "-gv", "set-clipboard"): (True, "off\n")}
        )
        mgr = make_manager(io, {"TMUX": "x"}, tmp_path)
        mgr.copy("hello")
        assert tmux_passthrough(osc52_sequence("hello")) in io.emitted

    def test_screen_emits_chunked_dcs_and_registers_buffer(self, tmp_path):
        io = FakeIO()
        mgr = make_manager(io, {"STY": "x"}, tmp_path)
        result = mgr.copy("hello")
        assert io.emitted == ["".join(screen_dcs_chunks(osc52_sequence("hello")))]
        assert (["screen", "-X", "register", ".", "hello"], None) in io.commands
        assert "screen-buffer" in result.methods

    def test_zellij_emits_plain_osc52_only(self, tmp_path):
        io = FakeIO()
        mgr = make_manager(io, {"ZELLIJ": "0"}, tmp_path)
        result = mgr.copy("hello")
        assert io.emitted == [osc52_sequence("hello")]
        assert io.commands == []  # zellij has no CLI buffer API
        assert result.ok


class TestFallbacks:
    def test_oversized_payload_skips_osc52_and_writes_file(self, tmp_path):
        io = FakeIO()
        mgr = make_manager(io, {}, tmp_path, osc52_limit_kb=1)
        text = "x" * 2048  # b64 of this is > 1 KB
        result = mgr.copy(text)
        assert io.emitted == []
        assert result.file is not None
        assert result.file.read_text() == text
        assert "file" in result.methods
        assert result.message  # user-visible notice required

    def test_emit_failure_falls_back_to_file(self, tmp_path):
        io = FakeIO(emit_fails=True)
        mgr = make_manager(io, {}, tmp_path)
        result = mgr.copy("hello")
        assert result.ok
        assert result.file is not None
        assert result.file.read_text() == "hello"

    def test_tmux_buffer_still_works_when_emit_fails(self, tmp_path):
        io = FakeIO(emit_fails=True)
        mgr = make_manager(io, {"TMUX": "x"}, tmp_path)
        result = mgr.copy("hello")
        assert "tmux-buffer" in result.methods
        assert result.ok

    def test_everything_failing_still_writes_file(self, tmp_path):
        io = FakeIO(emit_fails=True, run_fails=True)
        mgr = make_manager(io, {"TMUX": "x"}, tmp_path)
        result = mgr.copy("hello")
        assert result.methods == ["file"]
        assert result.file.read_text() == "hello"
        assert result.ok


class TestExplicitModes:
    def test_mode_command_pipes_to_stdin(self, tmp_path):
        io = FakeIO()
        mgr = make_manager(
            io, {}, tmp_path, mode="command", command="xclip -selection clipboard"
        )
        result = mgr.copy("hello")
        assert (["xclip", "-selection", "clipboard"], "hello") in io.commands
        assert "command" in result.methods
        assert io.emitted == []

    def test_mode_command_without_command_falls_back_to_file(self, tmp_path):
        io = FakeIO()
        mgr = make_manager(io, {}, tmp_path, mode="command", command=None)
        result = mgr.copy("hello")
        assert result.methods == ["file"]

    def test_mode_file_writes_file_only(self, tmp_path):
        io = FakeIO()
        mgr = make_manager(io, {}, tmp_path, mode="file")
        result = mgr.copy("hello")
        assert io.emitted == []
        assert io.commands == []
        assert result.file.read_text() == "hello"

    def test_mode_osc52_forced_even_inside_tmux(self, tmp_path):
        io = FakeIO()
        mgr = make_manager(io, {"TMUX": "x"}, tmp_path, mode="osc52")
        mgr.copy("hello")
        assert io.emitted == [osc52_sequence("hello")]
        assert io.commands == []

    def test_mode_tmux_forced_without_tmux_env(self, tmp_path):
        io = FakeIO(
            responses={("tmux", "show", "-gv", "set-clipboard"): (True, "on\n")}
        )
        mgr = make_manager(io, {}, tmp_path, mode="tmux")
        result = mgr.copy("hello")
        assert (["tmux", "load-buffer", "-"], "hello") in io.commands
        assert result.ok


class TestIsSsh:
    def test_ssh_connection(self):
        assert is_ssh({"SSH_CONNECTION": "10.0.0.1 22 10.0.0.2 22"})

    def test_ssh_tty(self):
        assert is_ssh({"SSH_TTY": "/dev/pts/3"})

    def test_local(self):
        assert not is_ssh({"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"})


class TestLocalClipboardArgv:
    def test_wayland_prefers_wl_copy(self):
        env = {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}
        assert local_clipboard_argv(env, fake_which("wl-copy", "xclip")) == ["wl-copy"]

    def test_wayland_falls_through_to_xclip_when_no_wl_copy(self):
        # A Wayland session almost always has XWayland ($DISPLAY) too.
        env = {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}
        assert local_clipboard_argv(env, fake_which("xclip")) == [
            "xclip",
            "-selection",
            "clipboard",
            "-i",
        ]

    def test_x11_xsel_fallback(self):
        env = {"DISPLAY": ":0"}
        assert local_clipboard_argv(env, fake_which("xsel")) == [
            "xsel",
            "--clipboard",
            "--input",
        ]

    def test_ssh_never_uses_local_tool(self):
        env = {"DISPLAY": ":0", "SSH_CONNECTION": "x"}
        assert local_clipboard_argv(env, fake_which("wl-copy", "xclip")) is None

    def test_no_display_returns_none(self):
        assert local_clipboard_argv({}, fake_which("xclip")) is None

    def test_display_but_no_tool_installed(self):
        assert local_clipboard_argv({"DISPLAY": ":0"}, fake_which()) is None


class TestAutoLocalClipboardTier:
    """§3.4: on a *local* GUI session, ``auto`` also drives a native clipboard
    tool — the reliable path on terminals (GNOME) that silently drop OSC 52."""

    def test_local_wayland_runs_wl_copy_alongside_osc52(self, tmp_path):
        io = FakeIO()
        env = {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}
        mgr = make_manager(io, env, tmp_path, which=fake_which("wl-copy"))
        result = mgr.copy("hello")
        assert osc52_sequence("hello") in io.emitted
        assert (["wl-copy"], "hello") in io.commands
        assert "wl-copy" in result.methods
        assert result.file is None  # a reliable tier landed; no file needed

    def test_local_gui_without_tool_warns_and_writes_file(self, tmp_path):
        io = FakeIO()
        env = {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}
        mgr = make_manager(io, env, tmp_path, which=fake_which())  # nothing installed
        result = mgr.copy("hello")
        # OSC 52 is still emitted, but it may be ignored here, so back it up.
        assert osc52_sequence("hello") in io.emitted
        assert result.file is not None
        assert result.file.read_text() == "hello"
        assert "file" in result.methods
        assert "wl-clipboard" in result.message or "xclip" in result.message

    def test_ssh_session_does_not_run_local_tool(self, tmp_path):
        io = FakeIO()
        env = {"DISPLAY": ":0", "SSH_CONNECTION": "x", "WAYLAND_DISPLAY": "wayland-0"}
        mgr = make_manager(io, env, tmp_path, which=fake_which("wl-copy"))
        result = mgr.copy("hello")
        assert io.commands == []  # OSC 52 is the right channel over SSH
        assert result.file is None  # osc52 is legitimate here, not "unreliable"
        assert result.methods == ["osc52"]

    def test_explicit_osc52_mode_skips_native_tool(self, tmp_path):
        io = FakeIO()
        env = {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}
        mgr = make_manager(
            io, env, tmp_path, which=fake_which("wl-copy"), mode="osc52"
        )
        result = mgr.copy("hello")
        assert io.commands == []  # the user forced OSC 52; respect it
        assert result.methods == ["osc52"]

    def test_local_tmux_still_reaches_native_clipboard(self, tmp_path):
        # GNOME + tmux, locally: the native tool is the reliable path.
        io = FakeIO(
            responses={("tmux", "show", "-gv", "set-clipboard"): (True, "off\n")}
        )
        env = {"TMUX": "x", "WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}
        mgr = make_manager(io, env, tmp_path, which=fake_which("wl-copy"))
        result = mgr.copy("hello")
        assert (["wl-copy"], "hello") in io.commands
        assert "wl-copy" in result.methods
