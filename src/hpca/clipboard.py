"""Clipboard with multiplexer-aware OSC 52 and guaranteed fallbacks (§3.4).

Copying must never silently fail. On every copy the manager attempts, in
parallel tiers: (1) an OSC 52 escape sequence (wrapped for the detected
multiplexer) so the user's *local* terminal gets the text even over SSH, and
(2) the multiplexer's own paste buffer where a CLI exists. If nothing
succeeded — or the payload exceeds the OSC 52 size limit — the text is written
to ``<app_dir>/clipboard.txt`` and the result message says so.

Terminal writes (``emit``) and subprocess execution (``run``) are injected so
the manager is fully testable and the TUI can route emissions through the
Textual driver (Textual's own ``copy_to_clipboard`` does not wrap for
multiplexers, which is why it is not used).
"""

from __future__ import annotations

import base64
import shlex
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from hpca.config import ClipboardSettings, app_dir

ESC = "\x1b"
SCREEN_DCS_PAYLOAD = 768  # screen truncates long DCS blocks; chunk below that

Emit = Callable[[str], None]
Run = Callable[..., tuple[bool, str]]


def detect_multiplexer(env: Mapping[str, str]) -> str | None:
    """Identify the terminal multiplexer from the environment.

    $TMUX is checked first: it is the most specific signal and $STY can leak
    through nested sessions.
    """
    if env.get("TMUX"):
        return "tmux"
    if env.get("STY"):
        return "screen"
    if env.get("ZELLIJ") is not None:
        return "zellij"
    return None


def osc52_sequence(text: str) -> str:
    """Plain OSC 52 sequence setting the system clipboard ("c" selection)."""
    payload = base64.b64encode(text.encode()).decode()
    return f"{ESC}]52;c;{payload}\x07"


def tmux_passthrough(sequence: str) -> str:
    """Wrap a sequence in tmux DCS passthrough (needs ``allow-passthrough on``)."""
    return f"{ESC}Ptmux;" + sequence.replace(ESC, ESC + ESC) + f"{ESC}\\"


def screen_dcs_chunks(sequence: str, chunk_size: int = SCREEN_DCS_PAYLOAD) -> list[str]:
    """Split a sequence into DCS passthrough blocks GNU screen will forward."""
    return [
        f"{ESC}P" + sequence[i : i + chunk_size] + f"{ESC}\\"
        for i in range(0, len(sequence), chunk_size)
    ]


def _default_run(argv: list[str], stdin: str | None = None) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            argv, input=stdin, capture_output=True, text=True, timeout=10
        )
    except (OSError, subprocess.TimeoutExpired):
        return (False, "")
    return (proc.returncode == 0, proc.stdout)


@dataclass
class CopyResult:
    methods: list[str] = field(default_factory=list)
    file: Path | None = None
    message: str = ""

    @property
    def ok(self) -> bool:
        return bool(self.methods)


class ClipboardManager:
    def __init__(
        self,
        settings: ClipboardSettings,
        *,
        emit: Emit,
        run: Run | None = None,
        env: Mapping[str, str] | None = None,
        fallback_dir: Path | None = None,
    ) -> None:
        self._settings = settings
        self._emit = emit
        self._run = run or _default_run
        if env is None:
            import os

            env = os.environ
        self._env = env
        self._fallback_dir = fallback_dir or app_dir()

    def copy(self, text: str) -> CopyResult:
        result = CopyResult()
        notes: list[str] = []

        mode = self._settings.mode
        if mode == "auto":
            mode = detect_multiplexer(self._env) or "osc52"

        if mode == "command":
            self._copy_via_command(text, result, notes)
        elif mode == "file":
            pass  # handled by the unconditional fallback below
        else:
            self._copy_via_osc52_tiers(mode, text, result, notes)

        needs_file = mode == "file" or not result.methods or "oversized" in notes
        if needs_file:
            try:
                self._fallback_dir.mkdir(parents=True, exist_ok=True)
                file = self._fallback_dir / "clipboard.txt"
                file.write_text(text)
                result.file = file
                result.methods.append("file")
                notes.append(f"written to {file}")
            except OSError as e:
                notes.append(f"file fallback failed: {e}")

        result.message = self._compose_message(result, notes)
        return result

    def _copy_via_command(
        self, text: str, result: CopyResult, notes: list[str]
    ) -> None:
        command = self._settings.command
        if not command:
            notes.append("clipboard.mode is 'command' but no command configured")
            return
        ok, _ = self._run(shlex.split(command), text)
        if ok:
            result.methods.append("command")
        else:
            notes.append(f"clipboard command failed: {command}")

    def _copy_via_osc52_tiers(
        self, mode: str, text: str, result: CopyResult, notes: list[str]
    ) -> None:
        sequence = osc52_sequence(text)
        payload_kb = (len(sequence) - len(osc52_sequence(""))) / 1024
        if payload_kb > self._settings.osc52_limit_kb:
            notes.append("oversized")
        else:
            if mode == "tmux" and not self._tmux_handles_osc52_natively():
                sequence = tmux_passthrough(sequence)
            elif mode == "screen":
                sequence = "".join(screen_dcs_chunks(sequence))
            if self._try_emit(sequence):
                result.methods.append("osc52")
            else:
                notes.append("OSC 52 emission failed")

        # Multiplexer paste buffer — works regardless of OSC 52 support.
        if mode == "tmux":
            ok, _ = self._run(["tmux", "load-buffer", "-"], text)
            if ok:
                result.methods.append("tmux-buffer")
            else:
                notes.append("tmux load-buffer failed")
        elif mode == "screen":
            ok, _ = self._run(["screen", "-X", "register", ".", text])
            if ok:
                result.methods.append("screen-buffer")
            else:
                notes.append("screen register failed")

    def _tmux_handles_osc52_natively(self) -> bool:
        ok, out = self._run(["tmux", "show", "-gv", "set-clipboard"])
        return ok and out.strip() in ("on", "external")

    def _try_emit(self, sequence: str) -> bool:
        try:
            self._emit(sequence)
        except Exception:
            return False
        return True

    @staticmethod
    def _compose_message(result: CopyResult, notes: list[str]) -> str:
        if "oversized" in notes:
            notes[notes.index("oversized")] = "payload exceeds OSC 52 size limit"
        if result.methods:
            message = "Copied via " + " + ".join(result.methods)
        else:
            message = "Copy FAILED on every tier"
        if notes:
            message += " (" + "; ".join(notes) + ")"
        return message
