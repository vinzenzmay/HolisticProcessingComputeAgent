"""Application settings (§7 of the project plan).

Settings live in ``<app_dir>/settings.json`` (see :func:`app_dir`) and are
edited both by the in-app settings menu and by hand, so loading is lenient:
missing keys fall back to defaults and unknown keys are ignored. Only malformed
JSON or values of the wrong type/range are reported as errors.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Annotated, Any, Iterable, Literal
from urllib.parse import urlparse

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
)

from hpca.thinking import ThinkingEffort

APP_DIR_NAME = ".HolisticProcessingComputeAgent"

# The directory a cluster gives you to actually work in. On the login nodes
# ``$HOME`` is NFS and ``~/work`` is not, which is why this matters at all:
# everything the app keeps — the databases, the logs, the checkpoints — is
# written there by preference, and `hpca.dbcache` exists to paper over what it
# costs when it cannot be. A laptop has no ``~/work`` and never grows one, so
# the same build lands in ``~`` there without being told.
WORK_DIR_NAME = "work"


def _default_app_dir() -> Path:
    """Where the data goes when nothing has said otherwise.

    Resolved from what is on disk rather than recorded at first start, which
    reaches the same answer with nothing to keep in sync — the first run
    creates the directory the rule names, and every run after that finds it
    exactly where the rule looks.

    An app dir that already exists wins over the rule, and that is the whole
    of the migration story: a user who has been running out of ``~`` goes on
    running out of ``~`` when ``~/work`` appears, and moving the directory is
    what moves the app. Nothing is abandoned by a mount showing up.
    """
    home = Path.home()
    work = home / WORK_DIR_NAME / APP_DIR_NAME
    plain = home / APP_DIR_NAME
    if (home / WORK_DIR_NAME).is_dir():
        for existing in (work, plain):
            if existing.is_dir():
                return existing
        return work
    return plain


def _configured_app_dir(root: Path) -> Path:
    """``app_dir`` out of ``root``'s settings file, if it names one.

    Read as raw JSON and not through `Settings`, because this runs *before*
    there is a settings file to speak of: the model's own loader asks
    :func:`settings_path`, which asks this, and validating the whole file to
    learn one string would make a single bad key cost the app its data
    directory. Anything unreadable, malformed, or not a string is no answer,
    and the caller keeps the directory it already had.

    One hop, deliberately: the file at the far end is the settings the app
    then runs on, and a chain of redirects is a loop waiting to be written by
    hand.
    """
    try:
        data = json.loads((root / "settings.json").read_text())
    except (OSError, json.JSONDecodeError):
        return root
    named = data.get("app_dir") if isinstance(data, dict) else None
    if not isinstance(named, str) or not named.strip():
        return root
    return Path(named).expanduser()


def app_dir() -> Path:
    """Application data directory.

    Three answers, in order. ``$HPCA_HOME`` is the override and wins outright
    — it is what the tests and `evals/smoke_pty.py` run behind, and a variable
    that could be overruled by a file inside the directory it names would be
    no override at all. Otherwise the rule above picks a root, and that root's
    settings file may redirect to somewhere else entirely.
    """
    override = os.environ.get("HPCA_HOME")
    if override:
        return Path(override)
    return _configured_app_dir(_default_app_dir())


def settings_path() -> Path:
    return app_dir() / "settings.json"


class SettingsError(Exception):
    """Raised when the settings file cannot be parsed or validated."""


class _Section(BaseModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)


# Request fields `extra_body` may never carry. Every one of them is decided by
# the client from the conversation it is having, and a settings file that could
# set them would not be configuring a backend, it would be silently answering a
# different question than the one the agent asked. Refused at load, the way
# PaletteSettings refuses a colour it cannot draw, rather than at request time
# where the failure is one 400 in the middle of a turn.
_RESERVED_BODY_KEYS = frozenset({"model", "messages", "stream", "tools"})


def _no_reserved_keys(body: dict[str, Any]) -> dict[str, Any]:
    clashes = sorted(_RESERVED_BODY_KEYS & set(body))
    if clashes:
        raise ValueError(
            f"extra_body may not set {', '.join(clashes)} — "
            "those are the client's, not the backend entry's"
        )
    return body


# What a backend needs *sent* that no other backend does. The OpenAI shape is
# a floor, not a ceiling: every server extends it, and the extensions disagree.
# Measured case, and the reason this exists — turning thinking off:
#
#   vLLM   chat_template_kwargs.enable_thinking = false   (what `llm.py` sends)
#   Ollama reasoning_effort = "none"                      (drops the above)
#
# Sending both unconditionally is not an option: Qwen3.8 under vLLM answers a
# reasoning_effort outside xhigh/medium/low with a 400 (`hpca.thinking`), so
# the value that fixes one backend breaks the other. It is a property of how
# the server was launched — exactly like `tool_protocol` below it — so it is
# configuration, and a generic one, because the next backend's quirk will not
# be this one.
ExtraBody = Annotated[dict[str, Any], AfterValidator(_no_reserved_keys)]


class LLMSettings(_Section):
    base_url: str = "http://localhost:8000/v1"
    api_key: str | None = None
    model: str = "qwen3-6b"
    constrained_decoding: Literal["auto", "on", "off"] = "auto"
    # How a tool call travels. "native" uses the backend's own tool-calling
    # channel — the shape agent-trained models were post-trained on, and it
    # moves the tool listing out of the prompt into the chat template.
    # "envelope" is the hand-rolled JSON decision object under a grammar
    # (§4.3), which works on any OpenAI-compatible backend, including one too
    # old to have a tool-call parser.
    #
    # Envelope stays the default, and v0.22.0 is where that stopped being a
    # portability argument and became a measured one. specs/specs-edit-eval.md §7.3:
    # on Qwen3.8-27B the shape error that made native lose on Qwen3.6 is gone
    # and native is now correct — but over n=72 it costs 24.5% more completion
    # tokens and 20.3% more wall for the same success rate, and §7.4 is the
    # bigger reason: a cut-off long write cannot be salvaged on that channel,
    # because the parser drops the fragment instead of returning it.
    #
    # Native is a supported, tested choice — set it here, or per backend
    # below. It needs the server started for it (vLLM:
    # --enable-auto-tool-choice plus a --tool-call-parser matching the model,
    # which is qwen3_coder for the Qwen3.8 the cluster serves, not the hermes
    # the Qwen3.6 notes named); a backend without it rejects the request with
    # a 400 rather than quietly degrading, so the failure is loud and the fix
    # is this setting (or the per-backend override on LLMBackend below).
    #
    # The parser is not a detail: it decides what a *cut-off* call looks like.
    # qwen3_coder drops the argument the generation died inside and reports
    # finish_reason "tool_calls" anyway — see middleware._hit_token_cap, which
    # is what keeps that from costing the turn.
    #
    # Deliberately a setting and not a probe: the protocol shapes the whole
    # conversation — the system prompt's respond-vs-tool guidance moves with
    # it — so a verdict that arrives mid-session would leave the prompt
    # describing a format the model can no longer emit. That mismatch is the
    # most expensive bug this area has had (specs/specs-edit-eval.md §7).
    tool_protocol: Literal["envelope", "native"] = "envelope"
    max_retries: int = 3
    request_timeout_s: int = 120
    # Reasoning models think in a separate channel, shown in the chat window's
    # thinking box. This is only the fallback for a caller that expresses no
    # opinion (the sub-agents that hard-code False, a bare client in a test):
    # a chat turn's thinking comes from the *session's* effort level
    # (`agent.default_thinking`, `/reasoning`, hpca.thinking), which reaches the
    # wire per request. It has to be per request and not a client property,
    # because one client is shared by every session on the same backend
    # (core.backends.client_for_backend keys them by base_url||model).
    #
    # Off by default, and worth keeping off: thinking is not generally better —
    # small models often route tools worse with it on — and it is slow.
    # Measured on Qwen3.6-27B: a turn took 133s thinking against 2.3s without.
    enable_thinking: bool = False
    # Tool calls allowed in one turn before the agent must stop and summarise.
    # Each run_bash call is one "look at the system" step; a real check needs
    # several, and debugging a script costs more. The
    # default is -1: no cap, the agent works until it is done. Set a positive
    # number to keep turns snappy or to bound autonomous work.
    max_tool_rounds: int = -1
    # Merged into every chat/completions body this client sends, last, so
    # it wins over what `_payload` built. See ExtraBody above.
    extra_body: ExtraBody = Field(default_factory=dict)


class AgentSettings(_Section):
    """What a session starts with, and falls back to if it never chose.

    ``default_mode`` is the interaction mode (§3.5): ``manual`` shows every
    script to the user before it runs, ``auto`` works until the task is done,
    and ``full-auto`` additionally waives the destructive-op approvals.

    ``default_thinking`` is the reasoning-effort dial (hpca.thinking), which
    has the same lifecycle: stored per session, empty there meaning "whatever
    this says", changeable mid-session with ``/reasoning``. It defaults to
    ``xhigh``, the slowest level and the one that has worked best in use on
    the cluster's Qwen3.8 — it costs a thinking pass before *every* decision
    in a turn, and that is the price paid for it (project.md §3.6). ``off``
    is one ``/reasoning`` away for a session that wants speed, and is what a
    backend with no reasoning channel should be set to.
    """

    default_mode: Literal["manual", "auto", "full-auto"] = "manual"
    default_thinking: ThinkingEffort = "xhigh"


class ClusterSettings(_Section):
    submit_host: str | None = None
    job_poll_seconds: int = 30


class SafetySettings(_Section):
    backup_limit_gb: float = 1
    trash_ttl_days: int = 7


class MemorySettings(_Section):
    # Token budget for the injected system-prompt memory scope. Hard for writes
    # — a full scope rejects new memories until the user condenses it — but
    # injection never truncates what is in the file. RAG has no budget.
    system_prompt_token_cap: int = 2400
    # Whether session_search may recall sessions of OTHER profiles. Off by
    # default: profiles exist to isolate what each context learns and sees.
    cross_profile_search: bool = False
    # RAG (retrieved memory): not injected wholesale, only the entries matching
    # the current request, within this per-turn prefetch budget.
    rag_prefetch_chars: int = 800
    rag_prefetch_count: int = 3
    # Curator: ages RAG entries out so retrieval does not decay as notes
    # accumulate. Archived entries are moved to <profile>.archive.md, never
    # deleted. 0 disables the pass.
    curator_interval_days: int = 7
    curator_stale_days: int = 30
    curator_archive_days: int = 90
    # Whether /conclude self-review may propose entirely NEW skills, as opposed
    # to patches to existing ones. On so the quality can be judged in practice;
    # set false if a small model's skill drafts prove not worth reviewing.
    propose_new_skills: bool = True


ClipboardMode = Literal["auto", "tmux", "screen", "zellij", "osc52", "command", "file"]


class ClipboardSettings(_Section):
    mode: ClipboardMode = "auto"
    command: str | None = None
    osc52_limit_kb: int = 74


class LoggingSettings(_Section):
    """Plain-text session transcripts for later analysis (see hpca.logs)."""

    enabled: bool = True
    # Default: <app dir>/chatlogs. Set to collect them somewhere else, e.g. on
    # a project share; ~ is expanded.
    dir: str | None = None


class RagSettings(_Section):
    embedding: str = "BAAI/bge-m3"
    embedding_base_url: str = "http://localhost:20000/v1"


class EndpointsSettings(_Section):
    """Auto-connect: cluster endpoint discovery (spec §6).

    On the cluster, HPCA reads vLLM manifest files (written by the launch
    scripts) from ``endpoints_dir`` and connects directly. Off the cluster
    that dir is empty, so it falls through to the localhost scan and prints an
    SSH-tunnel template built from ``login_host``/``login_user``.

    The default is the shared group directory: manifests written by anyone on
    the team are visible to everyone's HPCA, so a user connects to a server
    they did not launch without configuring anything. Job ids are unique
    cluster-wide, so manifests from different users cannot collide. Point this
    somewhere private (and set ``HPCA_ENDPOINTS_DIR`` for the launch scripts)
    to opt out.
    """

    endpoints_dir: str = "/data/cephfs-1/work/groups/cubi/tools/hpca_connections"
    # Optional ordered model-id substrings for full auto-connect to the top
    # available match; empty => the "auto if one, list if many" default.
    preferred_models: list[str] = []
    login_host: str = "hpc-login-2.cubi.bihealth.org"
    login_user: str | None = None

    def dir_path(self) -> Path:
        return Path(os.path.expanduser(self.endpoints_dir))

    def login_target(self) -> str:
        """``user@host`` for the tunnel template, or just ``host`` if unset."""
        if self.login_user:
            return f"{self.login_user}@{self.login_host}"
        return self.login_host


class DatabaseSettings(_Section):
    """Where the sqlite databases actually run (see specs/specs-db-local-cache.md).

    On a cluster node ``$HOME`` is NFS, where every sqlite call is network
    round-trips plus a remote fsync, and the TUI lags whenever the agent
    works. With ``local_cache`` on, the databases are copied to node-local
    storage at startup, used from there, and synced back to home periodically
    and on exit.
    """

    local_cache: bool = True
    # Where the working copies go. None => $TMPDIR (the per-job dir Slurm
    # sets, which is what makes them node-local), else the system temp dir.
    local_dir: str | None = None
    # How often the working copies are written back, and so the upper bound on
    # what a hard kill (walltime, node failure, SIGKILL) can lose. 0 syncs only
    # on exit.
    sync_interval_s: int = Field(default=60, ge=0)


class WatchSettings(_Section):
    """The watch boxes' own dials — the boxes, and not the cluster in them.

    Filed apart from `cluster` deliberately. That section says where the
    cluster *is* (``submit_host``) and how often squeue is asked; a peek is
    just as often a plain log file on this machine as a job's stdout, and a
    key that trims a tail read off any path has no business living under a
    heading about Slurm. One key is a small section, but the file's shape is
    already a section per feature — rag, logging, endpoints, database — and a
    reader looking for the watch panel's settings looks for the watch panel.
    """

    # How much of a log's tail the Enter peek ships (`watches.peek`).
    #
    # 300 was the right figure while the answer was a toast: it expired on a
    # timer and had to stay small enough to be read in the seconds it lasted.
    # The tail now lands in a scrollable window instead (`client._peeked`), so
    # the constraint the 300 was chosen against is gone — and what was left
    # was a cap that cut a traceback off above the line naming the exception.
    # 2000 is a couple of screenfuls: enough for the end of a stack trace or
    # the last few progress lines, still nowhere near the gigabyte of progress
    # bars the read window is sized to avoid.
    #
    # ``gt=0``: zero would ship an empty peek for every box, which reads as
    # the log being empty — a different fact from one nobody asked to see.
    peek_chars: int = Field(default=2000, gt=0)


_COLOUR = re.compile(r"^(?:#[0-9a-fA-F]{6}|[0-9]|[1-9][0-9]|1[0-9]{2}|2[0-4][0-9]|25[0-5])$")


def _colour(value: str) -> str:
    """One palette entry, checked here rather than in the repaint.

    A colour is either an xterm-256 index (``"215"``) or a hex triple
    (``"#ffaf5f"``). Refused at load, where the answer is a line beside the
    field that named it — the alternative is a malformed escape sequence
    reaching a terminal in raw mode, which does not report a bad setting so
    much as stop being a terminal.
    """
    if not _COLOUR.match(value):
        raise ValueError(
            f"{value!r} is not a colour: use an xterm index 0-255 or #rrggbb"
        )
    return value


Colour = Annotated[str, AfterValidator(_colour)]


class PaletteSettings(_Section):
    """The colours, by the job each one does rather than by what it is.

    Named for roles because that is the part that has to stay true: `warn` is
    whatever colour warnings are, and a theme that makes it blue has made a
    choice, not a mistake. A key called `yellow` would be a lie the first time
    somebody took the offer this section exists to make.

    The defaults are the dark palette the app has always drawn, with three
    corrections: the chrome is xterm 73 rather than 44, whose chroma put the
    least important thing on the screen ahead of the most; `warn` is 172 rather
    than 179, far enough from `user` that a warning and one's own words stop
    reading as relatives; and `muted`/`faint` are real greys rather than the
    `DIM` attribute, whose rendering was the terminal theme's opinion and
    differed from one to the next on the same screen.
    """

    # The two that carry prose. `agent` holds most of what is on screen, so it
    # is the calmer of the two; `user` is what a scrollback is searched for.
    agent: Colour = "255"
    user: Colour = "215"
    # Rules, focused pane titles, key hints — structure rather than content.
    chrome: Colour = "73"
    # The signal three. Nothing decorative is drawn in these on purpose: the
    # whole value of `ok` is that seeing it means something is actually well.
    ok: Colour = "71"
    warn: Colour = "172"
    danger: Colour = "167"
    # Secondary and tertiary text. Both a step brighter than the greys this
    # started with (245/240): a turn's working is drawn entirely in `faint`,
    # and #585858 on a dark terminal is under 3:1 against the ground — a step
    # row was legible on the screen it was picked on and murky on the next
    # one. Tertiary means quieter than the prose, not harder to read than it.
    muted: Colour = "250"
    faint: Colour = "245"
    # The working row's drop, head first (`ui.rain.spinner`). Shorter than four
    # is a shorter trail, not an error — one entry is a plain blinking cell.
    # It is off the signal green because a spinner says "busy", not "well", and
    # a green that also means busy is a green that no longer means well.
    spinner: list[Colour] = Field(
        default=["73", "66", "23", "236"], min_length=1, max_length=4
    )
    # The background the pane that just took focus is washed in; see
    # `focus_flash_seconds`.
    flash: Colour = "23"
    # The background every notification is drawn on: a toast is not part of
    # the interactive UI, and a ground of its own says so and draws the eye.
    # Off the flash's teal, because a toast is not a pane taking focus.
    overlay: Colour = "236"


class DisplaySettings(_Section):
    """What the front-end draws *with*, as opposed to what the core does.

    The odd section out, because nothing in this process reads it: rule 2 of
    §4.2 keeps the settings file out of a front-end's reach, so these travel
    over the wire instead — `protocol.DisplaySettings`, carried on `hello` and
    restated after a save (`core.service._apply_settings`). They are gathered
    into one section for exactly that reason: the payload that crosses *is*
    this section, so adding a display key here is the whole of adding one, and
    no other part of the tree can reach the wire by being renamed into it.
    """

    # Whether a chat row's label carries the date and time it was said —
    # ``▸ you 22-08-2026 13:04:47`` against a bare ``▸ you``. On, because the
    # stamp is what tells a turn's question from its answer when both landed
    # in the same minute (`ui.state.STAMP_FORMAT` says why the seconds are
    # there), and a log read back the next day is read for when as much as for
    # what. Off is for the narrow terminal, and for the reader who wants the
    # conversation and not the clock.
    chat_stamps: bool = True
    # Whether the screen behind "Really quit?" rains (`ui.rain`). That one
    # question clears the frame it was asked over, and this fills the black it
    # leaves. On, because it costs nothing anybody is waiting on — and off,
    # because it is a repaint ten times a second of a screen made mostly of
    # escape sequences, which is a real thing to spend down a slow link and a
    # fair thing to decline.
    #
    # Named for the one dialog it appears on. The other questions here — stop
    # this turn, delete this session — clear the screen just the same and stay
    # black, so a key called `confirm_rain` would promise three screens it
    # does not paint.
    quit_rain: bool = True
    # How often that screen is repainted while it falls. Frames a second.
    #
    # Sixty by default, because falling is the whole of what it does and ten a
    # second reads as stepping. It is a setting and not a constant because
    # what it costs is a property of the *link*, not of the effect: the
    # painter sends only changed rows, so 60 is about 313 KiB/s against 88 at
    # 10 — nothing on a local terminal, enough to feel down a slow tunnel to a
    # login node. `ui.rain.FPS` has the measured table.
    #
    # Bounded at both ends. Below 1 there is no frame to book and the field
    # would hang mid-drop; above 120 the frames are closer together than the
    # glyphs change, so it is bytes bought for nothing at all.
    #
    # The literal rather than `ui.rain.FPS`, the way the pulse period
    # below carries its own: this module is read by a core that may be
    # running with no front-end in the process at all.
    quit_rain_fps: int = Field(default=60, ge=1, le=120)
    # How long one breath of the decision prompt's answer line takes
    # (`ui.ansi.pulse`). Seconds, fractional.
    #
    # ``gt=0`` and nothing else. Zero divides by zero in the sweep and a
    # negative runs it backwards, so both are refused here — where a refusal
    # is one line beside the editor that named the field, rather than an
    # exception out of a repaint loop with the terminal in raw mode. The upper
    # end is left open: a very long period is a line that barely moves, which
    # is a legitimate thing to ask for. The lower end is not clamped either,
    # but the repaint is only booked every `ui.ansi.PULSE_INTERVAL` (0.1s), so
    # a period under about 0.2s aliases into a flicker rather than a breath.
    decision_pulse_seconds: float = Field(default=1.0, gt=0)
    # The colours, all of them.
    palette: PaletteSettings = PaletteSettings()
    # How long the pane that just took focus is washed in `palette.flash`.
    #
    # A hold and not a fade. A decay needs frames to decay over and this is
    # over in one or two, so the ramp it would walk is a ramp nobody sees; what
    # the eye is actually caught by is the movement, which a step gives it more
    # cheaply than a gradient. Two repaints per focus change, one of which had
    # to happen anyway to redraw the title rule.
    #
    # Zero turns it off, which is why the floor is `ge` and not `gt`: a user
    # who finds the flash startling on a large terminal should be able to say
    # so, and the persistent focus mark on the title rule is still there
    # underneath it. Capped at a second because past about a quarter of one it
    # has stopped being a flash and started being a state, and a state that
    # says "focus arrived here recently" is not a thing this UI has.
    focus_flash_seconds: float = Field(default=0.1, ge=0, le=1.0)
    # Blank rows drawn under each chat row — a message, a turn's working, a
    # notice — and so also the gap between the foot of the conversation and
    # the message box, since the chat hangs from the bottom of its pane.
    #
    # One, because a chat is a column of paragraphs and the only thing saying
    # where one stops was the weight of the next `you` / `hpca` nameplate: it
    # is enough on a two-line exchange and not enough once a reply runs to
    # twenty, where the eye has to walk back up to find the boundary. A blank
    # line is what prose has always used for that.
    #
    # Zero is off, and off is a real answer rather than a grudging one: on a
    # short terminal every blank is a line of conversation that is not on
    # screen, and somebody reading a long turn on a 24-row window is trading
    # a boundary they can already see for rows they cannot spare.
    #
    # Capped at 8. Past there the pane is mostly gap — one message a screen —
    # which is not a spacious log, it is a broken one, and a cap is cheaper
    # than the bug report from whoever typed 800 to see what happened.
    #
    # Only the chat. The sessions and watchers columns are lists of one-line
    # titles that were never hard to tell apart, and doubling their height
    # would cost the chat the rows `_heights` gives it.
    spacer_lines: int = Field(default=1, ge=0, le=8)


class LLMBackend(_Section):
    """One entry of the configured backend catalog (manage-LLMs screen).

    There is no ``enable_thinking`` here any more. It was a per-backend flag on
    the reasoning that thinking is a property of the model — true, but the
    thing a user actually adjusts is how hard *this conversation* should think,
    and answering that by editing the catalog changed every session on that
    backend at once. It is now a per-session level (hpca.thinking, ``/reasoning``).
    Entries written before v0.22.0 still carry the key; ``extra="ignore"`` on
    ``_Section`` drops it on load, so an old settings file needs no migration.
    """

    model: str
    base_url: str
    api_key: str | None = None
    max_model_len: int | None = None
    # Whether this server can carry a call on its own tool channel is a
    # property of how it was launched, not of the client — one entry may be a
    # vLLM started with --enable-auto-tool-choice while the next is an older
    # one that 400s on `tools`. None means "whatever llm.tool_protocol says",
    # which is what every entry written before v0.22.0 has.
    tool_protocol: Literal["envelope", "native"] | None = None
    # This entry's own extra fields, merged over `llm.extra_body` rather
    # than replacing it: the base can carry what every backend here needs
    # and an entry only states its difference. Empty means the base as-is.
    extra_body: ExtraBody = Field(default_factory=dict)


def llm_settings_for(backend: LLMBackend, base: LLMSettings) -> LLMSettings:
    """The LLMSettings for a client talking to ``backend``: the backend's
    connection fields over the base's client behavior (retries, timeout,
    constrained decoding, tool budget). Used for per-session clients."""
    settings = base.model_copy(deep=True)
    settings.base_url = backend.base_url
    settings.model = backend.model
    settings.api_key = backend.api_key
    if backend.tool_protocol is not None:
        settings.tool_protocol = backend.tool_protocol
    if backend.extra_body:
        settings.extra_body = {**settings.extra_body, **backend.extra_body}
    return settings


class Settings(_Section):
    llm: LLMSettings = LLMSettings()
    agent: AgentSettings = AgentSettings()
    backends: list[LLMBackend] = []
    known_llm_ports: list[int] = []
    llm_api_keys: list[str] = []
    cluster: ClusterSettings = ClusterSettings()
    safety: SafetySettings = SafetySettings()
    memory: MemorySettings = MemorySettings()
    clipboard: ClipboardSettings = ClipboardSettings()
    editor: str | None = None
    rag: RagSettings = RagSettings()
    logging: LoggingSettings = LoggingSettings()
    endpoints: EndpointsSettings = EndpointsSettings()
    database: DatabaseSettings = DatabaseSettings()
    watches: WatchSettings = WatchSettings()
    display: DisplaySettings = DisplaySettings()
    # Where everything above is kept, when the default is not where you want
    # it. Null is "wherever `app_dir` decides", which is ``~/work`` on a
    # cluster node and ``~`` on a machine that has no ``~/work``; a path here
    # overrides both, and ``~`` in it is expanded.
    #
    # It is a real field and not merely a key `app_dir` happens to read,
    # because :meth:`save` writes the whole model back over the file — an
    # unmodelled key would survive being hand-edited exactly until the
    # settings editor next saved, and then the app would silently move house.
    #
    # Read before this model is loaded (`_configured_app_dir` parses the raw
    # JSON), so the value is only ever consulted from the file the rule found:
    # writing it into the settings of the directory it points *at* says
    # nothing, and points at nothing.
    app_dir: str | None = None

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        path = path or settings_path()
        if not path.exists():
            return cls()
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError as e:
            raise SettingsError(f"Malformed JSON in {path}: {e}") from e
        try:
            return cls.model_validate(data)
        except ValidationError as e:
            raise SettingsError(f"Invalid settings in {path}: {e}") from e

    def save(self, path: Path | None = None) -> None:
        path = path or settings_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2) + "\n")

    def activate_backend(self, backend: LLMBackend) -> None:
        """Make a catalog entry the active/default backend (used on startup).

        Only connection fields change; client behavior (retries, timeout,
        constrained decoding) is kept.
        """
        self.llm.base_url = backend.base_url
        self.llm.model = backend.model
        self.llm.api_key = backend.api_key
        if backend.tool_protocol is not None:
            self.llm.tool_protocol = backend.tool_protocol
        # Carried for the same reason as tool_protocol: the bootstrap client
        # is built from `self.llm`, so an entry whose extra fields stayed in
        # the catalog would be correct for every session that pinned it and
        # wrong for the one that did not.
        if backend.extra_body:
            self.llm.extra_body = {**self.llm.extra_body, **backend.extra_body}

    def remember_llm_ports(self, base_urls: Iterable[str]) -> bool:
        """Record ports that have served an LLM, so scans probe them first.

        Which GPU node hosts a model changes often, the tunneled local port
        rarely does. Returns whether anything new was learned, i.e. whether
        the settings need saving.
        """
        learned = False
        for base_url in base_urls:
            port = urlparse(base_url).port
            if port is not None and port not in self.known_llm_ports:
                self.known_llm_ports.append(port)
                learned = True
        return learned

    def remember_llm_key(self, api_key: str | None) -> bool:
        """Add an API key to the pool if new. Returns whether it was newly
        learned (i.e. whether settings need saving).

        Keys already known to the app are tried against key-locked endpoints,
        so any key typed for one backend can unlock the others. Empty or
        missing keys are ignored; dedup is by exact string.
        """
        if not api_key:
            return False
        if api_key in self.llm_api_keys:
            return False
        self.llm_api_keys.append(api_key)
        return True

    def is_active(self, backend: LLMBackend) -> bool:
        return (
            self.llm.base_url == backend.base_url
            and self.llm.model == backend.model
        )
