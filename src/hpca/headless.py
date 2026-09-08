"""The second front-end: one prompt in, one reply out, no terminal.

`hpca -p "..."` is this module. It is the same runtime the TUI runs — the
`Core` of `hpca.core.boot`, over the same `InProcessConnection` — with a script
where the screen was. Nothing here draws, and nothing in the core knows the
difference; that the two front-ends share a runtime *unchanged* is the whole
claim of the protocol split (§4.2), and this is the thing that tests it.

What it exists for is an agent driving HPCA rather than a person: another
model, a CI step, a `for` loop. That user cannot answer a prompt, cannot read a
toast, and cannot tell "the model said nothing" from "the process crashed" —
which is what the four decisions below are about.

**Nobody can be asked, so somebody has to have decided.** Three events park
work waiting for a human: `decision.requested` (a gated tool call),
`confirm.requested` (triage offering a signature it learned) and
`compact.proposed`. Answering none of them is a hang. The answers are:

* a gated call is **approved**, and this is the decision that was got wrong
  first, so it is worth the paragraph. Refusing looks like the safe default and
  is not one. Measured, on the first real run of this module: `edit_file` on a
  one-line bug fix was refused, and the model — correctly following the denial
  message, which asks for "an amended operation" — did the same edit through
  `run_bash` with `sed -i`. The edit happened either way. What refusing
  actually bought was the *worse* of the two routes: `edit_file` keeps a trash
  backup (§5.3) and puts a diff in the log, and `sed -i` keeps neither.
  So the gate is not a safety property when nobody is behind it; it is a
  question, and an unanswered question just moves the work somewhere less
  observable. The real dial is `--mode` — `manual` gates every execution tool,
  `auto` only the destructive ones, `full-auto` none — and the real net is the
  trash layer, which is what `full-auto`'s own docstring says it is relying on.
  `--refuse-gated` restores the refusal for a run that genuinely must not
  write; it tells the model not to reroute, which is guidance and not a
  guarantee, because `run_bash` runs arbitrary bash and always could.
* a triage offer is **declined**. It proposes writing a log signature into the
  user's own store, which is a side effect on the next human's session, from a
  poll this run did not ask for.
* a compaction is **accepted**. It is the only one whose refusal costs the
  turn, and it changes nothing outside the conversation.

**A pinned backend, named rather than assumed.** The session is created with an
explicit `LLMEntry.label` (`SessionNew.backend`), resolved out of the catalog
before anything is submitted. Not because naming is tidier: `service.startup()`
runs `backends.auto_connect()`, which on a cluster node may activate a backend
this run never asked for, between the moment settings were read and the moment
the turn starts. A session that pinned its backend keeps it
(`backends.max_model_len_for`, `backend_for`); one that did not, does not.

**Exit codes are the interface.** stdout is the reply and nothing else, so it
can be captured; progress and warnings go to stderr; and the difference between
"the model answered" and "no LLM was reachable" is a number rather than a
string the caller has to match on.

The one thing a run needs told about the wider system: give sub-agent runs
their own ``$HPCA_HOME``. Two processes on one app dir is survivable — the
second loses the `db.lease` race and runs directly on home
(specs/specs-db-local-cache.md §2.5) — but its sessions land in the sidebar of
whoever is using the TUI, which is rarely what was meant.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# The exit codes, and they are part of the interface — a caller distinguishes
# these cases without parsing anything. 2 follows `evals/edit_eval.py`, which
# already spends it on an unreachable backend.
EXIT_OK = 0
EXIT_USAGE = 1
EXIT_NO_BACKEND = 2
EXIT_TIMEOUT = 3
EXIT_TURN_FAILED = 4

# Handed back on a refused tool call, when `--refuse-gated` asked for one.
# Written for the model, not for a log. The second sentence is there because
# without it the first one is an invitation: the denial message the graph wraps
# this in asks for "an amended operation", and a model that reads "refused"
# and "amend" together will reach for `run_bash` and do the same thing with
# `sed -i` — which is what it did, the first time this module ran. Nothing here
# can stop that (`run_bash` runs arbitrary bash); saying so is what turns a
# reroute into a deliberate act rather than the obvious next step.
REFUSAL_REASON = (
    "Refused: this is an unattended run with nobody to approve it. "
    "Do not do the same thing another way — a shell command that edits or "
    "deletes the file is the same operation without the backup. "
    "Work around it, or stop and say what you would need approval for."
)


@dataclass
class Outcome:
    """What the run has to say for itself, in one object.

    Assembled whether the turn worked or not, because the caller that most
    needs the tool list and the refusals is the one whose turn failed.
    """

    session_id: str = ""
    reply: str | None = None
    error: str | None = None
    exit_code: int = EXIT_OK
    # One entry per tool exchange, in order: {"tool", "target", "failed"}.
    # Taken from the `Part`s of the chat entries rather than counted off
    # `turn.activity`, which cannot be counted: `interrupt()` re-runs its node
    # on resume, so an approved call reports "running edit_file" twice and a
    # tally built from that says the agent edited the file two times.
    tools: list[dict[str, Any]] = field(default_factory=list)
    refusals: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def as_json(self) -> str:
        return json.dumps(
            {
                "session_id": self.session_id,
                "reply": self.reply,
                "error": self.error,
                "exit_code": self.exit_code,
                "tools": self.tools,
                "refusals": self.refusals,
                "warnings": self.warnings,
                "usage": {
                    "prompt_tokens": self.prompt_tokens,
                    "completion_tokens": self.completion_tokens,
                },
            },
            indent=2,
        )


class _Session:
    """The event pump, and the four answers it gives without asking anyone.

    A class rather than a closure because the pump and the submit half both
    need the same handful of facts, and passing six mutable cells around a
    coroutine is how they end up disagreeing.
    """

    def __init__(self, wire, *, approve: bool, quiet: bool) -> None:
        self.wire = wire
        self.approve = approve
        self.quiet = quiet
        self.outcome = Outcome()
        self.done = asyncio.Event()
        self.created = asyncio.Event()
        self.catalog: list[dict] | None = None
        self.catalog_ready = asyncio.Event()
        self._last_activity = ""
        # Chat rows by `Entry.seq`, because a row is revised in place: the
        # same thinking entry comes again with one more part filled in, and
        # keeping the latest by seq is what turns that stream into a record.
        self._rows: dict[int, list[dict]] = {}

    # ----------------------------------------------------------------- output

    def note(self, text: str) -> None:
        """Progress, onto stderr, where it cannot contaminate the reply."""
        if not self.quiet:
            print(text, file=sys.stderr, flush=True)

    # ------------------------------------------------------------- the pump

    async def pump(self) -> None:
        from hpca import protocol as P

        async for env in self.wire:
            handler = getattr(self, f"_on_{env.type.replace('.', '_')}", None)
            if handler is not None:
                with contextlib.suppress(Exception):
                    await handler(P, env.payload)

    async def _on_llm_catalog(self, P, payload: dict) -> None:
        self.catalog = payload.get("entries") or []
        self.catalog_ready.set()

    async def _on_session_created(self, P, payload: dict) -> None:
        self.outcome.session_id = payload["row"]["session_id"]
        self.created.set()

    async def _on_turn_activity(self, P, payload: dict) -> None:
        activity = payload.get("activity") or ""
        # Only the changes: the core restates the same activity as a turn
        # ticks, and a line per restatement buries the ones that mean
        # something. Progress only — what the turn *did* comes from the parts.
        if activity and activity != self._last_activity:
            self._last_activity = activity
            self.note(f"… {activity}")

    async def _on_chat_append(self, P, payload: dict) -> None:
        entry = payload.get("entry") or {}
        seq = entry.get("seq") or 0
        if seq:
            self._rows[seq] = entry.get("parts") or []

    # A revision of a row already sent, and the same thing to us: keep the
    # latest parts for that seq and let the order fall out of the seq.
    _on_chat_update = _on_chat_append

    def _calls(self) -> list[dict[str, Any]]:
        return [
            {
                "tool": part.get("tool") or "",
                "target": part.get("target") or "",
                "failed": bool(part.get("failed")),
            }
            for _seq, parts in sorted(self._rows.items())
            for part in parts
            if part.get("kind") == "call"
        ]

    async def _on_turn_usage(self, P, payload: dict) -> None:
        self.outcome.prompt_tokens = payload.get("prompt_tokens") or 0
        self.outcome.completion_tokens += payload.get("completion_tokens") or 0

    async def _on_turn_finished(self, P, payload: dict) -> None:
        self.outcome.reply = payload.get("reply")
        self.outcome.tools = self._calls()
        self.done.set()

    async def _on_turn_failed(self, P, payload: dict) -> None:
        self.outcome.error = payload.get("error")
        self.outcome.exit_code = EXIT_TURN_FAILED
        self.outcome.tools = self._calls()
        self.done.set()

    async def _on_notify(self, P, payload: dict) -> None:
        text = payload.get("text") or ""
        if payload.get("severity") in ("warning", "error"):
            self.outcome.warnings.append(text)
        self.note(f"[{payload.get('severity')}] {text}")

    # ------------------------------------------- the answers nobody can give

    async def _on_decision_requested(self, P, payload: dict) -> None:
        what = str(payload.get("payload") or {})[:200]
        self.note(("approved: " if self.approve else "refused: ") + what)
        if not self.approve:
            self.outcome.refusals.append(what)
        await self.wire.send(
            P.DecisionResolve(
                session_id=payload["session_id"],
                approved=self.approve,
                reason="" if self.approve else REFUSAL_REASON,
            )
        )

    async def _on_confirm_requested(self, P, payload: dict) -> None:
        await self.wire.send(P.ConfirmResolve(id=payload["id"], confirmed=False))

    async def _on_compact_proposed(self, P, payload: dict) -> None:
        self.note("… accepting a compaction")
        await self.wire.send(
            P.CompactResolve(session_id=payload["session_id"], action="accept")
        )


def _ad_hoc_backend(settings, args) -> None:
    """Make `--base-url/--model` a catalog entry, and the active one.

    An entry rather than only `settings.llm`, because everything downstream
    that answers "how big is this window" and "which backend is this session
    on" reads the catalog (`backends.labels`, `max_model_len_for`), and a bare
    `settings.llm` is answerable by neither.
    """
    from hpca.config import LLMBackend

    entry = LLMBackend(
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        max_model_len=args.window,
        extra_body=json.loads(args.extra_body) if args.extra_body else {},
    )
    settings.backends = [
        b
        for b in settings.backends
        if not (b.base_url == entry.base_url and b.model == entry.model)
    ] + [entry]
    settings.activate_backend(entry)


def _resolve_label(catalog: list[dict], settings, wanted: str | None) -> str | None:
    """The label to pin the session to, or None with an explanation printed.

    ``wanted`` must match exactly when given: a near-miss silently pinning the
    wrong model is the failure this whole step exists to prevent, so the error
    lists what there was to choose from instead of guessing.
    """
    labels = [entry["label"] for entry in catalog]
    if wanted is not None:
        if wanted in labels:
            return wanted
        print(
            f"hpca: no backend labelled {wanted!r}. "
            f"Configured: {', '.join(labels) or '(none)'}",
            file=sys.stderr,
        )
        return None
    for entry in catalog:
        if (
            entry.get("base_url") == settings.llm.base_url
            and entry.get("model") == settings.llm.model
        ):
            return entry["label"]
    # Nothing in the catalog matches what is active — a first run with no
    # configured backends. Let the core pick, which is what `backend=None`
    # means, rather than refusing to start over a pin nobody asked for.
    return ""


async def run(args) -> Outcome:
    """One turn, start to finish, and the runtime around it."""
    if args.home:
        os.environ["HPCA_HOME"] = str(Path(args.home).expanduser().resolve())

    from hpca import protocol as P
    from hpca.config import Settings, app_dir
    from hpca.core.boot import Core, logs_to_file
    from hpca.transport import InProcessConnection

    settings = Settings.load()
    if args.base_url or args.model:
        if not (args.base_url and args.model):
            print(
                "hpca: --base-url and --model go together", file=sys.stderr
            )
            return Outcome(exit_code=EXIT_USAGE)
        _ad_hoc_backend(settings, args)
    if args.mode:
        settings.agent.default_mode = args.mode
    if args.max_rounds is not None:
        settings.llm.max_tool_rounds = args.max_rounds
    # A run that lives for one turn has nothing to gain from copying three
    # databases to a node-local disk and back, and something to lose: the
    # lease, which an interactive session on the same app dir wants.
    settings.database.local_cache = False

    ui_end, core_end = InProcessConnection.pair()
    no_backend = asyncio.Event()
    core = await Core.start(
        settings=settings,
        profile=args.profile,
        app_dir=app_dir(),
        wire=core_end,
        on_no_backend=no_backend.set,
    )
    session = _Session(ui_end, approve=not args.refuse_gated, quiet=args.quiet)
    with logs_to_file("headless.log"):
        core.run()
        pump = asyncio.ensure_future(session.pump())
        try:
            await _drive(P, session, args, settings, no_backend)
        finally:
            pump.cancel()
            with contextlib.suppress(Exception, asyncio.CancelledError):
                await pump
            with contextlib.suppress(Exception):
                await ui_end.close()
            await core.stop(say=lambda text: None)
    return session.outcome


async def _drive(P, session: _Session, args, settings, no_backend) -> None:
    """The command half: pin, open, submit, wait.

    Split from `run` so that the shutdown above it happens on every path out of
    here, including a timeout — a run that leaves the checkpointer open loses
    the turn it just paid for.
    """
    outcome = session.outcome

    await session.wire.send(P.LLMList())
    with contextlib.suppress(asyncio.TimeoutError):
        await asyncio.wait_for(session.catalog_ready.wait(), 30)
    label = _resolve_label(session.catalog or [], settings, args.backend)
    if label is None:
        outcome.exit_code = EXIT_USAGE
        return

    if args.session:
        outcome.session_id = args.session
    else:
        await session.wire.send(
            P.SessionNew(profile=args.profile, backend=label or None)
        )
        try:
            await asyncio.wait_for(session.created.wait(), 60)
        except asyncio.TimeoutError:
            outcome.error = "the core never answered session.new"
            outcome.exit_code = _no_backend_or(no_backend, EXIT_TURN_FAILED)
            return
    session.note(f"session {outcome.session_id}")

    await session.wire.send(P.SessionOpen(session_id=outcome.session_id))
    await session.wire.send(
        P.ModeSet(session_id=outcome.session_id, mode=args.mode)
    )
    await session.wire.send(
        P.TurnSubmit(
            session_id=outcome.session_id,
            text=args.prompt,
            forced_skill=args.skill,
        )
    )
    try:
        await asyncio.wait_for(session.done.wait(), args.timeout)
    except asyncio.TimeoutError:
        outcome.error = f"no answer within {args.timeout:g}s"
        outcome.exit_code = EXIT_TIMEOUT
        return
    if outcome.exit_code == EXIT_TURN_FAILED:
        outcome.exit_code = _no_backend_or(no_backend, EXIT_TURN_FAILED)


def _no_backend_or(no_backend, fallback: int) -> int:
    """`EXIT_NO_BACKEND` when startup found nothing answering.

    Checked here rather than before the turn because the probe runs
    concurrently with it: waiting for the verdict up front would add a round
    trip to every run to answer a question that only matters when something has
    already gone wrong.
    """
    return EXIT_NO_BACKEND if no_backend.is_set() else fallback


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """The `-p` half of `hpca`'s argv, kept here with what reads it."""
    parser.add_argument(
        "-p",
        "--print",
        dest="prompt",
        metavar="TEXT",
        help="run one turn headlessly, write the reply to stdout, and exit",
    )
    parser.add_argument(
        "--session", help="continue this session id instead of making one"
    )
    parser.add_argument(
        "--backend", help="pin to this catalog label (see the manage-LLMs screen)"
    )
    parser.add_argument("--base-url", help="an ad-hoc backend: its OpenAI base url")
    parser.add_argument("--model", help="an ad-hoc backend: the model to ask for")
    parser.add_argument("--api-key", help="an ad-hoc backend: its key, if it wants one")
    parser.add_argument(
        "--window",
        type=int,
        help="an ad-hoc backend's context length, for backends that do not say",
    )
    parser.add_argument(
        "--extra-body",
        metavar="JSON",
        help='an ad-hoc backend\'s extra request fields, e.g. \'{"reasoning_effort":"none"}\'',
    )
    parser.add_argument(
        "--mode",
        default="auto",
        choices=("manual", "auto", "full-auto"),
        help="interaction mode for the turn (default: %(default)s)",
    )
    parser.add_argument(
        "--refuse-gated",
        action="store_true",
        help="refuse gated tool calls instead of approving them (see --mode)",
    )
    parser.add_argument("--skill", help="force this skill for the turn")
    parser.add_argument(
        "--max-rounds",
        type=int,
        help="stop the agent after this many tool calls (default: unbounded)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=900.0,
        help="seconds to wait for the turn (default: %(default)s)",
    )
    parser.add_argument("--home", help="the app dir to run under ($HPCA_HOME)")
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="write one JSON object to stdout instead of the bare reply",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="no progress on stderr"
    )


def main(args) -> int:
    """Run one turn, write it to stdout, and return the process exit code.

    stdout carries exactly one thing — the reply, or the JSON object — so that
    `$(hpca -p ...)` is the answer and nothing else. Everything a human would
    have read on the way is already on stderr.
    """
    if not args.prompt:
        print("hpca: -p/--print needs something to ask", file=sys.stderr)
        return EXIT_USAGE
    outcome = asyncio.run(run(args))
    if args.as_json:
        print(outcome.as_json())
    else:
        if outcome.reply:
            print(outcome.reply)
        if outcome.error:
            print(f"hpca: {outcome.error}", file=sys.stderr)
    return outcome.exit_code
