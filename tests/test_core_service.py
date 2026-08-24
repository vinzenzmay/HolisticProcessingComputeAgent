"""The four services, assembled and driven by commands alone.

Each service has its own tests; this one exists to prove they compose — that a
`turn.submit` arriving as a protocol command reaches the graph with the right
client, the right prompt and the right tool context, and comes back out as
events. It is the closest thing to running the real thing without a terminal,
and it is what would catch two services agreeing on a name and disagreeing on
what it means.
"""

from __future__ import annotations

import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from hpca.config import LLMBackend, Settings, settings_path
from hpca.core.service import AgentService, build_service
from hpca.db import connect, init_db
from hpca.episodic import EpisodicStore
from hpca.llm import ChatResponse
from hpca.jobs import JobStore
from hpca.memory_ops import MemoryOp
from hpca.agent.struggle import STRUGGLE_KIND
from hpca.profiles import MemoryScope, Profile
from hpca.protocol import (
    PROTOCOL_VERSION,
    BackendProbe,
    BackendRemove,
    BackendScan,
    BackendSet,
    Command,
    CommandList,
    CommandRun,
    CompactResolve,
    ConfirmResolve,
    DecisionResolve,
    JobCancel,
    LLMList,
    MemoryResolve,
    ModeSet,
    Notify,
    ProcessKill,
    ProfileCreate,
    ProfileDelete,
    ProfileDuplicate,
    ProfileGet,
    ProfileList,
    ProfileSave,
    ProfileSet,
    SessionClose,
    SessionDelete,
    SessionFocus,
    SessionFork,
    SessionList,
    SessionMove,
    SessionNew,
    SessionOpen,
    SessionRename,
    SessionRetitle,
    SessionRollback,
    SettingsGet,
    SettingsSave,
    Shutdown,
    SkillDelete,
    SkillDraft,
    SkillGet,
    SkillList,
    SkillSave,
    ThinkingSet,
    TurnInterrupt,
    TurnSubmit,
    TurnUnqueue,
    WatchDrop,
    WatchMove,
    WatchPeek,
)
from hpca.sessions import SessionStore
from hpca.skills import (
    Skill,
    load_own_skills,
    load_project_skills,
    load_skills,
    write_skill,
)
from hpca.transcript import RESULT_RULE
from hpca.ui.rain import KATAKANA as SPINNER_GLYPHS
from hpca.watches import KIND_JOB, KIND_LOG, WatchStore
from tests.ui_harness import connected


def respond(text="done"):
    return json.dumps({"action": "respond", "response": text})


class FakeLLM:
    """Answers with whatever was queued, and records what it was asked."""

    def __init__(self, outputs=None):
        self._outputs = list(outputs or [respond()])
        self.prompts: list[list[dict]] = []
        self.closed = False

    async def chat(self, messages, *, json_schema=None, **kwargs):
        self.prompts.append(messages)
        answer = self._outputs.pop(0) if self._outputs else respond()
        # A queued answer may be a whole `ChatResponse` when the test cares
        # about what rides alongside the content — reasoning, token counts.
        if isinstance(answer, ChatResponse):
            return answer
        return ChatResponse(content=answer)

    async def supports_constrained_decoding(self):
        return True

    async def close(self):
        self.closed = True


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def project(tmp_path):
    """Where project-level skills live for these tests.

    Pinned rather than left to default to the working directory, which under
    pytest is the repository: a `/skill-creator` test writing a project skill
    would otherwise create `.hpca/skills` in the checkout.
    """
    root = tmp_path / "project"
    root.mkdir()
    return root


@pytest.fixture
def conn(home):
    connection = connect(home / "hpca.db")
    init_db(connection)
    return connection


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def saver():
    """The checkpointer, out where a test can hand it to a second core.

    Which is what "restarted" means here: the same stores and the same
    threads, a new process. Only that can show what a core keeps in memory
    versus what it reads back — the difference automatic titling turns on.
    """
    return InMemorySaver()


@pytest.fixture
def service(home, conn, llm, saver, project):
    async def db(fn):
        return fn(conn)

    built = build_service(
        settings=Settings.load(),
        app_dir=home,
        db=db,
        conn=conn,
        checkpointer=saver,
        llm=llm,
        project_root=project,
    )
    return built


class FakeSlurm:
    """Enough of a cluster to answer a cancel. Records what was asked."""

    def __init__(self, error: Exception | None = None) -> None:
        self.cancelled: list[str] = []
        self._error = error

    async def cancel(self, job_id: str) -> None:
        if self._error is not None:
            raise self._error
        self.cancelled.append(job_id)


@pytest.fixture
def slurm():
    return FakeSlurm()


@pytest.fixture
def cluster_service(home, conn, llm, slurm, project):
    """The same runtime, with a cluster behind it — `job.cancel` needs one."""

    async def db(fn):
        return fn(conn)

    return build_service(
        settings=Settings.load(),
        app_dir=home,
        db=db,
        conn=conn,
        checkpointer=InMemorySaver(),
        llm=llm,
        slurm=slurm,
        project_root=project,
    )


@pytest.fixture
def session(conn):
    return SessionStore(conn).create(profile="default", title="a session")


def subscribe(service):
    """A subscriber's queue, wound past the frames every client is handed.

    `hello` is the first one, and a parked decision follows it; both are
    handshake, not news, so the tests that assert what a *command* produced
    start after them. `TestHandshake` is where they are asserted directly.
    """
    queue = service.subscribe()
    while not queue.empty():
        queue.get_nowait()
    return queue


async def drain(queue):
    """Everything emitted so far, without waiting for more."""
    events = []
    while not queue.empty():
        events.append(queue.get_nowait())
    return events


def kinds(events):
    return [type(e).__name__ for e in events]


async def relay(queue, wire):
    """Everything the core has said so far, played into a real front-end.

    The one test vehicle in this file that is not an assertion about events:
    the core's frames go over a real connection to a real `UIClient`, in the
    order the core emitted them, so what is asserted afterwards is the screen
    a user would be looking at.
    """
    while not queue.empty():
        await wire.tell(queue.get_nowait())


def spinner_lines(wire) -> list[str]:
    """The working rows on screen — one while a turn runs, none after."""
    return [line for line in wire.frame() if any(f in line for f in SPINNER_GLYPHS)]


class TestAssembly:
    def test_it_builds_without_a_terminal(self, service):
        assert service is not None

    async def test_a_subscriber_receives_what_the_core_says(self, service, session):
        queue = subscribe(service)
        await service.handle(TurnSubmit(session_id=session.session_id, text="hi"))
        events = await drain(queue)
        assert "TurnStarted" in kinds(events)

    async def test_two_subscribers_both_see_it(self, service, session):
        first, second = subscribe(service), subscribe(service)
        await service.handle(TurnSubmit(session_id=session.session_id, text="hi"))
        assert kinds(await drain(first)) == kinds(await drain(second))

    async def test_an_unsubscribed_queue_stops_receiving(self, service, session):
        queue = subscribe(service)
        service.unsubscribe(queue)
        await service.handle(TurnSubmit(session_id=session.session_id, text="hi"))
        assert await drain(queue) == []


class TestHandshake:
    """What a client is told the moment it subscribes, before it asks anything.

    Two things, and both are state a front-end cannot work out for itself: who
    the core is (§4.2 `hello`), and any decision already parked and waiting for
    an answer (§4.4). The second is the fix for a real latent bug — a parked
    decision used to be UI-process memory, so a restart left a session stuck on
    an interrupt with nothing on screen to answer it.
    """

    async def test_hello_is_the_first_frame(self, service):
        queue = service.subscribe()
        first = queue.get_nowait()
        assert type(first).__name__ == "Hello"
        assert first.version == PROTOCOL_VERSION
        assert first.profile == "default"
        # A digest rather than the settings themselves: a front-end must be
        # able to notice they changed without being handed the api keys.
        assert first.settings_digest

    async def test_hello_carries_the_settings_the_ui_draws_with(self, service):
        # The carve-out the digest implies: a front-end may not read the file,
        # and a digest cannot answer "draw this how". On `hello` because that
        # is the frame that lands before anything has been drawn — a setting
        # arriving later would be a redraw the user sees.
        service._deps.settings.display.chat_stamps = False
        service._deps.settings.display.decision_pulse_seconds = 4.0
        first = service.subscribe().get_nowait()
        assert first.display.chat_stamps is False
        assert first.display.decision_pulse_seconds == 4.0

    async def test_and_carries_nothing_else_of_the_settings(self, service):
        # Not the tree: it holds api keys and endpoint addresses, and a
        # process that only needs to know whether to print a timestamp has no
        # business being handed them. Adding a display key is a deliberate
        # edit here, exactly as dropping one would be.
        first = service.subscribe().get_nowait()
        assert set(type(first.display).model_fields) == {
            "chat_stamps",
            "decision_pulse_seconds",
            "focus_flash_seconds",
            "palette",
            "quit_rain",
            "quit_rain_fps",
            "spacer_lines",
        }
        # And the palette is colours and nothing else — no paths, no names, no
        # anything a front-end could be handed by calling it a display key.
        assert set(type(first.display.palette).model_fields) == {
            "agent",
            "chrome",
            "danger",
            "faint",
            "flash",
            "muted",
            "ok",
            "spinner",
            "user",
            "warn",
        }

    async def test_the_digest_follows_the_settings(self, home, conn, llm):
        async def db(fn):
            return fn(conn)

        def greeting(settings):
            built = build_service(
                settings=settings,
                app_dir=home,
                db=db,
                conn=conn,
                checkpointer=InMemorySaver(),
                llm=llm,
            )
            return built.subscribe().get_nowait()

        settings = Settings.load()
        before = greeting(settings)
        settings.agent.default_mode = "auto"
        assert greeting(settings).settings_digest != before.settings_digest

    async def test_the_handshake_goes_only_to_the_arriving_client(self, service):
        established = subscribe(service)
        service.subscribe()  # a second client arrives
        # The greeting is that client's, not a broadcast: a reconnect must not
        # make every other front-end re-run its version check.
        assert await drain(established) == []

    async def test_a_parked_decision_is_re_emitted_to_a_new_client(
        self, service, session
    ):
        service._scheduler._decisions[session.session_id] = {"tool": "run_bash"}
        events = await drain(service.subscribe())
        parked = only(events, "DecisionRequested")
        assert parked.session_id == session.session_id
        assert parked.payload == {"tool": "run_bash"}
        # After the greeting: a client runs its version check first.
        assert kinds(events).index("Hello") < kinds(events).index(
            "DecisionRequested"
        )

    async def test_a_client_arriving_with_nothing_parked_gets_only_hello(
        self, service, session
    ):
        assert kinds(await drain(service.subscribe())) == ["Hello"]


class TestTurns:
    async def test_a_submitted_turn_reaches_the_model_and_comes_back(
        self, service, session, llm
    ):
        queue = subscribe(service)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="which BAMs?")
        )
        # Waited for by condition, not by counting loop turns: a finished turn
        # goes through the session log and the episodic index on their own
        # threads before it is announced, and no number of `sleep(0)`s is a
        # thread hop. See `wait_for`.
        await wait_for(queue, "TurnFinished")
        assert llm.prompts, "the model was never asked"

    async def test_the_prompt_carries_the_sessions_own_profile(
        self, service, conn, llm
    ):
        """A profile's memories reach the system message of its own turns.

        Every piece of this is tested in isolation — `system_prompt_text` in
        `test_profiles.py`, `orchestrator_system_prompt` in `test_prompts.py`
        — and the join in `service.system_prompt_for` was not: this test used
        to assert that the content was a non-empty string, which is true of a
        prompt with no memories in it at all (specs-ui-coverage.md §3.14 row
        2). If the memories stopped arriving, nothing on screen would change
        and nothing here would fail.

        Two profiles, because "its own" is the claim: a turn in a background
        session must be rendered for *that* session's profile rather than for
        whichever one the core happens to be working under.
        """
        working = Profile.load("default")
        working.add_memory(
            "the working profile's own note", scope=MemoryScope.SYSTEM_PROMPT
        )
        working.save()
        theirs = Profile.create("bioinformatics")
        theirs.add_memory(
            "BAMs live on /scratch/cohort", scope=MemoryScope.SYSTEM_PROMPT
        )
        theirs.save()
        session = SessionStore(conn).create(
            profile="bioinformatics", title="somebody else's"
        )

        await service.handle(TurnSubmit(session_id=session.session_id, text="hello"))
        await _settle()

        system = llm.prompts[0][0]
        assert system["role"] == "system"
        assert "BAMs live on /scratch/cohort" in system["content"]
        assert "the working profile's own note" not in system["content"]

    async def test_the_user_message_reaches_the_model_with_its_sidecar(
        self, service, session, llm
    ):
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="count the reads")
        )
        await _settle()
        last = llm.prompts[0][-1]
        assert "count the reads" in (last.get("api_content") or last["content"])

    async def test_a_request_that_resembles_a_past_struggle_warns_first(
        self, service, session, llm
    ):
        """§4.4, and the fourth thing whose only caller was `tui/app.py`.

        The model gets the same notes as a memory-context block either way
        (`recall_lines`); this is the half addressed to the *user*, and it is
        worth saying before the turn rather than after it fails again.
        """
        profile = Profile.load(session.profile)
        profile.add_memory(
            "STAR ran out of memory\nkeywords: star",
            scope=MemoryScope.RAG,
            kind=STRUGGLE_KIND,
        )
        profile.save()
        service._memory.invalidate(session.profile)
        queue = subscribe(service)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="run star again")
        )
        warnings = [
            e
            for e in await drain(queue)
            if type(e).__name__ == "Notify" and e.severity == "warning"
        ]
        assert any("struggled with this before" in w.text for w in warnings)

    async def test_and_an_unremarkable_one_does_not(self, service, session, llm):
        queue = subscribe(service)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="run star again")
        )
        assert "Notify" not in kinds(await drain(queue))

    async def test_submitting_to_a_session_that_does_not_exist_says_so(self, service):
        queue = subscribe(service)
        await service.handle(TurnSubmit(session_id="nope", text="hi"))
        events = await drain(queue)
        assert kinds(events) == ["Notify"]
        assert events[0].severity == "warning"

    async def test_interrupting_nothing_is_harmless(self, service, session):
        await service.handle(TurnInterrupt(session_id=session.session_id))

    async def test_stopping_a_turn_re_states_the_chat_it_leaves_behind(
        self, service, session, llm
    ):
        # A stopped turn keeps its work, but the row for the call it died
        # inside is on screen waiting for a result that is never coming, and a
        # delta cannot un-draw a row. The reset that follows is the same case
        # a rollback is — an open of what the session holds now, which after
        # this change is the message and everything under it.
        release = await park_turn(service, llm, session.session_id)
        queue = subscribe(service)
        await service.handle(TurnInterrupt(session_id=session.session_id))
        entries = only(await drain(queue), "ChatReset").entries
        assert [e.kind for e in entries] == ["user", "event"]
        assert entries[0].text == "running"
        # And the marker saying where it stopped, so the reader scrolling back
        # can tell a turn that was stopped from one that answered nothing —
        # in the short form, not the paragraph the model is handed.
        assert entries[1].text == "stopped by the user"
        release.set()
        await service.stop()

    async def test_and_the_message_is_not_handed_back(
        self, service, session, llm
    ):
        # It used to be: the turn was rolled out of the thread and the
        # sentence came back to be edited. Now the sentence is in the
        # conversation — see the reset above — and handing it to the entry box
        # as well would have the user send it twice.
        release = await park_turn(service, llm, session.session_id)
        queue = subscribe(service)
        await service.handle(TurnInterrupt(session_id=session.session_id))
        assert "TurnInterrupted" not in kinds(await drain(queue))
        release.set()
        await service.stop()

    async def test_unless_the_stop_beat_the_message_into_the_thread(
        self, service, session, monkeypatch
    ):
        # The window a turn is announced in before the graph has written
        # anything. There is no exchange to keep, so the sentence goes back to
        # the box it was typed in rather than being lost — the one case left
        # for `turn.interrupted`.
        import asyncio

        from hpca.core import scheduler as scheduler_module

        async def never_gets_there(graph, **kwargs):
            await asyncio.Event().wait()

        monkeypatch.setattr(scheduler_module, "run_turn", never_gets_there)
        queue = subscribe(service)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="a lost sentence")
        )
        # As far as a turn gets here: it knows where the thread stood, and
        # nothing has been added to it.
        while not service._scheduler.can_interrupt(session.session_id):
            await asyncio.sleep(0)
        await service.handle(TurnInterrupt(session_id=session.session_id))
        events = await drain(queue)
        handed_back = only(events, "TurnInterrupted")
        assert handed_back.text == "a lost sentence"
        # Addressed: the user may be looking at another session by now, and
        # the message waits in the one it was typed in, as a draft.
        assert handed_back.session_id == session.session_id
        # After the reset, which has just re-stated the chat it belonged to.
        assert kinds(events).index("ChatReset") < kinds(events).index(
            "TurnInterrupted"
        )
        await service.stop()

    async def test_nothing_is_handed_back_when_nothing_was_stopped(
        self, service, session
    ):
        # A stale gesture — the turn finished while the key was on its way —
        # must not put a message into an entry box the user is typing in.
        queue = subscribe(service)
        await service.handle(TurnInterrupt(session_id=session.session_id))
        assert "TurnInterrupted" not in kinds(await drain(queue))


class TestStoppingATurnInEveryPhase:
    """The phases an abort has to cover, driven through a real graph.

    A turn does not spend its time waiting on the model. It spends it in
    tools, and — when the model keeps producing output the middleware refuses
    — going round a loop with no exit of its own. Both are what a user reaches
    for the stop gesture in, and refusing there leaves them watching a spinner
    they cannot answer.
    """

    def slow_tool(self):
        import asyncio

        from pydantic import BaseModel

        from hpca.agent.tools import Tool, ToolRegistry

        class SlowParams(BaseModel):
            text: str = ""

        entered, release = asyncio.Event(), asyncio.Event()

        async def handler(args, ctx):
            entered.set()
            await release.wait()
            return "finished at last"

        registry = ToolRegistry()
        registry.register(
            Tool(
                name="slow_tool",
                description="Takes its time",
                params=SlowParams,
                handler=handler,
            )
        )
        return registry, entered, release

    def service_with(self, home, conn, llm, tools):
        async def db(fn):
            return fn(conn)

        return build_service(
            settings=Settings.load(),
            app_dir=home,
            db=db,
            conn=conn,
            checkpointer=InMemorySaver(),
            llm=llm,
            tools=tools,
        )

    async def test_a_turn_running_a_tool_can_be_stopped(
        self, home, conn, session
    ):
        import asyncio

        tools, entered, release = self.slow_tool()
        llm = FakeLLM([calling("slow_tool"), respond("done")])
        service = self.service_with(home, conn, llm, tools)
        queue = subscribe(service)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="run the thing")
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        # The phase a long script spends its minutes in.
        assert service._scheduler.can_interrupt(session.session_id) is True

        await service.handle(TurnInterrupt(session_id=session.session_id))
        events = await drain(queue)
        assert "TurnFinished" in kinds(events)
        assert not service._scheduler.is_busy(session.session_id)
        # And the message it was working on stays where it was said.
        entries = only(events, "ChatReset").entries
        assert [e.text for e in entries if e.kind == "user"] == ["run the thing"]
        release.set()
        await service.stop()

    async def test_it_breaks_the_decision_retry_loop(
        self, home, conn, session
    ):
        # The phase with no exit of its own: the model keeps producing output
        # the middleware rejects and the turn goes round feeding the rejection
        # back. Cancelling lands on the loop's own await, so the next attempt
        # is never made.
        import asyncio

        llm = FakeLLM()
        entered, release = asyncio.Event(), asyncio.Event()
        rounds = {"n": 0}

        async def never_valid(messages, *, json_schema=None, **kwargs):
            rounds["n"] += 1
            if rounds["n"] >= 2:  # genuinely round the loop once first
                entered.set()
                await release.wait()
            return ChatResponse(content="Let me think about that some more.")

        llm.chat = never_valid
        service = self.service_with(home, conn, llm, None)
        queue = subscribe(service)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="do the impossible")
        )
        await asyncio.wait_for(entered.wait(), timeout=5)

        await service.handle(TurnInterrupt(session_id=session.session_id))
        assert "TurnFinished" in kinds(await drain(queue))
        assert rounds["n"] == 2  # the loop stopped where it stood
        assert not release.is_set()
        assert not service._scheduler.is_busy(session.session_id)
        await service.stop()


class TestAStoppedTurnKeepsItsWork:
    """The stop that stops the agent instead of throwing the turn away.

    Driven through a real graph and a real checkpointer, because the whole
    question is what the *thread* looks like afterwards. A turn stopped
    mid-tool is stopped between a decision and the result that answers it,
    and the next turn runs on whatever that left: keeping the work is only
    worth anything if the conversation it leaves behind is one a model can
    still be handed.
    """

    def tools(self):
        """One tool that answers at once, and one that never comes back."""
        import asyncio

        from pydantic import BaseModel

        from hpca.agent.tools import Tool, ToolRegistry

        class Params(BaseModel):
            text: str = ""

        ran = {"counted": 0, "slow": 0}
        entered, release = asyncio.Event(), asyncio.Event()

        async def counter(args, ctx):
            ran["counted"] += 1
            return "42 files"

        async def slow(args, ctx):
            ran["slow"] += 1
            entered.set()
            await release.wait()
            return "finished at last"

        registry = ToolRegistry()
        registry.register(
            Tool(name="count_files", description="Counts", params=Params,
                 handler=counter)
        )
        registry.register(
            Tool(name="slow_tool", description="Takes its time", params=Params,
                 handler=slow)
        )
        return registry, ran, entered, release

    async def stopped_mid_tool(self, home, conn, session):
        """A turn that ran one tool, then was stopped inside the next."""
        import asyncio

        registry, ran, entered, release = self.tools()
        llm = FakeLLM([
            calling("count_files"),
            calling("slow_tool"),
            respond("never said"),
        ])

        async def db(fn):
            return fn(conn)

        service = build_service(
            settings=Settings.load(),
            app_dir=home,
            db=db,
            conn=conn,
            checkpointer=InMemorySaver(),
            llm=llm,
            tools=registry,
        )
        queue = subscribe(service)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="count the files")
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        await service.handle(TurnInterrupt(session_id=session.session_id))
        release.set()
        return service, queue, llm, ran

    async def messages_of(self, service, session):
        state = await service._graph.aget_state(
            {"configurable": {"thread_id": session.session_id}}
        )
        return list((state.values or {}).get("messages", []))

    async def test_the_steps_it_got_through_are_still_there(
        self, home, conn, session
    ):
        # What the user asked for: the tool round that finished cost real
        # cluster time and is very often the half they wanted to read.
        service, queue, _, _ = await self.stopped_mid_tool(home, conn, session)
        entries = only(await drain(queue), "ChatReset").entries
        assert [e.kind for e in entries] == ["user", "thinking", "event"]
        assert [(p.tool, p.done) for p in entries[1].parts] == [
            ("count_files", True)
        ]
        # The call it was stopped inside is not among them: it never reached
        # the thread, so nothing about it survives to draw.
        assert "slow_tool" not in entries[1].text
        await service.stop()

    async def test_and_the_thread_it_leaves_is_one_a_model_can_read(
        self, home, conn, session
    ):
        # The hazard of keeping the work: a history whose last word is a call
        # nobody answered is malformed for most backends, and the model that
        # is handed one either errors or issues the call again. It cannot
        # happen here — the call and its result are appended in one state
        # update — and this is what says so out loud.
        from hpca.agent.history import is_tool_call_message

        service, _, _, _ = await self.stopped_mid_tool(home, conn, session)
        messages = await self.messages_of(service, session)
        for position, message in enumerate(messages):
            if is_tool_call_message(message):
                assert position + 1 < len(messages), "a call with no result"
                assert messages[position + 1]["content"].startswith("[tool")
        # And it ends on the note, so the model is told what happened rather
        # than left to infer it from a conversation that stops in mid-air.
        assert messages[-1]["content"].startswith("[stopped]")
        await service.stop()

    async def test_and_the_next_turn_runs_without_repeating_the_stopped_call(
        self, home, conn, session
    ):
        # The proof the change is safe: without it this could corrupt every
        # following turn on the session. The tool the user stopped is the one
        # thing that must not quietly happen anyway.
        service, queue, llm, ran = await self.stopped_mid_tool(home, conn, session)
        await drain(queue)  # the stop's own frames, `turn.finished` included
        llm._outputs = [respond("understood, stopping there")]
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="never mind, thanks")
        )
        events = await wait_for(queue, "TurnFinished")
        assert only(events, "TurnFinished").reply == "understood, stopping there"
        assert ran == {"counted": 1, "slow": 1}
        # And the turn was answered against the whole conversation, note and
        # all — the history is the one the stop left, not a rolled-back one.
        assert [m["role"] for m in llm.prompts[-1]] == [
            "system", "user", "assistant", "user", "user", "user"
        ]
        await service.stop()


class TestAStoppedTurnLeavesTheScreen:
    """The core's own frames, played into the real UI on the other end.

    Everywhere else in this file the assertion is about the events; here it
    has to be about the screen, because the bug this holds down was invisible
    in them. Every frame the interrupt sent was correct and none of them said
    the turn was *over*, so the working row went on spinning "LLM processing…
    (enter or esc esc to interrupt)" for a turn that no longer existed — and
    offering to stop it again. Nothing but a real client applying the real
    stream in order can catch that.
    """

    async def stopped(self, service, session, llm, wire):
        """A turn parked inside the model, then stopped. Returns the gate."""
        queue = service.subscribe()
        await service.handle(SessionList())
        await service.handle(SessionOpen(session_id=session.session_id))
        release = await park_turn(service, llm, session.session_id)
        await relay(queue, wire)
        assert spinner_lines(wire), "the turn should be on screen to begin with"
        await service.handle(TurnInterrupt(session_id=session.session_id))
        await relay(queue, wire)
        return release

    async def test_the_working_row_goes_when_the_turn_is_stopped(
        self, service, session, llm
    ):
        async with connected() as wire:
            release = await self.stopped(service, session, llm, wire)
            assert not spinner_lines(wire)
            release.set()
            await service.stop()

    async def test_and_the_session_is_no_longer_busy(
        self, service, session, llm
    ):
        # `Turn.busy` is what draws the row and `Turn.interruptible` is what
        # offers to stop it; both stayed true forever, so the sidebar kept the
        # session's "⟳" and the gesture claimed a stop it could not make.
        async with connected() as wire:
            release = await self.stopped(service, session, llm, wire)
            turn = wire.ui.session_for(session.session_id).turn
            assert (turn.busy, turn.interruptible) == (False, False)
            release.set()
            await service.stop()


class TestTypeAhead:
    """The queued-message channel: type ahead, see it, take it back.

    In the Textual front-end the queue was UI state; here the scheduler owns
    it, so everything the user can see or do about it has to cross the wire.
    """

    async def park(self, service, llm):
        """Hold the model inside a turn so the next message has to queue."""
        import asyncio

        entered, release = asyncio.Event(), asyncio.Event()
        answer = llm.chat

        async def gated(messages, **kwargs):
            entered.set()
            await release.wait()
            return await answer(messages, **kwargs)

        llm.chat = gated
        return entered, release

    async def test_a_message_typed_during_a_turn_shows_up_as_queued(
        self, service, session, llm
    ):
        import asyncio

        entered, release = await self.park(service, llm)
        await service.handle(TurnSubmit(session_id=session.session_id, text="first"))
        await asyncio.wait_for(entered.wait(), timeout=5)
        queue = subscribe(service)
        await service.handle(TurnSubmit(session_id=session.session_id, text="second"))

        appended = [e for e in await drain(queue) if type(e).__name__ == "ChatAppend"]
        assert [(e.entry.kind, e.entry.text) for e in appended] == [
            ("queued", "second")
        ]
        # Named by the core, which is the only side that may name a row.
        assert appended[0].entry.seq >= 1
        release.set()
        await service.stop()

    async def test_a_message_that_starts_at_once_is_not_a_queued_row(
        self, service, session
    ):
        # Nothing is waiting, so the turn starts — and a "queued" row would
        # then have to be un-drawn a moment later. The message is still drawn,
        # as the ordinary `user` row it already is.
        queue = subscribe(service)
        await service.handle(TurnSubmit(session_id=session.session_id, text="hi"))
        appended = [e for e in await drain(queue) if type(e).__name__ == "ChatAppend"]
        assert [(e.entry.kind, e.entry.text) for e in appended] == [("user", "hi")]
        await service.stop()

    async def test_cancelling_hands_the_text_back(self, service, session, llm):
        import asyncio

        entered, release = await self.park(service, llm)
        await service.handle(TurnSubmit(session_id=session.session_id, text="first"))
        await asyncio.wait_for(entered.wait(), timeout=5)
        queue = subscribe(service)
        await service.handle(TurnSubmit(session_id=session.session_id, text="second"))
        row = [e for e in await drain(queue) if type(e).__name__ == "ChatAppend"][0]

        # The UI cancels the row it was given, not a position it counted.
        await service.handle(
            TurnUnqueue(session_id=session.session_id, seq=row.entry.seq)
        )
        events = await drain(queue)
        assert kinds(events) == ["TurnUnqueued"]
        assert (events[0].seq, events[0].text) == (row.entry.seq, "second")
        release.set()
        await service.stop()

    async def test_cancelling_a_row_that_already_started_says_so(
        self, service, session
    ):
        # The turn ahead finished while the dialog was open. Refused the way
        # every un-carry-out-able command is refused: a warning, no state
        # change, and nothing for the UI to guess at.
        queue = subscribe(service)
        await service.handle(TurnUnqueue(session_id=session.session_id, seq=1))
        events = await drain(queue)
        assert kinds(events) == ["Notify"] and events[0].severity == "warning"
        await service.stop()

    async def test_the_queued_row_is_promoted_rather_than_redrawn(
        self, service, session, llm
    ):
        # End to end: the row the user sees waiting is the row that becomes
        # the message that ran — one name, two frames, no rebuild.
        import asyncio

        entered, release = await self.park(service, llm)
        await service.handle(TurnSubmit(session_id=session.session_id, text="first"))
        await asyncio.wait_for(entered.wait(), timeout=5)
        queue = subscribe(service)
        await service.handle(TurnSubmit(session_id=session.session_id, text="second"))
        row = [e for e in await drain(queue) if type(e).__name__ == "ChatAppend"][0]

        release.set()
        seen: list = []
        for _ in range(60):
            seen += await drain(queue)
            # By its seq, not by position: the turn ahead re-states its own
            # rows as it finishes, so several updates cross in this window.
            promotion = [
                e
                for e in seen
                if type(e).__name__ == "ChatUpdate"
                and e.entry.seq == row.entry.seq
            ]
            if promotion:
                break
            await _yield()
        else:
            raise AssertionError("the queued row was never promoted")
        assert (promotion[0].entry.kind, promotion[0].entry.text) == (
            "user",
            "second",
        )
        # And no second row was drawn for it.
        assert not [
            e
            for e in seen
            if type(e).__name__ == "ChatAppend" and e.entry.text == "second"
        ]
        await service.stop()


def only(events, name):
    """The single event of that type, or an assertion naming what did arrive."""
    found = [e for e in events if type(e).__name__ == name]
    assert len(found) == 1, f"{name} in {kinds(events)}"
    return found[0]


async def wait_for(queue, name, *, seconds=10):
    """Everything emitted up to and including the first ``name`` event.

    Condition-waiting rather than counting yields: a turn crosses a thread on
    its way through the log and the stores, and the `-n auto` run is exactly
    where a fixed number of loop turns stops being enough.
    """
    import asyncio

    loop = asyncio.get_running_loop()
    deadline = loop.time() + seconds
    events = []
    while loop.time() < deadline:
        events += await drain(queue)
        if any(type(e).__name__ == name for e in events):
            return events
        await asyncio.sleep(0.005)
    raise AssertionError(f"no {name} arrived; saw {kinds(events)}")


async def run_turn(service, session_id, text="hi"):
    """One turn, start to finish, driven the way a front-end drives it."""
    import asyncio

    queue = subscribe(service)
    try:
        await service.handle(TurnSubmit(session_id=session_id, text=text))
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 10
        seen = []
        while loop.time() < deadline:
            seen += await drain(queue)
            names = kinds(seen)
            if "TurnFailed" in names:
                raise AssertionError(seen[names.index("TurnFailed")].error)
            if "TurnFinished" in names:
                return
            await asyncio.sleep(0.005)
        raise AssertionError(f"the turn never finished; saw {kinds(seen)}")
    finally:
        service.unsubscribe(queue)


def gate_llm(llm):
    """Hold the *next* model call open. Returns (entered, release).

    The turn is then sitting somewhere a test can act on it — which is what
    "the spinner is up and the user presses escape twice" looks like from
    here.
    """
    import asyncio

    entered, release = asyncio.Event(), asyncio.Event()
    answer = llm.chat

    async def gated(messages, **kwargs):
        llm.chat = answer  # only this one call is held
        entered.set()
        await release.wait()
        return await answer(messages, **kwargs)

    llm.chat = gated
    return entered, release


async def park_turn(service, llm, session_id):
    """Hold a turn inside the model, so the session is busy. Returns the gate."""
    import asyncio

    entered, release = asyncio.Event(), asyncio.Event()
    answer = llm.chat

    async def gated(messages, **kwargs):
        entered.set()
        await release.wait()
        return await answer(messages, **kwargs)

    llm.chat = gated
    await service.handle(TurnSubmit(session_id=session_id, text="running"))
    await asyncio.wait_for(entered.wait(), timeout=5)
    llm.chat = answer  # only the first call is held
    return release


class TestSessionList:
    async def test_every_session_comes_back_in_the_stores_order(
        self, service, conn, session
    ):
        other = SessionStore(conn).create(profile="default", title="the other one")
        queue = subscribe(service)
        await service.handle(SessionList())
        rows = only(await drain(queue), "SessionRows").rows
        # The store's order, not one the core invents: newest first is a
        # decision `SessionStore.list_all` already made.
        assert [r.session_id for r in rows] == [
            s.session_id for s in SessionStore(conn).list_all()
        ]
        assert {r.title for r in rows} == {"a session", "the other one"}
        assert other.session_id in {r.session_id for r in rows}

    async def test_a_row_names_the_model_its_session_is_pinned_to(
        self, service, conn
    ):
        backend = json.dumps(
            {"model": "gemma-3-27b", "base_url": "http://localhost:20001/v1"}
        )
        pinned = SessionStore(conn).create(
            profile="default", title="pinned", backend=backend
        )
        queue = subscribe(service)
        await service.handle(SessionList())
        rows = {r.session_id: r for r in only(await drain(queue), "SessionRows").rows}
        assert rows[pinned.session_id].model == "gemma-3-27b"

    async def test_a_bootstrap_session_names_no_model(self, service, session):
        queue = subscribe(service)
        await service.handle(SessionList())
        rows = only(await drain(queue), "SessionRows").rows
        # Not the app's default model: the row says what this conversation is
        # pinned to, and it is pinned to nothing.
        assert rows[0].model == ""

    async def test_a_row_says_when_its_session_was_last_worked_in(
        self, service, session, conn
    ):
        queue = subscribe(service)
        await service.handle(SessionList())
        rows = only(await drain(queue), "SessionRows").rows
        assert rows[0].last_active == SessionStore(conn).get(
            session.session_id
        ).last_active
        assert rows[0].last_active, "a session that exists has been active once"

    async def test_and_a_turn_moves_it(self, service, session, conn, llm):
        # Submitting is the point it moves, not the point the answer lands: a
        # five-minute turn must not leave its row looking untouched for its
        # whole length.
        store = SessionStore(conn)
        store.touch(session.session_id, "2000-01-01T00:00:00+00:00")
        await service.handle(TurnSubmit(session_id=session.session_id, text="hi"))
        assert store.get(session.session_id).last_active > "2001"

    async def test_a_running_turn_marks_its_row(self, service, session, llm):
        release = await park_turn(service, llm, session.session_id)
        queue = subscribe(service)
        await service.handle(SessionList())
        assert only(await drain(queue), "SessionRows").rows[0].flags == ["working"]
        release.set()
        await service.stop()

    async def test_a_parked_decision_marks_its_row(self, service, session):
        # The other state a user working elsewhere has to be able to see. Set
        # on the scheduler because that is where a parked decision lives now.
        service._scheduler._decisions[session.session_id] = {"tool": "run_bash"}
        queue = subscribe(service)
        await service.handle(SessionList())
        assert only(await drain(queue), "SessionRows").rows[0].flags == ["decision"]


class TestReorderingTheSidebar:
    """`session.move` — alt+↑/alt+↓ on a sidebar row.

    The order is a fact about the database, which a front-end may not read
    (§4.2 rule 2), so the arrangement is made here and comes back as the
    sidebar. That is what makes it survive a restart, and what stops the next
    frame putting a locally-shuffled row back where it was.
    """

    def two(self, conn):
        """`session` fixture aside, a second row to trade places with. Newer,
        so it starts above it."""
        return SessionStore(conn).create(profile="default", title="the other one")

    async def test_a_move_rearranges_the_store_and_re_states_the_sidebar(
        self, service, session, conn
    ):
        self.two(conn)
        queue = subscribe(service)
        await service.handle(SessionMove(session_id=session.session_id, delta=-1))
        rows = only(await drain(queue), "SessionRows").rows
        assert [r.title for r in rows] == ["a session", "the other one"]
        # And the store agrees, which is the half that outlives the process.
        assert [s.title for s in SessionStore(conn).list_all()] == [
            "a session",
            "the other one",
        ]

    async def test_and_down_again_puts_it_back(self, service, session, conn):
        self.two(conn)
        await service.handle(SessionMove(session_id=session.session_id, delta=-1))
        queue = subscribe(service)
        await service.handle(SessionMove(session_id=session.session_id, delta=+1))
        rows = only(await drain(queue), "SessionRows").rows
        assert [r.title for r in rows] == ["the other one", "a session"]

    async def test_a_row_at_the_end_still_gets_the_sidebar_back(
        self, service, session, conn
    ):
        # Nothing moved, and the list is sent anyway: the front-end is
        # entitled to have moved the row itself while it waited, and this
        # frame is what puts it back.
        other = self.two(conn)
        queue = subscribe(service)
        await service.handle(SessionMove(session_id=other.session_id, delta=-1))
        rows = only(await drain(queue), "SessionRows").rows
        assert [r.title for r in rows] == ["the other one", "a session"]

    async def test_moving_a_row_that_is_gone_says_nothing(self, service, session):
        # Unlike every other session command, which warns through `_known`:
        # this one changes the order two rows are drawn in, and a keypress
        # against a sidebar that has just lost a row is worth a repaint rather
        # than an interruption.
        queue = subscribe(service)
        await service.handle(SessionMove(session_id="no-such-session", delta=-1))
        events = await drain(queue)
        assert "Notify" not in kinds(events)
        assert [r.title for r in only(events, "SessionRows").rows] == ["a session"]


class TestSessionNew:
    async def test_a_new_session_is_announced_and_then_listed(self, service):
        queue = subscribe(service)
        await service.handle(SessionNew(profile="default"))
        events = await drain(queue)
        # created first: the UI has to open it, and a sidebar cannot say which
        # of its lines is the new one. The profiles follow, because a row
        # there counts the conversations filed under each profile.
        assert kinds(events) == ["SessionCreated", "SessionRows", "ProfileRows"]
        created = events[0].row
        assert created.session_id in {r.session_id for r in events[1].rows}

    async def test_it_is_created_under_the_profile_asked_for(self, service, conn):
        queue = subscribe(service)
        await service.handle(SessionNew(profile="bioinformatics"))
        created = only(await drain(queue), "SessionCreated").row
        assert created.profile == "bioinformatics"
        stored = SessionStore(conn).get(created.session_id)
        assert stored is not None and stored.profile == "bioinformatics"

    async def test_the_backend_it_asks_for_is_pinned_to_it(self, service, conn):
        # A label out of the catalog the core answered `llm.list` with — the
        # thing `protocol.SessionNew` always documented and never accepted.
        service._deps.settings.backends = [
            LLMBackend(model="qwen3-32b", base_url="http://localhost:20001/v1")
        ]
        queue = subscribe(service)
        await service.handle(SessionNew(profile="default", backend="qwen3-32b"))
        created = only(await drain(queue), "SessionCreated").row
        assert created.model == "qwen3-32b"
        # Stored as the entry's JSON, so the choice survives the catalog entry
        # being dropped later: the label names it, the blob remembers it.
        assert "qwen3-32b" in SessionStore(conn).get(created.session_id).backend

    async def test_a_label_nothing_answers_to_is_said_out_loud(self, service):
        queue = subscribe(service)
        await service.handle(SessionNew(profile="default", backend="gone-away"))
        events = await drain(queue)
        # Still a session — an Enter keypress that produces nothing at all is
        # worse — but never a silent one: the old code answered an
        # unrecognised value by quietly using the bootstrap client.
        warning = only(events, "Notify")
        assert warning.severity == "warning" and "gone-away" in warning.text
        assert only(events, "SessionCreated").row.model == ""

    async def test_naming_no_backend_is_not_an_error(self, service):
        queue = subscribe(service)
        await service.handle(SessionNew(profile="default"))
        assert "Notify" not in kinds(await drain(queue))


class TestReusingAnEmptySession:
    """"(new session)" twice must not leave two empty rows.

    Textual reused an untouched session and *retagged* it to the profile and
    backend just chosen. Neither half can live in a front-end: no command in
    §4.1 changes an existing session's profile, and "untouched" is a question
    about the checkpointed thread, which §4.2 rule 2 puts out of its reach.

    The distinction that makes it safe is the same one automatic titling
    needs: a session still carrying the placeholder title was named by
    nobody, and one the user renamed is a conversation they meant to keep.
    """

    async def test_an_untouched_focused_session_is_handed_back(
        self, service, conn
    ):
        empty = SessionStore(conn).create(profile="default")
        await service.handle(SessionFocus(session_id=empty.session_id))
        queue = subscribe(service)
        await service.handle(SessionNew(profile="default"))
        created = only(await drain(queue), "SessionCreated").row
        assert created.session_id == empty.session_id
        # And no second empty row was added.
        assert len(SessionStore(conn).list_all()) == 1

    async def test_it_is_retagged_to_the_profile_just_chosen(
        self, service, conn
    ):
        empty = SessionStore(conn).create(profile="default")
        await service.handle(SessionFocus(session_id=empty.session_id))
        queue = subscribe(service)
        await service.handle(SessionNew(profile="bioinformatics"))
        assert only(await drain(queue), "SessionCreated").row.profile == (
            "bioinformatics"
        )
        assert SessionStore(conn).get(empty.session_id).profile == "bioinformatics"

    async def test_it_is_retagged_to_the_backend_just_chosen(
        self, service, conn
    ):
        service._deps.settings.backends = [
            LLMBackend(model="qwen3-32b", base_url="http://localhost:20001/v1")
        ]
        empty = SessionStore(conn).create(profile="default")
        await service.handle(SessionFocus(session_id=empty.session_id))
        queue = subscribe(service)
        await service.handle(SessionNew(profile="default", backend="qwen3-32b"))
        assert only(await drain(queue), "SessionCreated").row.model == "qwen3-32b"

    async def test_choosing_no_backend_leaves_a_pin_alone(self, service, conn):
        # "The core decides" is not "unpin this session": the user may have
        # picked that backend a moment ago, in this very session.
        pinned = json.dumps({"model": "qwen3-32b", "base_url": "http://x/v1"})
        empty = SessionStore(conn).create(profile="default", backend=pinned)
        await service.handle(SessionFocus(session_id=empty.session_id))
        await service.handle(SessionNew(profile="default"))
        assert SessionStore(conn).get(empty.session_id).backend == pinned

    async def test_a_session_someone_named_is_never_reused(self, service, conn):
        # Empty, but named — so it is a conversation the user meant to keep,
        # and retagging it would move it to a profile they did not choose.
        named = SessionStore(conn).create(profile="default", title="BAM QC")
        await service.handle(SessionFocus(session_id=named.session_id))
        queue = subscribe(service)
        await service.handle(SessionNew(profile="bioinformatics"))
        created = only(await drain(queue), "SessionCreated").row
        assert created.session_id != named.session_id
        assert SessionStore(conn).get(named.session_id).profile == "default"

    async def test_a_session_with_a_conversation_in_it_is_never_reused(
        self, service, conn
    ):
        used = SessionStore(conn).create(profile="default")
        await run_turn(service, used.session_id, "how many reads?")
        await service.handle(SessionFocus(session_id=used.session_id))
        queue = subscribe(service)
        await service.handle(SessionNew(profile="default"))
        assert only(await drain(queue), "SessionCreated").row.session_id != (
            used.session_id
        )

    async def test_a_busy_session_is_never_reused(self, service, conn, llm):
        empty = SessionStore(conn).create(profile="default")
        release = await park_turn(service, llm, empty.session_id)
        await service.handle(SessionFocus(session_id=empty.session_id))
        queue = subscribe(service)
        await service.handle(SessionNew(profile="bioinformatics"))
        assert only(await drain(queue), "SessionCreated").row.session_id != (
            empty.session_id
        )
        release.set()
        await service.stop()

    async def test_with_nothing_focused_a_session_is_simply_made(
        self, service, conn
    ):
        SessionStore(conn).create(profile="default")  # empty, but not on screen
        queue = subscribe(service)
        await service.handle(SessionNew(profile="default"))
        assert len(SessionStore(conn).list_all()) == 2
        assert "SessionCreated" in kinds(await drain(queue))


class TestSessionOpen:
    async def test_opening_sends_the_whole_transcript_once(self, service, session):
        await run_turn(service, session.session_id, "which BAMs?")
        queue = subscribe(service)
        await service.handle(SessionOpen(session_id=session.session_id))
        reset = only(await drain(queue), "ChatReset")
        assert reset.session_id == session.session_id
        assert [e.kind for e in reset.entries][:1] == ["user"]
        assert "which BAMs?" in reset.entries[0].text

    async def test_every_row_arrives_named_from_one(self, service, session):
        await run_turn(service, session.session_id, "hello")
        queue = subscribe(service)
        await service.handle(SessionOpen(session_id=session.session_id))
        reset = only(await drain(queue), "ChatReset")
        # A reset re-bases the numbering; the UI drops the names it held.
        assert [e.seq for e in reset.entries] == list(
            range(1, len(reset.entries) + 1)
        )

    async def test_and_each_one_says_when_it_happened(self, service, session):
        # The end-to-end of it: `graph._append_messages` stamps the message,
        # `build_entries` reads the stamp onto the entry, and `wire_entry`
        # carries it across by field name. Four layers, and the only place
        # they can be checked against each other is here.
        await run_turn(service, session.session_id, "which BAMs?")
        queue = subscribe(service)
        await service.handle(SessionOpen(session_id=session.session_id))
        reset = only(await drain(queue), "ChatReset")
        spoken = [e for e in reset.entries if e.kind in ("user", "assistant")]
        assert spoken, "the turn produced nothing to check"
        assert all(e.at for e in spoken), [(e.kind, e.at) for e in reset.entries]
        # And a turn's working names no instant: it folds several messages.
        assert all(e.at == "" for e in reset.entries if e.kind == "thinking")

    async def test_an_empty_session_opens_empty(self, service, session):
        queue = subscribe(service)
        await service.handle(SessionOpen(session_id=session.session_id))
        assert only(await drain(queue), "ChatReset").entries == []

    async def test_opening_says_how_full_the_window_already_is(
        self, service, session
    ):
        await run_turn(service, session.session_id, "hello")
        queue = subscribe(service)
        await service.handle(SessionOpen(session_id=session.session_id))
        events = await drain(queue)
        # Nothing has been sent this run, so the fill is derived from the
        # stored history — the same number a restart would show.
        assert only(events, "ContextEstimate").session_id == session.session_id

    async def test_opening_a_session_that_is_gone_says_so(self, service):
        queue = subscribe(service)
        await service.handle(SessionOpen(session_id="nope"))
        events = await drain(queue)
        assert kinds(events) == ["Notify"] and events[0].severity == "warning"

    async def test_a_session_reopened_mid_turn_keeps_its_type_ahead(
        self, service, session, llm
    ):
        # The known gap of specs-ui-replacement.md §4.2: the queue is core
        # state now, so a reset that leaves it out deletes the user's
        # type-ahead from under them.
        release = await park_turn(service, llm, session.session_id)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="and the CRAMs?")
        )
        queue = subscribe(service)
        await service.handle(SessionOpen(session_id=session.session_id))
        reset = only(await drain(queue), "ChatReset")
        assert [(e.kind, e.text) for e in reset.entries][-1] == (
            "queued",
            "and the CRAMs?",
        )
        assert reset.entries[-1].seq > 0, "a queued row must stay addressable"
        release.set()
        await service.stop()

    async def test_the_re_drawn_queued_row_is_the_one_a_cancel_names(
        self, service, session, llm
    ):
        release = await park_turn(service, llm, session.session_id)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="and the CRAMs?")
        )
        queue = subscribe(service)
        await service.handle(SessionOpen(session_id=session.session_id))
        row = only(await drain(queue), "ChatReset").entries[-1]

        await service.handle(
            TurnUnqueue(session_id=session.session_id, seq=row.seq)
        )
        assert only(await drain(queue), "TurnUnqueued").text == "and the CRAMs?"
        release.set()
        await service.stop()


def apply_deltas(events, session_id):
    """The chat a front-end holds after applying what it was sent, in order.

    A renderer in three lines, and deliberately strict about the two rules the
    protocol asks of it: a reset replaces everything, and an update addresses a
    row by `Entry.seq` and nothing else. An update naming a row nobody has is
    raised rather than turned into a new row — a client that invented one would
    hide exactly the desynchronisation this is here to catch.
    """
    rows: list = []
    for event in events:
        name = type(event).__name__
        if getattr(event, "session_id", None) != session_id:
            continue
        if name == "ChatReset":
            rows = list(event.entries)
        elif name == "ChatAppend":
            rows.append(event.entry)
        elif name == "ChatUpdate":
            for position, row in enumerate(rows):
                if row.seq == event.entry.seq:
                    rows[position] = event.entry
                    break
            else:
                raise AssertionError(
                    f"chat.update for row {event.entry.seq}, which was never drawn"
                )
    return rows


def calling(tool: str, **arguments) -> str:
    """One tool_call decision, as the envelope protocol carries it."""
    return json.dumps(
        {"action": "tool_call", "tool": tool, "arguments": arguments}
    )


class TestTheTurnAsItHappens:
    """What a front-end with a session open sees while the turn runs.

    This is the property the whole protocol is for: the conversation arrives as
    deltas, and re-opening the session afterwards produces the *same* rows. The
    second half is the one that is easy to get wrong — a `chat.reset` folds a
    turn's reasoning and tool calls into one working box, so live rows shaped
    any other way make the screen rearrange itself the moment the user comes
    back, which is the bug the old UI paid for by rebuilding the whole log
    every turn.
    """

    @pytest.fixture
    def target(self, home):
        path = home / "reads.tsv"
        path.write_text("sample\tcount\na\t7\n")
        return path

    async def run_with_a_call(self, service, session, llm, target, *, reasoning=""):
        """One turn that reads a file and then answers. Returns its events."""
        llm._outputs = [
            ChatResponse(
                content=calling("read_file", path=str(target)),
                reasoning=reasoning,
            ),
            respond("seven reads"),
        ]
        queue = subscribe(service)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="how many reads?")
        )
        return await wait_for(queue, "TurnFinished"), queue

    async def test_the_call_is_a_row_before_the_result_exists(
        self, service, session, llm, target
    ):
        events, _ = await self.run_with_a_call(service, session, llm, target)
        drawn = [
            e
            for e in events
            if type(e).__name__ in ("ChatAppend", "ChatUpdate")
            and e.entry.parts
        ]
        first = drawn[0].entry
        # The call is on screen while it is still running: one part, named,
        # and with nothing in the result half yet.
        assert [(p.kind, p.tool, p.done) for p in first.parts] == [
            ("call", "read_file", False)
        ]
        await service.stop()

    async def test_the_result_lands_in_the_row_the_call_drew(
        self, service, session, llm, target
    ):
        events, _ = await self.run_with_a_call(service, session, llm, target)
        drawn = [
            e
            for e in events
            if type(e).__name__ in ("ChatAppend", "ChatUpdate")
            and e.entry.parts
        ]
        call_row = drawn[0].entry.seq
        filled = [
            e.entry
            for e in drawn
            if e.entry.seq == call_row
            and any(p.done and p.result for p in e.entry.parts)
        ]
        assert filled, "the result never reached the row the call drew"
        assert "sample" in filled[0].parts[0].result
        # And it never became a row of its own: one exchange, one row.
        assert [type(e).__name__ for e in drawn if e.entry.seq != call_row] == []
        await service.stop()

    async def test_the_reply_arrives_without_anyone_asking_for_a_snapshot(
        self, service, session, llm, target
    ):
        events, _ = await self.run_with_a_call(service, session, llm, target)
        rows = apply_deltas(events, session.session_id)
        assert [r.kind for r in rows] == ["user", "thinking", "assistant"]
        assert (rows[0].text, rows[-1].text) == ("how many reads?", "seven reads")
        # Nothing resent the transcript to achieve that (§4.2 property 1).
        assert "ChatReset" not in kinds(events)
        await service.stop()

    async def test_re_opening_afterwards_yields_the_rows_the_deltas_built(
        self, service, session, llm, target
    ):
        # The one that matters: what the user watched happen and what they get
        # back when they return have to be the same rows, seq for seq.
        events, queue = await self.run_with_a_call(
            service, session, llm, target, reasoning="the counts are in column 2"
        )
        live = apply_deltas(events, session.session_id)

        await service.handle(SessionOpen(session_id=session.session_id))
        reopened = only(await drain(queue), "ChatReset").entries
        assert reopened == live
        await service.stop()

    async def test_the_models_reasoning_is_in_the_working_box(
        self, service, session, llm, target
    ):
        # Reasoning reaches nobody through a callback — it is checkpointed
        # state — so it arrives with the fold at the end of the turn, in the
        # place the fold gives it: ahead of the call it led to.
        events, _ = await self.run_with_a_call(
            service, session, llm, target, reasoning="column 2 holds the counts"
        )
        working = apply_deltas(events, session.session_id)[1]
        assert [p.kind for p in working.parts] == ["reasoning", "call"]
        assert working.parts[0].text == "column 2 holds the counts"
        assert working.reasoning_chars == len("column 2 holds the counts")
        await service.stop()

    async def test_the_backends_token_count_is_announced(
        self, service, session, llm
    ):
        llm._outputs = [
            ChatResponse(content=respond("done"), usage={"prompt_tokens": 4321})
        ]
        queue = subscribe(service)
        await service.handle(TurnSubmit(session_id=session.session_id, text="hi"))
        events = await wait_for(queue, "TurnFinished")
        # The context meter's measured half; without it the bar only ever
        # shows the estimate a re-open derives from stored history.
        assert only(events, "TurnUsage").prompt_tokens == 4321
        await service.stop()

    async def test_re_opening_mid_turn_re_binds_the_rows_it_is_drawing(
        self, service, session, llm, target
    ):
        # A reset renumbers every row on screen, including the working box a
        # turn is still filling. If the turn kept the old names, the update
        # carrying its result would address a row the client no longer has —
        # which `apply_deltas` refuses to invent, so this fails loudly.
        import asyncio

        entered, release = asyncio.Event(), asyncio.Event()
        answer, rounds = llm.chat, {"n": 0}

        async def gated(messages, **kwargs):
            rounds["n"] += 1
            if rounds["n"] == 2:  # the tool exchange is checkpointed by now
                entered.set()
                await release.wait()
            return await answer(messages, **kwargs)

        llm._outputs = [calling("read_file", path=str(target)), respond("seven")]
        llm.chat = gated
        queue = subscribe(service)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="how many reads?")
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        await service.handle(SessionOpen(session_id=session.session_id))
        release.set()

        live = apply_deltas(await wait_for(queue, "TurnFinished"), session.session_id)
        await service.handle(SessionOpen(session_id=session.session_id))
        assert only(await drain(queue), "ChatReset").entries == live
        await service.stop()

    async def test_a_turn_in_a_session_nobody_is_looking_at_still_says_so(
        self, service, conn, session, llm, target
    ):
        # The core does not know what is on screen and must not decide from it
        # (§4.2 rule 3): the event names its session and the UI drops what it
        # is not showing.
        other = SessionStore(conn).create(profile="default", title="the other one")
        await service.handle(SessionFocus(session_id=other.session_id))
        events, _ = await self.run_with_a_call(service, session, llm, target)
        assert [r.kind for r in apply_deltas(events, session.session_id)] == [
            "user",
            "thinking",
            "assistant",
        ]
        await service.stop()


class TestAParkedApproval:
    """A turn that stops to ask, driven through a real graph interrupt.

    The scheduler's own tests assert the bookkeeping against a fake; this one
    exists because the park is where the row reconciliation is hardest. A
    parked turn and its resume are *one* exchange as far as `build_entries` is
    concerned — one working box, holding the call, the answer to it, and
    whatever came after — so the two graph invocations have to draw one box
    between them or a re-open rearranges the screen.
    """

    async def park(self, service, session, llm):
        """Ask for something manual mode will not run unasked."""
        llm._outputs = [
            calling("run_bash", content_lines=["rm -rf /scratch/old"]),
            respond("left it alone"),
        ]
        queue = subscribe(service)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="clear the scratch dir")
        )
        return queue, await wait_for(queue, "DecisionRequested")

    async def test_the_graph_parking_asks(self, service, session, llm):
        _, events = await self.park(service, session, llm)
        asked = only(events, "DecisionRequested")
        assert asked.session_id == session.session_id
        assert asked.payload["tool"] == "run_bash"
        # Manual mode gates execution as well as destruction (§3.5).
        assert asked.payload["kind"] in ("execution", "destructive")
        # Nothing says the turn finished: it cannot, until this is answered.
        assert "TurnFinished" not in kinds(events)
        await service.stop()

    async def test_answering_clears_the_prompt_and_finishes_the_turn(
        self, service, session, llm
    ):
        queue, events = await self.park(service, session, llm)
        await service.handle(
            DecisionResolve(
                session_id=session.session_id,
                approved=False,
                reason="that is the real data",
            )
        )
        events += await wait_for(queue, "TurnFinished")
        assert "DecisionCleared" in kinds(events)
        assert only(events, "TurnFinished").reply == "left it alone"
        await service.stop()

    async def test_the_refusal_lands_in_the_row_that_asked_for_it(
        self, service, session, llm
    ):
        queue, events = await self.park(service, session, llm)
        await service.handle(
            DecisionResolve(session_id=session.session_id, approved=False)
        )
        events += await wait_for(queue, "TurnFinished")
        rows = apply_deltas(events, session.session_id)
        # One working box across both halves of the turn, not two.
        assert [r.kind for r in rows] == ["user", "thinking", "assistant"]
        assert [(p.kind, p.tool, p.failed) for p in rows[1].parts] == [
            ("call", "run_bash", True)
        ]

        await service.handle(SessionOpen(session_id=session.session_id))
        assert only(await drain(queue), "ChatReset").entries == rows
        await service.stop()

    async def test_it_does_not_stall_another_sessions_queue(
        self, service, conn, session, llm
    ):
        # A parked thread cannot move until it is answered — but only its own.
        # The rule the scheduler is written around, asserted here through a
        # real interrupt rather than a fake result.
        queue, _ = await self.park(service, session, llm)
        other = SessionStore(conn).create(profile="default", title="the other one")
        await service.handle(TurnSubmit(session_id=other.session_id, text="hi"))

        finished = only(await wait_for(queue, "TurnFinished"), "TurnFinished")
        assert finished.session_id == other.session_id
        # And the parked one is still parked — its own queue is the only one
        # the unanswered question holds up.
        assert session.session_id in service._scheduler.pending_decisions()
        await service.stop()

    async def test_a_rollback_is_refused_while_it_waits(
        self, service, session, llm
    ):
        # The resume would land on message indices the cut had removed. The
        # answer is the reason, and it reaches the user as a warning.
        queue, _ = await self.park(service, session, llm)
        await service.handle(
            SessionRollback(session_id=session.session_id, index=0)
        )
        events = await drain(queue)
        assert "ChatReset" not in kinds(events)
        assert "decision" in only(events, "Notify").text
        await service.stop()

    async def test_the_resumed_turn_is_still_stoppable(
        self, service, session, llm
    ):
        # An approval splits one exchange into two turns. The second carries
        # no user message of its own, and without the anchor it is a spinner
        # nothing can answer — the case `tui/app.py:_interrupt_anchor` exists
        # for, and the acceptance list's "after an approval, the resumed turn
        # carries the same interrupt anchor".
        import asyncio

        queue, _ = await self.park(service, session, llm)
        entered, release = gate_llm(llm)
        await service.handle(
            DecisionResolve(session_id=session.session_id, approved=False)
        )
        await asyncio.wait_for(entered.wait(), timeout=5)
        assert service._scheduler.can_interrupt(session.session_id) is True

        await service.handle(TurnInterrupt(session_id=session.session_id))
        events = await drain(queue)
        assert "TurnFinished" in kinds(events)
        # And the whole exchange stays, the refused call included: the half
        # the user answered for is the last thing to throw away.
        entries = only(events, "ChatReset").entries
        assert [e.kind for e in entries] == ["user", "thinking", "event"]
        assert entries[0].text == "clear the scratch dir"
        assert [(p.tool, p.done) for p in entries[1].parts] == [("run_bash", True)]
        release.set()
        await service.stop()

    async def test_the_anchor_goes_when_the_exchange_does(
        self, service, session, llm
    ):
        queue, _ = await self.park(service, session, llm)
        assert session.session_id in service._scheduler._anchors
        await service.handle(
            DecisionResolve(session_id=session.session_id, approved=False)
        )
        await wait_for(queue, "TurnFinished")
        # Answered for good: the next turn must not be handed this one's
        # message, and there is nothing left of it to roll back to.
        assert service._scheduler._anchors == {}
        await service.stop()


class TestAResumeAfterACoreRestart:
    """The half of an approval that outlives the process that asked.

    The thread is parked at `interrupt()` in the checkpointer, so the question
    survives; the turn that asked it does not, and with it goes the record of
    which chat rows are that exchange's. Without something to bind to, the
    resume can only stay silent and let the next `chat.reset` show the reply
    — the gap specs-ui-replacement.md §4.2 records against M5.
    """

    def build(self, home, conn, checkpointer, llm):
        async def db(fn):
            return fn(conn)

        return build_service(
            settings=Settings.load(),
            app_dir=home,
            db=db,
            conn=conn,
            checkpointer=checkpointer,
            llm=llm,
        )

    async def test_the_answer_lands_in_the_rows_already_on_screen(
        self, home, conn, session
    ):
        checkpointer = InMemorySaver()
        first = self.build(home, conn, checkpointer, FakeLLM(
            [calling("run_bash", content_lines=["rm -rf /scratch/old"])]
        ))
        queue = subscribe(first)
        await first.handle(
            TurnSubmit(session_id=session.session_id, text="clear the scratch dir")
        )
        parked = only(
            await wait_for(queue, "DecisionRequested"), "DecisionRequested"
        )
        await first.stop()

        # A new core over the same store and the same checkpointer, holding
        # the question the old one was parked on.
        second = self.build(
            home, conn, checkpointer, FakeLLM([respond("left it alone")])
        )
        second._scheduler._decisions[session.session_id] = dict(parked.payload)
        second._scheduler._awaiting_approval.add(session.session_id)
        queue = subscribe(second)

        # The user opens the session to answer the prompt, which is what puts
        # the parked exchange on screen — and names its rows.
        await second.handle(SessionOpen(session_id=session.session_id))
        reset = only(await drain(queue), "ChatReset")
        # The message alone: a parked call is announced only once it is
        # answered for, so the thread holds nothing else yet.
        assert [e.kind for e in reset.entries] == ["user"]

        await second.handle(
            DecisionResolve(session_id=session.session_id, approved=False)
        )
        events = await wait_for(queue, "TurnFinished")
        rows = apply_deltas([reset] + events, session.session_id)
        # One working box across both cores, and the reply after it — not a
        # second box, and not silence until the next open.
        assert [r.kind for r in rows] == ["user", "thinking", "assistant"]
        assert rows[-1].text == "left it alone"
        assert [(p.tool, p.failed) for p in rows[1].parts] == [("run_bash", True)]

        await second.handle(SessionOpen(session_id=session.session_id))
        assert only(await drain(queue), "ChatReset").entries == rows
        await second.stop()


class TestAFailedTurn:
    """A turn that breaks instead of finishing.

    The rows it drew have no result to fold, so without reconciling them they
    settle exactly as the live path left them — and a re-opened session would
    then draw something else.
    """

    @pytest.fixture
    def target(self, home):
        path = home / "reads.tsv"
        path.write_text("sample\tcount\na\t7\n")
        return path

    async def failing(self, service, session, llm, target):
        """One round of real work, and then the backend goes away — a tunnel
        dropped mid-turn, which is what this looks like on a cluster."""
        llm._outputs = [
            ChatResponse(
                content=calling("read_file", path=str(target)),
                reasoning="column 2 holds the counts",
            )
        ]
        answer, rounds = llm.chat, {"n": 0}

        async def then_dies(messages, **kwargs):
            rounds["n"] += 1
            if rounds["n"] > 1:
                raise ConnectionError("connection refused")
            return await answer(messages, **kwargs)

        llm.chat = then_dies
        queue = subscribe(service)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="how many reads?")
        )
        return queue, await wait_for(queue, "TurnFailed")

    async def test_the_failure_is_reported_not_raised(
        self, service, session, llm, target
    ):
        _, events = await self.failing(service, session, llm, target)
        # What a front-end draws its error entry from (§3.2), carrying enough
        # to say what went wrong.
        assert "connection refused" in only(events, "TurnFailed").error
        # And the spinner goes: an empty activity is how a renderer drops it.
        assert any(
            type(e).__name__ == "TurnActivity" and e.activity == ""
            for e in events
        )
        await service.stop()

    async def test_the_rows_it_drew_are_re_stated_from_the_thread(
        self, service, session, llm, target
    ):
        # The working box is the fold's copy, not the half-drawn live one: the
        # reasoning that only ever existed in the checkpoint is in it, exactly
        # where a re-opened session puts it.
        _, events = await self.failing(service, session, llm, target)
        rows = apply_deltas(events, session.session_id)
        assert [r.kind for r in rows] == ["user", "thinking"]
        assert [p.kind for p in rows[1].parts] == ["reasoning", "call"]
        assert rows[1].parts[0].text == "column 2 holds the counts"
        assert rows[1].parts[1].done  # and its result, not a call still going
        await service.stop()

    async def test_the_session_is_free_again(
        self, service, session, llm, target
    ):
        await self.failing(service, session, llm, target)
        assert not service._scheduler.is_busy(session.session_id)
        assert service._scheduler.rewind_blocker(session.session_id) is None
        await service.stop()

    async def test_re_opening_yields_the_rows_the_deltas_left(
        self, service, session, llm, target
    ):
        # The property a failure must not be allowed to break: what the user
        # is looking at and what they get back on re-open are the same rows,
        # seq for seq. Unreconciled they are not — the box still holds a call
        # with no result, and the fold's reasoning is missing from it.
        queue, events = await self.failing(service, session, llm, target)
        live = apply_deltas(events, session.session_id)
        await service.handle(SessionOpen(session_id=session.session_id))
        assert only(await drain(queue), "ChatReset").entries == live
        await service.stop()


class TestSessionClose:
    async def test_closing_forgets_what_was_open(self, service, session):
        await service.handle(SessionFocus(session_id=session.session_id))
        queue = subscribe(service)
        await service.handle(SessionClose())
        assert service._deps.focused_session_id is None
        events = await drain(queue)
        # The panel is session-scoped, so it empties with the session, and the
        # sidebar restates itself without a working row.
        assert "SessionRows" in kinds(events) and "PanelUpdate" in kinds(events)

    async def test_closing_nothing_is_harmless(self, service):
        await service.handle(SessionClose())
        assert service._deps.focused_session_id is None


class TestSessionRename:
    async def test_a_rename_lands_and_is_listed(self, service, session, conn):
        queue = subscribe(service)
        await service.handle(
            SessionRename(session_id=session.session_id, title="BAM QC")
        )
        rows = only(await drain(queue), "SessionRows").rows
        assert rows[0].title == "BAM QC"
        assert SessionStore(conn).get(session.session_id).title == "BAM QC"

    async def test_an_empty_title_is_refused(self, service, session, conn):
        queue = subscribe(service)
        await service.handle(
            SessionRename(session_id=session.session_id, title="   ")
        )
        events = await drain(queue)
        assert kinds(events) == ["Notify"] and events[0].severity == "warning"
        assert SessionStore(conn).get(session.session_id).title == "a session"

    async def test_renaming_a_session_that_is_gone_says_so(self, service):
        queue = subscribe(service)
        await service.handle(SessionRename(session_id="nope", title="x"))
        assert (await drain(queue))[0].severity == "warning"


class TestSessionRetitle:
    async def test_the_model_names_the_conversation(self, service, session, llm):
        await run_turn(service, session.session_id, "how many reads?")
        llm._outputs = [json.dumps({"title": "Read counting"})]
        queue = subscribe(service)
        await service.handle(SessionRetitle(session_id=session.session_id))
        events = await wait_for(queue, "SessionRows")
        assert only(events, "SessionRows").rows[0].title == "Read counting"

    async def test_it_does_not_hold_up_the_next_command(
        self, service, session, llm
    ):
        # A generation takes seconds; a dispatch that awaited it would stall
        # every command behind it on the same socket.
        await run_turn(service, session.session_id, "hello")
        import asyncio

        release = asyncio.Event()
        answer = llm.chat

        async def gated(messages, **kwargs):
            await release.wait()
            return await answer(messages, **kwargs)

        llm.chat = gated
        await service.handle(SessionRetitle(session_id=session.session_id))
        queue = subscribe(service)
        await service.handle(SessionList())  # answered while titling waits
        assert "SessionRows" in kinds(await drain(queue))
        release.set()
        await service.stop()

    async def test_an_empty_conversation_has_nothing_to_summarize(
        self, service, session
    ):
        queue = subscribe(service)
        await service.handle(SessionRetitle(session_id=session.session_id))
        events = await wait_for(queue, "Notify")
        assert kinds(events) == ["Notify"] and events[0].severity == "warning"

    async def test_a_model_that_cannot_write_one_is_reported_not_raised(
        self, service, session, llm
    ):
        await run_turn(service, session.session_id, "hello")
        llm._outputs = []  # every attempt answers with the turn JSON instead
        queue = subscribe(service)
        await service.handle(SessionRetitle(session_id=session.session_id))
        events = await wait_for(queue, "Notify")
        assert any(
            type(e).__name__ == "Notify" and e.severity == "error" for e in events
        )
        assert "SessionRows" not in kinds(events)

    async def test_it_says_the_session_is_busy_writing_a_title(
        self, service, session, llm
    ):
        # A silent backend call is indistinguishable from a core that has
        # stopped answering. The same event a turn reports with, because what
        # the user needs to know is that the session is busy and since when.
        await run_turn(service, session.session_id, "hello")
        llm._outputs = [json.dumps({"title": "Read counting"})]
        queue = subscribe(service)
        await service.handle(SessionRetitle(session_id=session.session_id))
        events = await wait_for(queue, "SessionRows")
        activity = [e for e in events if type(e).__name__ == "TurnActivity"]
        assert [a.activity for a in activity] == ["writing a title", ""]
        assert activity[0].started_at

    async def test_the_new_name_lands_in_the_session_log(
        self, service, session, llm, home
    ):
        await run_turn(service, session.session_id, "hello")
        llm._outputs = [json.dumps({"title": "Read counting"})]
        queue = subscribe(service)
        await service.handle(SessionRetitle(session_id=session.session_id))
        await wait_for(queue, "SessionRows")
        assert "Read counting (by llm)" in _transcript(home)


def _transcript(home) -> str:
    """Every session log this core wrote, concatenated.

    The transcript is the durable record of a conversation, so a session that
    changes name halfway through one has to say so in it — otherwise the file
    and the sidebar disagree about what the conversation was called and only
    one of them survives the run.
    """
    logs = sorted((home / "chatlogs").glob("*.log"))
    return "\n".join(path.read_text() for path in logs)


class TestAutomaticTitling:
    """A conversation names itself after its first exchange.

    A real regression when this was written: `TurnScheduler` takes an
    `on_turn_result` hook for post-turn work and `build_service` passed none,
    so no session was ever titled and `session.retitle` was the whole of
    naming. Seven acceptance claims hang off it, and they are the tests below.

    Two questions have to be told apart for any of it to work, and they are
    answered by two different things. "Has anyone named this conversation?" is
    durable and is answered by the stored title still being the placeholder.
    "Is this one waiting for the model to name it?" belongs to this run only,
    and is a set of ids — which is what makes a reopened session safe.
    """

    async def test_the_column_shows_the_models_summary_not_the_first_message(
        self, service, conn, llm
    ):
        fresh = SessionStore(conn).create(profile="default")
        llm._outputs = [respond(), json.dumps({"title": "Read counting"})]
        await run_turn(service, fresh.session_id, "how many reads are in these BAMs?")
        assert SessionStore(conn).get(fresh.session_id).title == "Read counting"

    async def test_the_sidebar_is_restated_with_the_new_name(
        self, service, conn, llm
    ):
        fresh = SessionStore(conn).create(profile="default")
        llm._outputs = [respond(), json.dumps({"title": "Read counting"})]
        queue = subscribe(service)
        await run_turn(service, fresh.session_id, "how many reads?")
        rows = [e for e in await drain(queue) if type(e).__name__ == "SessionRows"]
        assert rows and rows[-1].rows[0].title == "Read counting"

    async def test_the_opening_message_names_it_until_the_model_does(
        self, service, conn, llm
    ):
        # The placeholder has to be replaced by *something* immediately: a row
        # reading "untitled" for as long as the first exchange takes is a row
        # the user cannot find again, and a failed title call needs a name to
        # fall back to.
        fresh = SessionStore(conn).create(profile="default")
        release = await park_turn(service, llm, fresh.session_id)
        assert SessionStore(conn).get(fresh.session_id).title == "running"
        release.set()
        await service.stop()

    async def test_a_long_first_message_is_cut_to_a_column_width(
        self, service, conn, llm
    ):
        fresh = SessionStore(conn).create(profile="default")
        llm._outputs = [respond(), "not a title"]  # titling fails; the cut shows
        await run_turn(service, fresh.session_id, "x" * 200)
        assert SessionStore(conn).get(fresh.session_id).title == "x" * 40

    async def test_each_new_conversation_is_titled(self, service, conn, llm):
        first = SessionStore(conn).create(profile="default")
        second = SessionStore(conn).create(profile="default")
        llm._outputs = [respond(), json.dumps({"title": "One"})]
        await run_turn(service, first.session_id, "one")
        llm._outputs = [respond(), json.dumps({"title": "Two"})]
        await run_turn(service, second.session_id, "two")
        stored = SessionStore(conn)
        assert stored.get(first.session_id).title == "One"
        assert stored.get(second.session_id).title == "Two"

    async def test_a_conversation_is_titled_once(self, service, conn, llm):
        fresh = SessionStore(conn).create(profile="default")
        llm._outputs = [respond(), json.dumps({"title": "Read counting"})]
        await run_turn(service, fresh.session_id, "how many reads?")
        # A second exchange with another title queued behind it: if titling
        # fired again, the turn would consume the reply and the title the row.
        llm._outputs = [respond(), json.dumps({"title": "Something else"})]
        await run_turn(service, fresh.session_id, "and how many were mapped?")
        assert SessionStore(conn).get(fresh.session_id).title == "Read counting"

    async def test_a_reopened_conversation_is_not_retitled(
        self, home, conn, llm, saver
    ):
        """A session whose first exchange happened in an earlier run.

        The set of sessions awaiting a title is this run's; a restart empties
        it. So a conversation carried over from a previous run is never
        renamed behind the user's back, however it was named then.
        """

        async def db(fn):
            return fn(conn)

        def core():
            return build_service(
                settings=Settings.load(),
                app_dir=home,
                db=db,
                conn=conn,
                checkpointer=saver,
                llm=llm,
            )

        fresh = SessionStore(conn).create(profile="default")
        first = core()
        llm._outputs = [respond(), json.dumps({"title": "Read counting"})]
        await run_turn(first, fresh.session_id, "how many reads?")
        await first.stop()

        restarted = core()
        llm._outputs = [respond(), json.dumps({"title": "Something else"})]
        await run_turn(restarted, fresh.session_id, "and mapped?")
        assert SessionStore(conn).get(fresh.session_id).title == "Read counting"
        await restarted.stop()

    async def test_a_slash_command_alone_does_not_trigger_a_title(
        self, service, conn, llm
    ):
        fresh = SessionStore(conn).create(profile="default")
        await service.handle(
            CommandRun(name="skills-list", session_id=fresh.session_id)
        )
        await _settle()
        # Never named, so never queued for a title: a command is not a turn,
        # and the conversation still has nothing in it to summarise.
        assert SessionStore(conn).get(fresh.session_id).title == "untitled"
        assert not llm.prompts

    async def test_a_hand_written_name_is_never_overwritten(
        self, service, conn, llm
    ):
        # The race that makes this worth a test: the user renames while the
        # first turn is still running, so the model's title arrives *after*
        # the name it must not replace.
        fresh = SessionStore(conn).create(profile="default")
        release = await park_turn(service, llm, fresh.session_id)
        await service.handle(
            SessionRename(session_id=fresh.session_id, title="BAM QC")
        )
        llm._outputs = [respond(), json.dumps({"title": "Something else"})]
        release.set()
        await _settle()
        assert SessionStore(conn).get(fresh.session_id).title == "BAM QC"
        await service.stop()

    async def test_a_failed_title_call_leaves_the_name_alone(
        self, service, conn, llm
    ):
        fresh = SessionStore(conn).create(profile="default")
        # Every attempt answers with something that is not a title.
        llm._outputs = [respond()]
        queue = subscribe(service)
        await run_turn(service, fresh.session_id, "how many reads?")
        assert SessionStore(conn).get(fresh.session_id).title == "how many reads?"
        # And silently: the user is reading the reply, and a toast about the
        # *title* on the back of it would be the loudest thing on screen for
        # the smallest reason. `t` is always there.
        assert not [
            e
            for e in await drain(queue)
            if type(e).__name__ == "Notify" and e.severity == "error"
        ]

    async def test_a_turn_parked_on_an_approval_is_not_named_yet(
        self, service, conn, llm
    ):
        # The exchange is not over — it continues in the resume — so naming it
        # here would summarise half a conversation and spend the one attempt.
        fresh = SessionStore(conn).create(profile="default")
        service._untitled.add(fresh.session_id)
        await service.after_turn(fresh, _Parked(), None)
        assert fresh.session_id in service._untitled

    async def test_the_model_written_name_lands_in_the_session_log(
        self, service, conn, llm, home
    ):
        fresh = SessionStore(conn).create(profile="default")
        llm._outputs = [respond(), json.dumps({"title": "Read counting"})]
        await run_turn(service, fresh.session_id, "how many reads?")
        assert "Read counting (by llm)" in _transcript(home)


class TestTheTranscriptLog:
    """`<app_dir>/chatlogs/*.log`: the durable, greppable record of a run.

    Written by the core and by nothing else, which is the whole of this
    class's reason to exist — `on_turn_result` names the transcript log as
    one of its three consumers and only titling was ever wired to it, so a
    core-driven run wrote a file with a rename line in it and no conversation.

    Every claim here was carried by `test_tui_thinking_logs.py` against the
    widget class that is about to be deleted.
    """

    async def test_a_turn_is_logged_with_its_thinking_and_its_answer(
        self, service, session, llm, home
    ):
        llm._outputs = [
            ChatResponse(content=respond("Four BAMs match."), reasoning="Count them.")
        ]
        await run_turn(service, session.session_id, "which BAMs?")
        text = _transcript(home)
        assert "] user\nwhich BAMs?" in text
        assert "thinking" in text and "Count them." in text
        assert "] agent\nFour BAMs match." in text
        # In the order it happened: the file is read top to bottom.
        assert text.index("which BAMs?") < text.index("Count them.")
        assert text.index("Count them.") < text.index("Four BAMs match.")

    async def test_the_call_is_written_to_the_session_log(
        self, service, session, llm, home
    ):
        target = home / "reads.tsv"
        target.write_text("sample\tcount\na\t7\n")
        llm._outputs = [calling("read_file", path=str(target)), respond("seven")]
        await run_turn(service, session.session_id, "how many reads?")
        text = _transcript(home)
        # One heading for the exchange, with the call and what it returned
        # under it — the same reading of the turn the chat draws, so the tool
        # is named once in both.
        assert text.count("— read_file") == 1
        assert str(target) in text
        assert RESULT_RULE in text and "sample" in text

    async def test_a_subagents_query_and_reply_are_logged_under_the_tool(
        self, home, conn, llm, saver
    ):
        async def db(fn):
            return fn(conn)

        service = build_service(
            settings=Settings.load(),
            app_dir=home,
            db=db,
            conn=conn,
            checkpointer=saver,
            llm=llm,
            tools=_subagent_tools(),
        )
        fresh = SessionStore(conn).create(profile="default", title="asking")
        llm._outputs = [
            calling("ask_docs", question="what is a BAM?"),
            "binary alignment map",
            respond("ok"),
        ]
        await run_turn(service, fresh.session_id, "explain BAMs")
        text = _transcript(home)
        # The model calls a tool makes on its own are the turn's too, and they
        # are logged under the tool's name rather than as the orchestrator's.
        assert "subagent:ask_docs query" in text
        assert "user: what is a BAM?" in text
        assert "subagent:ask_docs reply" in text
        assert "binary alignment map" in text
        await service.stop()

    async def test_a_parked_approval_is_written_down_before_it_is_answered(
        self, service, session, llm, home
    ):
        llm._outputs = [
            calling("run_bash", content_lines=["rm -rf /scratch/old"]),
            respond("left it alone"),
        ]
        queue = subscribe(service)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="clear the scratch dir")
        )
        await wait_for(queue, "DecisionRequested")
        # The half that has happened is in the file while the question is
        # still on screen: a core that dies waiting for an answer must not
        # take the conversation with it.
        assert "] user\nclear the scratch dir" in _transcript(home)
        await service.handle(
            DecisionResolve(session_id=session.session_id, approved=False)
        )
        await wait_for(queue, "TurnFinished")
        text = _transcript(home)
        # And the resume adds only what followed: one exchange, written once,
        # however many graph invocations it took.
        assert text.count("] user\nclear the scratch dir") == 1
        assert text.count("rm -rf /scratch/old") == 1
        assert "] agent\nleft it alone" in text
        await service.stop()

    async def test_reopening_a_session_does_not_re_log_it(
        self, service, session, llm, home
    ):
        await run_turn(service, session.session_id, "first question")
        await service.handle(SessionOpen(session_id=session.session_id))
        await _settle()
        await run_turn(service, session.session_id, "second question")
        text = _transcript(home)
        # Only the tail of each turn is written, so re-reading a conversation
        # never duplicates it. Logged entries are counted, not mentions.
        assert text.count("] user\nfirst question") == 1
        assert text.count("] user\nsecond question") == 1

    async def test_each_session_gets_its_own_file(
        self, service, session, conn, llm, home
    ):
        other = SessionStore(conn).create(profile="default", title="the other one")
        await run_turn(service, session.session_id, "first")
        await run_turn(service, other.session_id, "second")
        files = sorted((home / "chatlogs").glob("*.log"))
        assert len(files) == 2
        # And each holds its own conversation, not whichever was on screen.
        contents = [path.read_text() for path in files]
        assert any("first" in text and "second" not in text for text in contents)
        assert any("second" in text and "first" not in text for text in contents)

    async def test_errors_are_logged(self, service, session, llm, home):
        async def dead(messages, **kwargs):
            raise ConnectionError("backend unreachable")

        llm.chat = dead
        queue = subscribe(service)
        await service.handle(TurnSubmit(session_id=session.session_id, text="hi"))
        await wait_for(queue, "TurnFailed")
        text = _transcript(home)
        assert "backend unreachable" in text
        # And what the user asked before it broke: a transcript that drops the
        # question keeps no record of the exchange at all.
        assert "] user\nhi" in text
        await service.stop()

    async def test_logging_off_writes_nothing(self, home, conn, llm, saver):
        settings = Settings.load()
        settings.logging.enabled = False

        async def db(fn):
            return fn(conn)

        service = build_service(
            settings=settings,
            app_dir=home,
            db=db,
            conn=conn,
            checkpointer=saver,
            llm=llm,
        )
        fresh = SessionStore(conn).create(profile="default", title="quiet")
        await run_turn(service, fresh.session_id, "hello")
        assert not (home / "chatlogs").exists()
        await service.stop()

    async def test_switching_logging_off_takes_effect_at_once(
        self, service, session, llm, home
    ):
        await run_turn(service, session.session_id, "logged question")
        service._deps.settings.logging.enabled = False
        await run_turn(service, session.session_id, "private question")
        text = _transcript(home)
        assert "logged question" in text
        # The next turn, not the next session: the log is opened per turn.
        assert "private question" not in text


class TestTheEpisodicIndex:
    """Recall across sessions (redesign Phase 2), which is invisible until it
    is gone: nothing on screen changes when the index stops being written,
    and `session_search` simply never finds anything again."""

    async def test_a_turn_is_recallable_from_a_later_session(
        self, service, session, conn, llm
    ):
        llm._outputs = [respond("STAR needs about 32 GB for the human index.")]
        await run_turn(
            service, session.session_id, "how much memory does STAR need?"
        )
        later = SessionStore(conn).create(profile="default", title="later")
        hits = EpisodicStore(conn).search(
            "STAR memory", profile="default", exclude_session_id=later.session_id
        )
        assert [hit.session_id for hit in hits] == [session.session_id]
        # Reported bookends-first: what was asked, and what it came to.
        assert "how much memory does STAR need?" in hits[0].goal
        assert "32 GB" in hits[0].resolution

    async def test_the_agent_recalls_it_through_session_search(
        self, service, session, conn, llm
    ):
        llm._outputs = [respond("STAR needs about 32 GB for the human index.")]
        await run_turn(
            service, session.session_id, "how much memory does STAR need?"
        )
        later = SessionStore(conn).create(profile="default", title="later")
        llm._outputs = [
            calling("session_search", query="STAR memory"),
            respond("32 GB, as before"),
        ]
        queue = subscribe(service)
        await service.handle(
            TurnSubmit(session_id=later.session_id, text="and STAR again?")
        )
        await wait_for(queue, "TurnFinished")
        # The tool's answer is fed back as a tool result; no model call was
        # spent finding it.
        results = [
            str(message["content"])
            for prompt in llm.prompts
            for message in prompt
            if str(message["content"]).startswith("[tool result] session_search")
        ]
        assert results and "32 GB" in results[-1]
        await service.stop()

    async def test_tool_traffic_is_not_indexed(
        self, service, session, conn, llm, home
    ):
        target = home / "reads.tsv"
        target.write_text("sample\tcount\na\t7\n")
        llm._outputs = [
            ChatResponse(
                content=calling("read_file", path=str(target)),
                reasoning="the counts are in column 2",
            ),
            respond("seven reads"),
        ]
        await run_turn(service, session.session_id, "how many reads?")
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? "
            "ORDER BY turn_no",
            (session.session_id,),
        ).fetchall()
        # BM25 over tool vocabulary matches everything: only what the two
        # sides of the conversation said is indexed.
        assert [row["role"] for row in rows] == ["user", "assistant"]
        assert [row["content"] for row in rows] == [
            "how many reads?",
            "seven reads",
        ]

    async def test_a_stopped_turn_is_indexed_as_far_as_it_got(
        self, service, session, conn, llm
    ):
        # It used to leave nothing behind, because stopping a turn rolled its
        # messages out of the thread and indexing them would have described a
        # conversation that no longer existed. The work stays now, so the
        # opposite is what would be wrong: a conversation the user can scroll
        # back to and `session_search` cannot find.
        release = await park_turn(service, llm, session.session_id)
        await service.handle(TurnInterrupt(session_id=session.session_id))
        release.set()
        await _settle()
        rows = conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? "
            "ORDER BY turn_no",
            (session.session_id,),
        ).fetchall()
        # The question, and only the question: it was never answered, and the
        # "[stopped]" marker is an event rather than something either side
        # said (`_record_turn`).
        assert [(row["role"], row["content"]) for row in rows] == [
            ("user", "running")
        ]
        await service.stop()

    async def test_a_deleted_session_is_never_indexed_behind_its_deletion(
        self, service, session, conn, llm
    ):
        # The write crosses a thread, so the session can be gone by the time
        # it lands; recording then would leave rows search can find and
        # nothing can delete.
        entries = [_Entry("user", "a question"), _Entry("assistant", "an answer")]
        SessionStore(conn).delete(session.session_id)
        await service._record_turn(session, entries, None)
        assert conn.execute(
            "SELECT COUNT(*) AS n FROM messages WHERE session_id = ?",
            (session.session_id,),
        ).fetchone()["n"] == 0


class _Entry:
    """Enough of a transcript entry for the recording half to read."""

    def __init__(self, kind: str, text: str) -> None:
        self.kind, self.text = kind, text


def _subagent_tools():
    """A registry with one tool that runs its own model call (§4.2)."""
    from pydantic import BaseModel, Field

    from hpca.agent.tools import Tool, ToolRegistry

    class AskParams(BaseModel):
        question: str = Field(description="What to ask the docs")

    async def ask(args, ctx):
        response = await ctx.llm.chat(
            [{"role": "user", "content": args.question}]
        )
        return response.content

    registry = ToolRegistry()
    registry.register(
        Tool(
            name="ask_docs",
            description="Ask the docs",
            params=AskParams,
            handler=ask,
        )
    )
    return registry


class _Parked:
    """A turn result that stopped at an approval."""

    interrupt = {"tool": "run_bash"}
    messages: list = []


class TestSessionDelete:
    async def test_the_row_and_its_history_go(self, service, session, conn):
        await run_turn(service, session.session_id, "hello")
        queue = subscribe(service)
        await service.handle(SessionDelete(session_id=session.session_id))
        events = await drain(queue)
        assert only(events, "SessionRows").rows == []
        assert SessionStore(conn).get(session.session_id) is None
        state = await service._graph.aget_state(
            {"configurable": {"thread_id": session.session_id}}
        )
        assert not (state.values or {}).get("messages")

    async def test_a_deleted_sessions_watches_go_with_it(
        self, service, session, conn
    ):
        from hpca.watches import WatchStore

        WatchStore(conn).add(
            session_id=session.session_id, kind="log", target="/tmp/run.log"
        )
        await service.handle(SessionDelete(session_id=session.session_id))
        # Left behind they would be boxes no session can show while the
        # pollers went on stat-ing their files forever.
        assert WatchStore(conn).list(session_id=session.session_id) == []

    async def test_it_is_forgotten_by_search_too(self, service, session, conn):
        from hpca.episodic import EpisodicStore

        store = EpisodicStore(conn)
        store.record(
            session_id=session.session_id,
            profile="default",
            entries=[("user", "the patient identifier")],
        )
        assert store.search("patient", limit=5)
        await service.handle(SessionDelete(session_id=session.session_id))
        # Patient-data environments: a deleted conversation must not resurface
        # through a search either.
        assert store.search("patient", limit=5) == []

    async def test_nothing_queued_for_it_can_still_start(
        self, service, session, llm
    ):
        release = await park_turn(service, llm, session.session_id)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="queued behind it")
        )
        await service.handle(SessionDelete(session_id=session.session_id))
        assert service._scheduler.queued_texts_for(session.session_id) == []
        release.set()
        await service.stop()

    async def test_deleting_a_session_that_is_gone_says_so(self, service):
        queue = subscribe(service)
        await service.handle(SessionDelete(session_id="nope"))
        assert (await drain(queue))[0].severity == "warning"


class TestRollback:
    async def test_the_trimmed_conversation_comes_back_as_a_reset(
        self, service, session
    ):
        await run_turn(service, session.session_id, "first")
        await run_turn(service, session.session_id, "second")
        queue = subscribe(service)
        # Cut before the second user message — the index the core itself put
        # on that entry.
        await service.handle(
            SessionRollback(session_id=session.session_id, index=2)
        )
        reset = only(await drain(queue), "ChatReset")
        assert [e.text for e in reset.entries] == ["first", "done"]
        # Re-based numbering: the rows the cut removed cannot be addressed by
        # either side afterwards.
        assert [e.seq for e in reset.entries] == [1, 2]

    async def test_it_is_refused_while_a_turn_is_running(
        self, service, session, llm
    ):
        release = await park_turn(service, llm, session.session_id)
        queue = subscribe(service)
        await service.handle(
            SessionRollback(session_id=session.session_id, index=0)
        )
        events = await drain(queue)
        assert kinds(events) == ["Notify"] and events[0].severity == "warning"
        assert "a turn is running" in events[0].text
        release.set()
        await service.stop()

    async def test_an_index_the_thread_no_longer_has_is_refused(
        self, service, session
    ):
        await run_turn(service, session.session_id, "first")
        queue = subscribe(service)
        await service.handle(
            SessionRollback(session_id=session.session_id, index=99)
        )
        events = await drain(queue)
        # Refused rather than truncating somewhere the user never pointed at.
        assert kinds(events) == ["Notify"] and events[0].severity == "warning"
        assert "no longer" in events[0].text

    async def test_the_fill_is_restated_for_the_thread_that_is_left(
        self, service, session
    ):
        await run_turn(service, session.session_id, "first")
        queue = subscribe(service)
        await service.handle(
            SessionRollback(session_id=session.session_id, index=0)
        )
        events = await drain(queue)
        # The measured number described a thread that no longer exists.
        assert only(events, "ContextEstimate").used == 0

    async def test_rolling_back_a_session_that_is_gone_says_so(self, service):
        queue = subscribe(service)
        await service.handle(SessionRollback(session_id="nope", index=0))
        assert (await drain(queue))[0].severity == "warning"


class TestFork:
    async def test_the_branch_is_created_and_announced(
        self, service, session, conn
    ):
        await run_turn(service, session.session_id, "first")
        await run_turn(service, session.session_id, "second")
        queue = subscribe(service)
        await service.handle(SessionFork(session_id=session.session_id, index=2))
        events = await drain(queue)
        created = only(events, "SessionCreated").row
        assert created.title == "a session (fork)"
        assert created.session_id in {
            r.session_id for r in only(events, "SessionRows").rows
        }
        # The source keeps its whole history: this half is non-destructive.
        assert SessionStore(conn).get(session.session_id) is not None

    async def test_the_fork_holds_the_conversation_up_to_the_cut(
        self, service, session
    ):
        await run_turn(service, session.session_id, "first")
        await run_turn(service, session.session_id, "second")
        queue = subscribe(service)
        await service.handle(SessionFork(session_id=session.session_id, index=2))
        fork = only(await drain(queue), "SessionCreated").row

        await service.handle(SessionOpen(session_id=fork.session_id))
        reset = only(await drain(queue), "ChatReset")
        assert [e.text for e in reset.entries] == ["first", "done"]

    async def test_the_forks_profile_and_backend_come_from_the_source(
        self, service, conn
    ):
        blob = json.dumps(
            {"model": "qwen3-32b", "base_url": "http://localhost:20001/v1"}
        )
        store = SessionStore(conn)
        source = store.create(profile="bioinformatics", title="pinned", mode="plan")
        await run_turn(service, source.session_id, "first")
        # Pinned after the turn: a session pointed at a real endpoint would
        # dial it, and this test has no backend to answer.
        store.set_backend(source.session_id, blob)
        queue = subscribe(service)
        await service.handle(SessionFork(session_id=source.session_id, index=0))
        row = only(await drain(queue), "SessionCreated").row
        # Not on the wire, so a front-end cannot fork a conversation into a
        # profile the user never chose.
        assert (row.profile, row.mode, row.model) == (
            "bioinformatics",
            "plan",
            "qwen3-32b",
        )

    async def test_forking_is_allowed_while_the_source_is_busy(
        self, service, session, llm
    ):
        # Deliberately ungated where the rollback is not: a fork only reads a
        # snapshot of the source, and branching off mid-turn is the case it
        # exists for.
        await run_turn(service, session.session_id, "first")
        release = await park_turn(service, llm, session.session_id)
        queue = subscribe(service)
        await service.handle(SessionFork(session_id=session.session_id, index=0))
        assert "SessionCreated" in kinds(await drain(queue))
        release.set()
        await service.stop()

    async def test_an_index_the_thread_no_longer_has_leaves_no_session_behind(
        self, service, session, conn
    ):
        await run_turn(service, session.session_id, "first")
        before = len(SessionStore(conn).list_all())
        queue = subscribe(service)
        await service.handle(SessionFork(session_id=session.session_id, index=99))
        events = await drain(queue)
        assert kinds(events) == ["Notify"] and events[0].severity == "warning"
        assert len(SessionStore(conn).list_all()) == before

    async def test_a_failed_copy_leaves_no_session_behind(
        self, service, session, conn, monkeypatch
    ):
        # An empty session nobody asked for is worse than none: it would sit
        # in the sidebar looking like the fork worked.
        await run_turn(service, session.session_id, "first")

        async def explode(*a, **k):
            raise RuntimeError("checkpoint write failed")

        monkeypatch.setattr("hpca.core.service.fork_thread", explode)
        before = len(SessionStore(conn).list_all())
        queue = subscribe(service)
        await service.handle(SessionFork(session_id=session.session_id, index=0))
        events = await drain(queue)
        assert any(
            type(e).__name__ == "Notify" and e.severity == "error" for e in events
        )
        assert len(SessionStore(conn).list_all()) == before

    async def test_forking_a_session_that_is_gone_says_so(self, service):
        queue = subscribe(service)
        await service.handle(SessionFork(session_id="nope", index=0))
        assert (await drain(queue))[0].severity == "warning"


class TestAConversationEndToEnd:
    """List, make one, open it, talk, rewind, branch — commands only.

    The point of this one is that nothing else is used: no store call, no
    graph call, no reach into a service. If a front-end can send these seven
    frames and read the ones that come back, it has a conversation.
    """

    async def test_the_whole_round_trip(self, service, llm):
        queue = subscribe(service)

        await service.handle(SessionList())
        assert only(await drain(queue), "SessionRows").rows == []

        await service.handle(SessionNew(profile="default"))
        events = await drain(queue)
        session_id = only(events, "SessionCreated").row.session_id
        assert [r.session_id for r in only(events, "SessionRows").rows] == [
            session_id
        ]

        await service.handle(SessionOpen(session_id=session_id))
        assert only(await drain(queue), "ChatReset").entries == []

        await run_turn(service, session_id, "which BAMs are in /data?")
        await run_turn(service, session_id, "and the CRAMs?")
        await drain(queue)

        await service.handle(SessionFork(session_id=session_id, index=2))
        fork_id = only(await drain(queue), "SessionCreated").row.session_id

        await service.handle(SessionRollback(session_id=session_id, index=2))
        reset = only(await drain(queue), "ChatReset")
        assert (reset.session_id, [e.text for e in reset.entries]) == (
            session_id,
            ["which BAMs are in /data?", "done"],
        )

        # The fork kept the same two entries, and the source's rollback did
        # not touch it — two conversations now, numbered from 1 apiece.
        await service.handle(SessionOpen(session_id=fork_id))
        forked = only(await drain(queue), "ChatReset")
        assert forked.session_id == fork_id
        assert [(e.text, e.seq) for e in forked.entries] == [
            ("which BAMs are in /data?", 1),
            ("done", 2),
        ]
        await service.stop()


class TestFocus:
    async def test_focus_is_recorded_and_repaints(self, service, session):
        queue = subscribe(service)
        await service.handle(SessionFocus(session_id=session.session_id))
        assert service._deps.focused_session_id == session.session_id
        # force=True on focus: a client that just opened a session has a blank
        # column and needs the rows even if they did not change.
        assert "PanelUpdate" in kinds(await drain(queue))

    async def test_clearing_focus_is_allowed(self, service):
        await service.handle(SessionFocus(session_id=None))
        assert service._deps.focused_session_id is None


class TestConfirmations:
    async def test_a_question_is_asked_and_its_answer_runs_the_action(self, service):
        queue = subscribe(service)
        ran = []

        async def on_yes():
            ran.append(True)

        service.ask("s1", "Learn this signature?", on_yes)
        events = await drain(queue)
        assert kinds(events) == ["ConfirmRequested"]
        # Which conversation is being asked about, so a client can put the
        # question in it rather than over whatever the user is reading.
        assert events[0].session_id == "s1"
        await service.handle(ConfirmResolve(id=events[0].id, confirmed=True))
        assert ran == [True]

    async def test_a_no_runs_nothing(self, service):
        queue = subscribe(service)
        ran = []
        service.ask("s1", "Learn this?", lambda: _record(ran))
        events = await drain(queue)
        await service.handle(ConfirmResolve(id=events[0].id, confirmed=False))
        assert ran == []

    async def test_an_answer_to_a_question_nobody_asked_is_ignored(self, service):
        await service.handle(ConfirmResolve(id="q99", confirmed=True))

    async def test_the_same_answer_twice_runs_once(self, service):
        queue = subscribe(service)
        ran = []
        service.ask("s1", "Learn this?", lambda: _record(ran))
        key = (await drain(queue))[0].id
        await service.handle(ConfirmResolve(id=key, confirmed=True))
        await service.handle(ConfirmResolve(id=key, confirmed=True))
        assert ran == [True]

    async def test_a_failing_action_is_reported_not_raised(self, service):
        queue = subscribe(service)

        async def boom():
            raise RuntimeError("the signature file is read-only")

        service.ask("s1", "Learn this?", boom)
        key = (await drain(queue))[0].id
        await service.handle(ConfirmResolve(id=key, confirmed=True))
        assert any(
            type(e).__name__ == "Notify" and e.severity == "error"
            for e in await drain(queue)
        )


class TestTheTwoDials:
    """`mode.set` and `thinking.set`: the per-session dials of §4.1.

    Both are stored and then *restated as a sidebar row* rather than
    acknowledged, which is what lets a second client see the change too.
    """

    async def test_a_mode_lands_on_the_row_and_in_the_store(
        self, service, session, conn
    ):
        queue = subscribe(service)
        await service.handle(
            ModeSet(session_id=session.session_id, mode="full-auto")
        )
        assert only(await drain(queue), "SessionRows").rows[0].mode == "full-auto"
        assert SessionStore(conn).get(session.session_id).mode == "full-auto"

    async def test_a_mode_the_agent_does_not_have_is_refused(
        self, service, session, conn
    ):
        # The wire carries a plain string because the set of modes is the
        # agent's business; that is exactly why it has to be checked here.
        queue = subscribe(service)
        await service.handle(ModeSet(session_id=session.session_id, mode="yolo"))
        events = await drain(queue)
        assert kinds(events) == ["Notify"] and events[0].severity == "warning"
        assert SessionStore(conn).get(session.session_id).mode == ""

    async def test_setting_the_mode_of_a_session_that_is_gone_says_so(
        self, service
    ):
        queue = subscribe(service)
        await service.handle(ModeSet(session_id="gone", mode="auto"))
        assert only(await drain(queue), "Notify").severity == "warning"

    async def test_the_thinking_level_rides_the_sidebar_row(
        self, service, session, conn
    ):
        # The meter's `· think medium` reads it from here: an event of its own
        # could only describe the session that just changed, and the level of
        # whichever session is opened next is what has to be drawn.
        queue = subscribe(service)
        await service.handle(
            ThinkingSet(session_id=session.session_id, effort="medium")
        )
        events = await drain(queue)
        assert only(events, "SessionRows").rows[0].thinking == "medium"
        assert SessionStore(conn).get(session.session_id).thinking == "medium"
        assert only(events, "Notify").severity == "information"

    async def test_xhigh_says_that_it_does_not_work(self, service, session):
        # Offered because the model advertises it, not because it is usable —
        # and the headline is in the title so it lands even if the paragraph
        # under it is skimmed.
        queue = subscribe(service)
        await service.handle(
            ThinkingSet(session_id=session.session_id, effort="xhigh")
        )
        toast = only(await drain(queue), "Notify")
        assert toast.severity == "warning"
        assert "NOT USABLE" in toast.title
        assert toast.timeout and toast.timeout > 10

    async def test_a_level_the_served_model_has_never_heard_of_is_refused(
        self, service, session, conn
    ):
        # There is no "high", however much the name suggests one (hpca.thinking).
        queue = subscribe(service)
        await service.handle(
            ThinkingSet(session_id=session.session_id, effort="high")
        )
        assert only(await drain(queue), "Notify").severity == "warning"
        assert SessionStore(conn).get(session.session_id).thinking == ""


def entry(model="qwen3-32b", url="http://localhost:20001/v1"):
    return {"model": model, "base_url": url}


class TestBackendSet:
    """One command, two operations, told apart by whether a session is named."""

    async def test_one_session_is_pointed_at_another_model(
        self, service, session, conn
    ):
        queue = subscribe(service)
        await service.handle(
            BackendSet(session_id=session.session_id, backend=entry())
        )
        events = await drain(queue)
        assert only(events, "SessionRows").rows[0].model == "qwen3-32b"
        # Stored as the blob, so the choice survives the catalog entry going.
        assert "qwen3-32b" in SessionStore(conn).get(session.session_id).backend

    async def test_it_is_refused_while_that_session_is_mid_reply(
        self, service, session, conn, llm
    ):
        release = await park_turn(service, llm, session.session_id)
        queue = subscribe(service)
        await service.handle(
            BackendSet(session_id=session.session_id, backend=entry())
        )
        assert only(await drain(queue), "Notify").severity == "warning"
        assert SessionStore(conn).get(session.session_id).backend == ""
        release.set()
        await service.stop()

    async def test_another_sessions_turn_does_not_block_it(
        self, service, session, conn, llm
    ):
        # Clients are keyed per backend and checkpoints per thread, so nothing
        # the other turn is holding is disturbed by this.
        other = SessionStore(conn).create(profile="default", title="elsewhere")
        release = await park_turn(service, llm, other.session_id)
        queue = subscribe(service)
        await service.handle(
            BackendSet(session_id=session.session_id, backend=entry())
        )
        assert "SessionRows" in kinds(await drain(queue))
        release.set()
        await service.stop()

    async def test_no_session_sets_the_default_and_catalogues_it(self, service):
        # The global half writes the *settings*, so it outlives the run and
        # decides what the next session is created against.
        await service.handle(BackendSet(backend=entry()))
        settings = service._deps.settings
        assert settings.llm.model == "qwen3-32b"
        assert any(b.model == "qwen3-32b" for b in settings.backends)

    async def test_a_blob_the_settings_model_refuses_is_not_stored(
        self, service, session, conn
    ):
        queue = subscribe(service)
        await service.handle(
            BackendSet(session_id=session.session_id, backend={"nonsense": 1})
        )
        assert only(await drain(queue), "Notify").severity == "error"
        assert SessionStore(conn).get(session.session_id).backend == ""


class TestTheLLMCatalog:
    """`llm.list` → `llm.catalog`: the backends, drawable, without the keys.

    The event this protocol was missing. Nothing carried the configured
    backends to a front-end, so the new-session picker and the manage-LLMs
    screen could only be filled by a demo — and §4.2 rule 2 (the UI never
    reads the core's state) leaves an event as the only way to fill them.
    """

    async def test_the_configured_backends_come_back_as_drawable_rows(
        self, service
    ):
        service._deps.settings.backends = [
            LLMBackend(
                model="qwen3-32b",
                base_url="http://node07:20001/v1",
                max_model_len=32768,
            )
        ]
        queue = subscribe(service)
        await service.handle(LLMList())
        entry_row = only(await drain(queue), "LLMCatalog").entries[0]
        assert (entry_row.label, entry_row.model) == ("qwen3-32b", "qwen3-32b")
        assert entry_row.base_url == "http://node07:20001/v1"
        assert entry_row.max_model_len == 32768

    async def test_no_api_key_crosses_the_wire(self, service):
        service._deps.settings.backends = [
            LLMBackend(
                model="qwen3-32b",
                base_url="http://node07:20001/v1",
                api_key="sk-secret",
            )
        ]
        queue = subscribe(service)
        await service.handle(LLMList())
        catalog = only(await drain(queue), "LLMCatalog")
        assert "sk-secret" not in catalog.model_dump_json()
        # Nor any hint that one is still wanted: the key is set, and
        # ``needs_key`` means "still locked" (`protocol.LLMEntry`).
        assert catalog.entries[0].needs_key is False

    async def test_the_default_backend_is_marked(self, service):
        settings = service._deps.settings
        settings.backends = [
            LLMBackend(model="a", base_url="http://a/v1"),
            LLMBackend(model="b", base_url="http://b/v1"),
        ]
        settings.activate_backend(settings.backends[1])
        queue = subscribe(service)
        await service.handle(LLMList())
        assert [e.active for e in only(await drain(queue), "LLMCatalog").entries] == [
            False,
            True,
        ]

    async def test_nothing_is_probed_unless_the_client_asks(self, service):
        service._deps.settings.backends = [
            LLMBackend(model="a", base_url="http://a/v1")
        ]
        queue = subscribe(service)
        await service.handle(LLMList())
        catalog = only(await drain(queue), "LLMCatalog")
        assert catalog.probed is False
        assert catalog.entries[0].reachable is None

    async def test_the_probes_arrive_in_a_second_frame(self, service):
        # The screen must draw before the round trips land: an endpoint on a
        # node that has gone away costs the full timeout, and a picker blank
        # for that long is a picker nobody waits for.
        service._deps.settings.backends = [
            LLMBackend(model="a", base_url="http://a/v1")
        ]
        service._backends.probe_catalog = _answers({"a": True})
        queue = subscribe(service)
        await service.handle(LLMList(probe=True))
        events = await wait_for(queue, "LLMCatalog")
        first = [e for e in events if type(e).__name__ == "LLMCatalog"][0]
        assert first.probed is False
        probed = await wait_for(queue, "LLMCatalog")
        answer = [e for e in probed if type(e).__name__ == "LLMCatalog"][-1]
        assert answer.probed is True and answer.entries[0].reachable is True

    async def test_setting_the_default_restates_the_catalog(self, service):
        # The ★ has moved, and `set_default` adds an unlisted backend on its
        # way past — so this can be a new row as well as a moved mark.
        queue = subscribe(service)
        await service.handle(BackendSet(backend=entry()))
        catalog = only(await drain(queue), "LLMCatalog")
        assert [(e.model, e.active) for e in catalog.entries] == [
            ("qwen3-32b", True)
        ]

    async def test_a_run_with_no_backends_answers_with_an_empty_catalog(
        self, service
    ):
        # Not silence: "there is nothing to pick from" is the answer that lets
        # a front-end skip the picker rather than wait for a frame.
        queue = subscribe(service)
        await service.handle(LLMList())
        assert only(await drain(queue), "LLMCatalog").entries == []


def _answers(result):
    """A stand-in for a probe run: no socket, the answer already known."""

    async def probe():
        return dict(result)

    return probe


class TestTheProfileListing:
    """`profile.list` → `profile.rows`: an event where inference used to be.

    The profiles screen was assembled from `hello`'s profile plus whatever
    profiles the sidebar rows happened to name. That misses every profile with
    no session, and neither source can carry the memory count or the
    provenance — those live in files only the core reads.
    """

    async def test_every_profile_comes_back_with_what_the_screen_draws(
        self, service
    ):
        Profile.create("bioinformatics")
        queue = subscribe(service)
        await service.handle(ProfileList())
        rows = {r.name: r for r in only(await drain(queue), "ProfileRows").rows}
        assert set(rows) == {"default", "bioinformatics"}
        assert rows["default"].is_default is True
        assert rows["bioinformatics"].is_default is False

    async def test_a_profile_with_no_session_is_still_listed(self, service):
        # The whole reason inference was not good enough: nothing in the
        # sidebar names a profile nobody has had a conversation under.
        Profile.create("unused")
        queue = subscribe(service)
        await service.handle(ProfileList())
        rows = only(await drain(queue), "ProfileRows").rows
        assert "unused" in {r.name for r in rows}

    async def test_the_memories_are_counted(self, service):
        profile = Profile.create("bioinformatics")
        profile.add_memory("BAMs live on /scratch", scope=MemoryScope.SYSTEM_PROMPT)
        profile.save()
        queue = subscribe(service)
        await service.handle(ProfileList())
        rows = {r.name: r for r in only(await drain(queue), "ProfileRows").rows}
        assert rows["bioinformatics"].memories == 1

    async def test_a_copy_says_what_it_was_copied_from(self, service):
        Profile.create("base")
        Profile.duplicate("base", "specialised")
        queue = subscribe(service)
        await service.handle(ProfileList())
        rows = {r.name: r for r in only(await drain(queue), "ProfileRows").rows}
        assert rows["specialised"].copied_from == "base"

    async def test_the_working_profile_is_marked_and_is_not_the_default(
        self, service
    ):
        Profile.create("bioinformatics")
        await service.handle(ProfileSet(name="bioinformatics"))
        queue = subscribe(service)
        await service.handle(ProfileList())
        rows = {r.name: r for r in only(await drain(queue), "ProfileRows").rows}
        assert rows["bioinformatics"].working is True
        assert rows["bioinformatics"].is_default is False
        assert rows["default"].working is False

    async def test_creating_one_restates_the_listing(self, service):
        queue = subscribe(service)
        await service.handle(ProfileCreate(name="bioinformatics"))
        rows = only(await drain(queue), "ProfileRows").rows
        assert "bioinformatics" in {r.name for r in rows}

    async def test_deleting_one_restates_the_listing(self, service):
        Profile.create("doomed")
        queue = subscribe(service)
        await service.handle(ProfileDelete(name="doomed"))
        rows = only(await drain(queue), "ProfileRows").rows
        assert "doomed" not in {r.name for r in rows}

    async def test_the_sessions_are_counted(self, service, conn):
        # The other number on the row, and the one a front-end cannot work out
        # for itself: the sidebar holds the sessions it was sent, and rule 2
        # of §4.2 keeps the table out of its reach.
        Profile.create("bioinformatics")
        store = SessionStore(conn)
        store.create(profile="bioinformatics", title="one")
        store.create(profile="bioinformatics", title="two")
        queue = subscribe(service)
        await service.handle(ProfileList())
        rows = {r.name: r for r in only(await drain(queue), "ProfileRows").rows}
        assert rows["bioinformatics"].sessions == 2
        assert rows["default"].sessions == 0

    async def test_a_new_session_restates_the_listing(self, service):
        queue = subscribe(service)
        await service.handle(SessionNew(profile="default"))
        rows = {r.name: r for r in only(await drain(queue), "ProfileRows").rows}
        assert rows["default"].sessions == 1

    async def test_deleting_a_session_restates_the_listing(
        self, service, session
    ):
        queue = subscribe(service)
        await service.handle(SessionDelete(session_id=session.session_id))
        rows = {r.name: r for r in only(await drain(queue), "ProfileRows").rows}
        assert rows["default"].sessions == 0

    async def test_an_approved_memory_restates_the_listing(
        self, service, session, llm
    ):
        # The count is only ever sent when it is asked for, and nothing asks
        # again after a review: a profile that had just gained a memory went
        # on saying what it said when the client connected, until a restart.
        await run_turn(service, session.session_id, "how many reads?")
        llm._outputs = [proposals("the cohort is in /data/cohort")]
        queue = subscribe(service)
        await service.handle(
            CommandRun(
                name="memorize",
                args="where the cohort lives",
                session_id=session.session_id,
            )
        )
        await wait_for(queue, "MemoryProposals")
        await drain(queue)
        await service.handle(
            MemoryResolve(session_id=session.session_id, approved=[True])
        )
        rows = {r.name: r for r in only(await drain(queue), "ProfileRows").rows}
        assert rows["default"].memories == 1

    async def test_approving_two_memories_restates_the_listing_once(
        self, service, session, llm
    ):
        # One answer is one change to one count; a frame per approved item
        # would repaint the screen behind the review three times over.
        await run_turn(service, session.session_id, "how many reads?")
        llm._outputs = [proposals("the cohort is in /data/cohort", "reads are bams")]
        queue = subscribe(service)
        await service.handle(
            CommandRun(
                name="memorize",
                args="where the cohort lives",
                session_id=session.session_id,
            )
        )
        await wait_for(queue, "MemoryProposals")
        await drain(queue)
        await service.handle(
            MemoryResolve(session_id=session.session_id, approved=[True, True])
        )
        rows = {r.name: r for r in only(await drain(queue), "ProfileRows").rows}
        assert rows["default"].memories == 2

    async def test_a_hand_edited_memory_file_restates_the_listing(self, service):
        # The other way the count moves, and the one where the screen showing
        # it is usually the screen the editor was opened from.
        Profile.create("bioinformatics")
        edited = Profile.load("bioinformatics")
        edited.add_memory("BAMs live on /scratch", scope=MemoryScope.SYSTEM_PROMPT)
        edited.add_memory("fastqs live on /data", scope=MemoryScope.SYSTEM_PROMPT)
        queue = subscribe(service)
        await service.handle(
            ProfileSave(
                name="bioinformatics", kind="memories", text=edited.render()
            )
        )
        rows = {r.name: r for r in only(await drain(queue), "ProfileRows").rows}
        assert rows["bioinformatics"].memories == 2

    async def test_a_profile_that_will_not_load_is_listed_anyway(self, service):
        # The screen is partly *how* a broken profile gets opened and fixed,
        # and it cannot be opened from a screen that failed to draw.
        Profile.create("bioinformatics")
        Profile.path_for("bioinformatics").write_text("\x00 not a profile")
        queue = subscribe(service)
        await service.handle(ProfileList())
        rows = {r.name: r for r in only(await drain(queue), "ProfileRows").rows}
        assert "bioinformatics" in rows


class TestProfiles:
    async def test_the_working_profile_can_be_switched(self, service):
        Profile.create("bioinformatics")
        queue = subscribe(service)
        await service.handle(ProfileSet(name="bioinformatics"))
        events = await drain(queue)
        assert service._deps.profile == "bioinformatics"
        # `panel.update` is the one event that restates the working profile to
        # a client that connected before the switch; `hello` only greets.
        assert only(events, "PanelUpdate").profile == "bioinformatics"

    async def test_a_profile_that_does_not_exist_is_refused(self, service):
        queue = subscribe(service)
        await service.handle(ProfileSet(name="ghost"))
        assert only(await drain(queue), "Notify").severity == "warning"
        assert service._deps.profile == "default"

    async def test_one_is_created_and_then_copied(self, service):
        await service.handle(ProfileCreate(name="bioinformatics"))
        assert "bioinformatics" in Profile.list_profiles()
        await service.handle(
            ProfileDuplicate(name="rnaseq", source="bioinformatics")
        )
        assert "rnaseq" in Profile.list_profiles()

    async def test_a_name_that_collides_is_refused_in_the_stores_words(
        self, service
    ):
        queue = subscribe(service)
        await service.handle(ProfileCreate(name="default"))
        toast = only(await drain(queue), "Notify")
        assert toast.severity == "warning" and "already exists" in toast.text

    async def test_a_memory_file_edited_in_the_editor_is_written_back(
        self, service
    ):
        await service.handle(
            ProfileSave(
                name="default",
                kind="memories",
                text="## [rag]\n- the cohort lives in /data/cohort\n",
            )
        )
        assert any(
            "/data/cohort" in m.text for m in Profile.load("default").memories
        )

    async def test_deleting_moves_its_sessions_to_the_default(
        self, service, conn
    ):
        Profile.create("bioinformatics")
        moved = SessionStore(conn).create(
            profile="bioinformatics", title="theirs"
        )
        queue = subscribe(service)
        await service.handle(ProfileDelete(name="bioinformatics"))
        rows = {
            r.session_id: r
            for r in only(await drain(queue), "SessionRows").rows
        }
        assert rows[moved.session_id].profile == "default"
        assert "bioinformatics" not in Profile.list_profiles()

    async def test_the_default_profile_cannot_be_deleted(self, service):
        # It is where a deleted profile's sessions land, so removing it would
        # leave them pointing at nothing.
        queue = subscribe(service)
        await service.handle(ProfileDelete(name="default"))
        assert only(await drain(queue), "Notify").severity == "warning"
        assert "default" in Profile.list_profiles()

    async def test_a_profile_with_a_reply_in_progress_is_kept(
        self, service, conn, llm
    ):
        Profile.create("bioinformatics")
        busy = SessionStore(conn).create(
            profile="bioinformatics", title="working"
        )
        release = await park_turn(service, llm, busy.session_id)
        queue = subscribe(service)
        await service.handle(ProfileDelete(name="bioinformatics"))
        toast = only(await drain(queue), "Notify")
        assert toast.severity == "warning" and "reply in progress" in toast.text
        assert "bioinformatics" in Profile.list_profiles()
        release.set()
        await service.stop()

    async def test_a_profile_with_a_live_subprocess_is_kept(
        self, service, conn
    ):
        import os

        Profile.create("bioinformatics")
        owner = SessionStore(conn).create(
            profile="bioinformatics", title="scripted"
        )
        # This process: `running_session_ids` verifies the pid against the OS,
        # so a made-up one would be treated as the stale row it looks like.
        conn.execute(
            "INSERT INTO processes (pid, session_id, name, state) "
            "VALUES (?, ?, ?, 'running')",
            (os.getpid(), owner.session_id, "align.sh"),
        )
        conn.commit()
        queue = subscribe(service)
        await service.handle(ProfileDelete(name="bioinformatics"))
        assert "sub-process" in only(await drain(queue), "Notify").text
        assert "bioinformatics" in Profile.list_profiles()


class TestSkillFiles:
    async def test_a_skill_is_written_into_the_profiles_own_directory(
        self, service
    ):
        await service.handle(
            SkillSave(
                profile="default",
                name="qc-report",
                text="---\nname: qc-report\ndescription: run QC\n---\n\nsteps\n",
            )
        )
        assert [s.name for s in load_own_skills("default")] == ["qc-report"]

    async def test_saving_without_a_body_is_refused(self, service):
        # The shape is shared with `skill.delete`, where a body is meaningless.
        queue = subscribe(service)
        await service.handle(SkillSave(profile="default", name="qc-report"))
        assert only(await drain(queue), "Notify").severity == "warning"
        assert load_own_skills("default") == []

    async def test_one_of_the_profiles_own_is_deleted(self, service):
        write_skill(
            Skill(name="qc-report", description="run QC", triggers=[], body="s"),
            "default",
        )
        await service.handle(SkillDelete(profile="default", name="qc-report"))
        assert load_own_skills("default") == []

    async def test_deleting_one_that_is_not_there_says_so(self, service):
        queue = subscribe(service)
        await service.handle(SkillDelete(profile="default", name="ghost"))
        assert only(await drain(queue), "Notify").severity == "warning"

    async def test_a_global_skill_is_visible_everywhere_but_is_not_own(
        self, service
    ):
        # The level `skill.save` carries decides where the file lands, and
        # "global" is the one a second profile can also see.
        await service.handle(
            SkillSave(
                profile="default",
                name="cluster-etiquette",
                text="---\nname: cluster-etiquette\n---\n\nbe kind\n",
                level="global",
            )
        )
        assert load_own_skills("default") == [], "not the profile's own"
        assert "cluster-etiquette" in [s.name for s in load_skills("default")]
        assert "cluster-etiquette" in [s.name for s in load_skills("other")]

    async def test_a_project_skill_lands_in_the_working_directory(
        self, service, project
    ):
        await service.handle(
            SkillSave(
                profile="default",
                name="run-cohort",
                text="---\nname: run-cohort\n---\n\n1. sbatch\n",
                level="project",
            )
        )
        assert (project / ".hpca" / "skills" / "run-cohort.md").exists()
        assert load_own_skills("default") == []
        assert [s.name for s in load_project_skills(project_root=project)] == [
            "run-cohort"
        ]

    async def test_and_a_project_skill_is_removable(self, service, project):
        # The other half of "a project skill lands in cwd and is removable":
        # `skill.delete` looks in both directories a user owns.
        write_skill(
            Skill(name="run-cohort", description="", triggers=[], body="s"),
            "default",
            level="project",
            project_root=project,
        )
        await service.handle(SkillDelete(profile="default", name="run-cohort"))
        assert load_project_skills(project_root=project) == []

    async def test_a_skill_defaults_to_the_profiles_own_directory(self, service):
        # No level named: where a hand-edited file came from, and where a
        # self-review patch goes.
        await service.handle(
            SkillSave(
                profile="default",
                name="qc",
                text="---\nname: qc\n---\n\nsteps\n",
            )
        )
        assert [s.name for s in load_own_skills("default")] == ["qc"]


class TestEditableBodies:
    """The read half of every editor — `profile.get`, `skill.list`, `skill.get`.

    The invariant they exist for is not "a convenient way to fetch a file": it
    is that the *writes* beside them (`profile.save`, `skill.save`) write
    verbatim, so an editor that could not read the body it is opening over
    would save an empty buffer on top of it. Before these the front-end read
    the app dir itself, which is §4.2 rule 2 and the rule that lets the core
    sit behind a socket.
    """

    async def test_a_memory_file_arrives_exactly_as_it_is_on_disk(
        self, service
    ):
        # Verbatim, not `Profile.load(...).render()`: a line the parser could
        # not read is precisely why someone opens the editor, and a re-rendered
        # body would delete it behind their back.
        raw = (
            "---\nname: default\n---\n\n## [system_prompt]\n"
            "- kept\n((mangled\n"
        )
        path = Profile.path_for("default")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(raw)
        queue = subscribe(service)
        await service.handle(ProfileGet(name="default", kind="memories"))
        body = only(await drain(queue), "ProfileBody")
        assert body.name == "default" and body.kind == "memories"
        assert body.text == raw and body.error == ""

    async def test_an_archive_that_was_never_written_may_still_be_edited(
        self, service
    ):
        # Empty and no error: a fresh profile has no archive, and refusing to
        # open an editor over it would leave no way to write the first line.
        queue = subscribe(service)
        await service.handle(ProfileGet(name="default", kind="archive"))
        body = only(await drain(queue), "ProfileBody")
        assert body.text == "" and body.error == ""

    async def test_an_archive_is_read_back_whole(self, service):
        from hpca.curator import archive_path

        path = archive_path("default")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("- retired: the old cluster\n")
        queue = subscribe(service)
        await service.handle(ProfileGet(name="default", kind="archive"))
        assert "old cluster" in only(await drain(queue), "ProfileBody").text

    async def test_a_profile_that_does_not_exist_answers_with_a_refusal(
        self, service
    ):
        # An error rather than a `notify`: the editor asked for a named file
        # and has to learn it may not open. Empty error is the only permission.
        queue = subscribe(service)
        await service.handle(ProfileGet(name="ghost", kind="memories"))
        body = only(await drain(queue), "ProfileBody")
        assert body.error and body.text == ""

    async def test_what_is_fetched_is_what_a_save_then_writes(self, service):
        # The round trip the whole shape exists for: an editor opens over what
        # `profile.get` handed it and saves the buffer back, so anything the
        # fetch failed to carry is anything the save deletes.
        await service.handle(
            ProfileSave(
                name="default",
                kind="memories",
                text="## [rag]\n- the cohort lives in /data/cohort\n",
            )
        )
        queue = subscribe(service)
        await service.handle(ProfileGet(name="default", kind="memories"))
        body = only(await drain(queue), "ProfileBody")
        assert "/data/cohort" in body.text
        await service.handle(
            ProfileSave(
                name="default",
                kind="memories",
                text=body.text + "- and the reference is /data/ref\n",
            )
        )
        await service.handle(ProfileGet(name="default", kind="memories"))
        text = only(await drain(queue), "ProfileBody").text
        assert "/data/cohort" in text and "/data/ref" in text

    async def test_the_skill_menu_carries_names_and_descriptions(self, service):
        write_skill(
            Skill(name="qc-report", description="run QC", triggers=[], body="s"),
            "default",
        )
        queue = subscribe(service)
        await service.handle(SkillList(profile="default"))
        rows = only(await drain(queue), "SkillRows")
        assert rows.profile == "default"
        assert [(r.name, r.description) for r in rows.skills] == [
            ("qc-report", "run QC")
        ]

    async def test_the_menu_lists_only_what_this_profile_may_edit(
        self, service
    ):
        # A shared skill is not one profile's to change: `skill.save` writes
        # into the profile's own directory, so offering it here would fork it.
        write_skill(
            Skill(
                name="shared-one",
                description="every profile",
                triggers=[],
                body="s",
            ),
            "default",
            level="global",
        )
        write_skill(
            Skill(name="mine", description="just here", triggers=[], body="s"),
            "default",
        )
        queue = subscribe(service)
        await service.handle(SkillList(profile="default"))
        assert [
            r.name for r in only(await drain(queue), "SkillRows").skills
        ] == ["mine"]

    async def test_the_visible_scope_carries_the_shipped_skills_too(
        self, service
    ):
        # The regression the scope exists for: HPCA ships skills, so a menu
        # built from the profile's own reports `/plan` unknown on a fresh
        # install — with no profile skill in sight to explain it.
        queue = subscribe(service)
        await service.handle(SkillList(profile="default", scope="visible"))
        rows = only(await drain(queue), "SkillRows")
        assert rows.scope == "visible", "an answer says which question it took"
        assert "plan" in [r.name for r in rows.skills]
        assert {r.level for r in rows.skills} == {"builtin"}

    async def test_and_says_which_level_each_one_resolved_from(
        self, service, project
    ):
        # The level is what tells a front-end which rows it may offer to
        # remove: the shipped and shared ones are not one profile's to delete.
        write_skill(
            Skill(name="mine", description="", triggers=[], body="s"), "default"
        )
        write_skill(
            Skill(name="ours", description="", triggers=[], body="s"),
            "default",
            level="global",
        )
        write_skill(
            Skill(name="here", description="", triggers=[], body="s"),
            "default",
            level="project",
            project_root=project,
        )
        queue = subscribe(service)
        await service.handle(SkillList(profile="default", scope="visible"))
        rows = only(await drain(queue), "SkillRows")
        levels = {r.name: r.level for r in rows.skills}
        assert levels["mine"] == "profile"
        assert levels["ours"] == "global"
        assert levels["here"] == "project"
        assert levels["plan"] == "builtin"

    async def test_the_own_scope_is_still_only_what_may_be_written(
        self, service, project
    ):
        # Listing and writing have different scopes on purpose: an editor that
        # opens `skill.get` and saves `skill.save` may only be shown files
        # those two commands can reach.
        write_skill(
            Skill(name="mine", description="", triggers=[], body="s"), "default"
        )
        write_skill(
            Skill(name="ours", description="", triggers=[], body="s"),
            "default",
            level="global",
        )
        queue = subscribe(service)
        await service.handle(SkillList(profile="default"))
        rows = only(await drain(queue), "SkillRows")
        assert rows.scope == "own"
        assert [r.name for r in rows.skills] == ["mine"]

    async def test_a_skill_file_arrives_verbatim(self, service):
        raw = "---\nname: qc-report\ndescription: run QC\n---\n\n1. sort\n"
        await service.handle(
            SkillSave(profile="default", name="qc-report", text=raw)
        )
        queue = subscribe(service)
        await service.handle(SkillGet(profile="default", name="qc-report"))
        body = only(await drain(queue), "SkillBody")
        assert body.profile == "default" and body.name == "qc-report"
        assert body.text == raw and body.error == ""

    async def test_a_project_skill_opens_too_because_a_screen_offers_it(
        self, service, project
    ):
        """§9.9: `skill.get` reaches every level the screens call removable.

        `SkillSave` took a level and `delete_profile_skill` learnt about the
        project's, and this stayed own-only — so a skill created at the
        project level was drawn on the profile's skills screen, offered for
        editing, and answered Enter with "has no skill of its own".
        """
        raw = "---\nname: run-cohort\ndescription: the cohort run\n---\n\n1. go\n"
        await service.handle(
            SkillSave(
                profile="default", name="run-cohort", text=raw, level="project"
            )
        )
        queue = subscribe(service)
        await service.handle(SkillGet(profile="default", name="run-cohort"))
        body = only(await drain(queue), "SkillBody")
        assert body.text == raw and body.error == ""

    async def test_but_a_shared_one_is_still_refused(self, service):
        # Not this profile's to edit: a save would land in its own directory
        # and silently fork the file every other profile sees.
        write_skill(
            Skill(name="ours", description="", triggers=[], body="s"),
            "default",
            level="global",
        )
        queue = subscribe(service)
        await service.handle(SkillGet(profile="default", name="ours"))
        assert only(await drain(queue), "SkillBody").error

    async def test_a_skill_the_profile_does_not_own_is_refused(self, service):
        queue = subscribe(service)
        await service.handle(SkillGet(profile="default", name="ghost"))
        body = only(await drain(queue), "SkillBody")
        assert body.error and body.text == ""


class TestSettingsFile:
    """`settings.get` / `settings.save` — the one screen whose subject is the
    file itself, and the only command that can make an edit of it take effect.

    Validation is split deliberately: a front-end can tell whether text is
    JSON, but which fields exist and what they may hold is this model's
    question, so the core re-asks it and refuses in its own words. Applying has
    no other home at all — the old front-end wrote the file and toasted
    "applies on the next start", which was the absence of this command rather
    than a policy.
    """

    async def test_the_file_is_answered_as_it_stands(self, service):
        settings_path().write_text('{"editor": "hx"}')
        queue = subscribe(service)
        await service.handle(SettingsGet())
        body = only(await drain(queue), "SettingsBody")
        assert body.text == '{"editor": "hx"}' and body.error == ""

    async def test_a_file_that_was_never_written_answers_with_the_defaults(
        self, service
    ):
        # What `Settings.load` would use, so the editor opens on a truthful
        # starting point rather than an empty buffer the next save makes real.
        assert not settings_path().exists()
        queue = subscribe(service)
        await service.handle(SettingsGet())
        body = only(await drain(queue), "SettingsBody")
        assert Settings.model_validate_json(body.text).llm.model

    async def test_a_save_writes_the_file_and_restates_what_landed(
        self, service
    ):
        queue = subscribe(service)
        await service.handle(SettingsSave(text='{"editor": "hx"}'))
        body = only(await drain(queue), "SettingsBody")
        # Not the text that was sent: the core writes the validated model back
        # out, so every default the file omitted is now explicit.
        assert body.error == ""
        assert Settings.model_validate_json(body.text).editor == "hx"
        assert Settings.load().editor == "hx"

    async def test_the_running_core_adopts_what_was_saved(self, service):
        settings = service._deps.settings
        await service.handle(SettingsSave(text='{"editor": "hx"}'))
        # The same object, updated in place: whoever built the core passed it
        # in and still reads it, so rebinding would leave them describing a
        # file that no longer exists.
        assert service._deps.settings is settings
        assert settings.editor == "hx"

    async def test_a_shape_the_model_refuses_is_not_written(self, service):
        settings_path().write_text('{"editor": "hx"}\n')
        queue = subscribe(service)
        await service.handle(
            SettingsSave(text='{"database": {"sync_interval_s": -5}}')
        )
        body = only(await drain(queue), "SettingsBody")
        assert body.error.startswith("invalid: database.sync_interval_s")
        # One line, because it is drawn beside an editor that stays on screen.
        assert "\n" not in body.error
        # And the body still describes the file that is still there.
        assert body.text == '{"editor": "hx"}\n'
        assert Settings.load().editor == "hx"

    async def test_text_that_is_not_json_is_refused_in_the_cores_words(
        self, service
    ):
        queue = subscribe(service)
        await service.handle(SettingsSave(text="not json at all"))
        assert only(await drain(queue), "SettingsBody").error
        assert not settings_path().exists()

    async def test_a_changed_llm_section_rebuilds_the_clients(self, service):
        # The half only the core can do: the bootstrap client every un-pinned
        # session talks through was built from the old section.
        rebuilt: list[bool] = []

        async def reload(*, busy=False):
            rebuilt.append(busy)
            return True

        service._backends.reload = reload
        queue = subscribe(service)
        await service.handle(
            SettingsSave(
                text=json.dumps(
                    {"llm": {"base_url": "http://localhost:9/v1", "model": "m"}}
                )
            )
        )
        assert rebuilt == [False]
        # The ★ moved with it, so the catalog is restated.
        assert "LLMCatalog" in kinds(await drain(queue))

    async def test_a_save_that_leaves_the_llm_alone_rebuilds_nothing(
        self, service
    ):
        rebuilt: list[bool] = []

        async def reload(*, busy=False):
            rebuilt.append(busy)
            return True

        service._backends.reload = reload
        await service.handle(SettingsSave(text='{"editor": "hx"}'))
        assert rebuilt == []

    async def test_a_display_change_takes_effect_without_a_restart(
        self, service
    ):
        # The config editor is *in* the app, and a key whose whole subject is
        # what the screen looks like must apply on the frame after the save.
        queue = subscribe(service)
        await drain(queue)
        await service.handle(
            SettingsSave(
                text=json.dumps(
                    {"display": {"chat_stamps": False, "decision_pulse_seconds": 4}}
                )
            )
        )
        changed = only(await drain(queue), "DisplayChanged")
        assert changed.display.chat_stamps is False
        assert changed.display.decision_pulse_seconds == 4

    async def test_a_save_that_leaves_the_display_alone_restates_nothing(
        self, service
    ):
        # Restating it repaints every conversation's chat rows, which is not
        # the price of an edited log level.
        queue = subscribe(service)
        await drain(queue)
        await service.handle(SettingsSave(text='{"editor": "hx"}'))
        assert "DisplayChanged" not in kinds(await drain(queue))

    async def test_database_settings_are_said_to_wait_for_a_restart(
        self, service
    ):
        # The one section that genuinely cannot be applied: the databases are
        # open and every service holds a connection. Named, rather than the old
        # blanket toast that claimed it of the LLM as well.
        queue = subscribe(service)
        await service.handle(
            SettingsSave(text=json.dumps({"database": {"local_cache": False}}))
        )
        toast = only(await drain(queue), "Notify")
        assert toast.severity == "warning" and "next start" in toast.text


class TestBackendDiscovery:
    """The three commands the manage-LLMs screen and the connection form need:
    `backend.probe`, `backend.scan`, `backend.remove` — plus the by-label
    `backend.set` that lets a key-locked entry be pinned without its key ever
    crossing the wire.
    """

    def transport(self, routes):
        """An httpx transport routing on port; absent ports look unreachable."""
        import httpx

        def handle(request):
            answer = routes.get(request.url.port)
            if answer is None:
                raise httpx.ConnectError("refused", request=request)
            return answer(request)

        return httpx.MockTransport(handle)

    def serves(self, *models):
        import httpx

        body = {
            "object": "list",
            "data": [
                {"id": m, "object": "model", "max_model_len": 4096}
                for m in models
            ],
        }
        return lambda request: httpx.Response(200, json=body)

    def scanner(self, hits, *, wait=None):
        async def scan(ports, *, progress=None, on_found=None, api_keys=()):
            if wait is not None:
                wait.wait(5)
            for hit in hits:
                if on_found is not None:
                    on_found(hit)
            return list(hits)

        return scan

    def hit(self, port, model):
        from hpca.discover import DiscoveredBackend

        return DiscoveredBackend(
            base_url=f"http://127.0.0.1:{port}/v1", model=model, max_model_len=4096
        )

    async def test_an_endpoint_is_asked_what_it_serves(self, service):
        service._backends._probe_transport = self.transport(
            {20001: self.serves("qwen-a", "qwen-b")}
        )
        queue = subscribe(service)
        await service.handle(BackendProbe(base_url="http://localhost:20001/v1"))
        probed = only(await wait_for(queue, "BackendProbed"), "BackendProbed")
        assert probed.base_url == "http://localhost:20001/v1"
        # Several models is the picker case; one would auto-fill the form.
        assert [e.model for e in probed.models] == ["qwen-a", "qwen-b"]
        assert probed.needs_key is False

    async def test_an_endpoint_that_is_not_there_answers_all_the_same(
        self, service
    ):
        # The form is waiting on this frame; a silence would leave it checking
        # forever.
        service._backends._probe_transport = self.transport({})
        queue = subscribe(service)
        await service.handle(BackendProbe(base_url="http://localhost:20001/v1"))
        probed = only(await wait_for(queue, "BackendProbed"), "BackendProbed")
        assert probed.models == [] and probed.needs_key is False

    async def test_a_scan_fills_the_catalog_as_it_goes_and_then_closes(
        self, service
    ):
        service._backends._port_scanner = self.scanner(
            [self.hit(20001, "qwen-x"), self.hit(20002, "qwen-y")]
        )
        queue = subscribe(service)
        await service.handle(BackendScan())
        events = await wait_for(queue, "BackendScanned")
        catalogs = [e for e in events if type(e).__name__ == "LLMCatalog"]
        # One frame per hit, so the panel fills while the sweep is running —
        # not one frame at the end, which is what looks hung.
        assert [len(c.entries) for c in catalogs] == [0, 1, 2, 2]
        assert all(e.discovered for e in catalogs[-1].entries)
        scanned = only(events, "BackendScanned")
        assert scanned.found == 2 and scanned.cluster == 0
        # Something was found, so there is nothing to explain.
        assert scanned.notice == "" and scanned.help == ""

    async def test_an_empty_scan_off_the_cluster_carries_the_tunnel_recipe(
        self, service
    ):
        # The verdict is the part a front-end cannot reach: "nothing found"
        # means something different depending on whether anything configured
        # still answers, and that is the core's probe.
        service._backends._port_scanner = self.scanner([])
        queue = subscribe(service)
        await service.handle(BackendScan())
        scanned = only(await wait_for(queue, "BackendScanned"), "BackendScanned")
        assert scanned.found == 0
        assert "ssh -fN" in scanned.help and scanned.notice == ""

    async def test_a_slow_scan_does_not_stop_the_core_answering(self, service):
        # The sweep is tens of thousands of ports. Awaited on the dispatch
        # loop it would mean no session could speak until it finished.
        import threading

        release = threading.Event()
        service._backends._port_scanner = self.scanner([], wait=release)
        queue = subscribe(service)
        await service.handle(BackendScan())
        await service.handle(SessionList())
        assert "SessionRows" in kinds(await drain(queue))
        release.set()
        await wait_for(queue, "BackendScanned")

    async def test_a_configured_entry_is_removed_by_its_label(self, service):
        service._deps.settings.backends = [
            LLMBackend(model="qwen-a", base_url="http://a/v1"),
            LLMBackend(model="qwen-b", base_url="http://b/v1"),
        ]
        queue = subscribe(service)
        await service.handle(BackendRemove(label="qwen-a"))
        events = await drain(queue)
        assert only(events, "Notify").text == "Removed qwen-a"
        assert [e.label for e in only(events, "LLMCatalog").entries] == ["qwen-b"]
        assert [b.model for b in Settings.load().backends] == ["qwen-b"]

    async def test_removing_something_that_is_not_configured_is_inert(
        self, service
    ):
        # A discovered row is not in the file — which is why the old screen's
        # `r` did nothing on that panel.
        service._backends._port_scanner = self.scanner([self.hit(20001, "qwen-x")])
        queue = subscribe(service)
        await service.handle(BackendScan())
        await wait_for(queue, "BackendScanned")
        await service.handle(BackendRemove(label="qwen-x"))
        events = await drain(queue)
        assert [e.severity for e in events if isinstance(e, Notify)] == ["warning"]

    async def test_and_says_so_with_the_catalog_the_screen_must_redraw(
        self, service
    ):
        """A refused remove restates the catalog (specs-ui-coverage.md §9.1).

        The screen takes the row off itself when it sends `backend.remove` —
        it cannot wait a round trip to stop drawing a row the user just
        deleted — so a refusal that emitted only a warning left the entry
        gone from the screen, present in the file, and the toast talking
        about a row that was no longer there.
        """
        service._deps.settings.backends = [
            LLMBackend(model="qwen-a", base_url="http://a/v1")
        ]
        queue = subscribe(service)
        await service.handle(BackendRemove(label="not-configured"))
        events = await drain(queue)
        assert [e.label for e in only(events, "LLMCatalog").entries] == ["qwen-a"]
        assert [b.model for b in service._deps.settings.backends] == ["qwen-a"]

    async def test_a_session_is_pinned_by_label(self, service, session):
        # The form ctrl+l needs: the catalog carries no api_key, so an entry
        # sent back by value could not name a key-locked backend at all.
        service._deps.settings.backends = [
            LLMBackend(model="qwen-a", base_url="http://a/v1", api_key="sk-secret")
        ]
        queue = subscribe(service)
        await service.handle(
            BackendSet(label="qwen-a", session_id=session.session_id)
        )
        rows = {r.session_id: r for r in only(await drain(queue), "SessionRows").rows}
        assert rows[session.session_id].model == "qwen-a"
        stored = json.loads(SessionStore(service._deps.conn).get(
            session.session_id
        ).backend)
        # The key stayed where it already was.
        assert stored["api_key"] == "sk-secret"

    async def test_a_scan_hit_can_be_added_to_the_catalog_by_label(self, service):
        service._backends._port_scanner = self.scanner([self.hit(20001, "qwen-x")])
        queue = subscribe(service)
        await service.handle(BackendScan())
        await wait_for(queue, "BackendScanned")
        await service.handle(BackendSet(label="qwen-x"))
        assert [b.model for b in Settings.load().backends] == ["qwen-x"]

    async def test_a_label_nothing_answers_to_is_refused(self, service, session):
        queue = subscribe(service)
        await service.handle(
            BackendSet(label="ghost", session_id=session.session_id)
        )
        assert only(await drain(queue), "Notify").severity == "warning"


class TestWatchBoxes:
    async def test_peeking_a_log_answers_with_its_tail(
        self, service, session, conn, home
    ):
        log = home / "train.log"
        log.write_text("epoch 4/10\nloss 0.31\n")
        watch = WatchStore(conn).add(
            kind=KIND_LOG, target=str(log), label="train.log",
            session_id=session.session_id,
        )
        queue = subscribe(service)
        await service.handle(WatchPeek(watch_id=watch.id))
        peeked = only(await drain(queue), "WatchPeeked")
        # Named, because two peeks can cross and a bare string of text could
        # then be attributed to the wrong box.
        assert peeked.watch_id == watch.id and peeked.title == "train.log"
        assert "loss 0.31" in peeked.text

    async def test_a_log_that_cannot_be_read_answers_in_the_text(
        self, service, session, conn, home
    ):
        # What happened to the log is exactly what the user asked; it belongs
        # where the tail would have been, not in an error.
        watch = WatchStore(conn).add(
            kind=KIND_LOG, target=str(home / "never-written.log"),
            session_id=session.session_id,
        )
        queue = subscribe(service)
        await service.handle(WatchPeek(watch_id=watch.id))
        assert "could not read" in only(await drain(queue), "WatchPeeked").text

    async def test_peeking_a_job_answers_with_its_state_and_output(
        self, service, session, conn, home
    ):
        out = home / "slurm-42.out"
        out.write_text("srun: step 1 done\n")
        JobStore(conn).add(
            job_id="42", kind="sbatch", session_id=session.session_id,
            profile="default", script_key="align", stdout_path=str(out),
            stderr_path="",
        )
        watch = WatchStore(conn).add(
            kind=KIND_JOB, target="42", label="job 42",
            session_id=session.session_id,
        )
        WatchStore(conn).update(watch.id, state="RUNNING")
        queue = subscribe(service)
        await service.handle(WatchPeek(watch_id=watch.id))
        text = only(await drain(queue), "WatchPeeked").text
        assert "RUNNING" in text and "step 1 done" in text

    async def test_the_peek_ships_as_much_tail_as_the_settings_say(
        self, service, session, conn, home
    ):
        # The size is `watches.peek_chars` and the core is what applies it:
        # the file is on a node the front-end may not share, so trimming it
        # afterwards is not something the UI could do (§4.2 rule 2).
        log = home / "train.log"
        log.write_text("\n".join(f"epoch {i}" for i in range(500)))
        watch = WatchStore(conn).add(
            kind=KIND_LOG, target=str(log), session_id=session.session_id,
        )
        service._deps.settings.watches.peek_chars = 40
        queue = subscribe(service)
        await service.handle(WatchPeek(watch_id=watch.id))
        text = only(await drain(queue), "WatchPeeked").text
        assert len(text) <= 41  # the tail, plus the "…" that says it is one
        assert text.endswith("epoch 499")

    async def test_and_a_bigger_setting_ships_more_of_it(
        self, service, session, conn, home
    ):
        # The whole point of raising the default: a traceback whose last line
        # names the exception must not be cut off above it.
        log = home / "train.log"
        log.write_text("boom\n" * 200 + "ValueError: the cohort is empty")
        watch = WatchStore(conn).add(
            kind=KIND_LOG, target=str(log), session_id=session.session_id,
        )
        service._deps.settings.watches.peek_chars = 300
        queue = subscribe(service)
        await service.handle(WatchPeek(watch_id=watch.id))
        short = only(await drain(queue), "WatchPeeked").text
        service._deps.settings.watches.peek_chars = 2000
        queue = subscribe(service)
        await service.handle(WatchPeek(watch_id=watch.id))
        long = only(await drain(queue), "WatchPeeked").text
        assert len(long) > len(short)
        # Both keep the end; what the bigger figure buys is what led up to it.
        assert short.endswith("the cohort is empty")
        assert long.endswith("the cohort is empty")

    async def test_a_configured_peek_survives_a_settings_save(
        self, service, session, conn, home
    ):
        # Read per peek rather than captured at startup, because `settings.save`
        # swaps the running section in place.
        log = home / "train.log"
        log.write_text("\n".join(f"epoch {i}" for i in range(500)))
        watch = WatchStore(conn).add(
            kind=KIND_LOG, target=str(log), session_id=session.session_id,
        )
        await service.handle(
            SettingsSave(text=json.dumps({"watches": {"peek_chars": 30}}))
        )
        queue = subscribe(service)
        await service.handle(WatchPeek(watch_id=watch.id))
        assert len(only(await drain(queue), "WatchPeeked").text) <= 31

    async def test_peeking_a_box_that_is_gone_says_so(self, service):
        queue = subscribe(service)
        await service.handle(WatchPeek(watch_id=404))
        assert only(await drain(queue), "Notify").severity == "warning"

    async def test_dropping_removes_the_box_and_repaints_the_column(
        self, service, session, conn, home
    ):
        service._deps.focused_session_id = session.session_id
        watch = WatchStore(conn).add(
            kind=KIND_LOG, target=str(home / "train.log"), label="train.log",
            session_id=session.session_id,
        )
        queue = subscribe(service)
        await service.handle(WatchDrop(watch_id=watch.id))
        events = await drain(queue)
        assert WatchStore(conn).get(watch.id) is None
        assert only(events, "PanelUpdate").rows == []

    async def test_dropping_a_box_that_is_gone_says_so(self, service):
        queue = subscribe(service)
        await service.handle(WatchDrop(watch_id=404))
        assert only(await drain(queue), "Notify").severity == "warning"

    def column(self, service, session, conn, home, *names):
        """A column of watch boxes, top to bottom, on the focused session."""
        service._deps.focused_session_id = session.session_id
        store = WatchStore(conn)
        return [
            store.add(
                kind=KIND_LOG,
                target=str(home / f"{name}.log"),
                label=name,
                session_id=session.session_id,
            )
            for name in names
        ]

    async def test_moving_a_box_rearranges_the_column_and_repaints_it(
        self, service, session, conn, home
    ):
        _, b, _ = self.column(service, session, conn, home, "a", "b", "c")
        queue = subscribe(service)
        await service.handle(WatchMove(watch_id=b.id, delta=-1))
        rows = only(await drain(queue), "PanelUpdate").rows
        assert [r.title for r in rows] == ["b", "a", "c"]
        # The store is where it has to have landed: the panel is repainted
        # from that every two seconds, and read back from it after a restart.
        assert [
            w.label for w in WatchStore(conn).list(session_id=session.session_id)
        ] == ["b", "a", "c"]

    async def test_a_box_at_the_end_still_gets_the_column_back(
        self, service, session, conn, home
    ):
        # Nothing moved, and the frame is sent anyway — the front-end may have
        # moved the box itself while it waited (`ui.pane.Pane.reorder` does),
        # and this is what puts it back.
        a, _ = self.column(service, session, conn, home, "a", "b")
        queue = subscribe(service)
        await service.handle(WatchMove(watch_id=a.id, delta=-1))
        rows = only(await drain(queue), "PanelUpdate").rows
        assert [r.title for r in rows] == ["a", "b"]

    async def test_moving_a_box_that_is_gone_says_nothing(
        self, service, session, conn, home
    ):
        # The column repaints on a timer, so the box under the cursor can be
        # dropped between the keypress and the write. Not worth a warning: the
        # repaint that comes back already tells the user what is there.
        self.column(service, session, conn, home, "a")
        queue = subscribe(service)
        await service.handle(WatchMove(watch_id=404, delta=-1))
        events = await drain(queue)
        assert "Notify" not in kinds(events)
        assert [r.title for r in only(events, "PanelUpdate").rows] == ["a"]

    async def test_one_sessions_column_cannot_disturb_anothers(
        self, service, session, conn, home
    ):
        # Each session has its own column and only one is ever on screen; a
        # store-wide swap could put a box next to one from a conversation the
        # user is not even looking at.
        other = SessionStore(conn).create(profile="default", title="elsewhere")
        self.column(service, session, conn, home, "a", "b")
        store = WatchStore(conn)
        for name in ("x", "y"):
            store.add(
                kind=KIND_LOG,
                target=str(home / f"{name}.log"),
                label=name,
                session_id=other.session_id,
            )
        mine = store.list(session_id=session.session_id)[1]
        await service.handle(WatchMove(watch_id=mine.id, delta=-1))
        assert [w.label for w in store.list(session_id=session.session_id)] == [
            "b",
            "a",
        ]
        assert [w.label for w in store.list(session_id=other.session_id)] == [
            "x",
            "y",
        ]


class TestRunningWork:
    """`process.kill` and `job.cancel` — stopping what the agent started."""

    def _row(self, conn, session_id, pid, state="running"):
        conn.execute(
            "INSERT INTO processes (pid, session_id, name, state) "
            "VALUES (?, ?, 'align.sh', ?)",
            (pid, session_id, state),
        )
        conn.commit()

    async def test_a_process_no_runner_owns_is_killed_and_its_row_settled(
        self, service, session, conn
    ):
        # A background script from an earlier turn outlives the runner that
        # started it, so there is no monitor left to notice the signal — and
        # without settling the row the history would claim it runs forever.
        self._row(conn, session.session_id, 999_999)
        queue = subscribe(service)
        await service.handle(ProcessKill(pid=999_999))
        assert only(await drain(queue), "Notify").severity == "information"
        state = conn.execute(
            "SELECT state FROM processes WHERE pid = 999999"
        ).fetchone()["state"]
        assert state == "killed"

    async def test_the_runner_that_started_it_is_preferred(
        self, service, session, conn
    ):
        # Its monitor is what records how the process ended, so a kill it can
        # see settles the row properly instead of racing an UPDATE with it.
        class FakeRunner:
            def __init__(self):
                self.killed = []

            def owns(self, pid):
                return True

            async def kill(self, pid):
                self.killed.append(pid)

            async def wait(self, pid):
                return None

        class FakeCtx:
            pass

        ctx, runner = FakeCtx(), FakeRunner()
        ctx.runner = runner
        service._scheduler.tool_context = lambda session_id: ctx
        self._row(conn, session.session_id, 999_998)
        await service.handle(ProcessKill(pid=999_998))
        assert runner.killed == [999_998]

    async def test_a_pid_with_no_row_says_so(self, service):
        queue = subscribe(service)
        await service.handle(ProcessKill(pid=999_997))
        assert only(await drain(queue), "Notify").severity == "warning"

    async def test_a_process_that_already_ended_is_left_alone(
        self, service, session, conn
    ):
        self._row(conn, session.session_id, 999_996, state="exited")
        queue = subscribe(service)
        await service.handle(ProcessKill(pid=999_996))
        toast = only(await drain(queue), "Notify")
        assert toast.severity == "warning" and "exited" in toast.text

    async def test_a_job_is_cancelled_and_marked_provisionally(
        self, cluster_service, conn, slurm, session
    ):
        # scancel returns before the scheduler has acted; sacct confirms it on
        # the next poll, and until then CANCELLING is the honest answer.
        JobStore(conn).add(
            job_id="42", kind="sbatch", session_id=session.session_id,
            profile="default", script_key="align", stdout_path="",
            stderr_path="",
        )
        queue = subscribe(cluster_service)
        await cluster_service.handle(JobCancel(job_id="42"))
        assert slurm.cancelled == ["42"]
        assert JobStore(conn).get("42").state == "CANCELLING"
        assert "Notify" in kinds(await drain(queue))

    async def test_a_failed_cancel_leaves_the_state_alone(
        self, home, conn, llm, session
    ):
        async def db(fn):
            return fn(conn)

        service = build_service(
            settings=Settings.load(), app_dir=home, db=db, conn=conn,
            checkpointer=InMemorySaver(), llm=llm,
            slurm=FakeSlurm(error=RuntimeError("scancel failed: no such job")),
        )
        JobStore(conn).add(
            job_id="42", kind="sbatch", session_id=session.session_id,
            profile="default", script_key="align", stdout_path="",
            stderr_path="",
        )
        queue = subscribe(service)
        await service.handle(JobCancel(job_id="42"))
        assert only(await drain(queue), "Notify").severity == "error"
        assert JobStore(conn).get("42").state == "SUBMITTED"

    async def test_cancelling_without_a_cluster_says_so(self, service):
        queue = subscribe(service)
        await service.handle(JobCancel(job_id="42"))
        assert only(await drain(queue), "Notify").severity == "warning"


def proposals(*texts, scope="rag"):
    return json.dumps(
        {
            "proposals": [
                {"scope": scope, "kind": "fact", "text": text} for text in texts
            ]
        }
    )


class TestMemoryReview:
    """`/memorize`, `/conclude` and the `memory.resolve` that answers them."""

    async def test_a_note_becomes_proposals_and_only_the_approved_are_written(
        self, service, session, llm
    ):
        await run_turn(service, session.session_id, "how many reads?")
        llm._outputs = [proposals("the cohort is in /data/cohort", "not this")]
        queue = subscribe(service)
        await service.handle(
            CommandRun(
                name="memorize",
                args="where the cohort lives",
                session_id=session.session_id,
            )
        )
        offer = only(await wait_for(queue, "MemoryProposals"), "MemoryProposals")
        assert [p.text for p in offer.proposals] == [
            "the cohort is in /data/cohort",
            "not this",
        ]
        # Positional, and a short answer rejects the rest: an answer that never
        # arrived is not an approval.
        await service.handle(
            MemoryResolve(session_id=session.session_id, approved=[True])
        )
        kept = [m.text for m in Profile.load("default").memories]
        assert "the cohort is in /data/cohort" in kept
        assert "not this" not in kept

    async def test_the_answer_cannot_carry_a_memory_of_its_own(self):
        # The authoritative objects never leave the core, so there is nowhere
        # in the answer for an edited memory to ride back in.
        assert set(MemoryResolve.model_fields) == {"session_id", "approved"}

    async def test_memorize_without_a_note_says_what_it_wants(
        self, service, session
    ):
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="memorize", session_id=session.session_id)
        )
        toast = only(await drain(queue), "Notify")
        assert toast.severity == "warning" and "Usage" in toast.text

    async def test_an_answer_to_nothing_is_a_stale_screen_not_an_error(
        self, service, session
    ):
        queue = subscribe(service)
        await service.handle(
            MemoryResolve(session_id=session.session_id, approved=[True])
        )
        assert only(await drain(queue), "Notify").severity == "warning"

    async def test_conclude_reviews_the_conversation(
        self, service, session, llm
    ):
        await run_turn(service, session.session_id, "how many reads?")
        llm._outputs = [
            json.dumps(
                {
                    "proposals": [
                        {"kind": "memory", "text": "samtools is at /opt/bin",
                         "scope": "rag"}
                    ]
                }
            )
        ]
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="conclude", session_id=session.session_id)
        )
        offer = only(await wait_for(queue, "MemoryProposals"), "MemoryProposals")
        assert offer.proposals[0].text == "samtools is at /opt/bin"

    async def test_the_flagged_batch_is_offered_once_the_review_is_answered(
        self, service, session, llm
    ):
        # Two rounds of one pass: a session holds one unanswered set at a time,
        # so a second offer alongside the first would overwrite it.
        await run_turn(service, session.session_id, "how many reads?")
        service._memory.queue_edits(
            session.session_id,
            [MemoryOp(op="add", scope=MemoryScope.RAG, text="flagged fact")],
        )
        llm._outputs = [
            json.dumps(
                {
                    "proposals": [
                        {"kind": "memory", "text": "reviewed fact", "scope": "rag"}
                    ]
                }
            )
        ]
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="conclude", session_id=session.session_id)
        )
        await wait_for(queue, "MemoryProposals")
        await service.handle(
            MemoryResolve(session_id=session.session_id, approved=[True])
        )
        second = only(await drain(queue), "MemoryProposals")
        assert "flagged fact" in second.proposals[0].text
        await service.handle(
            MemoryResolve(session_id=session.session_id, approved=[True])
        )
        kept = [m.text for m in Profile.load("default").memories]
        assert {"reviewed fact", "flagged fact"} <= set(kept)

    async def test_the_memory_tool_reaches_the_queue_conclude_reads(
        self, service, session, llm
    ):
        """The `memory` tool's one wire: `ToolContext.queue_memory_edits`.

        The three tests above prime the queue in Python, which is exactly why
        none of them noticed that no turn could reach it — the tool answered
        "Memory flagging is not available in this context" while the system
        prompt went on telling the model to use it. This one goes the whole
        way: the model flags a fact mid-turn, and `/conclude` offers that fact.
        """
        llm._outputs = [
            calling(
                "memory",
                operations=[
                    {"op": "add", "scope": "rag", "text": "scratch is /work"}
                ],
            ),
            respond("noted"),
        ]
        queue = subscribe(service)
        await service.handle(
            TurnSubmit(session_id=session.session_id, text="remember that")
        )
        events = await wait_for(queue, "TurnFinished")
        results = [
            part.result
            for event in events
            for entry in [getattr(event, "entry", None)]
            if entry is not None
            for part in entry.parts
            if part.tool == "memory" and part.done
        ]
        assert results and all("Queued 1 memory change" in r for r in results)

        # And the fact is there to be offered when the user asks for it.
        llm._outputs = [json.dumps({"proposals": []})]
        await service.handle(
            CommandRun(name="conclude", session_id=session.session_id)
        )
        offer = only(await wait_for(queue, "MemoryProposals"), "MemoryProposals")
        assert "scratch is /work" in offer.proposals[0].text

    async def test_a_conversation_with_nothing_in_it_is_not_reviewed(
        self, service, session, llm
    ):
        # Cheaper than a generation that will propose nothing.
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="conclude", session_id=session.session_id)
        )
        toast = only(await wait_for(queue, "Notify"), "Notify")
        assert toast.severity == "warning" and "Nothing to conclude" in toast.text


class TestCompact:
    """`/compact`: write the fold, offer it, and — deliberately — do not reset
    the chat.

    The summary is the one thing this core produces that cannot be undone from
    the front-end, so it lands only on `compact.resolve` with `accept`. Every
    test here that folds says so explicitly.
    """

    async def accept(self, service, session_id):
        """The half of the exchange the user does."""
        await service.handle(
            CompactResolve(session_id=session_id, action="accept")
        )

    async def test_the_summary_is_offered_and_nothing_is_folded_yet(
        self, service, session, llm
    ):
        await run_turn(service, session.session_id, "how many reads?")
        llm._outputs = ["we counted the reads in the cohort"]
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="compact", session_id=session.session_id)
        )
        offer = only(await wait_for(queue, "CompactProposed"), "CompactProposed")
        assert "we counted the reads in the cohort" in offer.summary
        assert offer.folded > 0 and offer.attempt == 1
        # The point of the whole exchange: the thread is untouched until the
        # answer comes back.
        values = await service._thread_values(session.session_id)
        assert not values.get("compacted")

    async def test_accepting_folds_the_thread_and_says_so(
        self, service, session, llm
    ):
        await run_turn(service, session.session_id, "how many reads?")
        llm._outputs = ["we counted the reads in the cohort"]
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="compact", session_id=session.session_id)
        )
        await wait_for(queue, "CompactProposed")
        await self.accept(service, session.session_id)
        toast = only(await wait_for(queue, "Notify"), "Notify")
        assert "Context compacted" in toast.text
        values = await service._thread_values(session.session_id)
        assert values["compacted"]["upto"] > 0
        assert "we counted the reads" in values["compacted"]["summary"]["content"]

    async def test_discarding_leaves_the_conversation_alone(
        self, service, session, llm
    ):
        await run_turn(service, session.session_id, "how many reads?")
        llm._outputs = ["a summary"]
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="compact", session_id=session.session_id)
        )
        await wait_for(queue, "CompactProposed")
        await service.handle(
            CompactResolve(session_id=session.session_id, action="discard")
        )
        assert "discarded" in only(await wait_for(queue, "Notify"), "Notify").text
        values = await service._thread_values(session.session_id)
        assert not values.get("compacted")

    async def test_a_retry_carries_the_comment_and_the_rejected_text(
        self, service, session, llm
    ):
        await run_turn(service, session.session_id, "how many reads?")
        llm._outputs = ["the cohort was counted at"]
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="compact", session_id=session.session_id)
        )
        await wait_for(queue, "CompactProposed")
        llm._outputs = ["the cohort was counted at 40 samples"]
        await service.handle(
            CompactResolve(
                session_id=session.session_id,
                action="retry",
                comment="you cut it off — finish the sentence",
            )
        )
        again = only(await wait_for(queue, "CompactProposed"), "CompactProposed")
        assert again.attempt == 2
        assert "40 samples" in again.summary
        system = llm.prompts[-1][0]["content"]
        assert "you cut it off" in system
        assert "the cohort was counted at" in system  # what it is fixing
        # And still nothing is folded: a retry is not an acceptance.
        values = await service._thread_values(session.session_id)
        assert not values.get("compacted")

    async def test_the_offer_survives_a_review_nobody_answered(
        self, service, session, llm
    ):
        """Escape closes the screen and answers nothing, so `/compact` brings
        the same summary back rather than paying for a second one."""
        await run_turn(service, session.session_id, "how many reads?")
        llm._outputs = ["a summary"]
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="compact", session_id=session.session_id)
        )
        first = only(await wait_for(queue, "CompactProposed"), "CompactProposed")
        calls = len(llm.prompts)
        await service.handle(
            CommandRun(name="compact", session_id=session.session_id)
        )
        again = only(await wait_for(queue, "CompactProposed"), "CompactProposed")
        assert again.summary == first.summary
        assert len(llm.prompts) == calls  # no second generation

    async def test_a_new_instruction_is_a_new_summary(
        self, service, session, llm
    ):
        await run_turn(service, session.session_id, "how many reads?")
        llm._outputs = ["a summary"]
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="compact", session_id=session.session_id)
        )
        await wait_for(queue, "CompactProposed")
        llm._outputs = ["a summary that keeps the QC findings"]
        await service.handle(
            CommandRun(
                name="compact",
                args="keep the QC findings",
                session_id=session.session_id,
            )
        )
        again = only(await wait_for(queue, "CompactProposed"), "CompactProposed")
        assert "QC findings" in again.summary
        assert again.guidance == "keep the QC findings"

    async def test_a_summary_the_thread_outgrew_is_refused(
        self, service, session, llm
    ):
        """The review is on screen for as long as the user reads it, and the
        thread can be rewound behind it. Nothing lands, which is the property
        the whole exchange is for."""
        from hpca.agent.graph import rollback_thread

        await run_turn(service, session.session_id, "how many reads?")
        llm._outputs = ["a summary"]
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="compact", session_id=session.session_id)
        )
        await wait_for(queue, "CompactProposed")
        await rollback_thread(
            service._graph, session_id=session.session_id, keep=0
        )
        await self.accept(service, session.session_id)
        toast = only(await wait_for(queue, "Notify"), "Notify")
        assert toast.severity == "warning" and "no longer describes" in toast.text
        values = await service._thread_values(session.session_id)
        assert not values.get("compacted")

    async def test_an_answer_to_an_offer_that_is_gone_does_nothing(
        self, service, session
    ):
        # Two front-ends on one core, or a key pressed as the screen closed.
        await service.handle(
            CompactResolve(session_id=session.session_id, action="accept")
        )
        values = await service._thread_values(session.session_id)
        assert not values.get("compacted")

    async def test_the_chat_is_not_re_stated(self, service, session, llm):
        # The rollback next to it removes messages, so only a reset can
        # un-draw their rows. A fold writes a *view*: the stored history is
        # untouched and every row on screen still names a message the thread
        # has, so a reset here would be the per-turn rebuild §4.2 deletes.
        await run_turn(service, session.session_id, "how many reads?")
        llm._outputs = ["a summary"]
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="compact", session_id=session.session_id)
        )
        await wait_for(queue, "CompactProposed")
        await self.accept(service, session.session_id)
        events = await wait_for(queue, "Notify")
        assert "ChatReset" not in kinds(events)
        # The fill is restated, though: the measured count described the
        # unfolded prompt.
        assert "ContextEstimate" in kinds(events)

    async def test_the_instruction_after_the_command_steers_the_summary(
        self, service, session, llm
    ):
        await run_turn(service, session.session_id, "how many reads?")
        llm._outputs = ["a summary"]
        queue = subscribe(service)
        await service.handle(
            CommandRun(
                name="compact",
                args="keep the QC findings",
                session_id=session.session_id,
            )
        )
        offer = only(await wait_for(queue, "CompactProposed"), "CompactProposed")
        assert offer.guidance == "keep the QC findings"
        assert "keep the QC findings" in llm.prompts[-1][0]["content"]
        await self.accept(service, session.session_id)
        toast = only(await wait_for(queue, "Notify"), "Notify")
        assert "keep the QC findings" in toast.text

    async def test_a_session_parked_on_an_approval_is_not_folded(
        self, service, session, llm
    ):
        # Rewriting the thread's state under an unanswered decision is not
        # something to do quietly. (The gate M5a left in place.)
        await run_turn(service, session.session_id, "how many reads?")
        service._scheduler._decisions[session.session_id] = {"tool": "run_bash"}
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="compact", session_id=session.session_id)
        )
        toast = only(await wait_for(queue, "Notify"), "Notify")
        assert toast.severity == "warning" and "approval" in toast.text
        values = await service._thread_values(session.session_id)
        assert not values.get("compacted")

    async def test_an_empty_conversation_has_nothing_to_fold(
        self, service, session
    ):
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="compact", session_id=session.session_id)
        )
        assert "Nothing new to compact" in only(
            await wait_for(queue, "Notify"), "Notify"
        ).text

    async def test_a_fold_that_fails_leaves_the_thread_as_it_was(
        self, service, session, llm
    ):
        await run_turn(service, session.session_id, "how many reads?")

        async def broken(messages, **kwargs):
            raise RuntimeError("the backend went away")

        llm.chat = broken
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="compact", session_id=session.session_id)
        )
        assert only(await wait_for(queue, "Notify"), "Notify").severity == "error"
        values = await service._thread_values(session.session_id)
        assert not values.get("compacted")

    async def test_it_does_not_hold_up_the_next_command(
        self, service, session, llm
    ):
        # A summary is a generation; a dispatch that awaited it would stall
        # every command queued behind it on the same socket.
        await run_turn(service, session.session_id, "hello")
        import asyncio

        release = asyncio.Event()
        answer = llm.chat

        async def gated(messages, **kwargs):
            await release.wait()
            return await answer(messages, **kwargs)

        llm.chat = gated
        await service.handle(
            CommandRun(name="compact", session_id=session.session_id)
        )
        queue = subscribe(service)
        await service.handle(SessionList())
        assert "SessionRows" in kinds(await drain(queue))
        release.set()
        await service.stop()


class TestTheOtherSlashCommands:
    async def test_thinking_with_a_level_sets_it(self, service, session, conn):
        # `/thinking low` is `thinking.set` typed instead of picked, and goes
        # through the same handler so the two cannot drift.
        await service.handle(
            CommandRun(name="thinking", args="low", session_id=session.session_id)
        )
        assert SessionStore(conn).get(session.session_id).thinking == "low"

    async def test_thinking_without_one_says_what_there_is(
        self, service, session
    ):
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="thinking", session_id=session.session_id)
        )
        toast = only(await drain(queue), "Notify")
        assert toast.title == "Thinking effort"
        for level in ("off", "low", "medium", "xhigh"):
            assert level in toast.text

    async def test_skills_list_names_every_level_it_can_see(
        self, service, session
    ):
        write_skill(
            Skill(name="qc-report", description="run QC", triggers=[], body="s"),
            "default",
        )
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="skills-list", session_id=session.session_id)
        )
        toast = only(await drain(queue), "Notify")
        assert "Skills · profile “default”" == toast.title
        assert "qc-report" in toast.text and "run QC" in toast.text
        # The built-ins ship with hpca and are marked as not the profile's own.
        assert "(built-in)" in toast.text

    async def test_skills_list_answers_for_the_sessions_own_profile(
        self, service, conn
    ):
        Profile.create("bioinformatics")
        theirs = SessionStore(conn).create(
            profile="bioinformatics", title="theirs"
        )
        write_skill(
            Skill(name="cohort-qc", description="qc", triggers=[], body="s"),
            "bioinformatics",
        )
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="skills-list", session_id=theirs.session_id)
        )
        toast = only(await drain(queue), "Notify")
        assert "bioinformatics" in toast.title and "cohort-qc" in toast.text

    async def test_skill_remove_with_a_name_deletes_it(self, service, session):
        write_skill(
            Skill(name="qc-report", description="run QC", triggers=[], body="s"),
            "default",
        )
        await service.handle(
            CommandRun(
                name="skill-remove", args="qc-report", session_id=session.session_id
            )
        )
        assert load_own_skills("default") == []

    async def test_skill_remove_without_one_offers_what_may_go(
        self, service, session
    ):
        # Global and shipped skills are not offered: removing one would change
        # every other profile that sees it.
        write_skill(
            Skill(name="qc-report", description="run QC", triggers=[], body="s"),
            "default",
        )
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="skill-remove", session_id=session.session_id)
        )
        toast = only(await drain(queue), "Notify")
        assert "qc-report" in toast.text
        assert "read_skill" not in toast.text

    async def test_skill_creators_form_belongs_to_the_front_end(
        self, service, session
    ):
        # The form is the front-end's; the *draft* is a model call and so is
        # the core's, which is what `skill.draft` carries. What comes back out
        # of the editing arrives as `skill.save`.
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="skill-creator", args="watch a jupyter run")
        )
        toast = only(await drain(queue), "Notify")
        assert toast.severity == "warning"
        assert "skill.draft" in toast.text and "skill.save" in toast.text

    async def test_a_session_scoped_command_with_no_session_says_so(
        self, service
    ):
        queue = subscribe(service)
        await service.handle(CommandRun(name="compact"))
        assert only(await drain(queue), "Notify").severity == "warning"

    async def test_a_session_scoped_command_naming_a_gone_session_says_so(
        self, service
    ):
        queue = subscribe(service)
        await service.handle(CommandRun(name="conclude", session_id="gone"))
        assert "That session is gone." == only(await drain(queue), "Notify").text

    async def test_an_unknown_slash_command_is_reported(self, service, session):
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="frobnicate", session_id=session.session_id)
        )
        toast = only(await drain(queue), "Notify")
        assert toast.severity == "warning" and "frobnicate" in toast.text

    async def test_a_recognised_command_is_counted_for_the_menus_sort(
        self, service, session, conn
    ):
        from hpca.db import command_use_counts

        await service.handle(
            CommandRun(name="skills-list", session_id=session.session_id)
        )
        await service.handle(
            CommandRun(name="frobnicate", session_id=session.session_id)
        )
        counts = command_use_counts(conn)
        assert counts.get("skills-list") == 1
        # An unknown one must not teach the menu a name nothing can run.
        assert "frobnicate" not in counts


class TestSkillDrafting:
    """`/skill-creator <what it should do>` — specs-ui-acceptance.md, "Skills".

    The form belongs to the front-end and the draft cannot: a draft is a
    generation, and only the core makes those. So the request crosses as
    `skill.draft`, one model call happens here, and three fields come back to
    be edited and confirmed — nothing is written until they are.
    """

    DRAFT = json.dumps(
        {
            "name": "watch-run",
            "description": "when a notebook run needs watching",
            "body": "1. squeue -u $USER\n2. tail the log",
        }
    )

    async def test_a_request_comes_back_as_three_fields_to_edit(
        self, service, llm
    ):
        llm._outputs = [self.DRAFT]
        queue = subscribe(service)
        await service.handle(
            SkillDraft(profile="default", request="watch a jupyter run")
        )
        drafted = only(await wait_for(queue, "SkillDrafted"), "SkillDrafted")
        assert (drafted.name, drafted.error) == ("watch-run", "")
        assert "squeue" in drafted.body
        assert drafted.request == "watch a jupyter run", "echoed, to be re-tried"

    async def test_and_nothing_is_written_by_it(self, service, llm):
        # Which is why it is a draft and not a save: the user still confirms.
        llm._outputs = [self.DRAFT]
        queue = subscribe(service)
        await service.handle(
            SkillDraft(profile="default", request="watch a jupyter run")
        )
        await wait_for(queue, "SkillDrafted")
        assert load_own_skills("default") == []

    async def test_the_request_and_the_conversation_reach_the_drafter(
        self, service, session, llm
    ):
        # "Write a skill for what we just did" is the common case, so the
        # transcript goes with the request.
        queue = subscribe(service)
        await service.handle(
            TurnSubmit(
                session_id=session.session_id, text="the gpu partition is a100"
            )
        )
        await wait_for(queue, "TurnFinished")
        llm._outputs = [self.DRAFT]
        await service.handle(
            SkillDraft(
                profile="default",
                request="submitting to the gpu queue",
                session_id=session.session_id,
            )
        )
        await wait_for(queue, "SkillDrafted")
        asked = json.dumps(llm.prompts[-1])
        assert "submitting to the gpu queue" in asked
        assert "a100" in asked, "the conversation went with it"

    async def test_the_wait_is_named_on_the_session(self, service, session, llm):
        # The spinner has to say what it is waiting for: a silent minute in a
        # conversation the user did not start is what this event prevents.
        llm._outputs = [self.DRAFT]
        queue = subscribe(service)
        await service.handle(
            SkillDraft(
                profile="default",
                request="watch a run",
                session_id=session.session_id,
            )
        )
        events = await wait_for(queue, "SkillDrafted")
        said = [e for e in events if type(e).__name__ == "TurnActivity"]
        assert [e.activity for e in said][0] == "drafting a skill"
        assert said[-1].activity == "", "and the wait is ended"

    async def test_a_failed_draft_is_still_answered(self, service, llm):
        # The empty form opens behind it, so silence is the one thing this
        # command may not answer with.
        llm._outputs = ["not json at all", "still not json"]
        queue = subscribe(service)
        await service.handle(
            SkillDraft(profile="default", request="watch a jupyter run")
        )
        drafted = only(await wait_for(queue, "SkillDrafted"), "SkillDrafted")
        assert drafted.error and drafted.name == ""
        assert drafted.request == "watch a jupyter run"

    async def test_an_empty_request_asks_the_model_for_nothing(
        self, service, llm
    ):
        # A bare `/skill-creator` is an empty form, and an empty form costs no
        # generation.
        queue = subscribe(service)
        await service.handle(SkillDraft(profile="default", request="  "))
        await _settle()
        assert llm.prompts == []
        assert only(await drain(queue), "Notify").severity == "warning"

    async def test_a_draft_counts_as_the_command_it_is(self, service, llm, conn):
        # `/skill-creator` reaches the core as this and nothing else, so this
        # is where the menu's frequency sort learns it was used.
        from hpca.db import command_use_counts

        llm._outputs = [self.DRAFT]
        queue = subscribe(service)
        await service.handle(SkillDraft(profile="default", request="watch a run"))
        await wait_for(queue, "SkillDrafted")
        assert command_use_counts(conn).get("skill-creator") == 1


class TestTheCommandCounts:
    """How often each slash command was run, served back (`command.counts`).

    The core has counted `command_usage` since M2 and nothing carried the
    numbers across, so a front-end's "/" menu could only be in definition
    order — rule 2 of §4.2 puts the table itself out of its reach.
    """

    async def test_they_are_answered_when_asked_for(self, service, session):
        await service.handle(
            CommandRun(name="skills-list", session_id=session.session_id)
        )
        queue = subscribe(service)
        await service.handle(CommandList())
        assert only(await drain(queue), "CommandCounts").counts == {
            "skills-list": 1
        }

    async def test_and_restated_as_soon_as_one_changes(self, service, session):
        # Which is why they do not ride on `hello`: the greeting is stated
        # once, and these numbers change every time the user runs a command.
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="skills-list", session_id=session.session_id)
        )
        assert only(await drain(queue), "CommandCounts").counts == {
            "skills-list": 1
        }
        await service.handle(
            CommandRun(name="skills-list", session_id=session.session_id)
        )
        assert only(await drain(queue), "CommandCounts").counts == {
            "skills-list": 2
        }

    async def test_the_greeting_carries_none_of_this(self, service):
        assert "counts" not in service.subscribe().get_nowait().model_dump()

    async def test_an_unknown_command_restates_nothing(self, service, session):
        # It is not counted, so there is nothing new to say — and a menu must
        # not learn a name nothing can run.
        queue = subscribe(service)
        await service.handle(
            CommandRun(name="frobnicate", session_id=session.session_id)
        )
        assert [e for e in await drain(queue) if type(e).__name__ ==
                "CommandCounts"] == []


class TestRobustness:
    async def test_a_command_that_fails_is_a_notify_not_an_exception(
        self, service, session, monkeypatch
    ):
        # One bad frame must not be able to end a session: the far side of
        # this is a socket.
        queue = subscribe(service)

        def explode(*a, **k):
            raise RuntimeError("scheduler is on fire")

        monkeypatch.setattr(service._scheduler, "submit_user", explode)
        await service.handle(TurnSubmit(session_id=session.session_id, text="hi"))
        events = await drain(queue)
        assert any(
            type(e).__name__ == "Notify" and "on fire" in e.text for e in events
        )

    async def test_a_command_with_no_handler_is_reported(self, service):
        # §4.1 is dispatched in full now, so the fallback needs a command from
        # outside it to be reached at all. It still has to exist: silence would
        # let a front-end wait forever for something that was never going to
        # happen.
        class Unheard(Command):
            pass  # no TYPE, so it claims no place in the registry

        queue = subscribe(service)
        await service.handle(Unheard())
        assert only(await drain(queue), "Notify").severity == "warning"

    async def test_answering_a_decision_nobody_is_parked_on_is_harmless(
        self, service
    ):
        queue = subscribe(service)
        await service.handle(DecisionResolve(session_id="s1", approved=True))
        # Not an error: a stale answer is exactly what arrives when a turn
        # resolved between the prompt being drawn and the key being pressed.
        assert [e for e in await drain(queue) if type(e).__name__ == "Notify"] == []


class TestStartup:
    """What happens once, before anyone types anything.

    All of it lived in `tui/app.py`'s `on_mount` and had exactly one caller
    each (`specs-ui-coverage.md` §3.1, §3.3), so deleting that module deleted
    the behaviour with it: the trash was never swept, the curator never ran,
    and nothing ever discovered — or checked — a backend.
    """

    @staticmethod
    def backup(home, name="kept.txt", *, at=None):
        """One backup in the app dir's trash, optionally aged past its TTL."""
        from hpca.trash import TrashManager

        source = home / name
        source.write_text("before")
        entry = TrashManager(
            home / "trash", backup_limit_bytes=1024**3
        ).backup(source)
        directory = entry.trashed_path.parent
        if at is not None:
            meta = directory / "meta.json"
            payload = json.loads(meta.read_text())
            payload["trashed_at"] = at
            meta.write_text(json.dumps(payload))
        return directory

    @pytest.fixture
    def trashed(self, home):
        return self.backup(home, at="2001-01-01T00:00:00+00:00")

    @pytest.fixture
    def startup_check(self, monkeypatch):
        """Turn the backend check back on (the suite disables it wholesale —
        `tests/conftest.py`), for the tests that are about the check."""
        monkeypatch.setattr(AgentService, "startup_backend_check", True)

    async def test_the_trash_is_swept(self, service, trashed):
        queue = subscribe(service)
        await service.startup()
        assert not trashed.exists()
        assert "Trash" in only(await drain(queue), "Notify").text

    async def test_a_backup_inside_its_ttl_is_left_alone(self, service, home):
        entry = self.backup(home)
        queue = subscribe(service)
        await service.startup()
        assert entry.exists()
        assert await drain(queue) == []  # nothing to say

    async def test_a_trash_that_cannot_be_swept_does_not_stop_startup(
        self, service, monkeypatch
    ):
        def explode(self, ttl_days):
            raise OSError("the NFS home went away")

        monkeypatch.setattr("hpca.trash.TrashManager.cleanup", explode)
        assert await service.startup() is True

    async def test_the_curator_runs(self, service, monkeypatch):
        ran: list[int] = []
        monkeypatch.setattr(
            service._memory, "run_curator_if_due", lambda: ran.append(1) or {}
        )
        await service.startup()
        assert ran == [1]

    async def test_the_curator_decides_for_itself_whether_it_is_due(
        self, service, monkeypatch
    ):
        # The interval is the curator's own business (`curator_interval_days`,
        # 0 disables it); startup's job is to give it the one chance to ask.
        passes: list[int] = []
        monkeypatch.setattr(
            "hpca.curator.run", lambda *a, **k: passes.append(1) or {}
        )
        service._deps.settings.memory.curator_interval_days = 0
        await service.startup()
        assert passes == []

    async def test_it_connects_before_it_checks(
        self, service, monkeypatch, startup_check
    ):
        """Order, not both: the backend the check probes is the one
        auto-connect just activated, so a cluster LLM counts as connected."""
        order: list[str] = []

        async def auto_connect(**kwargs):
            order.append("auto-connect")

        async def ensure_connected():
            order.append("check")
            return True

        monkeypatch.setattr(service._backends, "auto_connect", auto_connect)
        monkeypatch.setattr(service._backends, "ensure_connected", ensure_connected)
        assert await service.startup() is True
        assert order == ["auto-connect", "check"]

    async def test_nothing_answering_is_the_answer_startup_returns(
        self, service, monkeypatch, startup_check
    ):
        # What the front-end opens the manage-LLMs screen on.
        async def nothing():
            return False

        monkeypatch.setattr(service._backends, "ensure_connected", nothing)
        assert await service.startup() is False

    async def test_the_check_can_be_switched_off(
        self, service, monkeypatch, startup_check
    ):
        """The suite's own escape hatch (`tests/conftest.py`), because under
        it nothing ever answers and a probe on every core would be a network
        call in a hermetic test."""
        probed: list[int] = []

        async def probe():
            probed.append(1)
            return False

        monkeypatch.setattr(service._backends, "ensure_connected", probe)
        monkeypatch.setattr(AgentService, "startup_backend_check", False)
        assert await service.startup() is True
        assert probed == []


class TestShutdown:
    async def test_stop_cancels_the_poll_timers(self, service):
        service.start_timers()
        assert service._timers, "no timers were installed"
        await service.stop()
        # A poll firing after the databases close would raise into a dead
        # loop, so the timers go first and are actually awaited out.
        assert service._timers == []

    async def test_stop_is_idempotent(self, service):
        await service.stop()
        await service.stop()

    async def test_the_shutdown_command_stops_it(self, service):
        await service.handle(Shutdown())
        assert service._stopped is True


async def _record(sink):
    sink.append(True)


async def _yield():
    import asyncio

    await asyncio.sleep(0)


async def _settle():
    import asyncio

    for _ in range(40):
        await asyncio.sleep(0)
