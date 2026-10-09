"""User tools (hpca.user_tools) and the agent's doors to them (selfmod_tools).

A user tool is a Python file in ``<app_dir>/tools`` that the running agent
loads into its own registry. What is pinned here: what a file must be to get
in, that a reload changes the live registry and nothing else, that a broken
edit never costs the working version, and that new code reaches the agent
only past the approval a user would expect.
"""

from __future__ import annotations

import json
import re
import textwrap

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from pydantic import BaseModel, Field

from hpca.agent.builtin_tools import default_tool_registry
from hpca.agent.graph import build_graph, run_turn
from hpca.agent.modes import requires_execution_approval
from hpca.agent.selfmod_tools import add_selfmod_tools, tool_file
from hpca.agent.tools import Tool, ToolRegistry
from hpca.llm import ChatResponse
from hpca.user_tools import (
    APPROVED_FILE,
    UserTools,
    hpca_source,
    load_file,
    preview,
    summary,
    tool_files,
    user_tools_dir,
)

HELLO = '''
"""Greets someone."""

from __future__ import annotations

from pydantic import BaseModel, Field

from hpca.agent.context import ToolContext
from hpca.agent.tools import Tool, ToolRegistry


class HelloParams(BaseModel):
    who: str = Field(description="Who to greet")


async def hello(args: HelloParams, ctx: ToolContext) -> str:
    return f"{GREETING} {args.who}"


GREETING = "hello"


def register(registry: ToolRegistry) -> None:
    registry.register(
        Tool(name="hello", description="Greet someone", params=HelloParams, handler=hello)
    )
'''


def tool_source(name="hello", greeting="hello", *, destructive=False, body=None):
    """A tool file like HELLO, with the knobs the tests turn."""
    handler = body or f'return f"{greeting} {{args.who}}"'
    return textwrap.dedent(
        f'''
        from pydantic import BaseModel, Field

        from hpca.agent.tools import Tool


        class Params(BaseModel):
            who: str = Field(description="Who")


        async def handler(args, ctx):
            {handler}


        def register(registry):
            registry.register(
                Tool(
                    name="{name}",
                    description="A test tool",
                    params=Params,
                    handler=handler,
                    destructive={destructive},
                )
            )
        '''
    )


@pytest.fixture
def home(monkeypatch, tmp_path):
    monkeypatch.setenv("HPCA_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def tools_dir(home):
    directory = user_tools_dir()
    directory.mkdir()
    return directory


@pytest.fixture
def registry():
    registry = default_tool_registry()
    return registry


@pytest.fixture
def user_tools(registry, tools_dir):
    loader = UserTools(registry)
    add_selfmod_tools(registry, loader)
    return loader


def write(directory, name, text):
    path = directory / name
    path.write_text(text)
    return path


def approve(directory):
    """What a reload in an earlier run would have recorded: every file on
    disk, as it is now."""
    from hpca.user_tools import _digest

    files = {p.name: _digest(p.read_bytes()) for p in tool_files(directory)}
    (directory / APPROVED_FILE).write_text(json.dumps({"version": 1, "files": files}))


def start(user_tools):
    """A start after everything on disk was approved."""
    approve(user_tools.directory)
    return user_tools.load_now()


def build_core(home):
    """The real assembly over an app dir, with nothing behind the model."""
    from hpca.config import Settings
    from hpca.core.service import build_service
    from hpca.db import connect, init_db

    conn = connect(home / "hpca.db")
    init_db(conn)

    async def db(fn):
        return fn(conn)

    return build_service(
        settings=Settings.load(),
        app_dir=home,
        db=db,
        conn=conn,
        checkpointer=InMemorySaver(),
        llm=object(),
        project_root=home,
    )


# A file whose top level leaves a mark, so a test can tell that it never ran.
def marking(marker, name="hello"):
    return f"open({str(marker)!r}, 'w').close()\n" + tool_source(name=name)


class Args(BaseModel):
    who: str = Field(description="Who")


async def call(registry, name, **arguments):
    tool = registry.get(name)
    return await tool.handler(tool.params.model_validate(arguments), None)


class TestWhereThingsAre:
    def test_the_directory_is_in_the_app_dir(self, home):
        assert user_tools_dir() == home / "tools"

    def test_the_package_is_the_code_running_now(self):
        package, checkout = hpca_source()
        assert (package / "agent" / "tools.py").is_file()
        # The suite runs from an editable checkout, so there is one around it.
        assert checkout is not None and (checkout / "pyproject.toml").is_file()

    def test_only_plain_python_files_are_tool_files(self, tools_dir):
        for name in ["a.py", "_helper.py", ".hidden.py", "notes.md", "b.py"]:
            write(tools_dir, name, "")
        assert [p.name for p in tool_files(tools_dir)] == ["a.py", "b.py"]

    def test_a_missing_directory_is_no_tools(self, home):
        assert tool_files(home / "nope") == []


class TestLoadingAFile:
    def test_a_good_file_registers_its_tools(self, tools_dir):
        loaded = load_file(write(tools_dir, "hello.py", HELLO), reserved={})
        assert loaded.ok, loaded.problems
        assert loaded.names() == ["hello"]
        assert loaded.tools[0].user_file == tools_dir / "hello.py"

    async def test_future_annotations_resolve(self, tools_dir):
        """The template the skill teaches uses ``from __future__ import
        annotations``; its parameter model must still validate."""
        loaded = load_file(write(tools_dir, "hello.py", HELLO), reserved={})
        tool = loaded.tools[0]
        args = tool.params.model_validate({"who": "you"})
        assert await tool.handler(args, None) == "hello you"

    @pytest.mark.parametrize(
        "source, expected",
        [
            ("def register(registry)\n    pass\n", "SyntaxError"),
            ("def register(registry):\n    x = undefined\n", "line 2: x = undefined"),
            ("x = 1\n", "no `register(registry)` function"),
            ("def register(registry):\n    pass\n", "registered no tools"),
            ("import sys\nsys.exit(3)\n", "SystemExit"),
            (tool_source(name="has space"), "letters, digits and underscores"),
            (tool_source(name="read_file"), "taken by a built-in tool"),
            (
                tool_source().replace('description="A test tool"', 'description=""'),
                "description is empty",
            ),
            (
                tool_source().replace("async def handler", "def handler"),
                "must be an `async def`",
            ),
            (tool_source().replace("params=Params", "params=dict"), "BaseModel"),
            (
                "def register(registry):\n    registry.register('not a tool')\n",
                "takes an hpca.agent.tools.Tool, not a str",
            ),
        ],
    )
    def test_what_keeps_a_file_out(self, tools_dir, source, expected):
        loaded = load_file(
            write(tools_dir, "t.py", source), reserved={"read_file": "a built-in tool"}
        )
        assert not loaded.ok
        assert loaded.tools == []
        assert expected in "\n".join(loaded.problems)


class TestLoadAndReload:
    def test_startup_loads_what_was_approved(self, registry, user_tools, tools_dir):
        write(tools_dir, "hello.py", HELLO)
        report = start(user_tools)
        assert report.added == ["hello"]
        assert "hello" in registry.names()

    async def test_a_reload_picks_up_a_new_file(self, registry, user_tools, tools_dir):
        start(user_tools)
        write(tools_dir, "hello.py", HELLO)
        assert "hello" not in registry.names()
        report = await user_tools.reload()
        assert report.added == ["hello"]
        assert await call(registry, "hello", who="you") == "hello you"

    async def test_a_reload_replaces_a_changed_tool(
        self, registry, user_tools, tools_dir
    ):
        write(tools_dir, "t.py", tool_source(greeting="hi"))
        start(user_tools)
        write(tools_dir, "t.py", tool_source(greeting="hey"))
        report = await user_tools.reload()
        assert report.changed == ["hello"]
        assert await call(registry, "hello", who="you") == "hey you"

    async def test_an_untouched_file_is_unchanged(self, user_tools, tools_dir):
        write(tools_dir, "t.py", tool_source())
        start(user_tools)
        report = await user_tools.reload()
        assert report.unchanged == ["hello"] and not report.changed

    async def test_a_deleted_file_takes_its_tools_away(
        self, registry, user_tools, tools_dir
    ):
        path = write(tools_dir, "t.py", tool_source())
        start(user_tools)
        path.unlink()
        report = await user_tools.reload()
        assert report.removed == ["hello"]
        assert "hello" not in registry.names()

    async def test_a_renamed_tool_replaces_the_old_name(
        self, registry, user_tools, tools_dir
    ):
        write(tools_dir, "t.py", tool_source(name="old"))
        start(user_tools)
        write(tools_dir, "t.py", tool_source(name="new"))
        report = await user_tools.reload()
        assert report.added == ["new"] and report.removed == ["old"]
        assert "old" not in registry.names() and "new" in registry.names()

    async def test_a_broken_edit_keeps_the_working_version(
        self, registry, user_tools, tools_dir
    ):
        """Mid-session, a bad edit must not cost the tool it was improving."""
        write(tools_dir, "t.py", tool_source(greeting="hi"))
        start(user_tools)
        write(tools_dir, "t.py", "def register(registry):\n    oops(\n")
        report = await user_tools.reload()
        assert [f.path.name for f in report.failed] == ["t.py"]
        assert report.kept[tools_dir / "t.py"] == ["hello"]
        assert await call(registry, "hello", who="you") == "hi you"
        assert "Its previous version stays loaded: hello" in report.render()

    async def test_a_name_another_file_holds_is_refused(
        self, registry, user_tools, tools_dir
    ):
        write(tools_dir, "a.py", tool_source(greeting="from a"))
        write(tools_dir, "b.py", tool_source(greeting="from b"))
        report = await user_tools.reload()
        assert [f.path.name for f in report.failed] == ["b.py"]
        assert "the user tool in a.py" in report.render()
        assert await call(registry, "hello", who="x") == "from a x"

    async def test_built_ins_are_never_touched(self, registry, user_tools, tools_dir):
        before = set(registry.names())
        write(tools_dir, "t.py", tool_source(name="run_bash"))
        report = await user_tools.reload()
        assert report.failed and set(registry.names()) == before
        assert registry.get("run_bash").user_file is None


class TestStartupLoadsOnlyApprovedCode:
    """A restart must not walk around the reload approval."""

    def test_a_never_approved_file_never_runs(self, registry, user_tools, tools_dir):
        marker = tools_dir.parent / "ran"
        write(tools_dir, "draft.py", marking(marker))
        report = user_tools.load_now()
        assert not marker.exists()
        assert "hello" not in registry.names()
        assert report.failed[0].unapproved
        assert "never approved" in report.render()

    def test_a_file_changed_since_approval_never_runs(
        self, registry, user_tools, tools_dir
    ):
        marker = tools_dir.parent / "ran"
        write(tools_dir, "t.py", tool_source())
        approve(tools_dir)
        write(tools_dir, "t.py", marking(marker))
        report = user_tools.load_now()
        assert not marker.exists()
        assert "hello" not in registry.names()
        assert "changed since it was last approved" in report.render()

    async def test_a_reload_approves_for_the_next_start(self, user_tools, tools_dir):
        write(tools_dir, "hello.py", HELLO)
        await user_tools.reload()
        fresh = ToolRegistry()
        assert UserTools(fresh).load_now().added == ["hello"]

    async def test_a_broken_edit_is_not_approved(self, user_tools, tools_dir):
        """The previous version stays live this run, but the broken bytes on
        disk are not what was approved, so the next start skips them."""
        write(tools_dir, "t.py", tool_source())
        await user_tools.reload()
        write(tools_dir, "t.py", "def register(registry):\n    oops(\n")
        await user_tools.reload()
        report = UserTools(ToolRegistry()).load_now()
        assert report.failed[0].unapproved

    async def test_a_deleted_file_leaves_the_record(self, user_tools, tools_dir):
        path = write(tools_dir, "t.py", tool_source())
        await user_tools.reload()
        path.unlink()
        await user_tools.reload()
        assert user_tools.approved() == {}

    def test_an_unreadable_record_approves_nothing(self, registry, user_tools, tools_dir):
        write(tools_dir, "t.py", tool_source())
        (tools_dir / APPROVED_FILE).write_text("{not json")
        user_tools.load_now()
        assert "hello" not in registry.names()

    def test_the_record_is_not_a_tool_file(self, user_tools, tools_dir):
        approve(tools_dir)
        assert tool_files(tools_dir) == []

    async def test_the_status_says_what_is_waiting(self, registry, user_tools, tools_dir):
        write(tools_dir, "draft.py", tool_source())
        user_tools.load_now()
        text = await call(registry, "check_user_tool")
        assert "draft.py (did not load: never approved" in text


class TestPending:
    def test_a_new_file_is_shown_whole(self, user_tools, tools_dir):
        write(tools_dir, "hello.py", HELLO)
        text = preview(user_tools.pending())
        assert "=== hello.py (new) ===" in text
        assert "async def hello" in text

    async def test_a_changed_file_is_shown_as_a_diff(self, user_tools, tools_dir):
        write(tools_dir, "t.py", tool_source(greeting="hi"))
        start(user_tools)
        write(tools_dir, "t.py", tool_source(greeting="hey"))
        text = preview(user_tools.pending())
        assert "(changed)" in text
        assert '-    return f"hi {args.who}"' in text
        assert '+    return f"hey {args.who}"' in text

    def test_a_removed_file_is_named(self, user_tools, tools_dir):
        path = write(tools_dir, "t.py", tool_source())
        start(user_tools)
        path.unlink()
        assert "t.py (removed)" in preview(user_tools.pending())

    def test_nothing_pending_says_so(self, user_tools):
        assert preview(user_tools.pending()) == ""
        assert "nothing has changed" in summary(user_tools.pending())

    def test_a_long_preview_keeps_its_end(self, user_tools, tools_dir):
        body = "\n".join(f"# line {i}" for i in range(5000))
        write(tools_dir, "big.py", body + "\n# the very end\n")
        text = preview(user_tools.pending())
        assert "characters omitted" in text
        assert text.endswith("# the very end")


class TestCheckUserTool:
    async def test_no_file_says_where_things_are(self, registry, user_tools, tools_dir):
        text = await call(registry, "check_user_tool")
        package, _ = hpca_source()
        assert str(package) in text
        assert f"User tools directory: {tools_dir}" in text

    async def test_the_status_says_why_a_file_is_not_loaded(
        self, registry, user_tools, tools_dir
    ):
        write(tools_dir, "bad.py", "def register(registry):\n    x = nope\n")
        start(user_tools)
        text = await call(registry, "check_user_tool")
        assert "bad.py (did not load: NameError" in text

    async def test_a_check_loads_nothing_into_the_agent(
        self, registry, user_tools, tools_dir
    ):
        write(tools_dir, "hello.py", HELLO)
        text = await call(registry, "check_user_tool", file="hello.py")
        assert "loads cleanly and registers: hello" in text
        assert "Not loaded into the agent yet" in text
        assert "hello" not in registry.names()

    async def test_a_test_call_runs_the_staged_tool(
        self, registry, user_tools, tools_dir
    ):
        write(tools_dir, "hello.py", HELLO)
        text = await call(
            registry,
            "check_user_tool",
            file="hello",
            tool="hello",
            arguments={"who": "you"},
        )
        assert text.endswith("hello you")
        assert "hello" not in registry.names()

    async def test_arguments_may_be_a_json_string(self, registry, user_tools, tools_dir):
        write(tools_dir, "hello.py", HELLO)
        text = await call(
            registry,
            "check_user_tool",
            file="hello.py",
            tool="hello",
            arguments='{"who": "you"}',
        )
        assert text.endswith("hello you")

    async def test_arguments_that_do_not_fit_are_not_run(
        self, registry, user_tools, tools_dir
    ):
        ran = tools_dir.parent / "ran"
        body = f"open({str(ran)!r}, 'w').close(); return 'x'"
        write(tools_dir, "t.py", tool_source(body=body))
        text = await call(
            registry, "check_user_tool", file="t.py", tool="hello", arguments={}
        )
        assert "NOT made" in text and "who" in text
        assert not ran.exists()

    async def test_a_raising_tool_reports_its_line(
        self, registry, user_tools, tools_dir
    ):
        write(tools_dir, "t.py", tool_source(body="raise RuntimeError('squeue is gone')"))
        text = await call(
            registry,
            "check_user_tool",
            file="t.py",
            tool="hello",
            arguments={"who": "x"},
        )
        assert "RuntimeError: squeue is gone" in text
        assert re.search(r"\(line \d+: raise RuntimeError", text)

    async def test_a_file_that_does_not_load_lists_its_problems(
        self, registry, user_tools, tools_dir
    ):
        write(tools_dir, "t.py", tool_source(name="run_bash"))
        text = await call(registry, "check_user_tool", file="t.py")
        assert "does NOT load" in text and "built-in tool" in text

    async def test_a_loaded_file_says_so(self, registry, user_tools, tools_dir):
        write(tools_dir, "t.py", tool_source())
        start(user_tools)
        text = await call(registry, "check_user_tool", file="t.py")
        assert "already loaded and callable" in text
        write(tools_dir, "t.py", tool_source(greeting="hey"))
        text = await call(registry, "check_user_tool", file="t.py")
        assert "An older version is loaded" in text

    async def test_a_missing_file(self, registry, user_tools):
        text = await call(registry, "check_user_tool", file="nothing.py")
        assert "create_file" in text


class TestToolFile:
    def test_a_bare_name_is_in_the_directory(self, user_tools, tools_dir):
        assert tool_file("fairshare", user_tools, None) == tools_dir / "fairshare.py"
        assert tool_file("fairshare.py", user_tools, None) == tools_dir / "fairshare.py"

    def test_a_full_path_in_the_directory(self, user_tools, tools_dir):
        assert tool_file(str(tools_dir / "x.py"), user_tools, None) == tools_dir / "x.py"

    def test_a_path_elsewhere_is_refused(self, user_tools, home):
        with pytest.raises(ValueError, match="not in the user tools directory"):
            tool_file(str(home / "x.py"), user_tools, None)

    @pytest.mark.parametrize("name", ["_helper.py", "notes.md"])
    def test_files_that_never_load_are_refused(self, user_tools, name):
        with pytest.raises(ValueError):
            tool_file(name, user_tools, None)


class TestApproval:
    """New code enters the agent past the approval a user would expect."""

    def args(self, registry, **arguments):
        return registry.get("check_user_tool").params.model_validate(arguments)

    def test_reloading_always_asks(self, registry, user_tools):
        tool = registry.get("reload_user_tools")
        assert tool.gates(tool.params(), None)

    def test_the_reload_approval_shows_the_code(self, registry, user_tools, tools_dir):
        write(tools_dir, "hello.py", HELLO)
        tool = registry.get("reload_user_tools")
        sentence = tool.describe_call(tool.params(), None)
        assert "hello.py (new)" in sentence and "async def" not in sentence
        assert "async def hello" in tool.show_call(tool.params(), None)

    async def test_the_code_rides_the_approval_as_its_script(
        self, registry, user_tools, tools_dir
    ):
        """Through the real graph: the source is the scrolling block, the
        sentence is the pinned one — the approval prompt's contract."""

        class FakeLLM:
            async def chat(self, messages, **kwargs):
                return ChatResponse(
                    content=json.dumps(
                        {"action": "tool_call", "tool": "reload_user_tools", "arguments": {}}
                    )
                )

            async def supports_constrained_decoding(self):
                return True

        write(tools_dir, "hello.py", HELLO)
        graph = build_graph(llm=FakeLLM(), tools=registry, checkpointer=InMemorySaver())
        result = await run_turn(graph, session_id="s", user_text="load it")
        assert result.interrupt["kind"] == "destructive"
        assert "async def hello" in result.interrupt["script"]
        assert "hello.py (new)" in result.interrupt["details"]
        assert "hello" not in registry.names()  # nothing loads before the yes

    def test_a_plain_check_does_not_ask(self, registry, user_tools, tools_dir):
        tool = registry.get("check_user_tool")
        assert not tool.gates(self.args(registry), None)
        assert not tool.gates(self.args(registry, file="t.py"), None)

    def test_a_test_call_of_an_unchecked_tool_asks(
        self, registry, user_tools, tools_dir
    ):
        write(tools_dir, "t.py", tool_source())
        tool = registry.get("check_user_tool")
        args = self.args(registry, file="t.py", tool="hello", arguments={"who": "x"})
        assert tool.gates(args, None)
        assert "what it changes is unknown" in tool.describe_call(args, None)

    async def test_a_checked_harmless_tool_does_not_ask(
        self, registry, user_tools, tools_dir
    ):
        write(tools_dir, "t.py", tool_source())
        await call(registry, "check_user_tool", file="t.py")
        tool = registry.get("check_user_tool")
        args = self.args(registry, file="t.py", tool="hello", arguments={"who": "x"})
        assert not tool.gates(args, None)

    async def test_a_checked_destructive_tool_asks(self, registry, user_tools, tools_dir):
        write(tools_dir, "t.py", tool_source(destructive=True))
        await call(registry, "check_user_tool", file="t.py")
        tool = registry.get("check_user_tool")
        args = self.args(registry, file="t.py", tool="hello", arguments={"who": "x"})
        assert tool.gates(args, None)
        assert "marked it destructive" in tool.describe_call(args, None)

    async def test_an_edit_after_the_check_asks_again(
        self, registry, user_tools, tools_dir
    ):
        write(tools_dir, "t.py", tool_source())
        await call(registry, "check_user_tool", file="t.py")
        write(tools_dir, "t.py", tool_source(greeting="changed"))
        tool = registry.get("check_user_tool")
        args = self.args(registry, file="t.py", tool="hello", arguments={"who": "x"})
        assert tool.gates(args, None)

    def test_manual_mode_treats_user_code_as_execution(
        self, registry, user_tools, tools_dir
    ):
        write(tools_dir, "t.py", tool_source())
        start(user_tools)
        user_tool = registry.get("hello")
        assert user_tool.executes(user_tool.params(who="x"), None)
        check = registry.get("check_user_tool")
        assert check.executes(self.args(registry, file="t.py"), None)
        assert not check.executes(self.args(registry), None)
        assert not registry.get("read_file").executes(
            registry.get("read_file").params(path="x"), None
        )
        assert requires_execution_approval("manual", "hello", executes=True)
        assert not requires_execution_approval("auto", "hello", executes=True)


class TestGraph:
    async def test_a_call_whose_tool_was_unloaded_is_answered_not_raised(self):
        """Parked for approval, then reloaded away: the turn goes on."""

        class FakeLLM:
            def __init__(self):
                self.outputs = [
                    json.dumps(
                        {
                            "action": "tool_call",
                            "tool": "mine",
                            "arguments": {"who": "x"},
                        }
                    ),
                    json.dumps({"action": "respond", "response": "ok"}),
                ]
                self.calls = []

            async def chat(self, messages, **kwargs):
                self.calls.append(messages)
                return ChatResponse(content=self.outputs.pop(0))

            async def supports_constrained_decoding(self):
                return True

        async def handler(args, ctx):
            return "ran"

        registry = ToolRegistry()
        registry.register(
            Tool(
                name="mine",
                description="d",
                params=Args,
                handler=handler,
                destructive=True,
            )
        )
        llm = FakeLLM()
        graph = build_graph(llm=llm, tools=registry, checkpointer=InMemorySaver())
        first = await run_turn(graph, session_id="s", user_text="go")
        assert first.interrupt is not None
        registry.remove("mine")
        result = await run_turn(
            graph, session_id="s", resume=Command(resume={"approved": True})
        )
        assert result.reply == "ok"
        last = llm.calls[-1]
        assert any("no longer loaded" in str(m.get("content")) for m in last)


class TestService:
    """The real assembly: user tools load at startup, and say so if not."""

    @pytest.fixture
    def build(self, home):
        return lambda: build_core(home)

    def test_the_default_registry_has_the_selfmod_tools(self, build):
        service = build()
        names = service.user_tools._registry.names()
        assert "check_user_tool" in names and "reload_user_tools" in names

    def test_an_approved_file_is_loaded_at_startup(self, build, tools_dir):
        write(tools_dir, "hello.py", HELLO)
        approve(tools_dir)
        service = build()
        assert "hello" in service.user_tools._registry.names()

    async def test_a_file_that_did_not_load_is_told_at_startup(self, build, tools_dir):
        from hpca.protocol import Notify

        write(tools_dir, "bad.py", "def register(registry):\n    nope()\n")
        approve(tools_dir)
        service = build()
        queue = service.subscribe()
        await service.startup()
        told = []
        while not queue.empty():
            event = queue.get_nowait()
            if isinstance(event, Notify):
                told.append(event)
        assert any(
            "bad.py" in n.title and "NameError" in n.text for n in told
        ), told


class TestReloadToolsCommand:
    """`/reload-tools`: the user's own reload — the command is the consent."""

    @pytest.fixture
    def service(self, home):
        return build_core(home)

    def notifies(self, queue):
        from hpca.protocol import Notify

        out = []
        while not queue.empty():
            event = queue.get_nowait()
            if isinstance(event, Notify):
                out.append(event)
        return out

    async def test_a_draft_waits_quietly_at_startup(self, home, tools_dir):
        write(tools_dir, "draft.py", tool_source())
        service = build_core(home)
        queue = service.subscribe()
        await service.startup()
        told = self.notifies(queue)
        waiting = [n for n in told if "waiting for approval" in n.title]
        assert len(waiting) == 1 and "draft.py" in waiting[0].text
        assert "/reload-tools" in waiting[0].text
        assert waiting[0].severity == "information"

    async def test_the_command_loads_and_approves(self, service, tools_dir):
        import asyncio

        from hpca.protocol import CommandRun

        write(tools_dir, "hello.py", HELLO)
        queue = service.subscribe()
        await service.handle(CommandRun(name="reload-tools", args="", session_id=None))
        for _ in range(200):
            told = [n for n in self.notifies(queue) if n.title == "User tools"]
            if told:
                break
            await asyncio.sleep(0.01)
        assert told and "hello (new)" in told[0].text
        assert "hello" in service.user_tools._registry.names()
        assert "hello.py" in service.user_tools.approved()

    def test_the_menu_offers_what_the_core_answers(self):
        from hpca.core.service import SLASH_COMMANDS
        from hpca.ui.commands import BUILTINS

        assert "reload-tools" in {c.name for c in BUILTINS}
        assert {c.name for c in BUILTINS} <= SLASH_COMMANDS


class TestTheSkill:
    @pytest.fixture
    def skill(self, home):
        from hpca.skills import load_skills

        return next(s for s in load_skills("default") if s.name == "new-tool")

    def test_it_ships(self, skill):
        assert skill.level == "builtin" and not skill.problems

    def test_it_follows_the_steps_in_order(self, skill):
        body = skill.body
        steps = [
            "check_user_tool` with no arguments",  # 1. find the source
            "where HPCA's source is",
            "agent/context.py",  # 2. learn the conventions
            "create_file",  # 3. draft
            "`grillme` skill",  # 4. grill
            "edit_file",  # 5. implement
            "reload_user_tools",  # 6. check, then load
        ]
        positions = [body.index(step) for step in steps]
        assert positions == sorted(positions)

    def test_every_tool_it_names_exists(self, skill, registry, user_tools):
        """The skill drives the agent by tool name; a renamed tool would turn
        a step into a call that fails."""
        from hpca.agent.doc_tools import add_doc_tools
        from hpca.agent.file_tools import add_file_tools
        from hpca.agent.skill_tools import add_skill_tools

        add_skill_tools(add_doc_tools(add_file_tools(registry)))
        named = set(re.findall(r"`([a-z_]+)`", skill.body))
        tools = {n for n in named if "_" in n and n.split("_")[0] in {
            "check", "reload", "read", "run", "create", "edit"}}
        assert {"check_user_tool", "reload_user_tools", "read_skill"} <= tools
        assert tools <= set(registry.names()), tools - set(registry.names())

    def test_the_example_in_it_is_a_working_tool_file(self, skill, tools_dir):
        """The draft template the skill teaches has to load as written."""
        code = re.search(r"```python\n(.*?)```", skill.body, re.S).group(1)
        code = textwrap.dedent(code)
        loaded = load_file(write(tools_dir, "fairshare.py", code), reserved={})
        assert loaded.ok, loaded.problems
        assert loaded.names() == ["fairshare"]
