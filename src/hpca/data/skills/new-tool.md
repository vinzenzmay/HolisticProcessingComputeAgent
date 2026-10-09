---
name: new-tool
description: Build a new tool for yourself — find HPCA's source, learn its conventions,
  draft the tool as a user tool file, grill the user until it is exactly what they mean,
  implement it, test it, and hot-load it. Never edits HPCA's own source.
triggers: [new tool, create a tool, build a tool, user tool, self-modify]
---

You are adding a tool to yourself. It lives in the **user tools directory**, never in
HPCA's source: HPCA is updated from its main branch, and an edit to the source would be
overwritten or conflict. You only *read* HPCA's source — do not edit, create or delete
anything under it.

1. **Find the HPCA source.** Call `check_user_tool` with no arguments. It reports the
   HPCA package directory (the code running you), the checkout around it if there is
   one, and the user tools directory. Confirm you can read the package with `read_file`
   on `<package>/agent/tools.py`. If that fails, look for it with `run_bash` (e.g.
   `python3 -c 'import hpca; print(hpca.__file__)'`). If you still cannot read it, ask
   the user where HPCA's source is — do not guess, and do not go on without it.

2. **Learn how a tool is built here.** Read, in this order:
   - `<package>/agent/tools.py` — `Tool` and `ToolRegistry`, every field a tool has.
   - `<package>/agent/context.py` — `ToolContext`, everything a handler gets as `ctx`
     (runner, slurm, settings, workdir, …).
   - The one or two existing tool modules closest to what the user wants
     (`<package>/agent/*_tools.py`: `job_tools.py` for Slurm, `watch_tools.py` for
     logs, `run_bash` in `builtin_tools.py` for running a command through `ctx.runner`).
     Copy their idioms: a pydantic params model with `Field(description=...)` on every
     field, a one-line description, results that say in words what happened.
   - Any file already in the user tools directory — the user's own conventions.
   - If there is a checkout: §5.1 of its `project.md`.
   Read what you need to build this tool; you are learning conventions, not auditing.

3. **Draft the tool file.** Create `<user tools directory>/<tool_name>.py` with
   `create_file`: a module docstring saying what the tool is for, the params model, the
   `register` function with the name, description and `destructive` flag, and a handler
   that only raises `NotImplementedError`, with your plan written in it as comments.
   The shape:

   ```python
   """Fairshare and usage of a Slurm account, for deciding where to submit."""

   from __future__ import annotations

   from pydantic import BaseModel, Field

   from hpca.agent.context import ToolContext
   from hpca.agent.tools import Tool, ToolRegistry


   class FairshareParams(BaseModel):
       account: str = Field(description="Slurm account to report on")


   async def fairshare(args: FairshareParams, ctx: ToolContext) -> str:
       # plan: run `sshare -A <account> -l -P`, keep the header and the user's row,
       # and say the fairshare value in words
       raise NotImplementedError("drafted, not implemented yet")


   def register(registry: ToolRegistry) -> None:
       registry.register(
           Tool(
               name="fairshare",
               description="Fairshare and usage of a Slurm account",
               params=FairshareParams,
               handler=fairshare,
           )
       )
   ```

   Then call `check_user_tool` with `file` set to it: even the draft must load.

4. **Grill the user.** Fetch the `grillme` skill with `read_skill` and follow it exactly,
   with the draft as the thing being grilled: one question at a time, your recommended
   answer with each. Settle at least: the name; every input and its default; what the
   result says and how long it may get; whether it changes anything on the system (if
   it does, `destructive=True` — or `is_destructive_call` when only some calls do); what
   it does when each step fails. Look facts up yourself instead of asking — run the
   commands it will run with `run_bash` and read their real output. Update the draft
   with `edit_file` as answers come in. Do not implement until the user confirms you
   have reached a shared understanding.

5. **Implement it** in the same file with `edit_file`. A user tool must keep these rules:
   - The handler is `async def`, takes `(args, ctx)` and returns a `str` — the text the
     model reads. Say what happened, keep it short, and cut long output (keep the head
     and the tail, and say how much was cut).
   - Run commands through `ctx.runner` the way `run_bash` does, or with
     `asyncio.create_subprocess_exec`. Never `subprocess.run`, `time.sleep` or a slow
     file read on the event loop (wrap blocking work in `asyncio.to_thread`): the
     agent's screen is drawn from that loop.
   - Never `print`, `input()` or `sys.exit`. Report a failure by raising an exception
     with a clear message, or by returning text that says it failed.
   - Nothing runs at import time: the file's top level only defines things.
   - Import only the standard library, `pydantic` and `hpca.*` — the tool runs inside
     HPCA's own Python environment. Do not import other user tool files.
   - The name is never a built-in tool's name.

6. **Test it, then load it.**
   - `check_user_tool` with `file` — fix every problem it lists until the file loads.
   - `check_user_tool` with `file`, `tool` and `arguments` — a realistic call. Read the
     result the way the user will: is it exactly what they asked for? Try an edge case
     too (a bad input, an empty result). Fix with `edit_file` and check again.
   - After every edit, check the file once *without* `tool` before the next test call.
     A test call of a version nobody has checked asks the user first, because what it
     would change is unknown until the file has been loaded.
   - `reload_user_tools` — the user approves the code going into the running agent. The
     result must list your tool as new or changed; if a file did not load, fix it and
     reload again.
   - Call the tool for real once, unless it changes something — then ask the user
     first. Then tell the user its name and the file it lives in, and that:
     - it loads on every start, because the reload approved this version;
     - if the file is edited by hand, it stops loading at startup until they type
       `/reload-tools`;
     - deleting the file and typing `/reload-tools` removes it.
