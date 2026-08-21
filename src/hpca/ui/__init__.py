"""The row-oriented terminal UI, drawn without a framework.

    pixi run -e dev python -m hpca.ui.run --demo
    pixi run -e dev python -m hpca.ui.run --demo --chat 2000 --sessions 40

The shape is a header, four stacked rows — sessions, chat, the message box,
watchers — a footer, and the screens that open over them. The demo content is
synthetic and deliberately over-long, because the question this package exists
to answer is what the UI feels like once a turn has produced hundreds of steps.

The rendering model is pi-tui's, which is the reason for the exercise. pi-tui
is TypeScript, so what is ported here is the architecture, not the library:

* a component renders to ``list[str]`` at a known width. There is no style
  cascade, no auto-height measurement and no arrange pass, so there is no
  O(conversation) traversal to accidentally trigger — which is the whole of
  what makes the Textual version stutter;
* only the *visible* slice of a pane is ever turned into lines, so frame time
  is flat in the number of entries. The header prints it: that number staying
  put while you scroll a 2000-entry chat is the claim being tested;
* the screen is repainted differentially — lines that did not change are not
  written — inside a synchronised-output pair so the terminal shows each frame
  atomically instead of tearing.

Keys follow the ones already bound in the Textual app (app.py BINDINGS) so
that muscle memory survives the move: m manage llms, a profiles & learnings,
c config editor, r/t/d on a session, d unwatch, q quit. The one addition is
→/← to open and close an entry, which the row design needs and the column
design had no equivalent of.

The message box edits the way an editor does: it wraps rather than overflowing,
ctrl+arrow moves by word, ctrl+backspace and ctrl+delete cut one, and shift with
any motion marks text. That last one also fixed a real annoyance — an escape
sequence the key table did not know used to be read as the esc key followed by
its letters, so reaching for shift+arrow threw you out of the box and into the
chat. Escapes are now measured by shape and unknown ones are dropped whole.

The content is real from M2 on: ``client`` fills ``state`` in from protocol
events and the rows render that. From M3 there is a real core behind it —
``hpca --new-ui`` builds an ``AgentService`` over an ``InProcessConnection``
and ``run`` drives stdin, that connection and SIGWINCH on one asyncio loop.
What is deliberately still missing: mouse support, selecting text out of the *chat*,
the turn display and approval prompts. Those are arguments *against* leaving
Textual, so none of them is pretended solved — specs-ui-replacement.md §4 is
the full list of what is absent and the order it arrives in.

The modules, per specs-ui-replacement.md §3:

* ``ansi`` — the escape constants and the string helpers every other module
  pads, rules and highlights with;
* ``screen`` — the raw terminal: alternate screen, differential repaint;
* ``keys`` — one read off the wire into key names;
* ``editor`` — the shared multi-line buffer;
* ``pane`` — one navigable list of entries, each named by its own key;
* ``state`` — the plain dataclasses the rows are drawn from: sessions, chat,
  turn, context, and the intents a keypress becomes;
* ``client`` — the one module that knows the protocol: events become state,
  intents become commands;
* ``overlays`` — the screens that draw over the rows, one module each;
* ``app`` — layout, focus and key dispatch, with no I/O of its own;
* ``demo`` — the sample content, served by a loopback core that speaks the
  same protocol a real one does;
* ``run`` — the asyncio loop and the terminal's setup and teardown;
* ``boot`` — the core in this process: the databases, the ``AgentService``, the
  wire between them, and the order it all closes in.
"""
