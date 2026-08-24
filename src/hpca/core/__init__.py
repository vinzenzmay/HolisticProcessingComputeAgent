"""The agent runtime, with no front-end in it (specs/specs-core-process.md).

Everything here must run headless. The rule that keeps that true is simple and
worth stating once: **nothing under `hpca.core` may import Textual, and nothing
here may ask whether a session is on screen.** A module that needs to tell the
user something emits an event and lets whoever is rendering decide; a module
that needs to know what the user is looking at reads it from `session.focus`,
which arrived as a command.

That is not style. `tui/app.py` grew to 4875 lines precisely because agent
logic could reach for `self.notify`, `self._is_active_session` and a widget
query whenever it was convenient, and each of those reaches is a reason the
runtime cannot run anywhere else.
"""
