"""Add an LLM backend by hand, with a key.

The port scan lists a key-locked endpoint only as "(api key required)" — it
cannot read the model name or context length through the 401, and the literal
placeholder is not a usable model id. This form fills that gap: you supply the
key (and a URL, for an endpoint the scan never found), and on save it makes one
authenticated /v1/models call. That call both validates the key and, since it
returns the served model ids + context lengths, lets us auto-fill the model and
context you would otherwise have to type — so a manual key still yields
auto-detection.
"""

from __future__ import annotations

from textual import on
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.screen import ModalScreen
from textual.widgets import Input, Label, ListItem, ListView, Static

from hpca.config import LLMBackend
from hpca.discover import probe_endpoint


class _ModelPickScreen(ModalScreen[str | None]):
    """Choose which served model to configure when an endpoint offers several."""

    BINDINGS = [Binding("escape", "cancel", "cancel", priority=True)]

    DEFAULT_CSS = """
    _ModelPickScreen { align: center middle; }
    #pick-dialog {
        width: 70;
        height: auto;
        max-height: 20;
        border: heavy $accent;
        background: $surface;
        padding: 1 2;
    }
    #pick-hint { color: $text-muted; }
    """

    def __init__(self, models: list[str]) -> None:
        super().__init__()
        self._models = models

    def compose(self) -> ComposeResult:
        with Vertical(id="pick-dialog"):
            yield Static("This endpoint serves several models — pick one:")
            yield ListView(
                *(ListItem(Label(m)) for m in self._models), id="pick-list"
            )
            yield Static("(enter) choose · (escape) cancel", id="pick-hint")

    def on_mount(self) -> None:
        self.query_one("#pick-list", ListView).focus()

    @on(ListView.Selected, "#pick-list")
    def _chosen(self, event: ListView.Selected) -> None:
        index = self.query_one("#pick-list", ListView).index or 0
        self.dismiss(self._models[index])

    def action_cancel(self) -> None:
        self.dismiss(None)


class BackendFormScreen(ModalScreen["LLMBackend | None"]):
    """Collect base_url / model / key / context and validate before saving.

    Enter validates the endpoint (authenticated /v1/models) and, on success,
    auto-fills a blank model + context from the response. ctrl+s bypasses the
    check and saves exactly what was typed, for the rare case of an endpoint
    that is momentarily down but whose details you already know.
    """

    BINDINGS = [
        Binding("ctrl+s", "save_anyway", "save without checking"),
        Binding("escape", "cancel", "cancel", priority=True),
    ]

    DEFAULT_CSS = """
    BackendFormScreen { align: center middle; }
    #backend-dialog {
        width: 72;
        height: auto;
        border: heavy $accent;
        background: $surface;
        padding: 1 2;
    }
    #backend-dialog Label { margin-top: 1; }
    #backend-status { color: $text-muted; }
    #backend-hint { color: $text-muted; }
    """

    def __init__(
        self, *, base_url: str = "", model: str = "", editable_url: bool = True
    ) -> None:
        super().__init__()
        self._base_url = base_url
        self._model = model
        self._editable_url = editable_url

    def compose(self) -> ComposeResult:
        with Vertical(id="backend-dialog"):
            yield Static("Add LLM backend", id="backend-title")
            yield Label("Endpoint URL")
            yield Input(
                value=self._base_url,
                placeholder="http://localhost:20001/v1",
                disabled=not self._editable_url,
                id="backend-url",
            )
            yield Label("Model (leave blank to auto-detect from the endpoint)")
            yield Input(value=self._model, placeholder="auto", id="backend-model")
            yield Label("API key")
            yield Input(password=True, id="backend-key")
            yield Label("Context length (optional — auto-filled if left blank)")
            yield Input(placeholder="auto", id="backend-ctx")
            yield Static("", id="backend-status")
            yield Static(
                "(enter) validate & save · (ctrl+s) save without checking · "
                "(escape) cancel",
                id="backend-hint",
            )

    def on_mount(self) -> None:
        # Land on the first field the user actually needs to fill: the URL when
        # it is blank and editable (manual add), otherwise the model field.
        first = "backend-url" if self._editable_url and not self._base_url else "backend-model"
        self.query_one(f"#{first}", Input).focus()

    # ---------------------------------------------------------------- helpers

    def _set_status(self, text: str) -> None:
        self.query_one("#backend-status", Static).update(text)

    def _fields(self) -> tuple[str, str, str, str]:
        return (
            self.query_one("#backend-url", Input).value.strip(),
            self.query_one("#backend-model", Input).value.strip(),
            self.query_one("#backend-key", Input).value.strip(),
            self.query_one("#backend-ctx", Input).value.strip(),
        )

    def _parse_ctx(self, raw: str) -> int | None | bool:
        """Parsed context length, or False if the field is non-numeric."""
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return False

    # ----------------------------------------------------------------- submit

    @on(Input.Submitted)
    def _on_submitted(self, event: Input.Submitted) -> None:
        self.run_worker(self._attempt_save(), exclusive=True)

    async def _attempt_save(self) -> None:
        base_url, model, key, ctx_raw = self._fields()
        if not base_url:
            self._set_status("Enter the endpoint URL, e.g. http://localhost:20001/v1")
            return
        ctx_override = self._parse_ctx(ctx_raw)
        if ctx_override is False:
            self._set_status("Context length must be a whole number.")
            return

        self._set_status("Checking endpoint…")
        results = await probe_endpoint(base_url, api_key=key or None)

        if not results:
            self._set_status(f"No OpenAI-compatible API answered at {base_url}.")
            return
        if len(results) == 1 and results[0].needs_key:
            reason = "The key was rejected." if key else "This endpoint needs a key."
            self._set_status(f"{reason} Fix it, or press ctrl+s to save anyway.")
            return

        chosen = model
        if not chosen:
            if len(results) == 1:
                chosen = results[0].model
            else:
                chosen = await self.app.push_screen_wait(
                    _ModelPickScreen([r.model for r in results])
                )
                if not chosen:
                    self._set_status("Pick a model to continue.")
                    return

        matched = next((r for r in results if r.model == chosen), None)
        max_len = ctx_override or (matched.max_model_len if matched else None)
        self.dismiss(
            LLMBackend(
                model=chosen,
                base_url=base_url,
                api_key=key or None,
                max_model_len=max_len,
            )
        )

    def action_save_anyway(self) -> None:
        base_url, model, key, ctx_raw = self._fields()
        if not base_url or not model:
            self._set_status(
                "Saving without a check still needs a URL and a model id."
            )
            return
        ctx_override = self._parse_ctx(ctx_raw)
        if ctx_override is False:
            self._set_status("Context length must be a whole number.")
            return
        self.dismiss(
            LLMBackend(
                model=model,
                base_url=base_url,
                api_key=key or None,
                max_model_len=ctx_override or None,
            )
        )

    def action_cancel(self) -> None:
        self.dismiss(None)
