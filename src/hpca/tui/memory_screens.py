"""Modals for the memory workflows (§6.3)."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Static

from hpca.agent.conclude import MemoryProposal


class TierSelectScreen(ModalScreen[int | None]):
    """`\\memorize`: pick the tier for the new memory; default is tier 2."""

    BINDINGS = [
        Binding("1", "pick(1)", "tier 1"),
        Binding("2", "pick(2)", "tier 2"),
        Binding("enter", "pick(2)", "tier 2 (default)", priority=True),
        Binding("escape", "cancel", "cancel", priority=True),
    ]

    DEFAULT_CSS = """
    TierSelectScreen { align: center middle; }
    #tier-dialog {
        width: 60;
        height: auto;
        border: heavy $accent;
        background: $surface;
        padding: 1 2;
    }
    #tier-hint { color: $text-muted; }
    """

    def __init__(self, text: str) -> None:
        super().__init__()
        self._text = text

    def compose(self) -> ComposeResult:
        with Vertical(id="tier-dialog"):
            yield Static("Memorize into which tier?", id="tier-title")
            yield Static(Content(self._text), id="tier-text")
            yield Static(
                "(1) global standing note · (2) profile memory [default] · "
                "(esc) cancel",
                id="tier-hint",
            )

    def action_pick(self, tier: int) -> None:
        self.dismiss(tier)

    def action_cancel(self) -> None:
        self.dismiss(None)


class MemoryProposalScreen(ModalScreen[bool]):
    """`\\conclude`: approve or reject one proposed memory."""

    BINDINGS = [
        Binding("y", "accept", "accept"),
        Binding("n", "reject", "reject"),
        Binding("escape", "reject", "reject", priority=True),
    ]

    DEFAULT_CSS = """
    MemoryProposalScreen { align: center middle; }
    #proposal-dialog {
        width: 70;
        height: auto;
        max-height: 80%;
        border: heavy $accent;
        background: $surface;
        padding: 1 2;
    }
    #proposal-hint { color: $text-muted; }
    """

    def __init__(self, proposal: MemoryProposal, index: int, total: int) -> None:
        super().__init__()
        self._proposal = proposal
        self._index = index
        self._total = total

    def compose(self) -> ComposeResult:
        p = self._proposal
        with Vertical(id="proposal-dialog"):
            yield Static(
                f"Proposed memory {self._index}/{self._total} — "
                f"tier {p.tier}, {p.kind}",
                id="proposal-title",
            )
            yield Static(Content(p.text), id="proposal-text")
            yield Static("(y) keep · (n) discard", id="proposal-hint")

    def action_accept(self) -> None:
        self.dismiss(True)

    def action_reject(self) -> None:
        self.dismiss(False)
