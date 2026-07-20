"""Modals for the memory workflows (§6.3)."""

from __future__ import annotations

from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical
from textual.content import Content
from textual.screen import ModalScreen
from textual.widgets import Static

from hpca.agent.conclude import MemoryProposal
from hpca.memory_ops import MemoryOp


class TierSelectScreen(ModalScreen[int | None]):
    """`\\memorize`: pick the tier for the new memory; default is tier 2."""

    BINDINGS = [
        Binding("escape", "cancel", "cancel", priority=True),
        Binding("1", "pick(1)", "tier 1"),
        Binding("2", "pick(2)", "tier 2"),
        Binding("enter", "pick(2)", "tier 2 (default)", priority=True),
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
                "(esc) cancel · (1) global standing note · "
                "(2) profile memory [default]",
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


class MemoryBatchScreen(ModalScreen[bool]):
    """The `memory` tool's batch: approve or reject it whole (§6.3, P3).

    Whole-batch rather than per-operation, because a batch is often a trade —
    remove two stale entries to make room for one new one — and approving
    half of that leaves memory in a state nobody chose.
    """

    BINDINGS = [
        Binding("y", "accept", "accept"),
        Binding("n", "reject", "reject"),
        Binding("escape", "reject", "reject", priority=True),
    ]

    DEFAULT_CSS = """
    MemoryBatchScreen { align: center middle; }
    #batch-dialog {
        width: 76;
        height: auto;
        max-height: 80%;
        border: heavy $accent;
        background: $surface;
        padding: 1 2;
    }
    #batch-hint { color: $text-muted; }
    #batch-warning { color: $warning; }
    """

    def __init__(self, operations: list[MemoryOp], flagged: list[str]) -> None:
        super().__init__()
        self._operations = operations
        self._flagged = flagged

    def compose(self) -> ComposeResult:
        count = len(self._operations)
        with Vertical(id="batch-dialog"):
            yield Static(
                f"The agent proposes {count} memory change"
                f"{'' if count == 1 else 's'}",
                id="batch-title",
            )
            yield Static(
                Content(
                    "\n".join(f"• {op.describe()}" for op in self._operations)
                ),
                id="batch-text",
            )
            if self._flagged:
                yield Static(
                    Content(
                        "⚠ text matches a prompt-injection pattern "
                        f"({', '.join(self._flagged[:2])}) — read it closely"
                    ),
                    id="batch-warning",
                )
            yield Static("(y) apply all · (n) discard", id="batch-hint")

    def action_accept(self) -> None:
        self.dismiss(True)

    def action_reject(self) -> None:
        self.dismiss(False)


class ReflectionScreen(ModalScreen[bool]):
    """One self-review proposal: a memory, a struggle note, or a skill
    change (redesign Phase 4). Skill changes show what would be written, so
    approving a patch never means approving text nobody read."""

    BINDINGS = [
        Binding("y", "accept", "accept"),
        Binding("n", "reject", "reject"),
        Binding("escape", "reject", "reject", priority=True),
    ]

    DEFAULT_CSS = """
    ReflectionScreen { align: center middle; }
    #reflect-dialog {
        width: 76;
        height: auto;
        max-height: 80%;
        border: heavy $accent;
        background: $surface;
        padding: 1 2;
    }
    #reflect-hint { color: $text-muted; }
    #reflect-source { color: $text-muted; }
    """

    def __init__(self, proposal, index: int, total: int) -> None:
        super().__init__()
        self._proposal = proposal
        self._index = index
        self._total = total

    def compose(self) -> ComposeResult:
        with Vertical(id="reflect-dialog"):
            yield Static(
                f"Self-review proposal {self._index}/{self._total} — "
                f"{self._proposal.describe()}",
                id="reflect-title",
            )
            yield Static(Content(self._proposal.text), id="reflect-text")
            if self._proposal.keywords:
                yield Static(
                    Content("matches: " + ", ".join(self._proposal.keywords)),
                    id="reflect-source",
                )
            yield Static("(y) keep · (n) discard", id="reflect-hint")

    def action_accept(self) -> None:
        self.dismiss(True)

    def action_reject(self) -> None:
        self.dismiss(False)
