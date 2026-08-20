"""What every screen drawn over the rows has in common."""

from __future__ import annotations


class Overlay:
    """A screen drawn over the rows. ``handle`` returning False closes it."""

    title = ""

    def render(self, width: int, height: int) -> list[str]:  # pragma: no cover
        raise NotImplementedError

    def paste(self, text: str) -> None:
        """A block the terminal handed over whole. Ignored unless the screen
        has somewhere to put text — the key list has not."""

    def footer(self) -> list[tuple[str, str]]:
        return [("esc", "back")]
