"""The slash commands and the menu that offers them (§4.3 items 24 and 25).

Pure functions over plain data, deliberately: what a typed line *means* has to
be decidable without a core, a database or a terminal, and the menu is then one
more thing `RowUI.render` can be asked for as a string.

Three rules from `specs-ui-acceptance.md`, "Slash-command menu", live here:

* **Substring, not prefix.** ``skill`` finds every command with "skill" in the
  name, which is how a user who remembers half a name finds the whole one.
* **The menu closes as soon as the token contains whitespace.** The command is
  settled at that point and what follows is its arguments — or, after a
  shift+enter, the body of a multi-line message. An open menu owns ↑/↓, and
  leaving it up made every draft that opens with "/" untraversable for as long
  as it was being written.
* **Nothing here is markup.** A skill's description is user-written and is
  drawn as the characters it is made of; the Textual menu had to assemble
  styled spans to stop a bracket being eaten, and this renderer has no parser
  to protect it from.

**Frequency ordering, and where the numbers come from.** The core records
`command_usage` on every `command.run` and rule 2 of §4.2 says the UI never
reads the database — so the counts arrive as an event (`command.counts`),
asked for on connect and restated whenever one changes, and `RowUI.menu` hands
the mapping to `matching`. With no counts yet the sort collapses to definition
order, which is what a table nobody has run anything from would give anyway.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

from hpca.ui.ansi import BOLD, CYAN, DIM, RESET, REVERSE, cut, pad, rule, safe

# Both prefixes, everywhere. `\` is what a user whose keyboard layout puts `/`
# behind a modifier reaches for, and the Textual entry accepted it from the
# start ("`\memorize` (backslash) also works").
PREFIXES = ("/", "\\")


@dataclass(frozen=True)
class Command:
    """One entry in the menu: what to type, and what typing it does."""

    name: str
    usage: str
    # Whether HPCA ships it. The menu marks these and nothing else, because the
    # two things it mixes — built-ins and the profile's skills — are otherwise
    # indistinguishable, and bolding both would bury the names it exists to
    # help pick between.
    builtin: bool = True
    # Whether the command acts on one conversation. The profile-scoped ones
    # (`/skills-list`, `/skill-creator`, `/skill-remove`) are still *sent* with
    # the open session's id where there is one, because that is how the core
    # decides which profile is asking (`core.service._list_skills`); this flag
    # is what the UI checks before refusing a command with nothing open.
    session: bool = True


# The seven built-ins (§4.3 item 24), in definition order — which is also the
# order the menu shows them in until the first `command.counts` arrives.
# The strings are `tui/app.py`'s COMMANDS, kept word for word: they are the
# only documentation most of these commands have.
BUILTINS: tuple[Command, ...] = (
    Command(
        "memorize",
        "/memorize <note> — form memories from the note and this conversation",
    ),
    Command("conclude", "/conclude — propose memories from this conversation"),
    Command(
        "compact",
        "/compact [what to keep / what you do next] — fold this conversation "
        "into a summary and free the context",
    ),
    Command(
        "skill-creator",
        "/skill-creator [what it should do] — add a skill; with a request the "
        "model drafts it into the form first",
        session=False,
    ),
    Command(
        "skills-list",
        "/skills-list — list this profile's skills",
        session=False,
    ),
    Command(
        "skill-remove",
        "/skill-remove — remove one of this profile's skills",
        session=False,
    ),
    Command(
        "thinking",
        "/thinking — how hard this session's model reasons "
        "(off / low / medium / xhigh)",
    ),
)

BUILTIN_NAMES = frozenset(x.name for x in BUILTINS)

# What the menu's rule says it is for. The keys are named because the two that
# fill a partial command are the two nobody guesses.
MENU_TITLE = "commands"
MENU_HINT = "bold = built-in · ↑↓ select · ⇥ or ⏎ complete"


def skill_command(name: str, description: str = "") -> Command:
    """One of the profile's skills, as the menu draws it."""
    usage = f"/{name} — {description}" if description else f"/{name}"
    return Command(name, usage, builtin=False)


def all_commands(skills: Iterable[tuple[str, str]] = ()) -> list[Command]:
    """The built-ins plus every visible skill, built-ins winning a name clash.

    "Visible" is the wide `skill.list` scope — the shipped skills, the shared
    ones, the profile's own and the project's — which is what makes `/plan`
    offerable on an install whose profile has no skills of its own.

    A skill whose name carries whitespace is left out rather than listed and
    unreachable: `split` takes the name up to the first space, so `/<skill>`
    could never name it.
    """
    out = list(BUILTINS)
    for name, description in skills:
        if not name or name in BUILTIN_NAMES:
            continue
        if any(ch.isspace() for ch in name):
            continue
        if any(x.name == name for x in out):
            continue  # two profiles' lists overlapping, or a duplicate file
        out.append(skill_command(name, description))
    return out


def typed_name(draft: str) -> str | None:
    """The command name being typed, or None when there is no menu to show.

    None covers three different "no": the draft is not a command at all, the
    name is settled (a space follows it), or the draft is a multi-line message
    whose first line merely happens to start with a slash.
    """
    stripped = draft.lstrip()
    if not stripped.startswith(PREFIXES):
        return None
    typed = stripped[1:]
    if any(ch.isspace() for ch in typed):
        return None
    return typed


def split(text: str) -> tuple[str, str] | None:
    """``/name the rest`` as ``(name, rest)``, or None when it is not one.

    The name is taken up to the first whitespace and the rest is handed on
    unsplit, which is `protocol.CommandRun`'s bargain: what ``rest`` means is
    the handler's business — a note, a level, a skill name, nothing.
    """
    stripped = text.strip()
    if not stripped.startswith(PREFIXES):
        return None
    name, _, rest = stripped[1:].partition(" ")
    return name.strip(), rest.strip()


def matching(
    typed: str,
    commands: Sequence[Command],
    counts: Mapping[str, int] | None = None,
) -> list[Command]:
    """Every command whose name contains ``typed``, best first.

    "Best" is most-used first and then definition order. ``counts`` is what
    `command.counts` last said (`RowUI.command_counts`); with none — a core
    that has counted nothing, or an answer that has not arrived — this is a
    stable filter and the order is the order `all_commands` built.
    """
    needle = typed.lower()
    found = [x for x in commands if needle in x.name.lower()]
    if not counts:
        return found
    order = {x.name: i for i, x in enumerate(commands)}
    found.sort(key=lambda x: (-counts.get(x.name, 0), order[x.name]))
    return found


def menu_rows(
    matches: Sequence[Command], index: int, width: int, height: int
) -> list[str]:
    """The menu, exactly ``min(len, height)`` rows of exactly ``width`` cells.

    Every row is built plain, cut to the width and padded *before* any SGR is
    wrapped around it, which is the same discipline `ansi.pad` exists for: a
    description is arbitrary text off a user's disk, and a row whose real width
    depends on what was in it is a row that shifts the differential repaint.
    """
    if not matches or height < 1 or width < 4:
        return []
    shown = list(matches[: max(1, height)])
    # Keep the highlighted row on screen when the list is longer than the box.
    if index >= len(shown):
        start = min(index - len(shown) + 1, len(matches) - len(shown))
        shown = list(matches[start : start + len(shown)])
        index -= start
    out = []
    for at, command in enumerate(shown):
        marker = "▸ " if at == index else "  "
        body = safe(command.usage).replace("\n", " ")
        row = pad(marker + body, width)
        head = f"{marker}/{command.name}"
        if command.builtin and row.startswith(head):
            row = BOLD + CYAN + row[: len(head)] + RESET + row[len(head) :]
        out.append(REVERSE + row + RESET if at == index else row)
    return out


def menu_title(matches: Sequence[Command], width: int) -> str:
    """The rule over the menu, with the hint dropped on a narrow terminal."""
    label = f"{MENU_TITLE} ({len(matches)})"
    hint = MENU_HINT if width >= len(label) + len(MENU_HINT) + 8 else ""
    return DIM + rule(label, width, hint) + RESET


def unknown(name: str) -> str:
    """What a mistyped command is answered with. Kept here so the phrase the
    UI shows and the phrase a test looks for cannot drift apart."""
    return f"unknown command: /{cut(name, 40)} — fix it or press esc"
