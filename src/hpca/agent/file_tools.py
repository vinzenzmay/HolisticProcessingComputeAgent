"""File operation tools (§5.1, §5.3): all destructive paths trash-backed.

Every tool here names its target with a plain path (``hpca.paths``), absolute
or relative to the session's working directory — the shape every file tool in
every agent corpus the model was trained on has. Deletions and overwrites are
HITL-gated; a plain move/copy to a fresh target is not (conditional
``is_destructive_call``).

``restore_file`` is the other half of the trash (§5.3): without it the backup
a deletion writes is only reachable by the user digging through the app dir by
hand, which is exactly what happened the first time someone asked for a file
back.

``create_file`` and ``edit_file`` are the two tools here that write content,
and they split "new file" from "change a file" rather than overlapping.
``edit_file`` exists for the cost: without it the only way to change a
400-line script is to re-send all 400 lines through ``create_script``, paying
for the file twice in one window. ``create_file`` exists because prose had no
tool at all — ``create_script`` writes only into the scripts dir under a
language suffix — so a specs.md had to go through a ``cat << 'EOF'`` heredoc
in run_bash.

Both are held to the same rules as everything else that writes: a script's
content faces §5.2's syntax and code-vs-docs gate before it lands, checked on
a scratch copy so a refusal leaves nothing broken behind. ``edit_file``
overwrites, so it also backs the previous content up and gates; ``create_file``
refuses an existing path instead, which is what keeps it out of §5.3
entirely.
"""

from __future__ import annotations

import os
import re
import time
import unicodedata
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from hpca.agent import hints
from hpca.agent.context import ToolContext
from hpca.agent.history import carries_elision_marker
from hpca.agent.tools import Tool, ToolRegistry
from hpca.paths import PathError, resolve_path
from hpca.trash import TrashEntry

# Enough for the model to recognise the file the user means without pasting a
# week of deletions into the context.
TRASH_LIST_LIMIT = 20


def _require_trash(ctx: ToolContext):
    if ctx.trash is None:
        raise RuntimeError("Trash manager is not configured in this session")
    return ctx.trash


def _target(value: str, ctx: ToolContext) -> Path:
    """The path a tool argument names, anchored at the session workdir.

    One line, kept as a function because every handler, every gating predicate
    and every describe helper in this module has to agree on it — and because
    ``register=False`` used to be the thing they disagreed about.
    """
    return resolve_path(value, ctx.workdir)


def _existing(value: str, ctx: ToolContext) -> Path:
    """``_target`` plus the check the destructive tools need before they act.

    Raises PathError (model-facing) rather than returning a sentinel: the
    handlers catch it and hand the sentence straight back to the model, which
    is the same shape as every other refusal here.
    """
    path = _target(value, ctx)
    if not path.exists():
        raise PathError(
            f"No such path: {path}. {hints.PATH_NOT_FOUND}"
        )
    return path


class DeleteFileParams(BaseModel):
    path: str = Field(description="Path of the file to delete")


def _delete_resolvable(args: DeleteFileParams, ctx: ToolContext) -> bool:
    # Gate only calls that can actually run; a path that is not there goes
    # straight to the error-feedback loop instead of asking the user to
    # approve a dud.
    try:
        return _target(args.path, ctx).exists()
    except Exception:
        return False


def _describe_delete(args: DeleteFileParams, ctx: ToolContext) -> str:
    path = _target(args.path, ctx)
    size = path.stat().st_size if path.exists() else 0
    backed_up = ctx.trash is not None and size < ctx.trash.backup_limit_bytes
    backup_note = (
        "will be recoverable from trash"
        if backed_up
        else "NO BACKUP (file exceeds the backup size limit) — irreversible"
    )
    return f"rm {path}\n({size} bytes; {backup_note})"


def _describe_move(args: MoveFileParams, ctx: ToolContext) -> str:
    source = _target(args.source_path, ctx)
    target = _move_target(args, ctx)
    note = " (OVERWRITES existing file — old file goes to trash)" if target.exists() else ""
    return f"mv {source} → {target}{note}"


def _describe_copy(args: CopyFileParams, ctx: ToolContext) -> str:
    source = _target(args.source_path, ctx)
    target = _copy_target(args, ctx)
    note = " (OVERWRITES existing file — old file goes to trash)" if target.exists() else ""
    return f"cp {source} → {target}{note}"


async def delete_file(args: DeleteFileParams, ctx: ToolContext) -> str:
    trash = _require_trash(ctx)
    try:
        path = _existing(args.path, ctx)
    except ValueError as exc:
        # Prefixed like every other refusal here: "NOT <verb>" is the shape the
        # model reads first, and a bare sentence reads like a result.
        return f"NOT deleted: {exc}"
    entry = trash.trash(path)
    if entry.method == "none":
        return (
            f"Deleted {path} WITHOUT backup (file was above the backup size "
            "limit); this cannot be undone."
        )
    # Naming the tool, not just the trash: the model has to know recovery is
    # something it can DO, or it tells the user to go dig in the app dir.
    return (
        f"Deleted {path} (backed up via {entry.method}; recoverable with "
        f"restore_file if the user asks for it back)."
    )


class RestoreFileParams(BaseModel):
    path: str = Field(
        default="",
        description=(
            "Original path of the deleted file, as delete_file reported it "
            "(its file name alone also works). Leave empty to list what is in "
            "the trash."
        ),
    )


def _local_time(trashed_at: str) -> str:
    try:
        return f"{datetime.fromisoformat(trashed_at).astimezone():%Y-%m-%d %H:%M}"
    except ValueError:
        return trashed_at


def _newest_first(entries: list[TrashEntry]) -> list[TrashEntry]:
    # trashed_at is an ISO UTC stamp, so it sorts lexicographically. Sorting on
    # it rather than on the entry directory's name keeps the order right for
    # entries written by any past version.
    return sorted(entries, key=lambda entry: entry.trashed_at, reverse=True)


def _trash_listing(entries: list[TrashEntry]) -> str:
    if not entries:
        return "The trash is empty — there is nothing to restore."
    shown = entries[:TRASH_LIST_LIMIT]
    lines = [
        f"{entry.original_path} (deleted {_local_time(entry.trashed_at)})"
        + ("" if entry.trashed_path else " — NO backup, not restorable")
        for entry in shown
    ]
    if len(entries) > len(shown):
        lines.append(f"... and {len(entries) - len(shown)} older entries")
    return "In the trash, newest first:\n" + "\n".join(lines)


def _matching_entries(entries: list[TrashEntry], wanted: str) -> list[TrashEntry]:
    """Trash entries for ``wanted``, newest first.

    Widening rather than exact-only: the model may quote the path it saw, the
    file name alone, or a fragment of a long path, and a restore that fails on
    a near-miss costs the user their file.
    """
    name = Path(wanted).name
    for candidates in (
        [e for e in entries if str(e.original_path) == wanted],
        [e for e in entries if e.original_path.name == name],
        [e for e in entries if wanted in str(e.original_path)],
    ):
        if candidates:
            return candidates
    return []


async def restore_file(args: RestoreFileParams, ctx: ToolContext) -> str:
    """Put a trashed file back where it was (§5.3 recovery).

    Never gated: restore only ever *creates* a file, and refuses outright when
    something already sits at the original path, so there is nothing for the
    user to approve. Ambiguity is handed back to the model instead of guessed
    at — restoring the wrong file over a name collision is not recoverable in
    turn.
    """
    trash = _require_trash(ctx)
    entries = _newest_first(trash.list())
    wanted = args.path.strip()
    if not wanted:
        return _trash_listing(entries)
    matches = _matching_entries(entries, wanted)
    if not matches:
        return f"Nothing in the trash matches {wanted!r}.\n{_trash_listing(entries)}"
    if len({entry.original_path for entry in matches}) > 1:
        return (
            f"{wanted!r} matches several deleted files — call restore_file "
            f"again with the full path of the one you want:\n"
            f"{_trash_listing(matches)}"
        )
    entry = matches[0]  # the most recent deletion of that path
    if entry.trashed_path is None:
        return (
            f"Cannot restore {entry.original_path}: it was deleted without a "
            "backup (it was above the backup size limit), so no copy was kept. "
            f"{hints.TRASH_NOT_RECOVERABLE}"
        )
    try:
        restored = trash.restore(entry)
    except FileExistsError:
        return (
            f"Cannot restore {entry.original_path}: a file already exists "
            f"there. {hints.RESTORE_PATH_OCCUPIED}"
        )
    except OSError as exc:
        return f"Could not restore {entry.original_path}: {exc}"
    return f"Restored {restored}."


class EditFileParams(BaseModel):
    path: str = Field(description="Path of the file to edit")
    # Line arrays for the same reason create_script takes them: the live model
    # fills string arrays reliably and mangles \n escapes in long strings.
    old_lines: list[str] = Field(
        min_length=1,
        description=(
            "The exact consecutive lines to replace, copied from read_file "
            "without their line numbers — indentation and spacing included. "
            "Include enough surrounding lines that they occur only once."
        ),
    )
    new_lines: list[str] = Field(
        default_factory=list,
        description=(
            "The lines to put in their place, one string per line. Send an "
            "empty array to delete the old lines."
        ),
    )


def _find_runs(haystack: list[str], needle: list[str]) -> list[int]:
    """Every index where ``needle`` occurs as a run of whole lines.

    Whole lines, not a substring of the file: the model is asked for lines, so
    matching them as lines is what makes "occurs twice" mean what it says — and
    keeps ``fi`` from matching inside ``pipefail``.
    """
    span = len(needle)
    return [
        index
        for index in range(len(haystack) - span + 1)
        if haystack[index : index + span] == needle
    ]


# The repair-and-fuzz layer below exists to absorb model imperfection inside
# the tool instead of bouncing errors back into the retry loop: the live model
# (a mid-size Qwen behind constrained decoding) reliably produces *almost*
# right arguments — an embedded "\n" inside one array element, line numbers
# copied straight from read_file's numbered listing, a smart quote where the
# file has ASCII — and every bounce costs a round-trip plus, usually, a full
# re-read of the file.

def _split_embedded_newlines(lines: list[str]) -> list[str]:
    """One string per line, even when the model packed several into one.

    The params ask for arrays of lines, but a model that thinks in blocks
    sometimes sends ["a\\nb"] for two lines. The intent is unambiguous, so
    split rather than refuse.
    """
    out: list[str] = []
    for line in lines:
        out.extend(line.split("\n"))
    return out


# The one check in this neighbourhood that refuses instead of repairing, and
# for the opposite reason: an elision marker in a payload is not an argument
# the tool can guess the intent of, it is the model quoting its own history
# back at itself. hpca.agent.history replaces a large payload in the assistant
# copy of a call with a descriptor of what it left out; a model asked to
# rewrite a file it had written earlier was observed sending that descriptor
# back as the new content, which put the marker on disk and shrank the file to
# whatever had survived the elision — and since the marker is valid text,
# nothing downstream noticed. So the check has to live here, in front of every
# write, and it has to say *why* the content is being refused: from inside the
# turn the marker reads like a line of the file.

# How much of the offending line comes back quoted. Enough to recognise which
# line it is, short enough that a long descriptor does not push the rest of the
# refusal — the part that says what to do instead — off the end of the result.
def _elision_refusal(
    lines: list[str], *, refusal: str, field: str, outcome: str
) -> str:
    """The refusal for lines carrying an elision marker, or "" if they are clean.

    Deliberately does NOT quote the offending line, which is the opposite of
    what every other refusal in this module does. Measured on the live 27B
    (evals/edit_eval.py, ``second_file_after_first``): a refusal that quoted the
    marker put the marker back into the context, the model copied it out of the
    refusal into its next call, and that call was refused in the same words —
    seventeen times in one run, until the decision budget died. The quote is a
    reinforcement loop, and the line number alone identifies the line just as
    well.

    What is left says the two things the model cannot work out from inside the
    turn: that this came from its own record rather than from the file, and
    that the file on disk is where the text still is. The heredoc escape is
    named because documentation *about* the marker is a legitimate file to want
    to write, and this check has no way to tell it apart.
    """
    for index, line in enumerate(lines):
        if not carries_elision_marker(line):
            continue
        return (
            f"{refusal}: {field} line {index + 1} is not file content — it is a "
            "placeholder the session history left in place of a payload it did "
            "not keep, so what you sent is your own record of an earlier call "
            f"rather than the file. {outcome} {hints.ELISION_REWRITE_FILE}"
        )
    return ""


# A line-number prefix as read_file (or `cat -n`, or an editor) renders it:
# optional indent, digits, then ":", a tab, or "|" — e.g. "12: x", "12\tx",
# " 12 | x".
_LINE_NUMBER_PREFIX = re.compile(r"^\s*\d+\s*(?::|\t|\|)\s?")


def _strip_line_number_prefixes(lines: list[str]) -> list[str] | None:
    """The lines without a copied numbered-listing prefix, or None.

    Only fires when EVERY element carries the prefix — one numbered line in
    ten is genuine content (a dict entry, a timestamp), ten in ten is a model
    copying a numbered listing. The caller additionally tries the unstripped
    lines first, so genuine content that actually matches is never mangled.
    """
    if not all(_LINE_NUMBER_PREFIX.match(line) for line in lines):
        return None
    return [_LINE_NUMBER_PREFIX.sub("", line, count=1) for line in lines]


# Fuzzy normalization, per line (modelled on PI's normalizeForFuzzyMatch):
# smart quotes → ASCII quotes, Unicode dashes (U+2010..U+2015, minus U+2212)
# → "-", special spaces (NBSP and friends) → " ". Applied after NFKC, which
# already folds most compatibility characters.
_FUZZY_TRANSLATE = str.maketrans(
    {
        # smart single quotes U+2018..U+201B
        **dict.fromkeys(map(ord, "\u2018\u2019\u201a\u201b"), "'"),
        # smart double quotes U+201C..U+201F
        **dict.fromkeys(map(ord, "\u201c\u201d\u201e\u201f"), '"'),
        # hyphen, non-breaking hyphen, figure/en/em dash, horizontal bar, minus
        **dict.fromkeys(
            map(ord, "\u2010\u2011\u2012\u2013\u2014\u2015\u2212"), "-"
        ),
        # NBSP, en/em-family spaces, narrow NBSP, math space, ideographic space
        **dict.fromkeys(
            map(
                ord,
                "\u00a0\u2002\u2003\u2004\u2005\u2006\u2007\u2008"
                "\u2009\u200a\u202f\u205f\u3000",
            ),
            " ",
        ),
    }
)


def _fuzzy_line(line: str) -> str:
    """A line as fuzzy matching sees it. Leading whitespace survives on
    purpose: indentation is meaning (Python), trailing whitespace never is."""
    return (
        unicodedata.normalize("NFKC", line).translate(_FUZZY_TRANSLATE).rstrip()
    )


# The matching ladder, strictest first: exact; trailing whitespace stripped;
# full fuzzy. A unique match at ANY level applies the edit — asking the model
# to fix a trailing space it cannot even see is a wasted round-trip.
_MATCH_LEVELS = (None, str.rstrip, _fuzzy_line)


def _match_ladder(file_lines: list[str], old_lines: list[str]) -> list[int]:
    """Hit indices at the strictest level that matches at all.

    Ambiguity is judged at the level that matched: two exact occurrences are
    two occurrences, and never get "rescued" by a looser level seeing three.
    """
    for transform in _MATCH_LEVELS:
        if transform is None:
            hay, needle = file_lines, old_lines
        else:
            hay = [transform(line) for line in file_lines]
            needle = [transform(line) for line in old_lines]
        hits = _find_runs(hay, needle)
        if hits:
            return hits
    return []


def _near_miss(file_lines: list[str], old_lines: list[str]) -> str:
    """Where the lines nearly match, for the usual whitespace near-miss.

    A model that mis-indents its copy gets back "line 12" instead of a flat
    "not found", which is the difference between fixing the call and re-reading
    the whole file.
    """
    stripped = [line.strip() for line in file_lines]
    wanted = [line.strip() for line in old_lines]
    hits = _find_runs(stripped, wanted)
    if not hits:
        return ""
    where = ", ".join(f"line {index + 1}" for index in hits[:5])
    return (
        f" The same text ignoring leading/trailing whitespace is at {where} — "
        "read the file again and copy the indentation exactly."
    )


def _edit_target(args: EditFileParams, ctx: ToolContext) -> Path:
    return _target(args.path, ctx)


def edit_target_path(args: EditFileParams, ctx: ToolContext) -> Path | None:
    """The file an edit_file call would touch, or None when it cannot resolve.

    The graph's per-file approval memory (§3.5 auto mode) keys on this: two
    calls that resolve to the same path are edits of the same file, whether the
    model wrote it absolute or relative to the workdir. Swallows resolution
    errors — an unresolvable call never skips a gate, it just fails normally.
    """
    try:
        return _edit_target(args, ctx).resolve()
    except Exception:
        return None


def _edit_resolvable(args: EditFileParams, ctx: ToolContext) -> bool:
    # Overwriting a file's content is destructive (§5.3), so every edit that
    # can actually run gates; a dud path goes to the error-feedback loop instead
    # of asking the user to approve something that will not happen.
    try:
        return _edit_target(args, ctx).is_file()
    except Exception:
        return False


def _lines(count: int) -> str:
    return f"{count} line" if count == 1 else f"{count} lines"


def _tbd_note(content: str) -> str:
    """The unfinished-work reminder for a skeleton-then-fill write.

    The guidance teaches `TBD: ...` placeholder lines; this is the tool's
    side of that protocol. A 27B filling ten sections loses count (measured:
    sections left as TBD, or the model answering 'done' with two remaining),
    and the fix that costs nothing until it matters is the result naming what
    is left after every write.
    """
    remaining = [
        (i + 1, line.strip())
        for i, line in enumerate(content.split("\n"))
        if line.lstrip().startswith("TBD")
    ]
    if not remaining:
        return ""
    line_no, text = remaining[0]
    return (
        f" {len(remaining)} placeholder line(s) still to fill, next at "
        f"line {line_no}: {text!r}."
    )


def _change(args: EditFileParams) -> str:
    """The size of the change, in the one phrasing the prompt and the result
    both use: "3 lines → 2 lines", or "3 lines deleted" when nothing goes back.
    Counted after embedded-\\n repair, so the numbers match what lands."""
    old = _split_embedded_newlines(list(args.old_lines))
    new = _split_embedded_newlines(list(args.new_lines))
    if not new:
        return f"{_lines(len(old))} deleted"
    return f"{_lines(len(old))} → {_lines(len(new))}"


def _read_for_edit(path: Path) -> tuple[str, str, str]:
    """(LF-normalized text, bom, dominant line ending) of ``path``.

    Matching happens on LF-normalized, BOM-less text — the model copies lines
    out of read_file and never sees a BOM or a \\r — and the write restores
    both, so an edit does not silently re-terminate a CRLF file.
    """
    raw = path.read_bytes().decode()
    bom, text = ("\ufeff", raw[1:]) if raw.startswith("\ufeff") else ("", raw)
    # Dominant = first: a file that opens with \r\n is a CRLF file, whatever a
    # stray later line does (mirrors PI's detectLineEnding).
    crlf, lf = text.find("\r\n"), text.find("\n")
    ending = "\r\n" if crlf != -1 and lf != -1 and crlf < lf else "\n"
    return text.replace("\r\n", "\n").replace("\r", "\n"), bom, ending


def _describe_edit(args: EditFileParams, ctx: ToolContext) -> str:
    path = _edit_target(args, ctx)
    size = path.stat().st_size if path.is_file() else 0
    backed_up = ctx.trash is not None and size < ctx.trash.backup_limit_bytes
    backup_note = (
        "the current file is copied to trash first"
        if backed_up
        else "NO BACKUP (file exceeds the backup size limit) — irreversible"
    )
    return f"edit {path}\n({_change(args)}; {backup_note})"


def edit_preview(arguments: dict, ctx: object = None) -> str:
    """One edit as the user judges it: the lines out, then the lines in.

    The whole file is not the call — the change is — so this is what the
    approval prompt and the chat's call box show (see ``modes.script_preview``).
    Reads nothing and never raises: it runs on every call, and again whenever a
    parked turn resumes.
    """
    out = [f"- {line}" for line in arguments.get("old_lines") or []]
    into = [f"+ {line}" for line in arguments.get("new_lines") or []]
    return "\n".join(out + into)


async def edit_file(args: EditFileParams, ctx: ToolContext) -> str:
    """Replace an exact run of lines in a text file.

    The point is the file that stays: a one-line fix to a 400-line script costs
    the two lines, not the script twice over (once read, once rewritten). What
    it must not become is a way around the rest of §5: the previous content
    goes to the trash exactly as a deletion's would, and when the file is a
    script the edited content faces §5.2's gate before it is written — checked
    on a scratch copy, so a refused edit leaves the real file untouched rather
    than briefly broken.
    """
    from hpca.agent.builtin_tools import KIND_BY_SUFFIX, check_script_content

    try:
        path = _existing(args.path, ctx)
    except ValueError as exc:
        return f"NOT edited: {exc}"
    if path.is_dir():
        return (
            f"NOT edited: {path} is a directory. "
            f"{hints.EDIT_FILE_ON_DIRECTORY}"
        )
    try:
        text, bom, ending = _read_for_edit(path)
    except (UnicodeDecodeError, OSError) as exc:
        return f"NOT edited: {path} could not be read as text ({type(exc).__name__})."

    # Repair before matching: embedded \n split always (the intent is
    # unambiguous), numbered-prefix stripping only as a fallback when the
    # lines as sent match nothing — genuine "12: x" content that IS in the
    # file matches first and is never mangled.
    old_lines = _split_embedded_newlines(list(args.old_lines))
    new_lines = _split_embedded_newlines(list(args.new_lines))
    # new_lines only, and old_lines deliberately not: a file that already has a
    # marker written into it is repaired by matching that line and replacing
    # it, so old_lines has to be able to carry one. Guarding both would leave
    # the corruption unfixable by the tool that caused it.
    refused_marker = _elision_refusal(
        new_lines,
        refusal="NOT edited",
        field="new_lines",
        outcome="The file is unchanged.",
    )
    if refused_marker:
        return refused_marker
    file_lines = text.split("\n")
    hits = _match_ladder(file_lines, old_lines)
    if not hits:
        stripped = _strip_line_number_prefixes(old_lines)
        if stripped is not None:
            hits = _match_ladder(file_lines, stripped)
            if hits:
                old_lines = stripped
    if not hits:
        return (
            f"NOT edited: those lines are not in {path}."
            f"{_near_miss(file_lines, old_lines)}"
        )
    if len(hits) > 1:
        where = ", ".join(f"line {index + 1}" for index in hits[:5])
        return (
            f"NOT edited: those lines occur {len(hits)} times in {path} ({where}), "
            f"so which one you mean is ambiguous. {hints.EDIT_FILE_AMBIGUOUS}"
        )
    # A unique fallback-level match applies: the file's own lines outside the
    # run are untouched (whole-line replacement, so nothing gets normalized
    # that the edit did not touch), and new_lines land verbatim.
    start = hits[0]
    after = start + len(old_lines)
    edited = "\n".join(file_lines[:start] + new_lines + file_lines[after:])
    if edited == text:
        return (
            f"NOT edited: the replacement produces identical content — "
            f"old_lines and new_lines are the same. {hints.EDIT_FILE_NO_OP}"
        )

    warnings: list[str] = []
    kind = KIND_BY_SUFFIX.get(path.suffix)
    if kind is not None:
        # §5.2 on a scratch copy: the checkers read a file, and the real one
        # must not spend even a moment holding content the gate would refuse.
        ctx.scripts_dir.mkdir(parents=True, exist_ok=True)
        scratch = ctx.scripts_dir / f"edit_{time.time_ns()}{path.suffix}"
        scratch.write_text(edited)
        try:
            refused, warnings = await check_script_content(
                kind, scratch, edited, ctx, refusal="NOT edited"
            )
        finally:
            scratch.unlink(missing_ok=True)
        if refused:
            return refused

    entry = _require_trash(ctx).backup(path)
    # Write back what the file was, not what matching needed: dominant line
    # ending and BOM restored, so an edit never re-terminates a CRLF file.
    out = edited.replace("\n", ending) if ending != "\n" else edited
    path.write_bytes((bom + out).encode())
    extra = f" ({'; '.join(warnings)})" if warnings else ""
    # One short line on success (§ result-message diet): the standing undo
    # lecture cost tokens on every edit; the trash behavior itself is
    # unchanged, and only the irreversible case still warns. The read-back
    # clause is the one thing worth saying every time — see create_file, where
    # the same measured habit of confirming a write with a read_file costs a
    # round trip and re-inflates the context with lines the model just sent.
    no_backup = (
        ""
        if entry.trashed_path
        else " NO backup was kept (the file is above the backup size limit), "
        "so this cannot be undone."
    )
    return (
        f"Edited {path} at line {start + 1}: {_change(args)}{extra}."
        f" {hints.NO_READ_BACK}{no_backup}{_tbd_note(edited)}"
    )


class CreateFileParams(BaseModel):
    path: str = Field(
        description=(
            "Path for the new file, e.g. '/data/project/specs.md'. Missing "
            "parent directories are created"
        )
    )
    # Last, and an array of lines, for the two reasons the module and
    # hpca.agent.middleware give: long strings get their \n escapes mangled,
    # and nothing may follow a long array.
    content_lines: list[str] = Field(
        min_length=1,
        description="File content as an array of lines, one string per line",
    )


def _describe_create(args: CreateFileParams, ctx: ToolContext) -> str:
    """Where this call would put the file, and how much of it there is.

    create_file is not gated — it refuses rather than overwrites, so there is
    nothing for §5.3 to ask about — but every call is described, gated or not,
    because the chat row shows the description too. Without one the row falls
    back to the arguments, which say the same thing with less of it resolved. Side-effect-free like the other describers, and never raising is the
    caller's guarantee, not a reason to be careless here.
    """
    return f"create {_target(args.path, ctx)}\n({len(args.content_lines)} lines)"


async def create_file(args: CreateFileParams, ctx: ToolContext) -> str:
    """Write a new text file.

    The gap this fills is prose. ``create_script`` writes only into the scripts
    dir under a language suffix, and ``edit_file`` needs a file to already
    exist, so the one way to author a specs.md was a ``cat << 'EOF'`` heredoc
    through run_bash — a hundred lines of documentation squeezed through bash
    quoting, where a single stray line costs the whole file (that is exactly
    how the swallowed-argument bug in hpca.agent.middleware surfaced).

    It creates and does not overwrite: an existing path is refused and pointed
    at edit_file. That keeps it non-destructive by construction — nothing for
    §5.3 to gate, so writing a document does not stop for approval — and keeps
    "change a file" as one tool rather than two ways in.
    """
    from hpca.agent.builtin_tools import KIND_BY_SUFFIX, check_script_content

    try:
        path = _target(args.path, ctx)
    except ValueError as exc:
        return f"NOT created: {exc}"
    # A parent that is a *file* is the one shape mkdir cannot make. Say which
    # component it is rather than letting a NotADirectoryError out: from
    # inside the turn the two look nothing alike.
    for parent in path.parents:
        if parent.exists():
            if not parent.is_dir():
                return (
                    f"NOT created: {parent} is a file, not a directory, so "
                    f"{path} cannot be made. {hints.CREATE_FILE_PARENT_IS_FILE}"
                )
            break
    if path.is_dir():
        return (
            f"NOT created: {path} is a directory. "
            f"{hints.CREATE_FILE_PATH_IS_DIR}"
        )
    if path.exists():
        return f"NOT created: {path} already exists. {hints.CREATE_FILE_EXISTS}"
    # Same embedded-\n repair as edit_file: the array is lines, but a model
    # that packs a block into one element still means the lines it contains.
    content_lines = _split_embedded_newlines(list(args.content_lines))
    # Ahead of the §5.2 scratch write below, not merely ahead of the real one:
    # a refused file must never have existed at the target path, and there is
    # nothing to gain from syntax-checking a payload that is not going to be
    # written either way.
    refused_marker = _elision_refusal(
        content_lines,
        refusal="NOT created",
        field="content_lines",
        outcome="Nothing was written.",
    )
    if refused_marker:
        return refused_marker
    content = "\n".join(content_lines) + "\n"

    warnings: list[str] = []
    kind = KIND_BY_SUFFIX.get(path.suffix)
    if kind is not None:
        # §5.2 on a scratch copy, as edit_file does: a second way to put
        # content into a script file must not be a second way around the gate,
        # and a refused file must never have existed at the real path.
        ctx.scripts_dir.mkdir(parents=True, exist_ok=True)
        scratch = ctx.scripts_dir / f"new_{time.time_ns()}{path.suffix}"
        scratch.write_text(content)
        try:
            refused, warnings = await check_script_content(
                kind, scratch, content, ctx, refusal="NOT created"
            )
        finally:
            scratch.unlink(missing_ok=True)
        if refused:
            return refused

    made_dirs = not path.parent.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    extra = f" ({'; '.join(warnings)})" if warnings else ""
    # A made directory is a caution, not a footnote. Every other file tool
    # fails loudly on a mistyped path because the target has to exist already;
    # this one silently makes whatever it is given, so a typo lands the file in
    # a plausible-looking wrong place and reports success. Naming it is what
    # gives the model — and the user reading the transcript — a chance to catch
    # a misspelling before the next ten calls build on it.
    made = (
        f" NOTE: {path.parent} did not exist and was created. "
        f"{hints.CREATE_FILE_MADE_DIRS}"
        if made_dirs
        else ""
    )
    # The read-back clause is worth its words: measured on the live 27B, a
    # successful create_file is routinely followed by a read_file on the file
    # just written, purely to confirm the write landed. That is a round trip
    # for nothing, and it pays for itself twice over in context — the file's
    # content, which the model supplied a moment ago, comes straight back in.
    return (
        f"Created {path} ({_lines(len(content_lines))}){extra}. "
        f"{hints.EDIT_NOT_REWRITE} "
        f"{hints.NO_READ_BACK}{made}{_tbd_note(content)}"
    )


class MoveFileParams(BaseModel):
    source_path: str = Field(description="Path of the file to move")
    dest_path: str = Field(
        description=(
            "Where to move it: a full target path, or an existing directory "
            "to move it into under its own name"
        )
    )


def _destination(source: Path, dest: str, ctx: ToolContext) -> Path:
    """`mv` semantics, because that is the shape the model already knows.

    An existing directory takes the file under its own name; anything else is
    the target path itself. The old interface asked for the directory and an
    optional new name as separate arguments, which is a distinction the shell
    does not make and the model kept getting on the wrong side of.
    """
    target = _target(dest, ctx)
    return target / source.name if target.is_dir() else target


def _move_target(args: MoveFileParams, ctx: ToolContext) -> Path:
    # Predicate/describe path only — read-only, as their purity rule requires.
    return _destination(_target(args.source_path, ctx), args.dest_path, ctx)


def _move_overwrites(args: MoveFileParams, ctx: ToolContext) -> bool:
    try:
        return _move_target(args, ctx).exists()
    except Exception:
        return False  # resolution errors surface in the handler instead


async def move_file(args: MoveFileParams, ctx: ToolContext) -> str:
    import shutil

    try:
        source = _existing(args.source_path, ctx)
    except ValueError as exc:
        return f"NOT moved: {exc}"
    target = _destination(source, args.dest_path, ctx)
    note = ""
    if target.exists():
        entry = _require_trash(ctx).trash(target)
        note = f" (previous {target.name} moved to trash via {entry.method})"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(source), target)
    return f"Moved {source} to {target}{note}."


class CopyFileParams(BaseModel):
    source_path: str = Field(description="Path of the file to copy")
    dest_path: str = Field(
        description=(
            "Where to copy it: a full target path, or an existing directory "
            "to copy it into under its own name"
        )
    )


def _copy_target(args: CopyFileParams, ctx: ToolContext) -> Path:
    # Predicate/describe path only — read-only, as their purity rule requires.
    return _destination(_target(args.source_path, ctx), args.dest_path, ctx)


def _copy_overwrites(args: CopyFileParams, ctx: ToolContext) -> bool:
    try:
        return _copy_target(args, ctx).exists()
    except Exception:
        return False


async def copy_file(args: CopyFileParams, ctx: ToolContext) -> str:
    import shutil

    try:
        source = _existing(args.source_path, ctx)
    except ValueError as exc:
        return f"NOT copied: {exc}"
    target = _destination(source, args.dest_path, ctx)
    note = ""
    if target.exists():
        entry = _require_trash(ctx).trash(target)
        note = f" (previous {target.name} moved to trash via {entry.method})"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return f"Copied {source} to {target}{note}."


# --------------------------------------------------------------- edit arms
#
# An experiment scaffold, not a shipped interface. HPCA's edit_file takes
# arrays of lines; every other agent harness (pi's `edit`, Claude Code's Edit,
# Anthropic's str_replace_based_edit_tool, Aider's SEARCH/REPLACE) takes a
# single exact string, and that is the shape agent-trained models were RL'd on.
# Arrays were chosen here against a real decoding failure — see EditFileParams
# — so which one wins on THIS backend is an empirical question, and this is how
# the eval asks it.
#
#   a (default) - today's list[str] old_lines / new_lines. Untouched.
#   b           - old_text / new_text, single strings. The trained shape.
#   c           - edits: [{old_text, new_text}], pi's exact schema: several
#                 changes to one file in one call.
#
# Only the *interface* differs. The matching ladder, the elision refusal, the
# 5.2 script gate, the trash backup and the BOM/line-ending restore are the
# same code for all three, so a difference in the numbers is a difference in
# how well the model can fill the arguments - which is the whole question.
EDIT_ARM = os.environ.get("HPCA_EDIT_ARM", "a").strip().lower() or "a"


def _text_to_lines(text: str) -> list[str]:
    """A string argument as the shared core wants it.

    Empty means "delete these lines", matching pi's newText: "" - which is why
    this cannot simply be ``text.split()``, since that yields [""], a single
    blank line, and would turn every deletion into a blank.
    """
    return text.split("\n") if text else []


class EditFileParamsB(BaseModel):
    path: str = Field(description="Path of the file to edit")
    old_text: str = Field(
        min_length=1,
        description=(
            "The exact text to replace, copied from read_file without the "
            "line numbers - indentation, spacing and line breaks included. "
            "Include enough surrounding text that it occurs only once."
        ),
    )
    new_text: str = Field(
        default="",
        description=(
            "The text to put in its place. Send an empty string to delete "
            "the old text."
        ),
    )


class EditSpec(BaseModel):
    old_text: str = Field(
        min_length=1,
        description=(
            "The exact text to replace, copied from read_file without the "
            "line numbers. Include enough surrounding text that it occurs "
            "only once."
        ),
    )
    new_text: str = Field(
        default="",
        description=(
            "The text to put in its place. Empty string deletes the old text."
        ),
    )


class EditFileParamsC(BaseModel):
    path: str = Field(description="Path of the file to edit")
    edits: list[EditSpec] = Field(
        min_length=1,
        description=(
            "Every change to make to this file, applied in order. Put all of "
            "a file's changes in one call rather than calling edit_file "
            "repeatedly."
        ),
    )


def _arm_pairs(args) -> list[tuple[list[str], list[str]]]:
    """(old_lines, new_lines) per change, whichever arm's arguments arrived."""
    if isinstance(args, EditFileParamsC):
        return [
            (_text_to_lines(e.old_text), _text_to_lines(e.new_text))
            for e in args.edits
        ]
    return [(_text_to_lines(args.old_text), _text_to_lines(args.new_text))]


def _arm_change(args) -> str:
    pairs = _arm_pairs(args)
    if len(pairs) == 1:
        old, new = pairs[0]
        if not new:
            return f"{_lines(len(old))} deleted"
        return f"{_lines(len(old))} -> {_lines(len(new))}"
    return f"{len(pairs)} edits"


async def _edit_file_arm(args, ctx: ToolContext) -> str:
    """Arms b and c: same pipeline as ``edit_file``, different arguments.

    Deliberately a separate function rather than a refactor of ``edit_file``:
    arm a is the baseline every past measurement was taken against, and an
    experiment that quietly rewrites its control measures nothing.

    All-or-nothing across the list. A partial application would leave the file
    in a state neither the model nor the user asked for, and the model's next
    read would disagree with its own record of what it sent.
    """
    from hpca.agent.builtin_tools import KIND_BY_SUFFIX, check_script_content

    try:
        path = _existing(args.path, ctx)
    except ValueError as exc:
        return f"NOT edited: {exc}"
    if path.is_dir():
        return (
            f"NOT edited: {path} is a directory. "
            f"{hints.EDIT_FILE_ON_DIRECTORY}"
        )
    try:
        text, bom, ending = _read_for_edit(path)
    except (UnicodeDecodeError, OSError) as exc:
        return (
            f"NOT edited: {path} could not be read as text "
            f"({type(exc).__name__})."
        )

    pairs = _arm_pairs(args)
    for index, (_, new_lines) in enumerate(pairs):
        field = f"new_text (edit {index + 1})" if len(pairs) > 1 else "new_text"
        refused_marker = _elision_refusal(
            new_lines,
            refusal="NOT edited",
            field=field,
            outcome="The file is unchanged.",
        )
        if refused_marker:
            return refused_marker

    file_lines = text.split("\n")
    first_start = None
    for index, (old_lines, new_lines) in enumerate(pairs):
        where = f" (edit {index + 1})" if len(pairs) > 1 else ""
        hits = _match_ladder(file_lines, old_lines)
        if not hits:
            stripped = _strip_line_number_prefixes(old_lines)
            if stripped is not None:
                hits = _match_ladder(file_lines, stripped)
                if hits:
                    old_lines = stripped
        if not hits:
            return (
                f"NOT edited{where}: that text is not in {path}."
                f"{_near_miss(file_lines, old_lines)}"
            )
        if len(hits) > 1:
            spots = ", ".join(f"line {i + 1}" for i in hits[:5])
            return (
                f"NOT edited{where}: that text occurs {len(hits)} times in "
                f"{path} ({spots}), so which one you mean is ambiguous. "
                f"{hints.EDIT_FILE_AMBIGUOUS}"
            )
        start = hits[0]
        if first_start is None:
            first_start = start
        file_lines = (
            file_lines[:start] + new_lines + file_lines[start + len(old_lines):]
        )

    edited = "\n".join(file_lines)
    if edited == text:
        return (
            f"NOT edited: the replacement produces identical content - "
            f"old_text and new_text are the same. {hints.EDIT_FILE_NO_OP}"
        )

    warnings: list[str] = []
    kind = KIND_BY_SUFFIX.get(path.suffix)
    if kind is not None:
        ctx.scripts_dir.mkdir(parents=True, exist_ok=True)
        scratch = ctx.scripts_dir / f"edit_{time.time_ns()}{path.suffix}"
        scratch.write_text(edited)
        try:
            refused, warnings = await check_script_content(
                kind, scratch, edited, ctx, refusal="NOT edited"
            )
        finally:
            scratch.unlink(missing_ok=True)
        if refused:
            return refused

    entry = _require_trash(ctx).backup(path)
    out = edited.replace("\n", ending) if ending != "\n" else edited
    path.write_bytes((bom + out).encode())
    extra = f" ({'; '.join(warnings)})" if warnings else ""
    no_backup = (
        ""
        if entry.trashed_path
        else " NO backup was kept (the file is above the backup size limit), "
        "so this cannot be undone."
    )
    return (
        f"Edited {path} at line {(first_start or 0) + 1}: {_arm_change(args)}"
        f"{extra}. {hints.NO_READ_BACK}{no_backup}{_tbd_note(edited)}"
    )


def _describe_edit_arm(args, ctx: ToolContext) -> str:
    path = _target(args.path, ctx)
    size = path.stat().st_size if path.is_file() else 0
    backed_up = ctx.trash is not None and size < ctx.trash.backup_limit_bytes
    backup_note = (
        "the current file is copied to trash first"
        if backed_up
        else "NO BACKUP (file exceeds the backup size limit) - irreversible"
    )
    return f"edit {path}\n({_arm_change(args)}; {backup_note})"


def _edit_resolvable_arm(args, ctx: ToolContext) -> bool:
    try:
        return _target(args.path, ctx).is_file()
    except Exception:
        return False


def edit_preview_arm(arguments: dict, ctx: object = None) -> str:
    """The lines out then the lines in, as ``edit_preview`` renders arm a."""
    specs = arguments.get("edits")
    if not isinstance(specs, list):
        specs = [arguments]
    out = []
    for spec in specs:
        if not isinstance(spec, dict):
            continue
        out += [
            f"- {line}" for line in _text_to_lines(spec.get("old_text") or "")
        ]
        out += [
            f"+ {line}" for line in _text_to_lines(spec.get("new_text") or "")
        ]
    return "\n".join(out)


_ARM_B_DESCRIPTION = (
    "Change part of a text file in place: give the exact text to replace and "
    "what to put there, instead of rewriting the whole file. Read the file "
    "first and copy the text from it exactly"
)

_ARM_C_DESCRIPTION = (
    "Change part of a text file in place: give the exact text to replace and "
    "what to put there, instead of rewriting the whole file. Read the file "
    "first and copy the text from it exactly. Put every change to one file "
    "in a single call, as separate entries in edits"
)


def _arm_edit_tool():
    """The edit_file the current arm registers, or None for arm a."""
    if EDIT_ARM == "b":
        params, description = EditFileParamsB, _ARM_B_DESCRIPTION
    elif EDIT_ARM == "c":
        params, description = EditFileParamsC, _ARM_C_DESCRIPTION
    else:
        return None
    return Tool(
        name="edit_file",
        description=description,
        params=params,
        handler=_edit_file_arm,
        is_destructive_call=_edit_resolvable_arm,
        describe_call=_describe_edit_arm,
    )


def add_file_tools(registry: ToolRegistry) -> ToolRegistry:
    registry.register(
        Tool(
            name="delete_file",
            description="Delete a file by path (trash-backed)",
            params=DeleteFileParams,
            handler=delete_file,
            is_destructive_call=_delete_resolvable,
            describe_call=_describe_delete,
        )
    )
    registry.register(
        Tool(
            name="restore_file",
            description=(
                "Restore a file deleted earlier (by delete_file, or overwritten "
                "by move/copy) from the trash, using its original path; call "
                "with an empty path to list what can be restored"
            ),
            params=RestoreFileParams,
            handler=restore_file,
        )
    )
    registry.register(
        Tool(
            name="create_file",
            description=(
                "Write a NEW text file — specs, notes, a README, a config — "
                "at a path, one array element per line. Use this instead of "
                "echoing or heredoc'ing a file through run_bash. Past ~150 "
                "lines send a skeleton (headings + one `TBD: ...` line each) "
                "and fill per section with edit_file. It will not overwrite: "
                "to change a file that exists, call edit_file"
            ),
            params=CreateFileParams,
            handler=create_file,
            describe_call=_describe_create,
        )
    )
    registry.register(
        _arm_edit_tool()
        or Tool(
            name="edit_file",
            description=(
                "Change part of a text file in place: give the exact lines to "
                "replace and what to put there, instead of rewriting the whole "
                "file. Read the file first and copy the lines from it exactly"
            ),
            params=EditFileParams,
            handler=edit_file,
            is_destructive_call=_edit_resolvable,
            describe_call=_describe_edit,
        )
    )
    registry.register(
        Tool(
            name="move_file",
            description="Move or rename a file, like `mv`",
            params=MoveFileParams,
            handler=move_file,
            is_destructive_call=_move_overwrites,
            describe_call=_describe_move,
        )
    )
    registry.register(
        Tool(
            name="copy_file",
            description="Copy a file, like `cp`",
            params=CopyFileParams,
            handler=copy_file,
            is_destructive_call=_copy_overwrites,
            describe_call=_describe_copy,
        )
    )
    return registry
