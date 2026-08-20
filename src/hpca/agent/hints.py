"""The sentences in a tool result that are addressed to the model, not the user.

Every tool result does two jobs at once. It reports what happened — a path, a
line count, an exit code, a refusal and the state it left behind — and it tells
the model what to do next. The first half is news, and the user reading the
chat wants it; the second half is scaffolding for the following call, and
"Do not read it back to check." in the transcript tells the user nothing except
that the agent is being managed. The chat window drops the scaffolding before
it renders a step, and this module is where the two are told apart.

Named constants rather than a regex over the result text: the display layer has
to match what the handler actually said, byte for byte, and a pattern
maintained apart from the sentence it matches drifts the first time someone
rewords a refusal — silently, because nothing fails when a strip stops
matching. Importing the same string the handler returned cannot drift. Each
constant holds the sentence and nothing else; the space or newline that joins
it to the state it follows stays at the call site, so what the model receives
is unchanged by moving the text here.

A leaf on purpose: it imports nothing from ``hpca``, so the tool modules and
the TUI both depend on it without either having to depend on the other.

``MODEL_HINTS`` is the authoritative list, and it is ordered longest first
because stripping is by substring — a shorter hint that is a prefix of a longer
one, removed first, would leave the remainder of the longer one behind as a
fragment.
"""

from __future__ import annotations

# ---------------------------------------------------------------- writing files

CREATE_FILE_EXISTS = (
    "Change it with edit_file; to replace it wholesale, delete_file first "
    "(the old version stays recoverable from the trash)."
)

CREATE_FILE_DIR_KEY_IS_FILE = "Give the key of the directory the file belongs in."

CREATE_FILE_NAME_ESCAPES = (
    "Give a name relative to it, and register another directory if the file "
    "belongs somewhere else."
)

CREATE_FILE_MADE_DIRS = (
    "If you meant a directory that is already there, the path is misspelled — "
    "check it against what the user wrote before building on this file."
)

EDIT_NOT_REWRITE = "Change it with edit_file, not by writing it again."

NO_READ_BACK = "Do not read it back to check."

EDIT_FILE_ON_DIRECTORY = "Pass subpath to edit a file inside it."

EDIT_FILE_AMBIGUOUS = (
    "Call edit_file again with enough surrounding lines to pick out the one "
    "you want."
)

EDIT_FILE_NO_OP = (
    "If you meant to change something else, re-read the file and edit that."
)

# ------------------------------------------------------------ paths and the trash

SUBPATH_ESCAPES = "use a path inside the directory."

REGISTERED_PATH_MISSING = (
    "If the user meant a path that is already there, check the spelling "
    "against their message; otherwise create it before reading it."
)

TRASH_NOT_RECOVERABLE = "Tell the user it is not recoverable from HPCA's trash."

RESTORE_PATH_OCCUPIED = (
    "Move or rename that file first with move_file, then restore again — the "
    "backup is still in the trash. Undoing an edit always lands here, because "
    "the backup and the edited file share a path. Do not delete the file to "
    "clear the path: that trashes it under the same path too, and the restore "
    "would then bring back what you were undoing."
)

# ------------------------------------------------------------- elision placeholders

ELISION_REWRITE_FILE = (
    "Do not send that line again. Call read_file on the file and send the "
    "lines it gives back, or re-derive the content from whatever you built it "
    "from. If you did mean this literal text — documentation about the "
    "placeholder itself — write it with run_bash and a heredoc."
)

ELISION_REWRITE_SCRIPT = (
    "Do not send that line again. Re-derive the script from what you built it "
    "from, or read the file it came from back with read_file, and call "
    "create_script again with the real lines."
)

# ------------------------------------------------------------------------ scripts

SCRIPT_KEY_EXISTS = "change it with edit_file rather than creating it again"

SCRIPT_KEY_TAKEN = "pick a different key"

SCRIPT_SHEBANG_ONLY = (
    "Put each script line into its own content_lines array element and call "
    "create_script again."
)

BASH_STRICT_MODE = (
    "Runs fail-fast (set -euo pipefail): a failed command stops the script, "
    "so do not print success unconditionally."
)

SCRIPT_ALREADY_RUNNING = (
    "Wait for it — you will be told when it finishes — or kill it first."
)

RUN_BASH_IS_NOT_A_WRITER = (
    "To WRITE a file — notes, a specs document, a config — call create_file "
    "with the content as content_lines. To RUN real work, call create_script "
    "(it is syntax- and docs-checked), then start_background_script, or "
    "run_bash with {its_key}. If the content is too long for one call, write "
    "the first part with create_file, then add each further part with "
    "edit_file: put the file's current last line in old_lines, and that same "
    "line followed by the new lines in new_lines. Or, if this really is a "
    "look-around, make it shorter."
)

RUN_BASH_SHEBANG_ONLY = "Put each command on its own content_lines element."

RUN_BASH_TIMED_OUT = (
    "Narrow it (fewer directories, -maxdepth, pipe through head) or raise "
    "timeout_s, then run it again."
)

RUN_BASH_FAILED = "Read the error, fix the script, and call run_bash again."

# ----------------------------------------------------------------------- approval

SKIPPED_WITHOUT_REASON = (
    "Do not retry it, do not rephrase it, and do not attempt the same outcome "
    "via a different tool. Ask the user how to proceed."
)

SKIPPED_WITH_REASON = (
    "Treat that as the correction to make. Work it into a fixed version and "
    "put that up for approval — do not send back what was just refused. If "
    "the reason does not tell you enough to fix it, ask the user."
)

DENIED_WITH_REASON = (
    "Treat that as the correction to make. Propose an amended operation that "
    "answers it, or ask the user if the reason leaves you unsure — do not "
    "re-send what was just denied."
)

# --------------------------------------------------------------------------- docs

NO_SYMBOL_INDEX = "run index_docs first."

NO_SEMANTIC_SEARCH = "use lookup_symbol or read_manpage instead."

EMPTY_DOC_INDEX = "index something with index_docs first."

# ------------------------------------------------------------------ past sessions

SESSION_SEARCH_ARGUMENTS = "Give a query to search, or a session_id to read."

SESSION_TAIL = "(session tail; pass around=<turn number> to read elsewhere)"

SESSION_CONTEXT = (
    "(use session_search with session_id= and around=<turn> for context)"
)

# --------------------------------------------------------------- jobs and watches

CHECK_JOB_STATUS = "Check with job_status."

WATCH_NEEDS_A_PATH = "Find the log's full path first (ls / find), then pass that."

WATCH_TARGET_MISSING = "If that is not what you expected, check the path."

WATCH_JOB_UNKNOWN = "Check the id with squeue -u $USER."

WATCH_AMBIGUOUS = "Be specific."


# Longest first: the display layer strips by substring, and a shorter hint that
# is a prefix of a longer one would otherwise leave the remainder behind.
MODEL_HINTS: tuple[str, ...] = (
    RUN_BASH_IS_NOT_A_WRITER,
    RESTORE_PATH_OCCUPIED,
    ELISION_REWRITE_FILE,
    SKIPPED_WITH_REASON,
    ELISION_REWRITE_SCRIPT,
    DENIED_WITH_REASON,
    CREATE_FILE_MADE_DIRS,
    REGISTERED_PATH_MISSING,
    SKIPPED_WITHOUT_REASON,
    CREATE_FILE_EXISTS,
    BASH_STRICT_MODE,
    RUN_BASH_TIMED_OUT,
    CREATE_FILE_NAME_ESCAPES,
    SCRIPT_SHEBANG_ONLY,
    EDIT_FILE_AMBIGUOUS,
    EDIT_FILE_NO_OP,
    SCRIPT_ALREADY_RUNNING,
    SESSION_CONTEXT,
    SESSION_TAIL,
    WATCH_NEEDS_A_PATH,
    RUN_BASH_FAILED,
    SCRIPT_KEY_EXISTS,
    TRASH_NOT_RECOVERABLE,
    CREATE_FILE_DIR_KEY_IS_FILE,
    RUN_BASH_SHEBANG_ONLY,
    EDIT_NOT_REWRITE,
    WATCH_TARGET_MISSING,
    SESSION_SEARCH_ARGUMENTS,
    NO_SEMANTIC_SEARCH,
    EMPTY_DOC_INDEX,
    EDIT_FILE_ON_DIRECTORY,
    WATCH_JOB_UNKNOWN,
    SUBPATH_ESCAPES,
    NO_READ_BACK,
    CHECK_JOB_STATUS,
    NO_SYMBOL_INDEX,
    SCRIPT_KEY_TAKEN,
)

# Named here, and deliberately NOT stripped. Matching is by substring against
# whatever a tool returned — which includes a file the user asked to have read
# back — and at twelve characters "Be specific." is short enough to occur in
# someone's own prose. Cutting a sentence out of a document to spare the reader
# one line of steering is the wrong trade, and the line only ever appears on a
# watch name that matched two watches, which is rare. The constant still exists
# so the tool and this module agree on what the sentence is.
NOT_STRIPPED: tuple[str, ...] = (WATCH_AMBIGUOUS,)
