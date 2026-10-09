"""Indexing a directory of documents, in the background when it is large (§5.6.2).

`index_docs` with ``docs_dir`` used to embed every file inside one tool call:
fine for a manual, hours for a few thousand papers — with the session blocked
for all of them — and a second run re-embedded everything, because nothing
remembered what had already been done. Two changes:

* **Incremental.** `RagStore.indexed_files` records each file's size, mtime,
  content digest and embedding model. A file whose size and mtime match is
  skipped without being read; one whose bytes still match is skipped after the
  read. A different model re-embeds everything, since vectors from two models
  in one store would be compared as if they meant the same thing. A file gone
  from the directory has its chunks removed: an indexed directory mirrors it.
* **In the background.** Past `INLINE_LIMIT` documents to embed, the tool
  returns at once and `DocIndexer` runs the job as a task in the core. The user
  sees progress as notifications; the session that asked gets the outcome as a
  message (`TurnScheduler.submit_event`), the way a finished process reports.
  Asking again for the same directory reports progress. One job at a time.

In the core rather than a subprocess because the index belongs to the core: it
is opened from a node-local copy and synced home (`hpca.dbcache`), so a second
process writing the home copy would be overwritten by the next sync. Store
writes stay on the loop's thread, whose connection it is; reading and statting
files goes to a worker thread; embedding is HTTP and is awaited.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path

from hpca.embeddings import EmbeddingError
from hpca.protocol import Notify
from hpca.rag import RagStore, Recorded, index_text

logger = logging.getLogger("hpca.doc_index")

DOC_SUFFIXES = {".md", ".txt", ".rst", ".text"}
# Up to this many documents to embed, the tool does it in the call and answers
# with the outcome — a manual's worth is seconds, and an answer the model can
# act on in the same turn beats a message arriving later.
INLINE_LIMIT = 20
# Failures of the embedding backend in a row that stop a job. One is a hiccup;
# three is a backend that is down, and going on would write a failure line for
# each of the thousands of documents still to come.
MAX_BACKEND_FAILURES = 3
# How many progress notifications a background job raises: at each quarter.
PROGRESS_STEPS = 4


@dataclass(frozen=True)
class Candidate:
    """A file that may need embedding, as it was when the directory was read."""

    path: Path
    size: int
    mtime_ns: int


@dataclass
class IndexJob:
    """One directory being indexed, and how far it has got."""

    root: Path
    label: str  # the directory as the caller named it, for the messages
    profile: str  # whose index it goes into
    rag: RagStore
    embedder: object
    model: str
    recorded: dict[str, Recorded]
    todo: list[Candidate]
    gone: list[str]
    unchanged: int = 0
    done: int = 0
    indexed: int = 0
    empty: int = 0
    problems: list[str] = field(default_factory=list)
    # Why it stopped before the end, or "".
    stopped: str = ""
    finished: bool = False
    started: float = field(default_factory=time.monotonic)

    def progress(self) -> str:
        seconds = time.monotonic() - self.started
        took = f"{seconds:.0f} s" if seconds < 60 else f"{seconds / 60:.0f} min"
        return (
            f"Indexing {self.label!r} for profile {self.profile!r}: "
            f"{self.done} of {len(self.todo)} documents "
            f"done ({self.indexed} indexed, {len(self.problems)} failed) after "
            f"{took}."
        )

    def report(self) -> str:
        parts = []
        if self.stopped:
            parts.append(
                f"Stopped after {self.done} of {len(self.todo)} documents: "
                f"{self.stopped}. Indexing the directory again picks up where "
                "it stopped."
            )
        parts.append(
            f"Indexed {self.indexed} documents from {self.label!r} for search "
            f"in profile {self.profile!r}."
        )
        if self.unchanged:
            parts.append(
                f"{self.unchanged} unchanged since they were last indexed, skipped."
            )
        if self.gone:
            parts.append(
                f"Removed {len(self.gone)} that are no longer in the directory."
            )
        if self.empty:
            parts.append(f"{self.empty} had no text to index.")
        if self.problems:
            # The count leads, and the examples follow it. Reporting only the
            # first three read as a footnote on a success when in fact most of
            # a manual had been dropped, which is how a 100-of-1597 index came
            # to be announced as done.
            attempted = self.indexed + self.empty + len(self.problems)
            parts.append(
                f"{len(self.problems)} could NOT be indexed"
                + (f" (of {attempted} tried)" if attempted > len(self.problems) else "")
                + ": "
                + "; ".join(self.problems[:3])
                + ("; ..." if len(self.problems) > 3 else "")
            )
        return " ".join(parts)


def _scan(
    root: Path, recorded: dict[str, Recorded], model: str
) -> tuple[list[Candidate], int, list[str]]:
    """What in `root` needs embedding, how much does not, and what is gone.

    Off the loop: a few thousand stats on NFS are seconds.
    """
    todo: list[Candidate] = []
    unchanged = 0
    seen: set[str] = set()
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in DOC_SUFFIXES or not path.is_file():
            continue
        stat = path.stat()
        seen.add(str(path))
        known = recorded.get(str(path))
        if (
            known is not None
            and known.model == model
            and known.size == stat.st_size
            and known.mtime_ns == stat.st_mtime_ns
        ):
            unchanged += 1
            continue
        todo.append(Candidate(path, stat.st_size, stat.st_mtime_ns))
    gone = [source for source in recorded if source not in seen]
    return todo, unchanged, gone


async def prepare(
    root: Path, label: str, rag: RagStore, embedder, *, profile: str = "default"
) -> IndexJob:
    """A job for `root`: the directory read and compared with what is indexed
    in `profile`'s index, `rag`."""
    model = str(getattr(embedder, "model", "") or "")
    recorded = rag.recorded_under(str(root))
    todo, unchanged, gone = await asyncio.to_thread(_scan, root, recorded, model)
    return IndexJob(
        root=root,
        label=label,
        profile=profile,
        rag=rag,
        embedder=embedder,
        model=model,
        recorded=recorded,
        todo=todo,
        gone=gone,
        unchanged=unchanged,
    )


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()[:16]


async def _one(job: IndexJob, candidate: Candidate) -> EmbeddingError | None:
    """Index one file. Returns the backend's error if that is what failed it."""
    source = str(candidate.path)
    try:
        data = await asyncio.to_thread(candidate.path.read_bytes)
    except OSError as e:
        job.problems.append(f"{candidate.path.name}: {e.strerror or e}")
        return None
    mark = Recorded(candidate.size, candidate.mtime_ns, _digest(data), job.model)
    known = job.recorded.get(source)
    if known is not None and known.model == job.model and known.digest == mark.digest:
        # Touched, not changed: remember the new mtime so the next scan does
        # not read it again.
        job.rag.record(source, mark)
        job.unchanged += 1
        return None
    try:
        stored = await index_text(
            job.rag, job.embedder, source, data.decode(errors="replace")
        )
    except EmbeddingError as e:
        job.problems.append(f"{candidate.path.name}: embedding backend error: {e}")
        return e
    if stored:
        job.indexed += 1
    else:
        # Nothing in it to embed now, whatever it held before.
        job.rag.clear_source(source)
        job.empty += 1
    job.rag.record(source, mark)
    return None


async def run(job: IndexJob, on_step: Callable[[IndexJob], None] | None = None) -> None:
    """Embed what `job` found to do, one document at a time.

    Each document is stored and recorded before the next is read, so a job
    stopped part-way — a backend gone, the app closed — keeps everything it
    finished, and the next run of the same directory skips it.
    """
    try:
        for source in job.gone:
            job.rag.clear_source(source)
        failures = 0
        for candidate in job.todo:
            error = await _one(job, candidate)
            job.done += 1
            failures = failures + 1 if error is not None else 0
            if failures >= MAX_BACKEND_FAILURES:
                job.stopped = (
                    f"the embedding backend failed {failures} documents in a "
                    f"row ({error})"
                )
                return
            if on_step is not None:
                on_step(job)
    finally:
        job.finished = True


class DocIndexer:
    """Runs one `IndexJob` at a time as a task in the core."""

    def __init__(
        self,
        *,
        emit: Callable[[object], None] | None = None,
        submit_event: Callable[[str, str], None] | None = None,
    ) -> None:
        self._emit = emit
        self._submit_event = submit_event
        self.job: IndexJob | None = None
        self._task: asyncio.Task | None = None
        self._quarter = 0

    def busy(self) -> IndexJob | None:
        """The job still running, or None."""
        if self.job is not None and not self.job.finished:
            return self.job
        return None

    def start(self, job: IndexJob, session_id: str) -> None:
        if self.busy() is not None:
            raise RuntimeError("a document index job is already running")
        self.job = job
        self._quarter = 0
        self._task = asyncio.create_task(self._run(job, session_id))

    async def _run(self, job: IndexJob, session_id: str) -> None:
        try:
            await run(job, on_step=self._progress)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # a job that dies must still say so
            logger.exception("document indexing failed")
            job.stopped = f"indexing failed: {e}"
        report = job.report()
        self._say(
            report,
            "warning" if job.stopped or job.problems else "information",
        )
        if self._submit_event is not None and session_id:
            self._submit_event(session_id, f"[indexing finished] {report}")

    def _progress(self, job: IndexJob) -> None:
        quarter = job.done * PROGRESS_STEPS // max(1, len(job.todo))
        if self._quarter < quarter < PROGRESS_STEPS:
            self._quarter = quarter
            self._say(job.progress())

    def _say(self, text: str, severity: str = "information") -> None:
        if self._emit is not None:
            self._emit(Notify(severity=severity, title="Document indexing", text=text))

    async def stop(self) -> None:
        """Cancel a running job. What it finished is already stored."""
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError, Exception):
                await task
