"""Image inspector (§4.2): a firewalled sub-loop that looks so the caller need not.

The orchestrator asks a question about one to four files; this loop shows
them to the model in a fresh conversation, lets it ``zoom`` into grid cells,
and returns a bounded text answer. The images never enter the orchestrator's
context — the same contract as the doc-researcher — and that is what makes
this cheap to integrate: the conversation here is a local list that is
thrown away, so nothing that stores, folds, compacts or renders the session
(all of which assume text content) ever meets an image.

What code measured about each image (``hpca.images``: size, a stretched bit
depth, blankness, content running off an edge, SVG defects) is told to the
model next to the picture and returned with the answer, so the caller gets
the certain part even when the model's reading is wrong.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from hpca.agent.middleware import DecisionError, DirectResponse, decide
from hpca.agent.tools import Tool, ToolRegistry
from hpca.images import CELLS, ImageError, Picture
from hpca.llm import image_part

# Zoom rounds before the model must answer with what it has.
MAX_ROUNDS = 4
# Images one request may carry. The server refuses more (llm.a40-l40.sh,
# MM_LIMIT), so the loop detaches old ones rather than exceed it.
MAX_IMAGES_PER_REQUEST = 4
# The answer is bounded like every sub-agent's: it lands in the caller's
# context, which is what the firewall is protecting.
MAX_ANSWER_CHARS = 2000

FINAL_ROUND_NOTE = (
    "[no more zooms] Answer now, in the ANSWER / SEEN / UNSURE form, from what "
    "you have already seen."
)

VISION_SYSTEM = (
    "You inspect images for another agent that cannot see them. It asked a "
    "question; answer it from what is visible.\n"
    "Each image comes with facts measured by code — size, how its brightness "
    "was scaled, whether it is blank, whether content runs off an edge, "
    "defects in SVG source. Trust those over your impression.\n"
    "Each image is divided into a 3x3 grid: columns A, B, C from left to "
    "right, rows 1, 2, 3 from top to bottom (A1 is top-left, C3 "
    "bottom-right). When small text or detail matters and you cannot read "
    "it clearly, call zoom on the cell that holds it instead of guessing.\n"
    "Never invent text, numbers or labels you cannot actually read: say they "
    "are illegible. Reply in exactly this form:\n"
    "ANSWER: <the direct answer to the question>\n"
    "SEEN: <the specific things you saw that support it>\n"
    "UNSURE: <what you could not make out, or 'nothing'>"
)


class ZoomParams(BaseModel):
    image: int = Field(ge=1, description="Which image, numbered from 1")
    cell: str = Field(description="Grid cell A1..C3 (column A-C, row 1-3)")


def _zoom_handler(args: ZoomParams, ctx) -> str:
    # The loop below carries zoom out itself, because its result is an image
    # and a tool handler can only return text.
    raise RuntimeError("zoom is executed by the vision loop")


def _zoom_tools() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool(
            name="zoom",
            description=(
                "See one grid cell of an image at higher resolution: use it "
                "to read small text, tick labels, legends or fine detail"
            ),
            params=ZoomParams,
            handler=_zoom_handler,
        )
    )
    return registry


@dataclass
class _Zoom:
    image: int  # 1-based
    cell: str
    png: bytes | None  # None once detached


@dataclass
class _Session:
    """The sub-conversation, rendered fresh each round.

    Rendered rather than appended because what may be attached changes: the
    server takes MAX_IMAGES_PER_REQUEST images at most, so each new zoom
    detaches the previous one, and — when the overviews alone already fill
    the limit — the overviews of images other than the one being zoomed.
    A detached image stays in the conversation as a sentence saying so, so
    the model knows what it saw before and can ask for it again.
    """

    question: str
    pictures: list[Picture]
    attached: set[int] = field(default_factory=set)  # overview numbers, 1-based
    # What happened after the opening, in order: zooms, and the notes that
    # answer a zoom that could not be done (or ask for the answer).
    events: list[_Zoom | str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.attached = set(range(1, len(self.pictures) + 1))

    def add_zoom(self, number: int, cell: str) -> None:
        picture = self.pictures[number - 1]
        for earlier in self.events:
            if isinstance(earlier, _Zoom):
                earlier.png = None
        for other in sorted(self.attached - {number}):
            if len(self.attached) + 1 <= MAX_IMAGES_PER_REQUEST:
                break
            self.attached.discard(other)
        self.events.append(_Zoom(number, cell, picture.cell_png(cell)))

    def note(self, text: str) -> None:
        self.events.append(text)

    def messages(self) -> list[dict]:
        opening: list[dict] = [{"type": "text", "text": f"Question: {self.question}"}]
        for number, picture in enumerate(self.pictures, 1):
            heading = f"Image {number}: {picture.label}"
            facts = "".join(f"\n- {fact}" for fact in picture.facts)
            if number in self.attached:
                opening.append({"type": "text", "text": f"{heading}{facts}"})
                opening.append(image_part(picture.overview_png()))
            else:
                opening.append(
                    {
                        "type": "text",
                        "text": f"{heading}{facts}\n(no longer attached, to "
                        "stay within the image limit; zoom into it to see it "
                        "again)",
                    }
                )
        conversation = [
            {"role": "system", "content": VISION_SYSTEM},
            {"role": "user", "content": opening},
        ]
        for zoom in self.events:
            if isinstance(zoom, str):
                conversation.append({"role": "user", "content": zoom})
                continue
            conversation.append(
                {"role": "assistant", "content": f"(called zoom on image {zoom.image}, cell {zoom.cell})"}
            )
            label = f"[zoom] image {zoom.image}, cell {zoom.cell}"
            if zoom.png is None:
                conversation.append(
                    {
                        "role": "user",
                        "content": f"{label}: shown earlier, no longer attached.",
                    }
                )
            else:
                conversation.append(
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"{label}:"},
                            image_part(zoom.png),
                        ],
                    }
                )
        return conversation


def _header(pictures: list[Picture]) -> str:
    lines = []
    for number, picture in enumerate(pictures, 1):
        facts = "; ".join(picture.facts)
        lines.append(f"image {number}: {picture.label} — {facts}")
    return "\n".join(lines)


def _clip(text: str) -> str:
    text = text.strip()
    if len(text) <= MAX_ANSWER_CHARS:
        return text
    return text[:MAX_ANSWER_CHARS].rstrip() + " [answer cut at its length cap]"


async def inspect(
    llm,
    question: str,
    pictures: list[Picture],
    *,
    max_rounds: int | None = None,
) -> str:
    """Answer ``question`` about ``pictures``; the caller's text report.

    ``max_rounds`` defaults to MAX_ROUNDS as it stands at call time, so the
    vision eval's no-zoom arm can set the module value.
    """
    if max_rounds is None:
        max_rounds = MAX_ROUNDS
    header = _header(pictures)
    session = _Session(question, pictures)
    tools = _zoom_tools()
    for round_number in range(max_rounds + 1):
        if round_number == max_rounds:
            # Said rather than enforced by withdrawing the tool: the decision
            # grammar is built from the tool list, and the one it would build
            # from none is not one the other loops ever exercise.
            session.note(FINAL_ROUND_NOTE)
        try:
            decision = await decide(llm, session.messages(), tools)
        except DecisionError as exc:
            return f"{header}\n[image inspector failed: {exc}]"
        if isinstance(decision, DirectResponse):
            return f"{header}\n{_clip(decision.text)}"
        if round_number == max_rounds:
            break
        args = decision.arguments
        if not 1 <= args.image <= len(pictures):
            session.note(
                f"[zoom error] There is no image {args.image}; images are "
                f"numbered 1 to {len(pictures)}."
            )
            continue
        cell = args.cell.strip().upper()
        if cell not in CELLS:
            session.note(
                f"[zoom error] There is no cell {args.cell!r}; cells are A1 to C3."
            )
            continue
        try:
            session.add_zoom(args.image, cell)
        except ImageError as exc:
            session.note(f"[zoom error] {exc}")
    return f"{header}\n[image inspector ran out of rounds without an answer]"


# ------------------------------------------------------- the orchestrator's door


class InspectImageParams(BaseModel):
    question: str = Field(
        description=(
            "What to find out, as specifically as you can: 'is any axis label "
            "cut off?' gets a better answer than 'describe this'"
        )
    )
    page: int = Field(default=1, ge=1, description="Page to show of a PDF")
    # Last, like every array argument (§4.3).
    paths: list[str] = Field(
        min_length=1,
        max_length=MAX_IMAGES_PER_REQUEST,
        description="Image files: PNG, JPEG, TIFF, GIF, WebP, BMP, PDF or SVG",
    )


UNSEEING_BACKEND = (
    "The model behind this session cannot see images: its server refused a "
    "test image or answered without looking at it. Nothing was inspected. "
    "For an SVG, read_file shows its source; run_bash can report an image's "
    "size and format (`file`, `identify`)."
)


async def inspect_image(args: InspectImageParams, ctx) -> str:
    import asyncio

    from hpca.images import load
    from hpca.paths import PathError, resolve_path

    if ctx.llm is None:
        return "Image inspection is unavailable (no LLM in this context)."
    probe = getattr(ctx.llm, "supports_images", None)
    if probe is not None and not await probe():
        return UNSEEING_BACKEND
    pictures: list[Picture] = []
    problems: list[str] = []
    for value in args.paths:
        try:
            path = resolve_path(value, ctx.workdir)
            # Off the loop: a large TIFF or a PDF page takes real time to decode.
            pictures.append(await asyncio.to_thread(load, path, page=args.page))
        except (PathError, ImageError) as exc:
            problems.append(f"{value}: {exc}")
    notes = "".join(f"[not inspected] {problem}\n" for problem in problems)
    if not pictures:
        return notes.rstrip()
    return notes + await inspect(ctx.llm, args.question, pictures)


def add_inspect_image(registry: ToolRegistry) -> ToolRegistry:
    registry.register(
        Tool(
            name="inspect_image",
            description=(
                "Look at images (plots, figures, micrographs, PDF pages, SVGs) "
                "and answer a question about them. Use it to check a figure or "
                "SVG you made renders as intended — text cut off, overlaps, "
                "empty panels — and to read what an image shows"
            ),
            params=InspectImageParams,
            handler=inspect_image,
        )
    )
    return registry
