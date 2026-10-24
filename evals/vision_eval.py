#!/usr/bin/env python
"""Standalone eval: how well does the live model answer inspect_image?

Every task is a picture drawn here, in code, so its answer is known by
construction — a bar chart whose bar count was chosen, a run id whose
characters were chosen to be confusable (O/Q, 0/O, 1/l), an SVG whose label
was placed to run off the canvas or not. Each task goes through HPCA's real
``inspect_image`` (real hpca.images loading, real sub-loop, real middleware
decisions) against the live backend, and its answer is scored by a predicate.

It compares ARMS — the same tool with one part switched off — interleaved
task by task and repeat by repeat, so a backend that slows down mid-run
slows every arm alike:

    full       the tool as shipped
    nozoom     no zoom rounds: the overview alone
    nofacts    the model is not told what code measured

Before any number is produced the predicates are checked against answers
known to be right and answers known to be wrong, and every fixture is drawn
and checked to show what it should (an SVG renders non-blank — resvg needs
system fonts). A run that fails that aborts: a score from a predicate that
passes wrong answers is not a measurement. ``--self-check`` does only that,
with no backend, in about a second.

Usage:
    python evals/vision_eval.py --out vision.json [--repeats 3]
                                [--arms full,nozoom] [--tasks substr,...]
    python evals/vision_eval.py --self-check

Needs a backend that sees (llmServer/llm.a40-l40.sh, VISION). Like
test-live and edit_eval it costs real generations; run it deliberately.
Exit codes: 0 ok, 1 self-check failed, 2 backend unreachable or blind.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

_REPO_SRC = Path(__file__).resolve().parent.parent / "src"
if _REPO_SRC.is_dir() and str(_REPO_SRC) not in sys.path:
    sys.path.insert(0, str(_REPO_SRC))

from PIL import Image, ImageDraw, ImageFont  # noqa: E402

from hpca import images  # noqa: E402
from hpca.agent import vision  # noqa: E402
from hpca.agent.vision import InspectImageParams, inspect_image  # noqa: E402

LIVE_URL = os.environ.get("HPCA_TEST_LLM_URL", "http://localhost:20001/v1")
LIVE_KEY = os.environ.get("HPCA_TEST_LLM_KEY")
ARMS = ("full", "nozoom", "nofacts")


# ------------------------------------------------------------------ scoring


def headline(answer: str) -> str:
    """The part of an answer that commits to something: its first sentence,
    before any SEEN:/UNSURE: evidence.

    Scoring the whole reply punished right answers for their own evidence:
    "ANSWER: 5 bars, the tallest is KO-2. SEEN: labels ctrl, KO-1, KO-2, …"
    failed a check that wanted exactly one label named. Answers that bury the
    verdict after a preamble are scored wrong by this — acceptable, since the
    questions all ask for the answer first.
    """
    answer = re.sub(r"^\s*ANSWER:\s*", "", answer.strip())
    # Cut at the evidence fields whether or not ANSWER: was written — the
    # same run showed "KO-rep1 is drawn in red. SEEN: … WT-rep1 (blue) …".
    answer = re.split(r"\b(?:SEEN|UNSURE):", answer, maxsplit=1)[0]
    paragraph = answer.strip().split("\n\n", 1)[0]
    # And to its first sentence: "KO-rep1 is drawn in red. The red line is
    # the third entry, after WT-rep1 …" commits in the first one only.
    # A full stop counts only before whitespace, so 93.6% stays whole.
    return re.split(r"(?<=[.!?])\s+", paragraph.strip(), maxsplit=1)[0]


NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12,
}


def first_yes_no(answer: str) -> str | None:
    """The first standalone YES or NO, which the questions ask for up front."""
    match = re.search(r"\b(yes|no)\b", answer, re.IGNORECASE)
    return match.group(1).lower() if match else None


def says_yes_no(want: str) -> Callable[[str], bool]:
    return lambda answer: first_yes_no(headline(answer)) == want


def first_count(answer: str) -> int | None:
    """The first number in the answer, as digits or as a word."""
    match = re.search(
        r"\b(\d+|" + "|".join(NUMBER_WORDS) + r")\b", answer, re.IGNORECASE
    )
    if not match:
        return None
    token = match.group(1).lower()
    return int(token) if token.isdigit() else NUMBER_WORDS[token]


def says_count(want: int) -> Callable[[str], bool]:
    return lambda answer: first_count(headline(answer)) == want


def contains_exactly(token: str) -> Callable[[str], bool]:
    """``token`` as a whole word, case-sensitive: an id read with one wrong
    character is wrong, and so is the right id inside a longer one."""
    pattern = re.compile(r"(?<![\w-])" + re.escape(token) + r"(?![\w-])")
    return lambda answer: bool(pattern.search(headline(answer)))


POSITIONS = ("top-left", "top-right", "bottom-left", "bottom-right")


def names_position(want: str) -> Callable[[str], bool]:
    """The one panel position named — naming two is not an answer."""

    def check(answer: str) -> bool:
        text = headline(answer).lower().replace(" ", "-")
        named = {p for p in POSITIONS if p in text}
        return named == {want}

    return check


def names_one_of(want: str, options: list[str]) -> Callable[[str], bool]:
    """Exactly ``want`` among ``options`` is named (whole words, any case)."""

    def check(answer: str) -> bool:
        named = {
            o for o in options
            if re.search(r"(?<![\w-])" + re.escape(o) + r"(?![\w-])", headline(answer), re.IGNORECASE)
        }
        return named == {want}

    return check


def both(*checks: Callable[[str], bool]) -> Callable[[str], bool]:
    return lambda answer: all(check(answer) for check in checks)


# ----------------------------------------------------------------- fixtures


def font(size: int) -> ImageFont.FreeTypeFont:
    # Pillow's bundled font: the same glyphs on every machine, so a fixture
    # is the same picture on the dev box and on a compute node.
    return ImageFont.load_default(size=size)


def bar_chart(path: Path, labels: list[str], heights: list[int], title: str) -> None:
    width, height = 1200, 800
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((width // 2, 50), title, fill="black", font=font(36), anchor="mm")
    base = 700
    draw.line((80, base, width - 60, base), fill="black", width=2)
    slot = (width - 160) // len(labels)
    for index, (label, bar) in enumerate(zip(labels, heights)):
        left = 100 + index * slot + slot // 6
        draw.rectangle((left, base - bar, left + slot * 2 // 3, base), fill="steelblue")
        draw.text((left + slot // 3, base + 30), label, fill="black", font=font(24), anchor="mm")
    image.save(path)


def tiny_id(path: Path, run_id: str) -> None:
    image = Image.new("RGB", (3000, 2000), "white")
    draw = ImageDraw.Draw(image)
    draw.text((1500, 120), "Coverage by sample", fill="black", font=font(64), anchor="mm")
    for index, bar in enumerate((1300, 900, 1100)):
        draw.rectangle((400 + index * 700, 1800 - bar, 800 + index * 700, 1800), fill="steelblue")
    draw.text((2950, 1960), f"run id: {run_id}", fill="black", font=font(16), anchor="rs")
    image.save(path)


def panels(path: Path, empty: str) -> None:
    image = Image.new("RGB", (1000, 800), "white")
    draw = ImageDraw.Draw(image)
    boxes = {
        "top-left": (40, 40, 480, 380), "top-right": (520, 40, 960, 380),
        "bottom-left": (40, 420, 480, 760), "bottom-right": (520, 420, 960, 760),
    }
    for name, (x0, y0, x1, y1) in boxes.items():
        draw.rectangle((x0, y0, x1, y1), outline="black", width=2)
        if name == empty:
            continue
        points = [(x0 + 20 + i * 40, y1 - 30 - ((i * 37) % 200)) for i in range(10)]
        draw.line(points, fill="darkred", width=4)
    image.save(path)


def legend(path: Path, coloured: dict[str, str]) -> None:
    image = Image.new("RGB", (1200, 700), "white")
    draw = ImageDraw.Draw(image)
    for index, colour in enumerate(coloured.values()):
        points = [(80 + i * 60, 600 - index * 120 - ((i * 53 + index * 31) % 90)) for i in range(14)]
        draw.line(points, fill=colour, width=5)
    for index, (label, colour) in enumerate(coloured.items()):
        y = 40 + index * 26
        draw.line((960, y, 1000, y), fill=colour, width=5)
        draw.text((1010, y), label, fill="black", font=font(18), anchor="lm")
    image.save(path)


def spots(path: Path, count: int) -> None:
    """A 16-bit 'micrograph': dim noise-free background, ``count`` bright
    spots — the case where the stretch to 8 bit has to be said."""
    width = height = 512
    image = Image.new("I;16", (width, height))
    pixels = [300] * (width * height)
    centres = [(80 + (i * 97) % 350, 90 + (i * 151) % 330) for i in range(count)]
    for cx, cy in centres:
        for y in range(cy - 14, cy + 15):
            for x in range(cx - 14, cx + 15):
                if (x - cx) ** 2 + (y - cy) ** 2 <= 14 ** 2:
                    pixels[y * width + x] = 3800
    image.putdata(pixels)
    image.save(path)


SVG_NS = 'xmlns="http://www.w3.org/2000/svg"'


def label_svg(path: Path, text: str, x: int) -> None:
    path.write_text(
        f'<svg {SVG_NS} width="300" height="100" viewBox="0 0 300 100">'
        '<rect x="2" y="2" width="296" height="96" rx="10" fill="#e8f0fe" stroke="#1a73e8"/>'
        f'<text x="{x}" y="62" font-family="sans-serif" font-size="30">{text}</text>'
        "</svg>"
    )


def two_page_pdf(path: Path, page_two_value: str) -> None:
    first = Image.new("RGB", (850, 1100), "white")
    ImageDraw.Draw(first).text((425, 300), "Methods", fill="black", font=font(48), anchor="mm")
    second = Image.new("RGB", (850, 1100), "white")
    draw = ImageDraw.Draw(second)
    draw.text((425, 200), "Results", fill="black", font=font(48), anchor="mm")
    draw.text((425, 500), f"Mapping rate: {page_two_value}", fill="black", font=font(32), anchor="mm")
    first.save(path, save_all=True, append_images=[second], resolution=100)


@dataclass
class Task:
    name: str
    question: str
    files: list[str]
    draw: Callable[[Path], None]  # draws every file into the directory given
    check: Callable[[str], bool]
    right: list[str]  # answers the check must accept
    wrong: list[str]  # answers it must refuse
    page: int = 1
    # What a fixture must show to be the picture the task means; checked by
    # the self-check, since a renderer without fonts draws a blank SVG.
    fixture_ok: Callable[[Path], bool] = lambda _: True


def _svg_has_ink(directory: Path, name: str) -> bool:
    return not any("blank" in f for f in images.load(directory / name).facts)


def build_tasks() -> list[Task]:
    tasks: list[Task] = []

    labels, heights = ["ctrl", "KO-1", "KO-2", "KO-3", "rescue"], [310, 180, 520, 260, 400]
    tasks.append(Task(
        "bars-count-tallest",
        "How many bars does this chart have, and which label has the tallest bar? "
        "Give the number first.",
        ["bars.png"],
        lambda d: bar_chart(d / "bars.png", labels, heights, "Reads per sample"),
        both(says_count(5), names_one_of("KO-2", labels)),
        right=["5 bars; KO-2 is tallest.", "There are five bars and the tallest is KO-2.",
               "ANSWER: 5 bars, the tallest is KO-2. SEEN: labels ctrl, KO-1, KO-2, KO-3, rescue",
               "5, KO-2\n\nThe KO-2 bar is higher than rescue and ctrl."],
        wrong=["4 bars; KO-2 is tallest.", "5 bars; rescue is tallest.",
               "5 bars; KO-2 or rescue.", "Five bars, tallest KO-1.",
               "ANSWER: 5 bars, rescue is tallest. SEEN: KO-2 is second"],
    ))

    for run_id in ("QX-7741-b", "O0l1-8B3", "RNA-5S2Z"):
        tasks.append(Task(
            f"tiny-id-{run_id}",
            "What is the run id written in the image? Copy it character by character.",
            ["tiny.png"],
            lambda d, r=run_id: tiny_id(d / "tiny.png", r),
            contains_exactly(run_id),
            right=[f"The run id is {run_id}.", run_id],
            wrong=[run_id.replace(run_id[0], "#", 1), f"{run_id}X", run_id.lower()
                   if run_id.lower() != run_id else run_id.upper(), "illegible"],
        ))

    for empty in ("top-right", "bottom-left"):
        tasks.append(Task(
            f"empty-panel-{empty}",
            "One of the four panels has no data in it. Which one? Answer with one "
            "of: top-left, top-right, bottom-left, bottom-right.",
            ["panels.png"],
            lambda d, e=empty: panels(d / "panels.png", e),
            names_position(empty),
            right=[empty, f"The {empty.replace('-', ' ')} panel is empty."],
            wrong=[p for p in POSITIONS if p != empty] + [f"{empty} or top-left"],
        ))

    colours = {"WT-rep1": "#1f77b4", "WT-rep2": "#2ca02c", "KO-rep1": "#d62728", "KO-rep2": "#9467bd"}
    tasks.append(Task(
        "legend-red",
        "According to the legend, which sample is drawn in red?",
        ["legend.png"],
        lambda d: legend(d / "legend.png", colours),
        names_one_of("KO-rep1", list(colours)),
        right=["KO-rep1", "The red line is KO-rep1.",
               "KO-rep1 is drawn in red. SEEN: WT-rep1 (blue), WT-rep2 (green), KO-rep2",
               "KO-rep1 is drawn in red. It is the third entry, after WT-rep1 and WT-rep2."],
        wrong=["WT-rep1", "KO-rep2", "KO-rep1 and WT-rep2",
               "ANSWER: WT-rep2. SEEN: KO-rep1 is purple"],
    ))

    tasks.append(Task(
        "spots-16bit",
        "How many bright spots are in this micrograph? Give the number first.",
        ["spots.tif"],
        lambda d: spots(d / "spots.tif", 6),
        says_count(6),
        right=["6", "Six bright spots."],
        wrong=["5 spots", "seven", "There are 60 spots"],
    ))

    for name, x, cut in (("svg-clipped", 140, "yes"), ("svg-fits", 20, "no")):
        tasks.append(Task(
            name,
            "Is any of the text cut off at the edge of the drawing? Answer YES or "
            "NO first, then explain.",
            ["label.svg"],
            lambda d, x=x: label_svg(d / "label.svg", "pipeline passed", x),
            says_yes_no(cut),
            right=[f"{cut.upper()}. The label is ...", f"{cut}, because"],
            wrong=[f"{'no' if cut == 'yes' else 'yes'}. The label is ...", "It is hard to say."],
            fixture_ok=lambda d: _svg_has_ink(d, "label.svg"),
        ))

    tasks.append(Task(
        "pdf-page-2",
        "What mapping rate is reported on this page? Give the number as written.",
        ["report.pdf"],
        lambda d: two_page_pdf(d / "report.pdf", "93.6%"),
        contains_exactly("93.6%"),
        page=2,
        right=["93.6%", "The mapping rate is 93.6%."],
        wrong=["96.3%", "93.6", "Mapping rate: 93.5%"],
    ))

    tasks.append(Task(
        "compare-blank",
        "Which of the two images is empty: image 1 or image 2? Answer with the "
        "number first.",
        ["a.png", "b.png"],
        lambda d: (
            Image.new("RGB", (400, 300), "white").save(d / "a.png"),
            bar_chart(d / "b.png", ["x", "y"], [200, 300], "Two bars"),
        ),
        says_count(1),
        right=["1", "Image 1 is empty."],
        wrong=["2", "Image 2 is empty."],
    ))
    return tasks


# --------------------------------------------------------------- self-check


def self_check(tasks: list[Task]) -> list[str]:
    """Every problem that makes a score untrustworthy; empty means go."""
    problems = []
    names = [t.name for t in tasks]
    if len(set(names)) != len(names):
        problems.append("duplicate task names")
    for task in tasks:
        for answer in task.right:
            if not task.check(answer):
                problems.append(f"{task.name}: refuses a right answer {answer!r}")
        for answer in task.wrong:
            if task.check(answer):
                problems.append(f"{task.name}: accepts a wrong answer {answer!r}")
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            task.draw(directory)
            for name in task.files:
                try:
                    images.load(directory / name, page=task.page)
                except images.ImageError as exc:
                    problems.append(f"{task.name}: fixture {name} does not load: {exc}")
            if not task.fixture_ok(directory):
                problems.append(
                    f"{task.name}: fixture is not the picture the task means "
                    "(an SVG rendered blank: are any fonts installed?)"
                )
    return problems


# ---------------------------------------------------------------------- run


class CountingLLM:
    def __init__(self, llm):
        self._llm = llm
        self.calls = 0
        self.completion_tokens = 0

    async def chat(self, messages, **kwargs):
        self.calls += 1
        response = await self._llm.chat(messages, **kwargs)
        self.completion_tokens += int((response.usage or {}).get("completion_tokens") or 0)
        return response

    def __getattr__(self, name):
        return getattr(self._llm, name)


def answer_part(report: str, pictures: int) -> str:
    """The model's answer without the facts header the tool puts above it —
    the header is code's, and scoring it would credit the model for it."""
    return "\n".join(report.splitlines()[pictures:])


async def run_one(task: Task, arm: str, llm, workdir: Path) -> dict:
    counting = CountingLLM(llm)
    params = InspectImageParams(question=task.question, paths=task.files, page=task.page)
    ctx = SimpleNamespace(llm=counting, workdir=workdir)
    saved = vision.MAX_ROUNDS
    original_load = images.load
    try:
        if arm == "nozoom":
            vision.MAX_ROUNDS = 0
        if arm == "nofacts":
            def load_without_facts(*args, **kwargs):
                picture = original_load(*args, **kwargs)
                picture.facts = []
                return picture
            images.load = load_without_facts
        started = time.perf_counter()
        report = await inspect_image(params, ctx)
        elapsed = time.perf_counter() - started
    finally:
        vision.MAX_ROUNDS = saved
        images.load = original_load
    answer = answer_part(report, len(task.files))
    return {
        "task": task.name,
        "arm": arm,
        "ok": bool(task.check(answer)),
        "answer": answer.strip()[:400],
        "model_calls": counting.calls,
        "completion_tokens": counting.completion_tokens,
        "seconds": round(elapsed, 2),
    }


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    spread = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - spread), min(1.0, centre + spread))


def summarise(rows: list[dict], arms: list[str]) -> str:
    lines = [f"{'arm':<10} {'ok':>7} {'rate':>6} {'95% CI':>13} {'calls':>6} {'sec':>6}"]
    for arm in arms:
        mine = [r for r in rows if r["arm"] == arm]
        ok = sum(r["ok"] for r in mine)
        low, high = wilson(ok, len(mine))
        calls = sum(r["model_calls"] for r in mine) / max(1, len(mine))
        seconds = sum(r["seconds"] for r in mine) / max(1, len(mine))
        lines.append(
            f"{arm:<10} {ok:>3}/{len(mine):<3} {ok / max(1, len(mine)):>6.0%} "
            f"{f'{low:.0%}-{high:.0%}':>13} {calls:>6.2f} {seconds:>6.1f}"
        )
    lines.append("")
    lines.append("per task (ok/n by arm):")
    for task in dict.fromkeys(r["task"] for r in rows):
        cells = []
        for arm in arms:
            mine = [r for r in rows if r["task"] == task and r["arm"] == arm]
            cells.append(f"{arm}={sum(r['ok'] for r in mine)}/{len(mine)}")
        lines.append(f"  {task:<22} " + "  ".join(cells))
    return "\n".join(lines)


def make_live_llm():
    import httpx

    from hpca.config import LLMSettings
    from hpca.llm import LLMClient

    headers = {"Authorization": f"Bearer {LIVE_KEY}"} if LIVE_KEY else {}
    try:
        response = httpx.get(f"{LIVE_URL}/models", timeout=30, headers=headers)
    except Exception as exc:
        print(f"LLM backend at {LIVE_URL} is unreachable ({exc}).", file=sys.stderr)
        sys.exit(2)
    if response.status_code != 200:
        hint = " Export HPCA_TEST_LLM_KEY." if response.status_code == 401 else ""
        print(f"LLM backend at {LIVE_URL} answered {response.status_code}.{hint}", file=sys.stderr)
        sys.exit(2)
    model = os.environ.get("HPCA_TEST_LLM_MODEL") or response.json()["data"][0]["id"]
    print(f"Backend: {LIVE_URL}  model: {model}")
    return LLMClient(
        LLMSettings(
            base_url=LIVE_URL, model=model, api_key=LIVE_KEY,
            request_timeout_s=180, enable_thinking=False,
        )
    ), model


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default="vision_results.json")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--arms", default=",".join(ARMS), help=f"comma list of {ARMS}")
    parser.add_argument("--tasks", default="", help="comma list of name substrings to run")
    parser.add_argument("--label", default="")
    parser.add_argument("--self-check", action="store_true", help="check tasks only, no backend")
    args = parser.parse_args()

    arms = [a for a in args.arms.split(",") if a]
    unknown = set(arms) - set(ARMS)
    if unknown:
        parser.error(f"unknown arm(s): {', '.join(sorted(unknown))}")
    tasks = build_tasks()
    if args.tasks:
        wanted = [w for w in args.tasks.split(",") if w]
        tasks = [t for t in tasks if any(w in t.name for w in wanted)]
    problems = self_check(tasks)
    if problems:
        print("SELF-CHECK FAILED — no score would be trustworthy:", file=sys.stderr)
        for problem in problems:
            print(f"  {problem}", file=sys.stderr)
        return 1
    print(f"self-check: {len(tasks)} tasks, predicates and fixtures sound")
    if args.self_check:
        return 0

    llm, model = make_live_llm()
    if not await llm.supports_images():
        print("The backend cannot see images (vision probe failed).", file=sys.stderr)
        return 2
    rows: list[dict] = []
    started = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for task in tasks:
            (root / task.name).mkdir()
            task.draw(root / task.name)
        for repeat in range(args.repeats):
            for task in tasks:
                for arm in arms:
                    row = await run_one(task, arm, llm, root / task.name)
                    row["repeat"] = repeat
                    rows.append(row)
                    mark = "ok  " if row["ok"] else "FAIL"
                    print(f"[{repeat + 1}/{args.repeats}] {mark} {arm:<10} {task.name:<22} "
                          f"{row['model_calls']} calls {row['seconds']:>5.1f}s  "
                          f"{row['answer'][:70]!r}")
    await llm.close()
    summary = summarise(rows, arms)
    print("\n" + summary)
    Path(args.out).write_text(json.dumps({
        "label": args.label, "model": model, "url": LIVE_URL,
        "repeats": args.repeats, "arms": arms,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "rows": rows,
    }, indent=2))
    print(f"\nwritten to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
