"""Images as the vision sub-loop sees them (hpca.agent.vision).

Everything here is deterministic, per §1: code decodes, scales and measures;
the model only interprets. A file becomes one or more ``Picture``s — a raster
as itself, a PDF page or an SVG rendered — each carrying

* the PNG that goes on the wire, scaled to the budget the server enforces
  anyway (``MAX_EDGE``, the job's MM_MAX_PIXELS), so the bytes sent are the
  bytes seen and a 40-megapixel micrograph is not shipped to be thrown away;
* a working copy at up to ``WORK_EDGE`` for ``zoom``, which is the point of
  zooming: the overview has lost exactly the small text worth checking;
* ``facts`` — what code can say for certain (size, bit depth, how blank it
  is, what touches the edge, what an SVG gets wrong), stated to the model
  beside the image so it does not have to guess at them, and returned to the
  orchestrator with the answer.

Anything that changes the picture before the model sees it — a 16-bit range
stretched to 8, a multi-page PDF cut to one page, transparency flattened — is
written into the facts. A model that is not told its micrograph was contrast
stretched will report the stretch as the sample.
"""

from __future__ import annotations

import io
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageChops, ImageOps

# Long edge of what is sent. Matches the server cap (1536x1536,
# llm.a40-l40.sh MM_MAX_PIXELS), and about 2.3k tokens at ~32px per token.
MAX_EDGE = 1536
# Long edge of the copy zoom crops from: a 3x3 cell of it is ~1365px, so an
# image up to this size is zoomed at native resolution.
WORK_EDGE = 4096
# PDFs and SVGs carry no pixels; this is the resolution they are drawn at.
PDF_DPI = 150
SVG_ZOOM = 2.0  # thin lines and small labels survive the overview's scaling
# Decompression bombs: Pillow's own guard is a warning at this size, raised
# to an error here, since a file this large is a mistake, not a plot.
Image.MAX_IMAGE_PIXELS = 400_000_000

GRID_COLUMNS = "ABC"
GRID_ROWS = "123"
CELLS = [f"{c}{r}" for r in GRID_ROWS for c in GRID_COLUMNS]


class ImageError(Exception):
    """A file that cannot be shown: missing, unreadable, or not an image."""


@dataclass
class Picture:
    label: str  # how the model and the report name it: "plot.pdf, page 1"
    work: Image.Image  # RGB, long edge <= WORK_EDGE
    facts: list[str] = field(default_factory=list)

    def overview_png(self) -> bytes:
        return _png(_fit(self.work, MAX_EDGE))

    def overview_size(self) -> tuple[int, int]:
        return _fit(self.work, MAX_EDGE).size

    def cell_png(self, cell: str) -> bytes:
        # Native pixels, shrunk only to the budget, never enlarged: enlarging
        # a cell (up to 3x) was measured against not enlarging it and showed
        # no gain on 16px run ids (12/18 with, 14/18 without; vision eval,
        # Qwen3.8-27B-INT4, 2026-10-23), so it went.
        crop = self.work.crop(cell_box(self.work.size, cell))
        return _png(_fit(crop, MAX_EDGE))


def cell_box(size: tuple[int, int], cell: str) -> tuple[int, int, int, int]:
    """The pixel box of one 3x3 grid cell, columns A-C, rows 1-3.

    Named cells rather than coordinates because a small model names "B2" far
    more reliably than it writes four pixel offsets into an image it saw at a
    different scale. Each cell overlaps its neighbours by a tenth, so a label
    sitting on a cell boundary is whole in at least one of them.
    """
    cell = cell.strip().upper()
    if len(cell) != 2 or cell[0] not in GRID_COLUMNS or cell[1] not in GRID_ROWS:
        raise ImageError(f"No cell {cell!r}: cells are A1..C3 (columns A-C, rows 1-3)")
    width, height = size
    col, row = GRID_COLUMNS.index(cell[0]), GRID_ROWS.index(cell[1])
    pad_x, pad_y = width // 30, height // 30
    return (
        max(0, col * width // 3 - pad_x),
        max(0, row * height // 3 - pad_y),
        min(width, (col + 1) * width // 3 + pad_x),
        min(height, (row + 1) * height // 3 + pad_y),
    )


def load(path: Path, *, page: int = 1) -> Picture:
    """One file as a Picture. ``page`` is 1-based and only read for a PDF."""
    if not path.exists():
        raise ImageError(f"Nothing at {path}.")
    if path.is_dir():
        raise ImageError(f"{path} is a directory, not an image.")
    head = _head(path)
    if head.startswith(b"%PDF"):
        return _load_pdf(path, page)
    if _is_svg(path, head):
        return _load_svg(path)
    return _load_raster(path)


def sniff(path: Path) -> str | None:
    """A short name for the image kind of ``path`` ("PNG", "PDF", "SVG", …),
    or None when it is not one. Cheap: reads a header, decodes nothing."""
    try:
        head = _head(path)
    except OSError:
        return None
    if head.startswith(b"%PDF"):
        return "PDF"
    if _is_svg(path, head):
        return "SVG"
    for magic, kind in _MAGIC:
        if head.startswith(magic):
            if kind == "WEBP" and head[8:12] != b"WEBP":
                continue
            return kind
    return None


_MAGIC = [
    (b"\x89PNG\r\n\x1a\n", "PNG"),
    (b"\xff\xd8\xff", "JPEG"),
    (b"GIF87a", "GIF"),
    (b"GIF89a", "GIF"),
    (b"RIFF", "WEBP"),
    (b"II*\x00", "TIFF"),
    (b"MM\x00*", "TIFF"),
    (b"II+\x00", "TIFF"),  # BigTIFF
    (b"MM\x00+", "TIFF"),
    (b"BM", "BMP"),
]


def _head(path: Path) -> bytes:
    with path.open("rb") as handle:
        return handle.read(4096)


def _is_svg(path: Path, head: bytes) -> bool:
    if path.suffix.lower() in (".svg", ".svgz"):
        return True
    text = head.lstrip(b"\xef\xbb\xbf \t\r\n")
    return text.startswith(b"<") and b"<svg" in head


# ------------------------------------------------------------------ rasters


def _load_raster(path: Path) -> Picture:
    try:
        image = Image.open(path)
        image.load()
    except Image.DecompressionBombError as exc:
        raise ImageError(f"{path.name} is too large to open: {exc}") from exc
    except (OSError, SyntaxError, ValueError) as exc:
        raise ImageError(f"{path.name} is not an image this tool can read: {exc}") from exc
    facts = [f"{image.format or 'image'}, {image.width}x{image.height}, mode {image.mode}"]
    frames = getattr(image, "n_frames", 1)
    if frames > 1:
        facts.append(
            f"{frames} frames (a stack or animation); only frame 1 is shown"
        )
    rgb, notes = to_rgb(image)
    facts.extend(notes)
    return _picture(path.name, rgb, facts)


def to_rgb(image: Image.Image) -> tuple[Image.Image, list[str]]:
    """8-bit RGB, with whatever that took said out loud."""
    notes: list[str] = []
    image = ImageOps.exif_transpose(image) or image
    mode = image.mode
    if mode in ("I;16", "I;16B", "I;16L", "I;16N", "I", "F"):
        low, high = image.getextrema()
        notes.append(
            f"{'float' if mode == 'F' else '16/32-bit'} single channel, values "
            f"{low:g}..{high:g}, linearly stretched to 0..255 for display — "
            "brightness here is relative, not absolute"
        )
        scale = 255.0 / (high - low) if high > low else 0.0
        as_float = image.convert("I").convert("F") if mode != "F" else image
        image = as_float.point(lambda v: v * scale - low * scale).convert("L")
    if image.mode in ("RGBA", "LA", "PA") or (
        image.mode == "P" and "transparency" in image.info
    ):
        rgba = image.convert("RGBA")
        if rgba.getextrema()[3][0] < 255:
            notes.append("has transparency, shown flattened onto white")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba)
    if image.mode == "CMYK":
        notes.append("CMYK, converted to RGB")
    return image.convert("RGB"), notes


# --------------------------------------------------------------------- PDF


def _load_pdf(path: Path, page: int) -> Picture:
    import pypdfium2 as pdfium

    try:
        document = pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as exc:
        raise ImageError(f"{path.name} is not a PDF this tool can read: {exc}") from exc
    try:
        pages = len(document)
        if not 1 <= page <= pages:
            raise ImageError(f"{path.name} has {pages} page(s); there is no page {page}.")
        rendered = document[page - 1].render(scale=PDF_DPI / 72).to_pil()
    finally:
        document.close()
    facts = [f"PDF, {pages} page(s); page {page} rendered at {PDF_DPI} dpi"]
    if pages > 1:
        facts.append(f"other pages are not shown (pass page= to see one)")
    rgb, notes = to_rgb(rendered)
    return _picture(f"{path.name}, page {page}", rgb, facts + notes)


# --------------------------------------------------------------------- SVG


def _load_svg(path: Path) -> Picture:
    import resvg_py

    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        import gzip

        raw = gzip.decompress(raw)
    text = raw.decode("utf-8", errors="replace")
    facts = ["SVG source, rendered for display"]
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        # Not rendered: a renderer that tolerates the error would show a
        # picture of something the file does not say.
        raise ImageError(f"{path.name} is not well-formed XML: {exc}") from exc
    lint = svg_lint(root)
    try:
        lint += text_overflow(root, resources_dir=str(path.parent))
        png = resvg_py.svg_to_bytes(
            svg_string=text,
            zoom=SVG_ZOOM,
            background="#ffffff",
            resources_dir=str(path.parent),
        )
    except ValueError as exc:
        raise ImageError(f"{path.name} could not be rendered: {exc}") from exc
    image = Image.open(io.BytesIO(bytes(png)))
    image.load()
    rgb, notes = to_rgb(image)
    # No edge guess for an SVG: text_overflow measured the real thing, and the
    # guess mistakes a rounded frame drawn near the edge for a cut (measured:
    # it flagged a fitting label's badge in every run of the vision eval).
    return _picture(path.name, rgb, facts + lint + notes, edges=False)


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _number(value: str | None) -> float | None:
    if value is None:
        return None
    match = re.fullmatch(r"\s*(-?[\d.]+(?:e-?\d+)?)\s*(px)?\s*", value)
    return float(match.group(1)) if match else None


def svg_canvas(root: ET.Element) -> tuple[float, float, float, float] | None:
    """The drawing's canvas in its own units: the viewBox, else 0,0 to the
    numeric width/height, else None (the size is up to whoever displays it)."""
    view_box = root.get("viewBox")
    if view_box:
        try:
            x, y, w, h = (float(p) for p in re.split(r"[\s,]+", view_box.strip()))
        except ValueError:
            return None
        return (x, y, x + w, y + h) if w > 0 and h > 0 else None
    width, height = _number(root.get("width")), _number(root.get("height"))
    if width and height:
        return (0.0, 0.0, width, height)
    return None


def svg_lint(root: ET.Element) -> list[str]:
    """What is wrong with an SVG that the source alone can prove: the canvas
    it declares. Where text lands needs a renderer — :func:`text_overflow`."""
    if _local(root.tag) != "svg":
        return [f"root element is <{_local(root.tag)}>, not <svg>"]
    findings: list[str] = []
    view_box = root.get("viewBox")
    if view_box:
        if svg_canvas(root) is None:
            findings.append(f"viewBox {view_box!r} is not four numbers with a positive size")
    else:
        if svg_canvas(root) is not None:
            findings.append("no viewBox: the drawing will not scale with its container")
        else:
            findings.append(
                "neither viewBox nor numeric width/height: the canvas size is "
                "up to whoever displays it"
            )
    texts = sum(1 for element in root.iter() if _local(element.tag) == "text")
    if texts:
        findings.append(f"{texts} text element(s)")
    return findings


SVG_NAMESPACE = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NAMESPACE)  # re-serialise without ns0: prefixes
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
# Everything that is not text, hidden for the overflow render.
HIDE_ALL_BUT_TEXT = (
    "rect, circle, ellipse, line, polyline, polygon, path, image, use "
    "{ display: none; }"
)
# The overflow render's backgrounds. Two, because text of any colour shows on
# one of them — measured: a white label on a green badge was invisible on
# white alone and read as "no text". Forcing the text black through the
# stylesheet does not work: an inline style="fill:…" outranks it.
PROBE_BACKGROUNDS = (("#ffffff", (255, 255, 255)), ("#000000", (0, 0, 0)))
OVERFLOW_RENDER_EDGE = 2000  # px across the widened canvas


def text_overflow(root: ET.Element, *, resources_dir: str | None = None) -> list[str]:
    """Which canvas edges the drawing's text runs past, measured.

    The text is rendered alone on a canvas widened by a full canvas-size on
    every side, and its ink is measured back in the drawing's own units. The
    renderer does the layout — fonts, anchors, transforms — so nothing is
    estimated: a label that fits passes and one that does not is caught with
    how far it runs over. Replaces two guesses that were each wrong somewhere:
    checking text anchors against the canvas (blind to anything transformed,
    and to a label anchored inside that runs out), and looking for ink near
    the rendered edge (which took a rounded frame for a cut).
    """
    import copy

    import resvg_py

    box = svg_canvas(root)
    if box is None or not any(_local(e.tag) == "text" for e in root.iter()):
        return []
    x0, y0, x1, y1 = box
    width, height = x1 - x0, y1 - y0
    margin = max(width, height)
    scale = OVERFLOW_RENDER_EDGE / (max(width, height) + 2 * margin)
    probe = copy.deepcopy(root)
    probe.set("viewBox", f"{x0 - margin} {y0 - margin} {width + 2 * margin} {height + 2 * margin}")
    probe.set("width", str(round((width + 2 * margin) * scale)))
    probe.set("height", str(round((height + 2 * margin) * scale)))
    probe.attrib.pop("preserveAspectRatio", None)
    source = ET.tostring(probe, encoding="unicode")
    boxes = []
    for css_colour, rgb in PROBE_BACKGROUNDS:
        png = resvg_py.svg_to_bytes(
            svg_string=source,
            style_sheet=HIDE_ALL_BUT_TEXT,
            background=css_colour,
            resources_dir=resources_dir,
        )
        image = Image.open(io.BytesIO(bytes(png))).convert("RGB")
        found = _ink_mask(image, rgb).getbbox()
        if found is not None:
            boxes.append(found)
    ink = (
        (
            min(b[0] for b in boxes),
            min(b[1] for b in boxes),
            max(b[2] for b in boxes),
            max(b[3] for b in boxes),
        )
        if boxes
        else None
    )
    if ink is None:
        return [
            "the text elements drew nothing — no usable font on this machine, "
            "or the text is empty or hidden"
        ]
    left = ink[0] / scale + x0 - margin
    top = ink[1] / scale + y0 - margin
    right = ink[2] / scale + x0 - margin
    bottom = ink[3] / scale + y0 - margin
    tolerance = 2 / scale  # two pixels of the probe: antialiasing, not overflow
    over = {
        "left": x0 - left,
        "top": y0 - top,
        "right": right - x1,
        "bottom": bottom - y1,
    }
    findings = [
        f"text runs past the {side} edge of the canvas by about {amount:.3g} "
        f"units (canvas {width:g}x{height:g}) — it is cut off there"
        for side, amount in over.items()
        if amount > tolerance
    ]
    return findings


# ------------------------------------------------------------------ shared


def _picture(
    label: str, rgb: Image.Image, facts: list[str], *, edges: bool = True
) -> Picture:
    work = _fit(rgb, WORK_EDGE)
    if max(rgb.size) > MAX_EDGE:
        sent = _fit(rgb, MAX_EDGE).size
        facts.append(
            f"shown scaled to {sent[0]}x{sent[1]}; zoom shows a cell in more detail"
        )
    facts.extend(_ink_facts(work, edges=edges))
    return Picture(label=label, work=work, facts=facts)


def _fit(image: Image.Image, edge: int) -> Image.Image:
    if max(image.size) <= edge:
        return image
    scaled = image.copy()
    scaled.thumbnail((edge, edge), Image.Resampling.LANCZOS)
    return scaled


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _ink_mask(image: Image.Image, background: tuple) -> Image.Image:
    """255 where a pixel differs visibly from ``background``, else 0."""
    difference = ImageChops.difference(image, Image.new("RGB", image.size, background))
    return difference.convert("L").point(lambda v: 255 if v > INK_TOLERANCE else 0)


def _ink_facts(image: Image.Image, *, edges: bool = True) -> list[str]:
    """How empty the picture is, and which edges something runs into.

    Background is the commonest colour along the border, not "white": a plot
    on a grey panel or a full-bleed dark slide has its own.

    The edge test looks at a band, not the outermost pixel row. Measured on
    a label running off an SVG's right edge: resvg drops a glyph that does not
    fit whole, so the cut lands *between* letters and the last column is
    clean — the ink stopped 3px short of a 400px render. The band is at least
    EDGE_BAND_MIN to catch that, and at most EDGE_BAND_MAX so a figure saved
    with a normal margin passes (matplotlib's tight bbox pads by ~10px at
    100 dpi, more at higher).
    """
    width, height = image.size
    border: Counter = Counter()
    for box in (
        (0, 0, width, 1),
        (0, height - 1, width, height),
        (0, 0, 1, height),
        (width - 1, 0, width, height),
    ):
        strip = image.crop(box)
        for count, colour in strip.getcolors(strip.width * strip.height):
            border[colour] += count
    background = border.most_common(1)[0][0]
    ink = _ink_mask(image, background)
    blank = 1 - ink.histogram()[255] / (width * height)
    facts = []
    if blank > 0.995:
        facts.append("almost entirely background — effectively blank")
    elif blank > 0.97:
        facts.append(f"{blank:.0%} background — very sparse")
    if not edges:
        return facts
    band_x = min(EDGE_BAND_MAX, max(EDGE_BAND_MIN, round(width * 0.01)))
    band_y = min(EDGE_BAND_MAX, max(EDGE_BAND_MIN, round(height * 0.01)))
    bands = {
        "top": (ink.crop((0, 0, width, band_y)), True),
        "bottom": (ink.crop((0, height - band_y, width, height)), True),
        "left": (ink.crop((0, 0, band_x, height)), False),
        "right": (ink.crop((width - band_x, 0, width, height)), False),
    }
    touching = []
    for side, (band, horizontal) in bands.items():
        # Squash the band to one line along its edge: a position is inked
        # when anything in the band across from it is.
        line = band.resize(
            (band.width, 1) if horizontal else (1, band.height),
            Image.Resampling.BOX,
        )
        positions = line.width * line.height
        covered = (positions - line.histogram()[0]) / positions
        # A frame or a full-bleed band runs the whole edge and is design;
        # something cut off occupies part of it.
        if 0.002 < covered < 0.9:
            touching.append(side)
    if touching:
        facts.append(
            "content reaches the " + ", ".join(touching) + " edge(s) — "
            "possibly cut off there"
        )
    return facts


INK_TOLERANCE = 24  # luminance difference from background that counts as ink
EDGE_BAND_MIN = 4  # px; 1% of the side in between
EDGE_BAND_MAX = 8
