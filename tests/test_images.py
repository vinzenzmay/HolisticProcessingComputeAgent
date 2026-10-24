"""Tests for hpca.images: loading, normalising and measuring what the
vision sub-loop is shown. Everything here is deterministic; no model."""

import io

import pytest
from PIL import Image, ImageDraw

from hpca import images
from hpca.images import ImageError, cell_box, load, sniff, svg_canvas, svg_lint, text_overflow
import xml.etree.ElementTree as ET


def png_size(data: bytes) -> tuple[int, int]:
    return Image.open(io.BytesIO(data)).size


def save(image: Image.Image, path, **kwargs):
    image.save(path, **kwargs)
    return path


class TestSniff:
    @pytest.mark.parametrize(
        "fmt, kind", [("PNG", "PNG"), ("JPEG", "JPEG"), ("GIF", "GIF"),
                      ("WEBP", "WEBP"), ("TIFF", "TIFF"), ("BMP", "BMP")],
    )
    def test_raster_kinds_by_content(self, tmp_path, fmt, kind):
        # A misleading extension on purpose: the header decides.
        path = save(Image.new("RGB", (8, 8)), tmp_path / "image.dat", format=fmt)
        assert sniff(path) == kind

    def test_pdf_and_svg(self, tmp_path):
        pdf = save(Image.new("RGB", (8, 8)), tmp_path / "a.pdf", format="PDF")
        svg = tmp_path / "noext"
        svg.write_text('<?xml version="1.0"?>\n<svg xmlns="http://www.w3.org/2000/svg"/>')
        assert sniff(pdf) == "PDF"
        assert sniff(svg) == "SVG"

    def test_text_is_not_an_image(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("RIFF is a container format\n")
        assert sniff(path) is None


class TestRaster:
    def test_sixteen_bit_is_stretched_and_says_so(self, tmp_path):
        image = Image.new("I;16", (64, 64))
        image.putdata([100 + i for i in range(64 * 64)])
        picture = load(save(image, tmp_path / "micro.tif"))
        assert picture.work.mode == "RGB"
        assert picture.work.getextrema()[0] == (0, 255)
        assert any("values 100..4195" in f and "stretched" in f for f in picture.facts)

    def test_transparency_is_flattened_and_blankness_reported(self, tmp_path):
        picture = load(save(Image.new("RGBA", (50, 40), (0, 0, 0, 0)), tmp_path / "a.png"))
        assert picture.work.getpixel((0, 0)) == (255, 255, 255)
        assert any("transparency" in f for f in picture.facts)
        assert any("effectively blank" in f for f in picture.facts)

    def test_frames_beyond_the_first_are_named(self, tmp_path):
        frames = [Image.new("L", (16, 16), v) for v in (0, 128, 255)]
        path = tmp_path / "stack.tif"
        frames[0].save(path, save_all=True, append_images=frames[1:])
        assert any("3 frames" in f for f in load(path).facts)

    def test_large_image_is_scaled_for_the_wire_but_not_for_zoom(self, tmp_path):
        picture = load(save(Image.new("RGB", (3000, 1500), "white"), tmp_path / "big.png"))
        assert png_size(picture.overview_png()) == (images.MAX_EDGE, images.MAX_EDGE // 2)
        assert picture.work.size == (3000, 1500)
        assert any("shown scaled to 1536x768" in f for f in picture.facts)

    def test_not_an_image(self, tmp_path):
        path = tmp_path / "x.png"
        path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"garbage")
        with pytest.raises(ImageError, match="not an image"):
            load(path)

    def test_missing_and_directory(self, tmp_path):
        with pytest.raises(ImageError, match="Nothing at"):
            load(tmp_path / "absent.png")
        with pytest.raises(ImageError, match="directory"):
            load(tmp_path)


class TestEdges:
    def facts(self, tmp_path, image):
        return [f for f in load(save(image, tmp_path / "e.png")).facts if "edge" in f]

    def test_content_running_off_an_edge_is_flagged(self, tmp_path):
        image = Image.new("RGB", (800, 600), "white")
        ImageDraw.Draw(image).rectangle((100, 400, 200, 599), fill="blue")
        assert self.facts(tmp_path, image) == [
            "content reaches the bottom edge(s) — possibly cut off there"
        ]

    def test_a_cut_that_stops_short_of_the_last_pixel_still_counts(self, tmp_path):
        # resvg drops a glyph that does not fit, so a cut label ends a few
        # pixels before the edge rather than on it.
        image = Image.new("RGB", (400, 200), "white")
        ImageDraw.Draw(image).rectangle((300, 60, 396, 100), fill="black")
        assert "right" in self.facts(tmp_path, image)[0]

    def test_a_margin_and_a_frame_are_not_cuts(self, tmp_path):
        margin = Image.new("RGB", (800, 600), "white")
        ImageDraw.Draw(margin).rectangle((10, 10, 789, 589), outline="black")
        framed = Image.new("RGB", (800, 600), "white")
        ImageDraw.Draw(framed).rectangle((0, 0, 799, 599), outline="black", width=2)
        assert self.facts(tmp_path, margin) == []
        assert self.facts(tmp_path, framed) == []

    def test_background_is_not_assumed_white(self, tmp_path):
        image = Image.new("RGB", (400, 300), (30, 30, 30))
        ImageDraw.Draw(image).rectangle((100, 100, 200, 200), fill="yellow")
        facts = load(save(image, tmp_path / "dark.png")).facts
        assert not any("blank" in f or "edge" in f for f in facts)


class TestGrid:
    def test_cells_cover_the_image_with_overlap(self):
        assert cell_box((300, 300), "A1") == (0, 0, 110, 110)
        assert cell_box((300, 300), "c3") == (190, 190, 300, 300)

    @pytest.mark.parametrize("cell", ["D1", "A4", "B", "", "B22"])
    def test_unknown_cells_are_refused(self, cell):
        with pytest.raises(ImageError, match="A1..C3"):
            cell_box((300, 300), cell)

    def test_zoom_of_a_small_cell_is_native_not_enlarged(self, tmp_path):
        picture = load(save(Image.new("RGB", (600, 600), "white"), tmp_path / "a.png"))
        left, top, right, bottom = cell_box((600, 600), "B2")
        assert png_size(picture.cell_png("B2")) == (right - left, bottom - top)

    def test_zoom_of_a_large_cell_stays_within_budget(self, tmp_path):
        picture = load(save(Image.new("RGB", (4000, 4000), "white"), tmp_path / "a.png"))
        assert max(png_size(picture.cell_png("A1"))) <= images.MAX_EDGE


class TestPdf:
    def make(self, tmp_path, pages=2):
        first, *rest = [Image.new("RGB", (200, 100), c) for c in ("white", "red")[:pages]]
        path = tmp_path / "doc.pdf"
        first.save(path, save_all=True, append_images=rest)
        return path

    def test_first_page_by_default_and_page_count_reported(self, tmp_path):
        picture = load(self.make(tmp_path))
        assert picture.label == "doc.pdf, page 1"
        assert any("2 page(s)" in f for f in picture.facts)
        assert any("other pages are not shown" in f for f in picture.facts)

    def test_a_chosen_page(self, tmp_path):
        picture = load(self.make(tmp_path), page=2)
        assert picture.work.getpixel((50, 50))[0] > 200  # the red page

    def test_page_out_of_range(self, tmp_path):
        with pytest.raises(ImageError, match="no page 3"):
            load(self.make(tmp_path), page=3)


SVG_NS = 'xmlns="http://www.w3.org/2000/svg"'


class TestSvg:
    def test_renders_with_lint_facts(self, tmp_path):
        path = tmp_path / "fig.svg"
        path.write_text(
            f'<svg {SVG_NS} viewBox="0 0 200 100">'
            '<rect x="10" y="10" width="50" height="50" fill="blue"/>'
            '<text x="10" y="140">below</text></svg>'
        )
        picture = load(path)
        assert picture.work.size == (400, 200)  # SVG_ZOOM
        assert any("past the bottom edge" in f for f in picture.facts)

    def test_malformed_xml_is_refused_not_rendered(self, tmp_path):
        path = tmp_path / "bad.svg"
        path.write_text(f"<svg {SVG_NS}><rect></svg>")
        with pytest.raises(ImageError, match="not well-formed"):
            load(path)

    def test_lint_without_viewbox(self):
        root = ET.fromstring(f'<svg {SVG_NS} width="200" height="100"/>')
        assert any("no viewBox" in f for f in svg_lint(root))
        bare = ET.fromstring(f"<svg {SVG_NS}/>")
        assert any("neither viewBox" in f for f in svg_lint(bare))

    def test_canvas(self):
        assert svg_canvas(ET.fromstring(f'<svg {SVG_NS} viewBox="5 5 10 20"/>')) == (5, 5, 15, 25)
        assert svg_canvas(ET.fromstring(f'<svg {SVG_NS} width="30px" height="40"/>')) == (0, 0, 30, 40)
        assert svg_canvas(ET.fromstring(f'<svg {SVG_NS} viewBox="0 0 0 10"/>')) is None
        assert svg_canvas(ET.fromstring(f"<svg {SVG_NS}/>")) is None


def overflow(body: str, canvas: str = 'viewBox="0 0 300 100"') -> list[str]:
    return text_overflow(ET.fromstring(f"<svg {SVG_NS} {canvas}>{body}</svg>"))


class TestTextOverflow:
    """Measured by rendering the text alone on a widened canvas."""

    LABEL = '<text x="{x}" y="62" font-family="sans-serif" font-size="30">pipeline passed</text>'

    def test_a_label_that_fits_passes(self):
        assert overflow(self.LABEL.format(x=20)) == []

    def test_a_label_anchored_inside_that_runs_out_is_caught(self):
        (finding,) = overflow(self.LABEL.format(x=140))
        assert finding.startswith("text runs past the right edge of the canvas by about")

    def test_a_frame_near_the_edge_is_not_text(self):
        # The case the rendered-edge guess got wrong in every eval run.
        frame = '<rect x="2" y="2" width="296" height="96" rx="10" fill="none" stroke="black"/>'
        assert overflow(frame + self.LABEL.format(x=20)) == []

    def test_transforms_are_followed(self):
        moved_in = '<g transform="translate(0,-500)"><text x="10" y="550">ok</text></g>'
        assert overflow(moved_in, 'viewBox="0 0 100 100"') == []
        rotated = ('<g transform="rotate(90 50 50)"><text x="10" y="50" '
                   'font-size="20">a long rotated label</text></g>')
        assert "bottom edge" in overflow(rotated, 'viewBox="0 0 100 100"')[0]

    def test_text_of_any_colour_is_seen(self):
        # Inline style outranks a stylesheet, so the probe cannot recolour it.
        white = '<text x="60" y="25" style="fill:white" font-size="16">overflowing</text>'
        assert "right edge" in overflow(white, 'viewBox="0 0 100 40"')[0]

    def test_nothing_to_measure(self):
        assert overflow('<rect width="10" height="10"/>') == []
        assert overflow(self.LABEL.format(x=20), canvas="") == []  # no canvas

    def test_not_an_svg_root(self):
        assert svg_lint(ET.fromstring("<html/>")) == ["root element is <html>, not <svg>"]
