"""Tests for hpca.agent.vision: the image-inspector sub-loop and the
inspect_image tool, against a scripted fake model."""

import json
from types import SimpleNamespace

from PIL import Image

from hpca.agent import vision
from hpca.agent.vision import (
    FINAL_ROUND_NOTE,
    UNSEEING_BACKEND,
    InspectImageParams,
    inspect,
    inspect_image,
)
from hpca.images import load
from hpca.llm import ChatResponse, content_text
from hpca.logs import render_messages


class FakeLLM:
    def __init__(self, outputs, *, sees=True):
        self._outputs = list(outputs)
        self.calls = []
        self.sees = sees

    async def chat(self, messages, *, json_schema=None, **kwargs):
        self.calls.append(list(messages))
        return ChatResponse(content=self._outputs.pop(0))

    async def supports_constrained_decoding(self):
        return True

    async def supports_images(self):
        return self.sees


def respond(text):
    return json.dumps({"action": "respond", "response": text})


def zoom(image, cell):
    return json.dumps(
        {"action": "tool_call", "tool": "zoom", "arguments": {"image": image, "cell": cell}}
    )


def image_count(messages):
    return sum(
        1
        for m in messages
        if isinstance(m["content"], list)
        for part in m["content"]
        if part.get("type") == "image_url"
    )


def texts(messages):
    return "\n".join(content_text(m["content"]) for m in messages)


def picture(tmp_path, name="a.png", size=(300, 200)):
    path = tmp_path / name
    Image.new("RGB", size, "white").save(path)
    return load(path)


class TestInspect:
    async def test_direct_answer_carries_the_measured_facts(self, tmp_path):
        llm = FakeLLM([respond("ANSWER: it is empty")])
        report = await inspect(llm, "Is it empty?", [picture(tmp_path)])
        assert report.splitlines()[0].startswith("image 1: a.png — PNG, 300x200")
        assert "effectively blank" in report.splitlines()[0]
        assert report.endswith("ANSWER: it is empty")
        # the model saw the question, the facts and exactly one image
        assert image_count(llm.calls[0]) == 1
        shown = texts(llm.calls[0])
        assert "Question: Is it empty?" in shown and "effectively blank" in shown

    async def test_zoom_attaches_the_cell_and_detaches_the_previous_zoom(self, tmp_path):
        llm = FakeLLM([zoom(1, "B2"), zoom(1, "c3"), respond("done")])
        await inspect(llm, "q", [picture(tmp_path)])
        assert image_count(llm.calls[1]) == 2  # overview + B2
        assert "[zoom] image 1, cell B2:" in texts(llm.calls[1])
        assert image_count(llm.calls[2]) == 2  # overview + C3; B2 detached
        assert "cell B2: shown earlier, no longer attached" in texts(llm.calls[2])

    async def test_never_more_images_than_the_server_takes(self, tmp_path):
        pictures = [picture(tmp_path, f"{n}.png") for n in range(4)]
        llm = FakeLLM([zoom(3, "A1"), respond("done")])
        await inspect(llm, "q", pictures)
        assert image_count(llm.calls[0]) == 4
        second = llm.calls[1]
        assert image_count(second) == vision.MAX_IMAGES_PER_REQUEST
        # the overview that made room is not the one being zoomed into
        opening = second[1]["content"]
        assert "Image 1: 0.png" in opening[1]["text"]
        assert "no longer attached" in opening[1]["text"]
        image3 = [p for p in opening if p.get("type") == "text" and "Image 3" in p["text"]]
        assert "no longer attached" not in image3[0]["text"]

    async def test_a_bad_zoom_is_answered_not_raised(self, tmp_path):
        llm = FakeLLM([zoom(2, "B2"), zoom(1, "D9"), respond("ok")])
        report = await inspect(llm, "q", [picture(tmp_path)])
        assert report.endswith("ok")
        assert "There is no image 2" in texts(llm.calls[1])
        assert "There is no cell 'D9'" in texts(llm.calls[2])

    async def test_last_round_asks_for_the_answer(self, tmp_path):
        llm = FakeLLM([zoom(1, "A1"), zoom(1, "A2")])
        report = await inspect(llm, "q", [picture(tmp_path)], max_rounds=1)
        assert FINAL_ROUND_NOTE in texts(llm.calls[1])
        assert "ran out of rounds" in report

    async def test_long_answers_are_capped(self, tmp_path):
        llm = FakeLLM([respond("x" * 5000)])
        report = await inspect(llm, "q", [picture(tmp_path)])
        assert "[answer cut at its length cap]" in report
        assert len(report) < vision.MAX_ANSWER_CHARS + 300


class TestInspectImageTool:
    def ctx(self, tmp_path, llm):
        return SimpleNamespace(llm=llm, workdir=tmp_path)

    async def test_relative_paths_from_the_workdir(self, tmp_path):
        picture(tmp_path, "plot.png")
        llm = FakeLLM([respond("a plot")])
        out = await inspect_image(
            InspectImageParams(question="what?", paths=["plot.png"]), self.ctx(tmp_path, llm)
        )
        assert out.startswith("image 1: plot.png")
        assert out.endswith("a plot")

    async def test_blind_backend_is_said_plainly_and_nothing_is_sent(self, tmp_path):
        picture(tmp_path, "plot.png")
        llm = FakeLLM([], sees=False)
        out = await inspect_image(
            InspectImageParams(question="q", paths=["plot.png"]), self.ctx(tmp_path, llm)
        )
        assert out == UNSEEING_BACKEND
        assert llm.calls == []

    async def test_unreadable_files_are_reported_beside_the_rest(self, tmp_path):
        picture(tmp_path, "plot.png")
        (tmp_path / "notes.txt").write_text("hello")
        llm = FakeLLM([respond("seen")])
        out = await inspect_image(
            InspectImageParams(question="q", paths=["missing.png", "notes.txt", "plot.png"]),
            self.ctx(tmp_path, llm),
        )
        lines = out.splitlines()
        assert lines[0].startswith("[not inspected] missing.png: Nothing at")
        assert lines[1].startswith("[not inspected] notes.txt:")
        assert image_count(llm.calls[0]) == 1

    async def test_nothing_loadable_means_no_model_call(self, tmp_path):
        llm = FakeLLM([])
        out = await inspect_image(
            InspectImageParams(question="q", paths=["missing.png"]), self.ctx(tmp_path, llm)
        )
        assert out.startswith("[not inspected] missing.png")
        assert llm.calls == []

    async def test_no_llm(self, tmp_path):
        ctx = SimpleNamespace(llm=None, workdir=tmp_path)
        out = await inspect_image(InspectImageParams(question="q", paths=["a.png"]), ctx)
        assert "unavailable" in out

    def test_paths_come_last_in_the_schema(self):
        # §4.3: an array argument followed by nothing.
        assert list(InspectImageParams.model_json_schema()["properties"])[-1] == "paths"


def test_session_log_shows_images_as_placeholders():
    messages = [
        {"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * 10000}},
        ]},
    ]
    assert render_messages(messages) == "user: look\n[image]"


async def test_read_file_points_images_at_inspect_image(tmp_path):
    from hpca.agent.builtin_tools import ReadFileParams, read_file

    Image.new("RGB", (8, 8)).save(tmp_path / "plot.png")
    svg = tmp_path / "fig.svg"
    svg.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>')
    ctx = SimpleNamespace(workdir=tmp_path)
    out = await read_file(ReadFileParams(path="plot.png"), ctx)
    assert "is a PNG file, not text" in out and "inspect_image" in out
    # SVG source is text, and often what the caller wants to edit
    assert "<svg" in await read_file(ReadFileParams(path="fig.svg"), ctx)


def test_the_chat_row_names_what_was_inspected():
    from hpca.transcript import call_target

    assert call_target({"question": "q", "paths": ["out/plot.png"]}) == "plot.png"
    assert call_target({"question": "q", "paths": ["a.png", "b/c.svg", "d.pdf"]}) == "a.png +2"
    assert call_target({"question": "q"}) == ""
