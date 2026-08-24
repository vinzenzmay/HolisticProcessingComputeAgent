"""Tests for context compaction (redesign Phase 6)."""

import pytest

from hpca.agent import compact
from hpca.agent import history as history_module
from hpca.llm import ChatResponse


class FakeLLM:
    def __init__(self, summary="the user is aligning reads; STAR needs 40G"):
        self._summary = summary
        self.calls = []
        # What the summarizer asked for, not just what it said: the budget and
        # the thinking flag are two of the three reasons a summary came back
        # cut off, and neither shows up in the text.
        self.kwargs = []
        self.finish_reason = "stop"

    async def chat(self, messages, **kwargs):
        self.calls.append(messages)
        self.kwargs.append(kwargs)
        return ChatResponse(
            content=self._summary, finish_reason=self.finish_reason
        )


def history(count, chars=100):
    return [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i} " + "x" * chars}
        for i in range(count)
    ]


class TestShouldCompact:
    def test_off_without_a_known_window(self):
        assert not compact.should_compact(history(100), max_model_len=None)

    def test_off_for_a_short_history(self):
        assert not compact.should_compact(history(4), max_model_len=100)

    def test_on_when_the_estimate_passes_the_threshold(self):
        # 60 messages * ~100 chars = ~1500 tokens, over 70% of 2000
        assert compact.should_compact(history(60), max_model_len=2000)

    def test_off_with_plenty_of_room(self):
        assert not compact.should_compact(history(60), max_model_len=100_000)


def tool_history(pairs):
    """A history of completed tool rounds: call, result, call, result, ..."""
    messages = []
    for i in range(pairs):
        messages += history_module.tool_exchange(
            "read_file", {"registry_key": f"f{i}"}, f"12 lines from f{i}"
        )
    return messages


class TestSplit:
    def test_keeps_a_recent_tail_verbatim(self):
        older, keep = compact.split(history(60))
        assert len(keep) >= compact.KEEP_RECENT
        assert older + keep == history(60)
        assert keep[-1]["content"].startswith("m59")

    def test_the_tail_never_opens_on_an_orphaned_result(self):
        # 38 rounds puts the budget boundary (keep = 76 // 3 = 25) between a
        # call and its result; the cut steps back so the call comes along.
        messages = tool_history(38)
        older, keep = compact.split(messages)
        assert history_module.is_tool_call_message(keep[0])
        assert len(keep) == 26  # one more than the budget asked for
        assert older + keep == messages

    def test_an_aligned_boundary_is_left_alone(self):
        messages = tool_history(39)  # keep = 78 // 3 = 26, already on a call
        older, keep = compact.split(messages)
        assert history_module.is_tool_call_message(keep[0])
        assert len(keep) == 26
        assert older + keep == messages

    def test_short_history_is_all_kept(self):
        older, keep = compact.split(history(5))
        assert older == []
        assert len(keep) == 5


class TestSummarize:
    async def test_produces_one_user_message(self):
        summary = await compact.summarize(FakeLLM(), history(20))
        # the shape the model already knows
        assert summary.message["role"] == "user"
        assert summary.content.startswith(compact.SUMMARY_PREFIX)
        assert "STAR needs 40G" in summary.content

    async def test_prompt_asks_to_keep_identifiers(self):
        llm = FakeLLM()
        await compact.summarize(llm, history(20))
        system = llm.calls[0][0]["content"]
        assert "verbatim" in system
        assert "job ids" in system

    async def test_empty_summary_is_an_error(self):
        with pytest.raises(ValueError):
            await compact.summarize(FakeLLM(summary="   "), history(20))

    async def test_is_summary_recognizes_its_own_output(self):
        summary = await compact.summarize(FakeLLM(), history(20))
        assert compact.is_summary(summary.message)
        assert not compact.is_summary({"role": "user", "content": "hello"})

    async def test_summary_is_length_bounded(self):
        summary = await compact.summarize(FakeLLM(summary="x" * 9000), history(20))
        assert len(summary.content) < compact.MAX_SUMMARY_CHARS + 60


class TestWhatMadeSummariesLookBroken:
    """The `/compact` complaint: the summary on screen stopped mid-sentence.

    Three causes, all of them in this module and none of them visible to the
    user — which is why the review screen is also told when it happened
    (`Summary.truncated`).
    """

    async def test_the_generation_budget_matches_the_cap(self):
        """The first cause: the budget was twice the cap, so anything over
        ~200 words was written and then sliced by `[:cap]`."""
        llm = FakeLLM()
        await compact.summarize(llm, history(20))
        assert llm.kwargs[-1]["max_tokens"] == compact.budget(
            compact.MAX_SUMMARY_CHARS
        )
        await compact.summarize(llm, history(20), guidance="keep the paths")
        assert llm.kwargs[-1]["max_tokens"] == compact.budget(
            compact.MAX_GUIDED_SUMMARY_CHARS
        )

    async def test_the_summarizer_never_thinks(self):
        """The second: reasoning is billed against the same budget as the
        answer, so on a reasoning backend the summary is what runs out."""
        llm = FakeLLM()
        await compact.summarize(llm, history(20))
        assert llm.kwargs[-1]["enable_thinking"] is False

    async def test_a_cut_summary_says_it_was_cut(self):
        """The third: a backend that stops at max_tokens said so in
        `finish_reason` and nothing read it."""
        llm = FakeLLM()
        llm.finish_reason = "length"
        summary = await compact.summarize(llm, history(20))
        assert summary.truncated

    async def test_and_so_does_one_the_cap_cut(self):
        summary = await compact.summarize(FakeLLM(summary="x" * 9000), history(20))
        assert summary.truncated

    async def test_an_ordinary_summary_does_not(self):
        assert not (await compact.summarize(FakeLLM(), history(20))).truncated


class TestClip:
    def test_short_text_is_left_alone(self):
        assert compact.clip("all of it", 100) == ("all of it", False)

    def test_a_long_one_is_cut_at_a_sentence(self):
        text = "First sentence. Second one runs on and on and on and on."
        cut, lost = compact.clip(text, 30)
        assert lost
        assert cut.startswith("First sentence.")
        assert "runs on" not in cut

    def test_a_cut_says_so(self):
        cut, _ = compact.clip("First sentence. " + "x" * 100, 30)
        assert cut.endswith(compact.ELLIPSIS)

    def test_no_sentence_end_falls_back_to_a_word(self):
        cut, lost = compact.clip("aaa bbb ccc ddd eee fff ggg", 12)
        assert lost
        # never mid-word, which is what "the tool looks broken" was
        assert cut.replace(compact.ELLIPSIS, "").split()[-1] in ("aaa", "bbb", "ccc")


class TestRevision:
    """The retry: the user turned a summary down and said what was wrong."""

    async def test_the_comment_and_the_rejected_text_both_reach_the_model(self):
        llm = FakeLLM()
        await compact.summarize(
            llm,
            history(20),
            previous="STAR needs 40G and the run died at",
            comment="you cut it off — finish the sentence",
        )
        system = llm.calls[-1][0]["content"]
        assert "you cut it off" in system
        assert "STAR needs 40G and the run died at" in system

    async def test_the_comment_does_not_become_a_standing_instruction(self):
        """It is about this summary, not about the session: "make it shorter"
        must not be waiting in the folded view for the next ten turns."""
        summary = await compact.summarize(
            FakeLLM(), history(20), comment="make it shorter"
        )
        assert "make it shorter" not in summary.content
        assert compact.FOCUS_PREFIX not in summary.content

    async def test_a_retry_gets_the_roomier_cap(self):
        """The commonest complaint is that it stopped too early, so the second
        attempt is not held to the tighter of the two bounds."""
        llm = FakeLLM()
        await compact.summarize(llm, history(20), comment="finish it")
        assert llm.kwargs[-1]["max_tokens"] == compact.budget(
            compact.MAX_GUIDED_SUMMARY_CHARS
        )

    def test_the_body_handed_back_is_the_model_own_text(self):
        message = {
            "role": "user",
            "content": (
                f"{compact.SUMMARY_PREFIX}\nwhat happened\n\n"
                f"{compact.FOCUS_PREFIX}\nkeep the paths"
            ),
        }
        assert compact.summary_body(message) == "what happened"


class TestTranscript:
    def test_an_ordinary_message_is_clipped(self):
        text = compact.transcript([{"role": "user", "content": "y" * 5000}])
        assert len(text) < compact.MAX_MESSAGE_CHARS + 20

    def test_an_earlier_summary_survives_the_next_fold(self):
        """Folding twice must not shred the first summary down to a per-message
        excerpt: it already stands for the whole start of the session."""
        summary = {
            "role": "user",
            "content": f"{compact.SUMMARY_PREFIX}\n" + "s" * 2000,
        }
        text = compact.transcript([summary, {"role": "user", "content": "next"}])
        assert text.count("s") >= 2000


class TestGuidedSummarize:
    """/compact takes the text after it as an instruction for the summary:
    what to keep, or what the user is about to do next."""

    async def test_the_instruction_reaches_the_summarizer(self):
        llm = FakeLLM()
        await compact.summarize(
            llm, history(20), guidance="keep the STAR parameters exactly"
        )
        system = llm.calls[0][0]["content"]
        assert "keep the STAR parameters exactly" in system
        # and it is framed as outranking the default brevity target
        assert "word limit" in system

    async def test_the_instruction_is_kept_in_the_summary_message(self):
        """The declared next step stays in the model's view after the fold —
        that is how the summary "aligns with what comes next" on later turns."""
        summary = await compact.summarize(
            llm := FakeLLM(), history(20), guidance="next I run the full cohort"
        )
        assert compact.is_summary(summary.message)  # still a summary message
        assert "next I run the full cohort" in summary.content
        assert compact.FOCUS_PREFIX in summary.content
        assert llm.calls  # sanity: it really went through the model

    async def test_without_an_instruction_nothing_is_added(self):
        llm = FakeLLM()
        summary = await compact.summarize(llm, history(20))
        assert compact.FOCUS_PREFIX not in summary.content
        assert "word limit" not in llm.calls[0][0]["content"]

    async def test_a_guided_summary_may_be_longer(self):
        """Naming things to keep needs room for them; the cap still exists."""
        guided = await compact.summarize(
            FakeLLM(summary="x" * 9000), history(20), guidance="keep every path"
        )
        assert compact.MAX_SUMMARY_CHARS < len(guided.content)
        assert len(guided.content) < compact.MAX_GUIDED_SUMMARY_CHARS + 200

    async def test_a_blank_instruction_is_no_instruction(self):
        llm = FakeLLM()
        summary = await compact.summarize(llm, history(20), guidance="   ")
        assert compact.FOCUS_PREFIX not in summary.content
        assert "word limit" not in llm.calls[0][0]["content"]


class TestSidecarAwareEstimate:
    """wire_messages sends api_content when present, so the estimate must
    count it — those are exactly the messages carrying extra payload."""

    def test_api_content_counted(self):
        plain = [{"role": "user", "content": "x" * 400}]
        with_recall = [
            {"role": "user", "content": "x" * 400, "api_content": "x" * 1200}
        ]
        assert compact.estimate_tokens(with_recall) == 300
        assert compact.estimate_tokens(plain) == 100

    def test_compaction_triggers_on_the_real_size(self):
        messages = [
            {"role": "user", "content": "short", "api_content": "x" * 900}
            for _ in range(30)  # past KEEP_RECENT, or there is nothing to fold
        ]
        # 30 * 900 chars = ~6750 tokens; only ~37 by content alone
        assert compact.should_compact(messages, max_model_len=2000)


class TestNativeCallsInTheSummarizerView:
    def test_a_native_call_is_not_a_blank_line(self):
        # Its content is empty by construction — the call is in tool_calls — so
        # summarizing content alone would drop the action from the summary.
        call = history_module.tool_exchange(
            "edit_file", {"registry_key": "runner"}, "ok", call_id="c1"
        )[0]
        rendered = compact.transcript([call])
        assert "edit_file" in rendered
        assert "runner" in rendered

    def test_ordinary_messages_are_unchanged(self):
        assert compact.transcript(
            [{"role": "user", "content": "align the reads"}]
        ) == "user: align the reads"
