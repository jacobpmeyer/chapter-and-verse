"""summarize.py: length rules, the condense loop, and cost-estimate calibration.
The model is never called: call_model and the database are replaced with fakes."""

import re

import pytest

import summarize as S


def section_ranges(instructions: str) -> list[tuple[int, int]]:
    return [(int(a), int(b)) for a, b in re.findall(r"\((\d+)-(\d+) words\)", instructions)]


# --------------------------------------------------------------------------- #
# Length rules
# --------------------------------------------------------------------------- #

def test_word_count_ignores_markdown_symbols():
    md = "### Summary\n- **Fern:** signs as \"Jane Doe\" — twice.\n\n### Tone\nWry."
    assert S.word_count(md) == 9  # Summary Fern signs as Jane Doe twice Tone Wry


def test_chapter_section_ranges_fit_the_target():
    ranges = section_ranges(S.CHAPTER_INSTRUCTIONS)
    assert len(ranges) == 5
    lo, hi = S.CHAPTER_WORDS
    assert sum(a for a, _ in ranges) >= lo  # minimums pull it into the target range
    assert sum(b for _, b in ranges) <= hi  # maximums never add up past the hard cap


@pytest.mark.parametrize("tokens,expected", [
    (66_000, (600, 1000)), (150_000, (600, 1000)), (150_001, (900, 1500)),
    (258_640, (900, 1500)), (431_850, (1200, 2000)),
])
def test_book_summary_length_scales_with_the_book(tokens, expected):
    assert S.book_words(tokens)[0] == expected


@pytest.mark.parametrize("tokens", [66_000, 258_640, 431_850])
def test_book_section_ranges_fit_each_tier(tokens):
    (lo, hi), _ = S.book_words(tokens)
    ranges = section_ranges(S.book_instructions("the book above", tokens))
    assert len(ranges) == 6
    assert sum(a for a, _ in ranges) >= lo
    assert sum(b for _, b in ranges) <= hi
    assert f"{lo}-{hi} words" in S.book_instructions("the book above", tokens)


# --------------------------------------------------------------------------- #
# The condense loop
# --------------------------------------------------------------------------- #

class FakeModel:
    """Stands in for call_model: returns scripted replies and records prompts."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def __call__(self, client, prompt, max_tokens, ctx, purpose):
        self.prompts.append((purpose, prompt))
        return self.replies.pop(0), "fake-model"


def words(n):
    return " ".join(["word"] * n)


@pytest.fixture
def ctx():
    return S.CallContext(S.Usage(), conn=None, book_id=1)


def test_text_within_the_limit_is_returned_without_a_call(monkeypatch, ctx):
    fake = FakeModel([])
    monkeypatch.setattr(S, "call_model", fake)
    assert S.enforce_length(None, words(400), 450, ctx, "chapter 1") == words(400)
    assert fake.prompts == []


def test_long_text_is_condensed_once(monkeypatch, ctx):
    fake = FakeModel([words(380)])
    monkeypatch.setattr(S, "call_model", fake)
    assert S.word_count(S.enforce_length(None, words(520), 450, ctx, "chapter 1")) == 380
    assert [p for p, _ in fake.prompts] == ["condense"]
    assert "520 words" in fake.prompts[0][1] and "at most 405 words" in fake.prompts[0][1]


def test_condense_stops_after_two_tries_and_keeps_the_shortest(monkeypatch, ctx, capsys):
    fake = FakeModel([words(500), words(470)])
    monkeypatch.setattr(S, "call_model", fake)
    result = S.enforce_length(None, words(520), 450, ctx, "chapter 7")
    assert S.word_count(result) == 470
    assert len(fake.prompts) == S.MAX_CONDENSE_ATTEMPTS
    assert "chapter 7 is still 470 words" in capsys.readouterr().out


def test_a_condense_that_gets_longer_is_ignored(monkeypatch, ctx):
    fake = FakeModel([words(600), words(440)])
    monkeypatch.setattr(S, "call_model", fake)
    assert S.word_count(S.enforce_length(None, words(520), 450, ctx, "x")) == 440
    assert "520 words" in fake.prompts[1][1]  # second try condenses the original, not the longer one


# --------------------------------------------------------------------------- #
# Cost estimate calibration
# --------------------------------------------------------------------------- #

def stats(chapter_n=0, chapter_out=0.0, condense_n=0, condense_in=0.0, condense_out=0.0, book_n=0):
    out = {}
    if chapter_n:
        out["chapter_summary"] = {"n": chapter_n, "avg_in": 4000, "avg_out": chapter_out}
    if condense_n:
        out["condense"] = {"n": condense_n, "avg_in": condense_in, "avg_out": condense_out}
    if book_n:
        out["book_summary"] = {"n": book_n, "avg_in": 1, "avg_out": 1}
    return out


def test_estimate_uses_defaults_without_enough_history(monkeypatch):
    monkeypatch.setattr(S.db, "api_call_stats", lambda *a, **k: stats(chapter_n=S.MIN_HISTORY_CALLS - 1,
                                                                       chapter_out=5000))
    costs = S.CallCosts.load(conn=None)
    assert costs.source == "defaults"
    assert costs.chapter_output != 5000
    assert costs.condense_rate == S.DEFAULT_CONDENSE_RATE


def test_estimate_switches_to_logged_averages(monkeypatch):
    monkeypatch.setattr(S.db, "api_call_stats", lambda *a, **k: stats(
        chapter_n=38, chapter_out=915.6, condense_n=2, condense_in=1200, condense_out=700, book_n=2))
    costs = S.CallCosts.load(conn=None)
    assert costs.source.startswith("history (38")
    assert costs.chapter_output == 915.6
    assert costs.condense_rate == pytest.approx(2 / 40)
    assert (costs.condense_input, costs.condense_output) == (1200, 700)


def test_no_condense_history_means_a_zero_condense_rate(monkeypatch):
    monkeypatch.setattr(S.db, "api_call_stats", lambda *a, **k: stats(chapter_n=10, chapter_out=900))
    assert S.CallCosts.load(conn=None).condense_rate == 0


def test_usage_cost_uses_model_prices():
    u = S.Usage()
    u.add(1_000_000, 100_000)
    assert u.cost("claude-sonnet-5-5") == pytest.approx(2.00 + 1.00)
    assert u.calls == 1
