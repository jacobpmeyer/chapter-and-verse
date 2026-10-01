"""chunk.py: paragraph-safe packing, overlap, and tail handling."""

import random

import pytest

from chunk import chunk_chapter, split_paragraphs
from tokens import estimate_tokens

SIZES = dict(max_tokens=800, min_tokens=500, overlap_tokens=80, overlap_cap=150, min_tail_tokens=150)


def para(tokens: int, tag: str) -> str:
    """A paragraph of roughly `tokens` estimated tokens, starting with a unique tag."""
    text = f"{tag} " + "word " * 10_000
    return text[: round(tokens * 3.8)].strip()


def chapter(sizes: list[int]) -> tuple[str, list[str]]:
    paras = [para(t, f"P{i:03d}") for i, t in enumerate(sizes)]
    return "\n\n".join(paras), paras


def chunk_paras(passage) -> list[str]:
    return split_paragraphs(passage.content)


def test_split_paragraphs_ignores_blank_runs_and_whitespace():
    assert split_paragraphs("one\n\n\n\ntwo\n  \nthree\n") == ["one", "two", "three"]
    assert split_paragraphs("   ") == []


def test_empty_chapter_has_no_chunks():
    assert chunk_chapter("", **SIZES) == []


def test_short_chapter_is_a_single_chunk():
    text, _ = chapter([200, 200])
    chunks = chunk_chapter(text, **SIZES)
    assert len(chunks) == 1
    assert chunks[0].content == text


@pytest.mark.parametrize("seed", range(25))
def test_paragraphs_are_never_split_or_lost(seed):
    rng = random.Random(seed)
    text, paras = chapter([rng.randint(20, 400) for _ in range(rng.randint(5, 40))])
    chunks = chunk_chapter(text, **SIZES)
    seen = []
    for c in chunks:
        for p in chunk_paras(c):
            assert p in paras, "a chunk contains a fragment that isn't a whole paragraph"
            if p not in seen:
                seen.append(p)
    assert seen == paras, "every paragraph appears, in order"


@pytest.mark.parametrize("seed", range(25))
def test_chunks_respect_the_size_limits(seed):
    rng = random.Random(seed)
    text, _ = chapter([rng.randint(20, 300) for _ in range(rng.randint(10, 40))])
    chunks = chunk_chapter(text, **SIZES)
    for c in chunks[:-1]:
        # Over the max only by one paragraph, and only when the chunk was still under the min.
        if c.token_count > SIZES["max_tokens"]:
            without_last = estimate_tokens("\n\n".join(chunk_paras(c)[:-1]))
            assert without_last < SIZES["min_tokens"]
    # The last chunk may also absorb a small tail (< min_tail_tokens of new material)
    # rather than leave a runt chunk behind.
    assert chunks[-1].token_count <= SIZES["max_tokens"] + SIZES["min_tokens"] + SIZES["min_tail_tokens"]


@pytest.mark.parametrize("seed", range(25))
def test_overlap_is_whole_trailing_paragraphs_within_the_cap(seed):
    rng = random.Random(seed)
    text, _ = chapter([rng.randint(20, 300) for _ in range(rng.randint(10, 40))])
    chunks = chunk_chapter(text, **SIZES)
    for prev, nxt in zip(chunks, chunks[1:]):
        a, b = chunk_paras(prev), chunk_paras(nxt)
        n = next((k for k in range(len(a), 0, -1) if a[-k:] == b[:k]), 0)
        overlap = b[:n]
        assert n < len(a), "overlap never repeats the whole previous chunk"
        assert estimate_tokens("\n\n".join(overlap)) <= SIZES["overlap_cap"] + 2 * len(overlap)


def test_overlap_is_skipped_when_the_last_paragraph_exceeds_the_cap():
    text, paras = chapter([300, 300, 300, 300])  # each paragraph > 150-token cap
    chunks = chunk_chapter(text, **SIZES)
    assert len(chunks) == 2
    assert chunk_paras(chunks[1])[0] == paras[2]  # no repeated paragraph


def test_an_oversized_paragraph_becomes_its_own_chunk():
    text, paras = chapter([300, 300, 1500, 300, 300])
    chunks = chunk_chapter(text, **SIZES)
    giant = [c for c in chunks if paras[2] in chunk_paras(c)]
    assert len(giant) == 1
    assert chunk_paras(giant[0]) == [paras[2]] or chunk_paras(giant[0])[-1] == paras[2]


def test_a_small_tail_is_merged_into_the_previous_chunk():
    text, paras = chapter([400, 350, 60])  # 750 fits, then a 60-token runt
    chunks = chunk_chapter(text, **SIZES)
    assert len(chunks) == 1
    assert chunk_paras(chunks[0])[-1] == paras[2]


def test_chunk_indexes_are_sequential():
    text, _ = chapter([300] * 12)
    chunks = chunk_chapter(text, **SIZES)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
