"""Chapter -> passage chunks.

Rules:
  * Chunks never cross chapter boundaries, so every passage has a clean chapter
    for citations.
  * Paragraphs (blocks separated by a blank line) are packed greedily up to
    max_tokens. A paragraph is never split. One paragraph larger than
    max_tokens becomes its own oversized chunk. A chunk still under min_tokens
    may overshoot max_tokens by one paragraph instead of closing early.
  * Overlap: each new chunk starts with the trailing whole paragraphs of the
    previous chunk, up to ~overlap_tokens (never more than overlap_cap).
  * A small tail (under min_tail_tokens of new material) is folded into the
    previous chunk instead of producing a runt.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from config import settings
from tokens import estimate_tokens


@dataclass
class Passage:
    chunk_index: int  # position within the chapter
    content: str
    token_count: int


def split_paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]


def chunk_chapter(
    text: str,
    max_tokens: int = settings.chunk_max_tokens,
    min_tokens: int = settings.chunk_min_tokens,
    overlap_tokens: int = settings.chunk_overlap_tokens,
    overlap_cap: int = settings.chunk_overlap_cap,
    min_tail_tokens: int = settings.chunk_min_tail_tokens,
) -> list[Passage]:
    paras = [(p, estimate_tokens(p)) for p in split_paragraphs(text)]
    if not paras:
        return []

    chunks: list[list[tuple[str, int]]] = []
    current: list[tuple[str, int]] = []
    current_toks = 0
    new_in_current = 0  # paragraphs in `current` that aren't overlap

    def overlap_from(chunk: list[tuple[str, int]]) -> list[tuple[str, int]]:
        tail: list[tuple[str, int]] = []
        toks = 0
        for p, t in reversed(chunk):
            if toks >= overlap_tokens or toks + t > overlap_cap:
                break
            tail.insert(0, (p, t))
            toks += t
        # Never let the overlap be the whole previous chunk.
        return tail if len(tail) < len(chunk) else []

    for p, t in paras:
        # Close the chunk if adding p would overflow, unless it's still below
        # min_tokens (then allow a modest overshoot rather than a runt chunk).
        # An oversized paragraph always gets a chunk of its own.
        overflow = current_toks + t > max_tokens
        if current and new_in_current and overflow and (current_toks >= min_tokens or t > max_tokens):
            chunks.append(current)
            current = overlap_from(current)
            current_toks = sum(x[1] for x in current)
            new_in_current = 0
        current.append((p, t))
        current_toks += t
        new_in_current += 1

    if current and new_in_current:
        new_toks = sum(t for _, t in current[len(current) - new_in_current:])
        if chunks and new_toks < min_tail_tokens:
            chunks[-1].extend(current[len(current) - new_in_current:])
        else:
            chunks.append(current)

    passages = []
    for i, chunk in enumerate(chunks):
        content = "\n\n".join(p for p, _ in chunk)
        passages.append(Passage(i, content, estimate_tokens(content)))
    return passages
