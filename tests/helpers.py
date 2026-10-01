"""Helpers for database tests: books built in memory, handmade vectors, and a
fake embedder that needs no API."""

from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path

from config import settings
from extract import BookMeta, Chapter, ExtractedBook
from tests.conftest import prose

DIM = settings.embed_dim


def make_book(title="Wayward Book", chapters=3, path=None, uuid=None, content_hash="hash-1",
              authors=("Ada Author",), series=None, paragraphs_per_chapter=8,
              texts: list[str] | None = None) -> ExtractedBook:
    """An ExtractedBook with `chapters` chapters of filler prose (several passages each),
    or with the given chapter `texts`."""
    chs = []
    for i in range(1, (len(texts) if texts else chapters) + 1):
        text = texts[i - 1] if texts else \
            "\n\n".join(prose(120, seed=i * 10 + j) for j in range(paragraphs_per_chapter))
        chs.append(Chapter(i, f"Chapter {i}", f"## Chapter {i}\n\n{text}", round(len(text) / 3.8)))
    meta = BookMeta(title=title, authors=list(authors), series=series, series_index=1.0 if series else None,
                    language="en", calibre_uuid=uuid)
    return ExtractedBook(Path(path or f"/library/{authors[0]}/{title}/{title}.epub"), content_hash, meta, chs)


def axis(i: int, dim: int = DIM) -> list[float]:
    """A unit vector along axis i."""
    v = [0.0] * dim
    v[i] = 1.0
    return v


def blend(weights: dict[int, float], dim: int = DIM) -> list[float]:
    """A unit vector mixing axes, e.g. {0: 0.9, 1: 0.1}."""
    v = [0.0] * dim
    for i, w in weights.items():
        v[i] = w
    norm = math.sqrt(sum(x * x for x in v))
    return [x / norm for x in v]


class FakeEmbedder:
    """Bag-of-words hashed into DIM buckets, normalized. Deterministic, and texts
    that share words get similar vectors, which is enough to test search."""

    model_name = "fake-embed"
    dim = DIM

    def __init__(self):
        self.tokens_used = 0
        self.queries: list[str] = []

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in re.findall(r"[a-z]+", text.lower()):
            v[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        self.tokens_used += len(text.split())
        return [x / norm for x in v]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vec(text)
