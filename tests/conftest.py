"""Shared fixtures: a builder for small synthetic EPUBs, and filler text.

The EPUBs reproduce structures found in real books (a file per chapter, MOBI
conversions with chapters only as TOC anchors, part dividers, untitled
openings), so extraction can be tested without shipping copyrighted books.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import pytest
from ebooklib import epub

WORDS = ("the river ran past the house where the girls kept their secrets and the clocks counted "
         "down the weeks until something had to change in the quiet rooms upstairs").split()


def prose(words: int = 120, seed: int = 0) -> str:
    """Deterministic filler prose of about `words` words, as one paragraph."""
    out = [WORDS[(seed * 7 + i) % len(WORDS)] for i in range(words)]
    out[0] = out[0].capitalize()
    return " ".join(out) + "."


def paragraphs(n: int, words: int = 120, seed: int = 0) -> str:
    return "".join(f"<p>{prose(words, seed + i)}</p>" for i in range(n))


@dataclass
class Doc:
    file_name: str
    body: str  # inner HTML of <body>


@dataclass
class EpubSpec:
    title: str = "Test Book"
    authors: list[str] = field(default_factory=lambda: ["Ada Author"])
    language: str = "en"
    docs: list[Doc] = field(default_factory=list)
    # TOC entries: (title, href) or ((title, href), [children...])
    toc: list = field(default_factory=list)
    guide: list[dict] = field(default_factory=list)  # {"type": ..., "href": ...}
    # Calibre metadata written to <title>.opf next to the EPUB; None = no .opf
    opf: dict | None = None


def _toc_nodes(entries):
    nodes = []
    for i, entry in enumerate(entries):
        if isinstance(entry[0], tuple):  # ((title, href), children)
            (title, href), children = entry
            nodes.append((epub.Section(title, href=href), _toc_nodes(children)))
        else:
            title, href = entry
            nodes.append(epub.Link(href, title, f"toc{id(entry)}{i}"))
    return nodes


def _write_opf(path: Path, spec: EpubSpec) -> None:
    meta = spec.opf or {}
    creators = "".join(
        f'<dc:creator opf:role="aut">{a}</dc:creator>' for a in meta.get("authors", spec.authors)
    )
    extra = ""
    if "series" in meta:
        extra += f'<meta name="calibre:series" content="{meta["series"]}"/>'
    if "series_index" in meta:
        extra += f'<meta name="calibre:series_index" content="{meta["series_index"]}"/>'
    path.write_text(f"""<?xml version='1.0' encoding='utf-8'?>
<package xmlns="http://www.idpf.org/2007/opf" unique-identifier="uuid_id" version="2.0">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/" xmlns:opf="http://www.idpf.org/2007/opf">
    <dc:identifier opf:scheme="calibre" id="calibre_id">{meta.get("calibre_id", 1)}</dc:identifier>
    <dc:identifier opf:scheme="uuid" id="uuid_id">{meta.get("uuid", "00000000-0000-0000-0000-000000000001")}</dc:identifier>
    <dc:title>{meta.get("title", spec.title)}</dc:title>
    {creators}
    <dc:language>{meta.get("language", spec.language)}</dc:language>
    {extra}
  </metadata>
</package>""")


def build_epub(spec: EpubSpec, root: Path) -> Path:
    """Write an EPUB (and optional Calibre .opf) under root; returns the .epub path."""
    book = epub.EpubBook()
    book.set_identifier("test-id")
    book.set_title(spec.title)
    book.set_language(spec.language)
    for a in spec.authors:
        book.add_author(a)
    items = []
    for doc in spec.docs:
        item = epub.EpubHtml(title=doc.file_name, file_name=doc.file_name, lang=spec.language)
        item.content = f"<html><head><title>{doc.file_name}</title></head><body>{doc.body}</body></html>"
        book.add_item(item)
        items.append(item)
    book.toc = _toc_nodes(spec.toc)
    book.spine = items
    book.guide = list(spec.guide)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    folder = root / spec.authors[0] / spec.title
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{spec.title} - {spec.authors[0]}.epub"
    epub.write_epub(str(path), book)
    if spec.opf is not None:
        _write_opf(path.with_suffix(".opf"), spec)
    return path


@pytest.fixture
def make_epub(tmp_path):
    """build_epub bound to a per-test temporary directory."""
    return lambda spec: build_epub(spec, tmp_path)


# --------------------------------------------------------------------------- #
# Database fixtures (tests marked `db`)
# --------------------------------------------------------------------------- #
# A separate database, chapter_and_verse_test, is created on the same server
# for each test session and dropped afterwards. Every table is emptied before
# each test. Set TEST_DATABASE_URL to point elsewhere (CI does); otherwise it's
# derived from DATABASE_URL with the database name swapped.

TEST_DB_NAME = "chapter_and_verse_test"


def _test_db_url() -> str:
    explicit = os.getenv("TEST_DATABASE_URL")
    if explicit:
        return explicit
    from config import settings

    if not settings.database_url:
        pytest.skip("no DATABASE_URL or TEST_DATABASE_URL; database tests skipped")
    return urlunparse(urlparse(settings.database_url)._replace(path=f"/{TEST_DB_NAME}"))


@pytest.fixture(scope="session")
def test_db_url():
    import psycopg

    url = _test_db_url()
    name = urlparse(url).path.lstrip("/")
    if not name.endswith("_test"):  # never create, empty or drop a real database
        pytest.fail(f"refusing to use {name!r} for tests: the database name must end in _test")
    admin_url = urlunparse(urlparse(url)._replace(path="/postgres"))
    try:
        admin = psycopg.connect(admin_url, autocommit=True, connect_timeout=3)
    except psycopg.OperationalError as e:
        message = f"Postgres unavailable ({e.__class__.__name__}); run `docker compose up -d`"
        if os.getenv("REQUIRE_DB"):  # CI: a missing database must fail, not quietly skip
            pytest.fail(message)
        pytest.skip(message)
    admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    admin.execute(f'CREATE DATABASE "{name}"')
    yield url
    admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
    admin.close()


@pytest.fixture(scope="session")
def _schema(test_db_url):
    import db

    conn = db.connect(test_db_url)
    db.ensure_schema(conn)
    conn.close()


@pytest.fixture
def conn(test_db_url, _schema, monkeypatch):
    """A connection to an empty test database. Code that opens its own
    connections (db.connect()) is pointed at the test database too."""
    import db

    real_connect = db.connect
    monkeypatch.setattr(db, "connect", lambda url=None: real_connect(url or test_db_url))
    c = real_connect(test_db_url)
    c.execute("TRUNCATE books, chapters, chunks, api_calls RESTART IDENTITY CASCADE")
    yield c
    c.close()
