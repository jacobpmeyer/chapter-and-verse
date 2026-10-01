"""EPUB -> metadata + ordered, cleaned chapters (Markdown).

Chapter detection is TOC-driven rather than file-driven, because EPUBs in the
wild disagree about what a "file" is:
  * one file per chapter (most retail EPUBs),
  * a handful of huge files with chapters marked only by TOC anchors
    (MOBI conversions),
  * chapters split across several files.

Pipeline:
  1. Walk the spine in reading order and cut each document into *segments* at
     every TOC anchor that points into it. A new spine file also starts a
     segment.
  2. Decide which TOC entries are chapter boundaries. When the TOC contains
     numbered chapters ("1. ...", "Chapter 3"), unnumbered entries at that
     depth or deeper are treated as in-chapter section headings. Structural
     entries (Preface, Part, Appendix...) always stay boundaries.
  3. Drop front/back matter by guide type, file name, TOC title, and content
     heuristics. Every decision is recorded with a reason for --verbose.
  4. Group segments into chapters: a boundary segment opens a chapter, and an
     untitled segment continues the current one (or opens a new one if it
     starts with its own heading). Tiny chapters, like part dividers or
     full-page illustrations, are dropped.
"""

from __future__ import annotations

import hashlib
import re
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote

import ebooklib
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
from ebooklib import epub
from markdownify import markdownify

from tokens import estimate_tokens

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="ebooklib")
warnings.filterwarnings("ignore", category=FutureWarning, module="ebooklib")

MIN_CHAPTER_TOKENS = 120  # below this a "chapter" is a divider/illustration page
MIN_UNTITLED_TOKENS = 300  # untitled material before the first chapter must be substantial

SKIP_GUIDE_TYPES = {
    "cover", "title-page", "copyright-page", "toc", "colophon", "dedication",
    "notes", "index", "other.backmatter",
}
SKIP_TITLE_RE = re.compile(
    r"^\s*(cover|title( page)?|half[- ]?title|copyright|(table of )?contents|"
    r"dedication|epigraph|maps?\b|acknowledge?ments?|about the (author|publisher)s?|"
    r"also by|other (books|titles) by|by [A-Z]|praise|newsletter|sign[- ]?up|index|"
    r"(end)?notes|bibliography|further reading|credits|colophon|excerpt|preview|"
    r"reading group|discussion questions)",
    re.IGNORECASE,
)
SKIP_FILE_RE = re.compile(
    r"^(cover.*|title.*|halftitle.*|copyright.*|.*toc|contents|nav|endpaper.*|"
    r"adcard.*|ad_?chapter.*|torad.*|.*newsletter.*|about.*|alsoby.*|praise.*|"
    r"dedication.*|maps?\d*|fb2info|colophon|index)$",
    re.IGNORECASE,
)
NUMBERED_RE = re.compile(r"^\s*(chapter\s+\w+|\d+[.:)]?\s|[IVXLC]+[.:)]\s)", re.IGNORECASE)
STRUCTURAL_RE = re.compile(
    r"^\s*(preface|foreword|introduction|prologue|epilogue|afterword|appendix|"
    r"part\b|book\b|conclusion|postscript|coda|interlude|author.?s note)",
    re.IGNORECASE,
)
PART_RE = re.compile(r"^\s*(part|book)\b", re.IGNORECASE)
COPYRIGHT_RE = re.compile(r"all rights reserved|\bISBN\b", re.IGNORECASE)
CJK_RE = re.compile(r"[぀-ヿ㐀-鿿가-힯]")


@dataclass
class BookMeta:
    title: str
    authors: list[str]
    series: str | None = None
    series_index: float | None = None
    language: str | None = None
    calibre_uuid: str | None = None
    calibre_id: int | None = None


@dataclass
class Chapter:
    index: int  # 1-based position among kept chapters
    title: str
    content: str  # Markdown
    token_count: int


@dataclass
class ExtractedBook:
    path: Path
    content_hash: str
    meta: BookMeta
    chapters: list[Chapter]
    log: list[tuple[str, str, str]] = field(default_factory=list)  # (action, label, reason)

    @property
    def token_count(self) -> int:
        return sum(c.token_count for c in self.chapters)


@dataclass
class _TocEntry:
    title: str
    href: str  # spine file name, relative to the OPF dir
    anchor: str | None
    depth: int
    boundary: bool = True


@dataclass
class _Segment:
    href: str
    html: str
    toc: _TocEntry | None  # the TOC entry this segment starts at, if any


# --------------------------------------------------------------------------- #
# Discovery / hashing / metadata
# --------------------------------------------------------------------------- #

def find_epubs(library: Path) -> list[Path]:
    return sorted(library.rglob("*.epub"))


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


_NS = {
    "opf": "http://www.idpf.org/2007/opf",
    "dc": "http://purl.org/dc/elements/1.1/",
}


def _calibre_opf_meta(epub_path: Path) -> BookMeta | None:
    """Read Calibre's metadata.opf (named '<Title> - <Author>.opf' here) beside the EPUB."""
    candidates = [epub_path.with_suffix(".opf"), epub_path.parent / "metadata.opf"]
    opf = next((p for p in candidates if p.exists()), None)
    if opf is None:
        return None
    md = ET.parse(opf).getroot().find("opf:metadata", _NS)
    if md is None:
        return None

    def text(tag: str) -> str | None:
        el = md.find(tag, _NS)
        return el.text.strip() if el is not None and el.text else None

    metas = {m.get("name"): m.get("content") for m in md.findall("opf:meta", _NS)}
    ids = {
        (el.get(f"{{{_NS['opf']}}}scheme") or "").lower(): (el.text or "").strip()
        for el in md.findall("dc:identifier", _NS)
    }
    authors = [
        el.text.strip()
        for el in md.findall("dc:creator", _NS)
        if el.text and el.get(f"{{{_NS['opf']}}}role", "aut") == "aut"
    ]
    series_index = metas.get("calibre:series_index")
    return BookMeta(
        title=text("dc:title") or epub_path.stem,
        authors=authors,
        series=metas.get("calibre:series"),
        series_index=float(series_index) if series_index else None,
        language=text("dc:language"),
        calibre_uuid=ids.get("uuid"),
        calibre_id=int(ids["calibre"]) if ids.get("calibre", "").isdigit() else None,
    )


def _epub_meta(book: epub.EpubBook, path: Path) -> BookMeta:
    def first(name: str) -> str | None:
        vals = book.get_metadata("DC", name)
        return vals[0][0] if vals else None

    return BookMeta(
        title=first("title") or path.stem,
        authors=[v for v, _ in book.get_metadata("DC", "creator")],
        language=first("language"),
    )


# --------------------------------------------------------------------------- #
# TOC handling
# --------------------------------------------------------------------------- #

def _split_href(href: str) -> tuple[str, str | None]:
    file, _, anchor = unquote(href).partition("#")
    return file, anchor or None


def _flatten_toc(toc, depth: int = 0) -> list[_TocEntry]:
    out: list[_TocEntry] = []
    for node in toc:
        if isinstance(node, tuple):
            section, children = node
            if getattr(section, "href", None):
                out.append(_TocEntry(section.title or "", *_split_href(section.href), depth))
            out.extend(_flatten_toc(children, depth + 1))
        elif getattr(node, "href", None):
            out.append(_TocEntry(node.title or "", *_split_href(node.href), depth))
    return out


def _mark_boundaries(entries: list[_TocEntry]) -> None:
    """Demote section-level TOC entries when the TOC has numbered chapters.

    At the chapter depth, an unnumbered entry is demoted to an in-chapter
    section only if it points *inside* a file (an anchor), as MOBI conversions
    do for section headings. An entry that starts its own file is a structural
    unit (a part divider, an interlude) and stays a boundary.
    """
    by_depth: dict[int, int] = {}
    for e in entries:
        if NUMBERED_RE.match(e.title):
            by_depth[e.depth] = by_depth.get(e.depth, 0) + 1
    if not by_depth or max(by_depth.values()) < 3:
        return  # no numbering convention: every TOC entry is a boundary
    chapter_depth = max(by_depth, key=by_depth.get)
    for e in entries:
        if e.depth > chapter_depth:
            e.boundary = False
        elif e.depth == chapter_depth and e.anchor is not None:
            e.boundary = bool(
                NUMBERED_RE.match(e.title) or STRUCTURAL_RE.match(e.title) or SKIP_TITLE_RE.match(e.title)
            )


# --------------------------------------------------------------------------- #
# HTML helpers
# --------------------------------------------------------------------------- #

def _clean_part_title(title: str) -> str:
    """'- May 1970 -: 26 WEEKS' -> 'May 1970: 26 WEEKS'."""
    title = re.sub(r"\s*[-–—]\s*:", ":", title)
    return re.sub(r"\s+", " ", title.strip(" -–—")).strip()


def _is_book_title(title: str, book_title: str) -> bool:
    """True if a TOC entry is just the book's own title (a title page, not a part)."""
    norm = lambda t: re.sub(r"[^a-z0-9]", "", t.lower())
    return bool(norm(book_title)) and norm(title).startswith(norm(book_title)[:15])


def _cut_positions(html: str, anchors: list[str]) -> dict[str, int]:
    """Map anchor id -> offset of the start of the tag carrying that id."""
    pos = {}
    for a in anchors:
        m = re.search(r"""\sid\s*=\s*["']%s["']""" % re.escape(a), html)
        if m:
            pos[a] = html.rfind("<", 0, m.start())
    return pos


def _soup(html: str) -> BeautifulSoup:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style", "img", "svg", "figure", "head", "title"]):
        tag.decompose()
    for sup in soup("sup"):  # footnote markers
        if re.fullmatch(r"[\s\d*†‡§]*", sup.get_text()):
            sup.decompose()
    return soup


def _to_markdown(soup: BeautifulSoup) -> str:
    root = soup.body or soup
    md = markdownify(str(root), heading_style="ATX", strip=["a", "span", "font"], bullets="-")
    md = md.replace("\r", "").replace("­", "")  # CRs and soft hyphens
    md = re.sub(r"[ \t]+\n", "\n", md)
    md = re.sub(r"\n[ \t]+", "\n", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def _starts_with_heading(soup: BeautifulSoup) -> str | None:
    root = soup.body or soup
    for el in root.find_all(True):
        if el.name in ("h1", "h2", "h3"):
            return el.get_text(" ", strip=True) or None
        if el.name == "p" and el.get_text(strip=True):
            return None
    return None


def _link_density(soup: BeautifulSoup) -> float:
    total = len(soup.get_text(strip=True))
    if not total:
        return 0.0
    linked = sum(len(a.get_text(strip=True)) for a in soup("a"))
    return linked / total


def _content_skip_reason(md: str, soup: BeautifulSoup, language: str | None) -> str | None:
    toks = estimate_tokens(md)
    if toks < 1500 and COPYRIGHT_RE.search(md):
        return "copyright text"
    if toks > 20 and _link_density(soup) > 0.5:
        return "mostly links (table of contents)"
    letters = re.findall(r"\w", md)
    if (language or "").lower().startswith("en") and letters:
        if len(CJK_RE.findall(md)) / len(letters) > 0.3:
            return "foreign-language insert (likely ad)"
    return None


# --------------------------------------------------------------------------- #
# Main entry
# --------------------------------------------------------------------------- #

def extract_book(path: Path, content_hash: str | None = None) -> ExtractedBook:
    book = epub.read_epub(str(path), {"ignore_ncx": False})
    meta = _calibre_opf_meta(path) or _epub_meta(book, path)
    log: list[tuple[str, str, str]] = []

    toc = _flatten_toc(book.toc)
    _mark_boundaries(toc)
    toc_by_file: dict[str, list[_TocEntry]] = {}
    for e in toc:
        toc_by_file.setdefault(e.href, []).append(e)

    guide_skip = {
        _split_href(g.get("href", ""))[0]
        for g in book.guide
        if (g.get("type") or "").lower() in SKIP_GUIDE_TYPES and "#" not in (g.get("href") or "")
    }

    # 1. Segment the spine.
    segments: list[_Segment] = []
    for idref, _linear in book.spine:
        item = book.get_item_with_id(idref)
        # Some EPUBs (often older or converted ones) declare chapters as text/html,
        # which ebooklib doesn't classify as documents; accept both.
        if item is None or (item.get_type() != ebooklib.ITEM_DOCUMENT
                            and item.media_type not in ("text/html", "application/xhtml+xml")):
            continue
        href = item.get_name()
        html = item.get_content().decode("utf-8", errors="replace")
        entries = toc_by_file.get(href, [])
        # A TOC entry without an anchor points at the top of the file.
        top = next((e for e in entries if e.anchor is None), None)
        anchored = [e for e in entries if e.anchor]
        cuts = _cut_positions(html, [e.anchor for e in anchored])
        points = sorted((cuts[e.anchor], e) for e in anchored if e.anchor in cuts)
        if points and points[0][0] <= 0:
            top = top or points.pop(0)[1]
        start, current = 0, top
        for offset, entry in points:
            if offset > start:
                segments.append(_Segment(href, html[start:offset], current))
            start, current = offset, entry
        segments.append(_Segment(href, html[start:], current))

    # 2-4. Filter and group into chapters.
    chapters: list[dict] = []  # {"title", "parts": [md], "label"}
    for seg in segments:
        soup = _soup(seg.html)
        md = _to_markdown(soup)
        stem = Path(seg.href).stem
        label = seg.href + (f"#{seg.toc.anchor}" if seg.toc and seg.toc.anchor else "")
        title = seg.toc.title.strip() if seg.toc else None
        if title:
            label += f' "{title}"'

        if not md:
            if seg.toc:
                log.append(("skip", label, "empty (image-only or blank)"))
            continue
        reason = None
        if seg.toc is None or seg.toc.anchor is None:  # segment covers the file start
            if seg.href in guide_skip:
                reason = "EPUB guide marks it as front/back matter"
            elif SKIP_FILE_RE.match(stem):
                reason = f"file name '{stem}'"
        if reason is None and title and SKIP_TITLE_RE.match(title) and (seg.toc.boundary or not chapters):
            reason = "TOC title looks like front/back matter"
        reason = reason or _content_skip_reason(md, soup, meta.language)
        if reason:
            log.append(("skip", label, reason))
            # A skipped boundary also closes the current chapter so trailing
            # untitled material isn't glued onto the last real chapter.
            if seg.toc and seg.toc.boundary:
                chapters.append({"title": None, "parts": [], "label": label, "closed": True})
            continue

        if seg.toc and seg.toc.boundary:
            chapters.append({"title": title, "parts": [md], "label": label})
            continue
        heading = _starts_with_heading(soup)
        if heading and estimate_tokens(md) >= MIN_CHAPTER_TOKENS:
            chapters.append({"title": heading, "parts": [md], "label": label})
        elif chapters and not chapters[-1].get("closed"):
            chapters[-1]["parts"].append(md)
            log.append(("merge", label, f'continuation of "{chapters[-1]["title"]}"'))
        elif estimate_tokens(md) >= MIN_UNTITLED_TOKENS and not any(not c.get("closed") for c in chapters):
            # Substantial text before the first chapter (a prologue or frame
            # narration without a TOC entry). Skipped front matter before it
            # doesn't count as "a chapter has started".
            chapters.append({"title": heading or "Opening (untitled)", "parts": [md], "label": label})
        else:
            log.append(("skip", label, "untitled material outside any chapter"))

    # Part dividers ("Part I", "27 WEEKS", ...) are too short to be chapters,
    # but they carry structure: their text is prepended to the next chapter and
    # their title labels every chapter until the next divider, e.g.
    # "Chapter 6 (27 WEEKS)". A divider that is just the book's title page
    # doesn't count, and back matter like an appendix ends the current part.
    kept: list[Chapter] = []
    part: str | None = None
    carry: list[str] = []
    for ch in chapters:
        if ch.get("closed"):
            continue
        content = "\n\n".join(ch["parts"])
        toks = estimate_tokens(content)
        title = ch["title"]
        is_divider = bool(title) and estimate_tokens(ch["parts"][0]) < MIN_CHAPTER_TOKENS \
            and not _is_book_title(title, meta.title)
        if toks < MIN_CHAPTER_TOKENS:
            if is_divider:
                part = _clean_part_title(title)
                carry.append(content)
                log.append(("part", ch["label"], f"divider ({toks} tokens): labels the chapters that follow"))
            else:
                log.append(("skip", ch["label"], f"too short ({toks} tokens: divider/illustration)"))
            continue
        if is_divider:
            # A divider followed by untitled text: the divider names this chapter.
            part = None
            title = _clean_part_title(title)
        elif title and STRUCTURAL_RE.match(title) and not PART_RE.match(title):
            part = None  # e.g. Appendix, Epilogue: not inside the preceding part
        title = title or f"Chapter {len(kept) + 1}"
        if part and part.lower() not in title.lower():
            title = f"{title} ({part})"
        if carry:
            content = "\n\n".join(carry + [content])
            toks = estimate_tokens(content)
            carry = []
        kept.append(Chapter(len(kept) + 1, title, content, toks))
        log.append(("keep", ch["label"], f'chapter {len(kept)} "{title}", ~{toks} tokens'))

    if not kept:
        log.append(("warn", str(path.name), "no chapters found: check the EPUB with `extract --dry-run -v`"))
    return ExtractedBook(path, content_hash or file_hash(path), meta, kept, log)
