"""extract.py: TOC-based chapter detection, front/back-matter filtering, part
labels, and metadata, on small synthetic EPUBs modeled on real ones."""

from extract import _clean_part_title, _is_book_title, extract_book
from tests.conftest import Doc, EpubSpec, paragraphs


def titles(book):
    return [c.title for c in book.chapters]


def skipped(book):
    return {label: reason for action, label, reason in book.log if action == "skip"}


# --------------------------------------------------------------------------- #
# One file per chapter, with typical retail front and back matter
# --------------------------------------------------------------------------- #

def retail_book() -> EpubSpec:
    return EpubSpec(
        title="Retail Novel",
        docs=[
            Doc("cover.xhtml", '<div class="cover"></div>'),
            Doc("title.xhtml", "<h1>Retail Novel</h1><p>Ada Author</p>"),
            Doc("copyright.xhtml", "<p>Copyright 2024. All rights reserved. ISBN 978-0-00-000000-0.</p>"),
            Doc("contents.xhtml", '<p><a href="ch1.xhtml">One</a></p><p><a href="ch2.xhtml">Two</a></p>'
                                  '<p><a href="ch3.xhtml">Three</a></p>'),
            Doc("ch1.xhtml", "<h2>The Arrival</h2>" + paragraphs(3, seed=1)),
            Doc("ch2.xhtml", "<h2>The Middle</h2>" + paragraphs(3, seed=2)),
            Doc("ch3.xhtml", "<h2>The End</h2>" + paragraphs(3, seed=3)),
            Doc("acknowledgments.xhtml", "<h2>Acknowledgments</h2>" + paragraphs(2, seed=4)),
            Doc("adcard.xhtml", "<h2>Also by Ada Author</h2><p>Another Book</p>"),
        ],
        toc=[
            ("Cover", "cover.xhtml"),
            ("Copyright", "copyright.xhtml"),
            ("The Arrival", "ch1.xhtml"),
            ("The Middle", "ch2.xhtml"),
            ("The End", "ch3.xhtml"),
            ("Acknowledgments", "acknowledgments.xhtml"),
            ("Also by Ada Author", "adcard.xhtml"),
        ],
        guide=[{"type": "cover", "href": "cover.xhtml", "title": "Cover"},
               {"type": "toc", "href": "contents.xhtml", "title": "Contents"}],
    )


def test_one_file_per_chapter_keeps_only_the_story(make_epub):
    book = extract_book(make_epub(retail_book()))
    assert titles(book) == ["The Arrival", "The Middle", "The End"]
    assert [c.index for c in book.chapters] == [1, 2, 3]


def test_front_and_back_matter_skipped_with_reasons(make_epub):
    reasons = skipped(extract_book(make_epub(retail_book())))
    assert any("copyright" in label for label in reasons)
    assert any("contents" in label for label in reasons)
    assert any("acknowledgments" in label for label in reasons)
    assert any("adcard" in label for label in reasons)


def test_chapter_text_is_markdown_with_paragraph_breaks(make_epub):
    chapter = extract_book(make_epub(retail_book())).chapters[0]
    assert chapter.content.startswith("## The Arrival")
    assert chapter.content.count("\n\n") >= 3  # heading + 3 paragraphs
    assert "<p>" not in chapter.content
    assert chapter.token_count > 0


# --------------------------------------------------------------------------- #
# MOBI-style conversion: one big file, chapters only as TOC anchors, a flat TOC
# mixing parts, numbered chapters, and section headings
# --------------------------------------------------------------------------- #

def mobi_book() -> EpubSpec:
    def anchor(a, heading, n=2, seed=0):
        return f'<span id="{a}"></span><p><b>{heading}</b></p>' + paragraphs(n, seed=seed)

    body = (
        '<p><a href="text.html#pre">Preface</a></p><p><a href="text.html#c1">1. Reading</a></p>'  # inline TOC
        + anchor("pre", "Preface", seed=1)
        + '<span id="p1"></span><p><b>PART ONE: BASICS</b></p>'
        + anchor("c1", "1. Reading", seed=2)
        + anchor("s1", "Active Reading", seed=3)
        + anchor("c2", "2. Levels", seed=4)
        + anchor("s2", "Elementary Reading", seed=5)
        + anchor("c3", "3. Inspection", seed=7)
        + anchor("app", "Appendix A. Reading List", seed=6)
        + '<span id="idx"></span><p><b>Index</b></p><p>Aristotle, 12; Bacon, 40</p>'
    )
    toc = [("Preface", "text.html#pre"), ("PART ONE: BASICS", "text.html#p1"),
           ("1. Reading", "text.html#c1"), ("Active Reading", "text.html#s1"),
           ("2. Levels", "text.html#c2"), ("Elementary Reading", "text.html#s2"),
           ("3. Inspection", "text.html#c3"),
           ("Appendix A. Reading List", "text.html#app"), ("Index", "text.html#idx")]
    return EpubSpec(title="How to Read", docs=[Doc("text.html", body)], toc=toc)


def test_mobi_style_anchors_split_into_numbered_chapters(make_epub):
    book = extract_book(make_epub(mobi_book()))
    assert titles(book) == [
        "Preface",
        "1. Reading (PART ONE: BASICS)",
        "2. Levels (PART ONE: BASICS)",
        "3. Inspection (PART ONE: BASICS)",
        "Appendix A. Reading List",  # back matter isn't inside the preceding part
    ]


def test_unnumbered_toc_entries_become_sections_inside_chapters(make_epub):
    book = extract_book(make_epub(mobi_book()))
    reading = book.chapters[1]
    assert "Active Reading" in reading.content
    assert "Elementary Reading" in book.chapters[2].content
    merges = [label for action, label, _ in book.log if action == "merge"]
    assert any("Active Reading" in m for m in merges)


def test_mobi_index_and_inline_contents_skipped(make_epub):
    book = extract_book(make_epub(mobi_book()))
    text = "\n".join(c.content for c in book.chapters)
    assert "Aristotle, 12" not in text  # the index
    assert text.count("1. Reading") == 1  # the inline contents page before the first anchor is gone
    assert any("Index" in label for label in skipped(book))


def test_numbering_needs_at_least_three_numbered_chapters(make_epub):
    # With only two numbered entries there's no clear convention, so every TOC
    # entry stays a chapter rather than guessing which are sections.
    spec = mobi_book()
    spec.toc = [e for e in spec.toc if e[0] != "3. Inspection"]
    spec.docs[0].body = spec.docs[0].body.replace('<span id="c3"></span><p><b>3. Inspection</b></p>', "")
    assert "Active Reading (PART ONE: BASICS)" in titles(extract_book(make_epub(spec)))


# --------------------------------------------------------------------------- #
# Part dividers carry structure; an untitled opening is kept (Hendrix-style)
# --------------------------------------------------------------------------- #

def weeks_book() -> EpubSpec:
    return EpubSpec(
        title="Wayward Book",
        docs=[
            Doc("epigraph.xhtml", "<p>A short poem, a line or two.</p>"),
            Doc("contents.xhtml", '<p><a href="part1.xhtml">26 WEEKS</a></p>'),
            Doc("opening.xhtml", paragraphs(3, seed=9)),  # untitled frame narration, not in the TOC
            Doc("part1.xhtml", "<h1>- May 1970 -</h1><h2>26 WEEKS</h2>"),
            Doc("chapter1.xhtml", "<h2>Chapter 1</h2>" + paragraphs(3, seed=1)),
            Doc("chapter2.xhtml", "<h2>Chapter 2</h2>" + paragraphs(3, seed=2)),
            Doc("part2.xhtml", "<h2>27 WEEKS</h2><p>Baby now weighs over 2 lbs!</p>"),
            Doc("chapter3.xhtml", "<h2>Chapter 3</h2>" + paragraphs(3, seed=3)),
        ],
        toc=[("Epigraph", "epigraph.xhtml"), ("Contents", "contents.xhtml"),
             ("- May 1970 -: 26 WEEKS", "part1.xhtml"), ("Chapter 1", "chapter1.xhtml"),
             ("Chapter 2", "chapter2.xhtml"), ("27 WEEKS", "part2.xhtml"), ("Chapter 3", "chapter3.xhtml")],
        guide=[{"type": "toc", "href": "contents.xhtml", "title": "Contents"}],
    )


def test_part_labels_with_a_nested_toc(make_epub):
    spec = weeks_book()
    spec.toc = [("Epigraph", "epigraph.xhtml"), ("Contents", "contents.xhtml"),
                (("- May 1970 -: 26 WEEKS", "part1.xhtml"),
                 [("Chapter 1", "chapter1.xhtml"), ("Chapter 2", "chapter2.xhtml")]),
                (("27 WEEKS", "part2.xhtml"), [("Chapter 3", "chapter3.xhtml")])]
    assert titles(extract_book(make_epub(spec)))[1:] == ["Chapter 1 (May 1970: 26 WEEKS)",
                                                          "Chapter 2 (May 1970: 26 WEEKS)",
                                                          "Chapter 3 (27 WEEKS)"]


def test_untitled_opening_after_skipped_front_matter_is_kept(make_epub):
    # Regression: skipped Epigraph/Contents entries used to count as "a chapter
    # has started", so the opening frame was dropped.
    book = extract_book(make_epub(weeks_book()))
    assert book.chapters[0].title == "Opening (untitled)"


def test_part_dividers_label_following_chapters(make_epub):
    # Flat TOC: the dividers sit at the same depth as "Chapter N". Regression:
    # a divider that isn't titled "Part ..." used to be demoted to a section and
    # glued onto the end of the previous chapter.
    book = extract_book(make_epub(weeks_book()))
    assert titles(book)[1:] == ["Chapter 1 (May 1970: 26 WEEKS)",
                                "Chapter 2 (May 1970: 26 WEEKS)",
                                "Chapter 3 (27 WEEKS)"]


def test_divider_text_is_prepended_to_the_next_chapter(make_epub):
    book = extract_book(make_epub(weeks_book()))
    chapter3 = book.chapters[-1]
    assert "Baby now weighs over 2 lbs" in chapter3.content
    assert chapter3.content.index("Baby now weighs") < chapter3.content.index("Chapter 3")
    assert "Baby now weighs" not in book.chapters[2].content


# --------------------------------------------------------------------------- #
# A TOC whose top entry is the book's own title page (Morrie-style)
# --------------------------------------------------------------------------- #

def test_book_title_page_is_not_treated_as_a_part(make_epub):
    spec = EpubSpec(
        title="Tuesdays Book",
        docs=[Doc("section1.xhtml", "<p>Tuesdays Book: an old man and a lesson, by Ada Author</p>"),
              Doc("section2.xhtml", "<h2>The Curriculum</h2>" + paragraphs(3, seed=1)),
              Doc("section3.xhtml", "<h2>The Syllabus</h2>" + paragraphs(3, seed=2))],
        toc=[(("Tuesdays Book: an old man and a lesson by Ada Author", "section1.xhtml"),
              [("The Curriculum", "section2.xhtml"), ("The Syllabus", "section3.xhtml")])],
    )
    book = extract_book(make_epub(spec))
    assert titles(book) == ["The Curriculum", "The Syllabus"]
    assert not book.chapters[0].content.startswith("Tuesdays Book")


def test_chapters_declared_as_text_html_are_read(make_epub):
    # Regression: spine items with media type text/html (common in older or
    # converted EPUBs) were silently ignored, producing a book with no chapters.
    # mobi_book's single file is named .html, so ebooklib declares it text/html.
    assert len(extract_book(make_epub(mobi_book())).chapters) == 5  # Preface, 1-3, Appendix A


def test_a_book_with_no_chapters_is_flagged(make_epub):
    spec = EpubSpec(title="Empty", docs=[Doc("cover.xhtml", '<div class="cover"></div>')],
                    toc=[("Cover", "cover.xhtml")])
    book = extract_book(make_epub(spec))
    assert book.chapters == []
    assert any(action == "warn" for action, _, _ in book.log)


# --------------------------------------------------------------------------- #
# Inserts and metadata
# --------------------------------------------------------------------------- #

def test_foreign_language_ad_insert_is_skipped(make_epub):
    spec = retail_book()
    spec.docs.insert(5, Doc("insert.xhtml", "<h1>读累了记得休息一会哦</h1><p>电子书搜索下载 书单分享 书友学习交流 "
                                            "电子书打包资源分享 学习资源分享</p>" * 3))
    book = extract_book(make_epub(spec))
    assert titles(book) == ["The Arrival", "The Middle", "The End"]
    assert any("foreign-language" in r for r in skipped(book).values())


def test_metadata_comes_from_the_calibre_opf(make_epub):
    spec = retail_book()
    spec.opf = {"title": "Retail Novel", "authors": ["Ada Author", "Bo Coauthor"], "series": "The Saga",
                "series_index": "2", "uuid": "abc-123", "calibre_id": 42, "language": "eng"}
    meta = extract_book(make_epub(spec)).meta
    assert meta.authors == ["Ada Author", "Bo Coauthor"]
    assert (meta.series, meta.series_index) == ("The Saga", 2.0)
    assert (meta.calibre_uuid, meta.calibre_id, meta.language) == ("abc-123", 42, "eng")


def test_metadata_falls_back_to_the_epub_without_an_opf(make_epub):
    meta = extract_book(make_epub(retail_book())).meta
    assert meta.title == "Retail Novel"
    assert meta.authors == ["Ada Author"]
    assert meta.calibre_uuid is None


def test_content_hash_is_stable_and_content_sensitive(make_epub, tmp_path):
    path = make_epub(retail_book())
    assert extract_book(path).content_hash == extract_book(path).content_hash
    other = retail_book()
    other.title = "Retail Novel Two"
    assert extract_book(make_epub(other)).content_hash != extract_book(path).content_hash


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def test_clean_part_title():
    assert _clean_part_title("- May 1970 -: 26 WEEKS") == "May 1970: 26 WEEKS"
    assert _clean_part_title("  Part I. Worst Princess Ever ") == "Part I. Worst Princess Ever"
    assert _clean_part_title("— 27 WEEKS —") == "27 WEEKS"


def test_is_book_title():
    title = "Tuesdays With Morrie: An Old Man, a Young Man and Life's Greatest Lesson"
    assert _is_book_title("Tuesdays with Morrie:  an old man, a young man... by Mitch Albom", title)
    assert not _is_book_title("Part I. Worst Princess Ever", "The Devils")
    assert not _is_book_title("27 WEEKS", "")
