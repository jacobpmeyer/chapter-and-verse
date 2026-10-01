"""Indexing CLI.

    python index.py extract [--limit N] [--dry-run] [-v] [--dump DIR] [--force]   # whole library, free
    python index.py status                                              # stage + cost to finish, per book
    python index.py book <id | "title words"> [--dry-run] [--yes] [--redo-summaries]
    python index.py summaries <id | "title words"> [--chapter N | --book-only] [--out FILE]   # read them
    python index.py summarize (--book-id ID ... | --all) [--dry-run] [--yes]
    python index.py embed (--book-id ID ... | --all) [--dry-run] [--yes]
    python index.py search "query" [--book-id ID] [--level passage|chapter_summary|book_summary] [-k 8]
    python index.py backup [--label NAME] [--keep N] [--list]   # dated pg_dump to ~/Backups/chapter-and-verse
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from chunk import chunk_chapter
from config import settings
from extract import ExtractedBook, extract_book, file_hash, find_epubs


def _print_book(book: ExtractedBook, verbose: bool) -> tuple[int, int]:
    m = book.meta
    series = f" [{m.series} #{m.series_index:g}]" if m.series and m.series_index is not None else ""
    print(f"\n{m.title} — {', '.join(m.authors)}{series} ({m.language or '?'})")
    if verbose:
        for action, label, reason in book.log:
            print(f"    {action:5} {label}: {reason}")
    n_chunks = 0
    sizes: list[int] = []
    for ch in book.chapters:
        passages = chunk_chapter(ch.content)
        n_chunks += len(passages)
        sizes += [p.token_count for p in passages]
        if verbose:
            print(f"  {ch.index:3}. {ch.title[:60]:60} ~{ch.token_count:6} tok  {len(passages):3} chunks")
    size_note = f", chunk tokens min/median/max {min(sizes)}/{sorted(sizes)[len(sizes) // 2]}/{max(sizes)}" if sizes else ""
    print(f"  {len(book.chapters)} chapters, {n_chunks} passage chunks, ~{book.token_count:,} tokens{size_note}")
    if not book.chapters:
        print("  WARNING: no chapters were found in this EPUB. Run `python index.py extract --dry-run -v` "
              "to see why, before indexing it.")
    return n_chunks, book.token_count


def _dump(book: ExtractedBook, out: Path) -> None:
    d = out / re.sub(r"[^\w.-]+", "_", book.meta.title)[:80]
    d.mkdir(parents=True, exist_ok=True)
    for ch in book.chapters:
        (d / f"{ch.index:03d}.md").write_text(f"# {ch.title}\n\n{ch.content}\n")
        chunks = chunk_chapter(ch.content)
        (d / f"{ch.index:03d}.chunks.md").write_text(
            "\n\n".join(f"<<< chunk {p.chunk_index} (~{p.token_count} tok) >>>\n{p.content}" for p in chunks)
        )


def _extract_one(conn, path: Path, digest: str, verbose: bool = False, dump: str | None = None) -> ExtractedBook | None:
    """Extract one EPUB, print its summary, and save it (unless conn is None)."""
    try:
        book = extract_book(path, digest)
    except Exception as e:  # one bad EPUB shouldn't stop the run
        print(f"\n{path.name}: FAILED to extract: {e}")
        return None
    _print_book(book, verbose)
    if dump:
        _dump(book, Path(dump))
    if conn is not None:
        import db

        book_id = db.save_extracted_book(conn, book)
        print(f"  saved as book_id={book_id}")
    return book


def cmd_extract(args: argparse.Namespace) -> None:
    paths = find_epubs(settings.library_path)
    if args.limit:
        paths = paths[: args.limit]
    if not paths:
        sys.exit(f"No EPUBs found under {settings.library_path}")

    conn = None
    if not args.dry_run:
        import db

        conn = db.connect()
        db.ensure_schema(conn)

    totals = {"books": 0, "skipped": 0, "chapters": 0, "chunks": 0, "tokens": 0}
    for path in paths:
        digest = file_hash(path)
        if conn is not None:
            unchanged = db.stored_hash(conn, str(path)) == digest
            if unchanged and not args.force:
                print(f"\n{path.name}: unchanged, skipped")
                totals["skipped"] += 1
                continue
            if unchanged and db.has_paid_work(conn, str(path)):
                # Re-extracting would delete summaries/embeddings that cost money.
                print(f"\n{path.name}: unchanged and already summarized/embedded; --force skips it "
                      "(use `python index.py book <id> --redo-summaries` to rebuild deliberately)")
                totals["skipped"] += 1
                continue
        book = _extract_one(conn, path, digest, args.verbose, args.dump)
        if book is None:
            continue
        totals["books"] += 1
        totals["chapters"] += len(book.chapters)
        totals["chunks"] += sum(len(chunk_chapter(c.content)) for c in book.chapters)
        totals["tokens"] += book.token_count

    print(
        f"\n{'DRY RUN — nothing written. ' if args.dry_run else ''}"
        f"Processed {totals['books']} book(s), skipped {totals['skipped']} unchanged: "
        f"{totals['chapters']} chapters, {totals['chunks']} passage chunks, ~{totals['tokens']:,} tokens."
    )


def _remaining_costs(conn, client) -> dict[int, tuple[float, float]]:
    """book_id -> (summary cost, embedding cost) still to spend."""
    import db
    import embed
    import summarize

    plans = {p.book["id"]: p for p in summarize.plan(conn, client, None)}  # count_tokens is free
    out = {}
    for r in db.book_stages(conn):
        p = plans.get(r["id"])
        future = (len(p.pending_chapters) + p.needs_book_summary) if p else 0
        _n, _tokens, embed_cost = embed.estimate(conn, [r["id"]], future)
        out[r["id"]] = (p.est_cost if p else 0.0, embed_cost)
    return out


def _print_status(conn, client) -> None:
    import db

    costs = _remaining_costs(conn, client)  # also stores exact token counts, so fetch rows after
    rows = db.book_stages(conn)
    print(f"\n{'id':>3}  {'stage':12} {'tokens':>9}  {'summaries':>9}  {'to finish':>9}  title")
    total = 0.0
    for r in rows:
        tokens = f"{r['exact_token_count']:,}" if r["exact_token_count"] else f"~{r['token_count']:,}"
        progress = f"{r['chapter_summaries']}/{r['chapter_count']}"
        cost = sum(costs[r["id"]])
        total += cost
        print(f"{r['id']:>3}  {r['stage']:12} {tokens:>9}  {progress:>9}  {'$%.2f' % cost:>9}  "
              f"{r['title'][:60]} — {', '.join(r['authors'])}")
    known = {r["source_path"] for r in rows}
    new = [p for p in find_epubs(settings.library_path) if str(p) not in known]
    for path in new:
        print(f"{'-':>3}  {'not extracted':12} {'':>9}  {'':>9}  {'':>9}  {path.name}")
    print(f"\nCost to fully index everything remaining ≈ ${total:.2f} "
          f"(summaries: {settings.summary_model}, effort={settings.summary_effort}; "
          f"embeddings: {settings.embed_model})")
    if new:
        print(f"{len(new)} EPUB(s) not extracted yet: run `python index.py extract` (free).")
    print("Index one book with: python index.py book <id | \"title words\">")


def cmd_status(args: argparse.Namespace) -> None:
    import anthropic

    import db

    conn = db.connect()
    db.ensure_schema(conn)
    _print_status(conn, anthropic.Anthropic())


def _resolve_book(conn, selector: str) -> dict:
    """Exactly one book by id or title/author words, or exit with a helpful message."""
    import db

    matches = db.find_books(conn, selector)
    if not matches:
        sys.exit(f'No extracted book matches "{selector}". New EPUBs need `python index.py extract` first '
                 "(free); `python index.py status` lists everything.")
    if len(matches) > 1:
        print(f'"{selector}" matches {len(matches)} books; use an id:')
        for b in matches:
            print(f"  {b['id']:>3}  {b['title']} — {', '.join(b['authors'])}")
        sys.exit(1)
    return matches[0]


def cmd_summaries(args: argparse.Namespace) -> None:
    """Show a book's stored summaries as Markdown: paged in a terminal, or written to --out."""
    import pydoc

    import db
    import summarize

    conn = db.connect()
    book = _resolve_book(conn, args.selector)
    rows = conn.execute(
        """SELECT level, chapter_index, chapter_title, content, generated_by FROM chunks
           WHERE book_id = %s AND level IN ('book_summary', 'chapter_summary')
             AND (%s::int IS NULL OR level = 'chapter_summary' AND chapter_index = %s)
             AND NOT (%s AND level = 'chapter_summary')
           ORDER BY chapter_index NULLS FIRST""",
        (book["id"], args.chapter, args.chapter, args.book_only),
    ).fetchall()
    if not rows:
        if args.chapter and not 1 <= args.chapter <= book["chapter_count"]:
            sys.exit(f"[{book['id']}] {book['title']} has chapters 1-{book['chapter_count']}; "
                     f"there's no chapter {args.chapter}.")
        what = f"a summary for chapter {args.chapter}" if args.chapter else "summaries"
        sys.exit(f"[{book['id']}] {book['title']} doesn't have {what} yet "
                 f"(`python index.py book {book['id']}` generates them; `status` shows progress).")

    out = [f"# {book['title']} — {', '.join(book['authors'])}", ""]
    if book["book_summary_method"]:
        out.append(f"*Book summary written from: {book['book_summary_method'].replace('_', ' ')}*")
        out.append("")
    for r in rows:
        words = summarize.word_count(r["content"])
        if r["level"] == "book_summary":
            out.append(f"## Book summary  ({words} words, {r['generated_by']})")
        else:
            out.append(f"## {r['chapter_index']}. {r['chapter_title']}  ({words} words)")
        out += ["", r["content"], ""]
    text = "\n".join(out)

    if args.out:
        Path(args.out).write_text(text)
        print(f"Wrote {len(rows)} summaries to {args.out}")
    elif sys.stdout.isatty():
        pydoc.pager(text)  # uses $PAGER, or less; q to quit
    else:
        print(text)


def cmd_book(args: argparse.Namespace) -> None:
    import anthropic

    import db
    import embed
    import summarize

    conn = db.connect()
    db.ensure_schema(conn)
    book = _resolve_book(conn, args.selector)
    print(f"[{book['id']}] {book['title']} — {', '.join(book['authors'])}")

    # Refresh extraction if the EPUB changed since it was indexed (free, keeps the id).
    path = Path(book["source_path"])
    if not path.exists():
        print(f"  warning: {path} no longer exists; using the stored text")
    else:
        digest = file_hash(path)
        if digest != book["content_hash"]:
            print("  EPUB changed since extraction; re-extracting (summaries and embeddings will be redone)")
            if _extract_one(conn, path, digest) is None:
                sys.exit(1)

    if args.redo_summaries and not args.dry_run:
        db.delete_summaries(conn, book["id"])
        print("  existing summaries deleted; regenerating")
    stage = db.book_stages(conn, [book["id"]])[0]["stage"]
    print(f"  stage: {stage}")

    # One combined estimate (summaries + embeddings) and one confirmation.
    client = anthropic.Anthropic()
    print("Counting tokens (count_tokens is free)...")
    plans = summarize.plan(conn, client, [book["id"]])
    summary_cost = summarize.print_estimate(conn, plans) if plans else 0.0
    future = sum(len(p.pending_chapters) + p.needs_book_summary for p in plans)
    n_embed, embed_tokens, embed_cost = embed.estimate(conn, [book["id"]], future)
    if n_embed:
        embed.print_estimate(n_embed, embed_tokens, embed_cost)
    if not plans and not n_embed:
        print("\nNothing to do: this book is fully indexed.")
        return
    print(f"\nTOTAL for this book ≈ ${summary_cost + embed_cost:.2f}")
    if args.dry_run:
        print("DRY RUN: nothing generated.")
        return
    if not summarize.confirm(args.yes):
        print("Aborted.")
        return

    if plans and not summarize.execute(client, plans):
        sys.exit("Summaries didn't finish; embeddings skipped. Re-run the same command to resume.")
    if n_embed:
        print("\nEmbedding...")
        if not embed.execute(conn, [book["id"]]):
            sys.exit(1)
    print(f"\n[{book['id']}] {book['title']}: stage {db.book_stages(conn, [book['id']])[0]['stage']}")


def _require_selection(args: argparse.Namespace, command: str) -> None:
    """Refuse library-wide paid runs unless --all is given; show the status table instead."""
    if args.all or args.book_id:
        return
    import anthropic

    import db

    conn = db.connect()
    db.ensure_schema(conn)
    _print_status(conn, anthropic.Anthropic())
    sys.exit(f"\n{command} needs --book-id ID (repeatable) or --all. "
             "Prefer `python index.py book <id>` to index one book end to end.")


def cmd_summarize(args: argparse.Namespace) -> None:
    import summarize

    _require_selection(args, "summarize")
    summarize.run(None if args.all else args.book_id, yes=args.yes, dry_run=args.dry_run)


def cmd_embed(args: argparse.Namespace) -> None:
    import embed

    _require_selection(args, "embed")
    embed.run(None if args.all else args.book_id, yes=args.yes, dry_run=args.dry_run)


def cmd_search(args: argparse.Namespace) -> None:
    """Debugging aid for Phase 3: raw vector search, no LLM involved."""
    from pgvector import Vector

    import db
    import embed

    conn = db.connect()
    embedder = embed.get_embedder()
    levels = [args.level] if args.level else ["passage", "chapter_summary", "book_summary"]
    hits = db.search_chunks(conn, Vector(embedder.embed_query(args.query)), embedder.model_name, levels,
                            args.book_id, args.k)
    if not hits:
        print("No embedded chunks match those filters.")
    for h in hits:
        where = "book summary" if h["level"] == "book_summary" else f"ch {h['chapter_index']} {h['chapter_title']!r}"
        snippet = " ".join(h["content"].split())[:160]
        print(f"{h['similarity']:.3f}  [{h['level']}] {h['book_title'][:30]}, {where}\n       {snippet}…")


def cmd_backup(args: argparse.Namespace) -> None:
    import backup

    directory = backup.backup_dir()
    if args.list:
        existing = backup.list_backups(directory)
        for p in existing:
            print(f"  {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")
        print(f"{len(existing)} backup(s) in {directory}")
        return
    try:
        path = backup.create_backup(args.label)
    except backup.BackupError as e:
        sys.exit(f"Backup failed: {e}")
    print(f"Backed up to {path} ({path.stat().st_size / 1e6:.1f} MB, verified)")
    if args.keep:
        for old in backup.prune(directory, args.keep):
            print(f"  removed old backup {old.name}")
    print(f"Restore with (replaces the current database contents):\n  {backup.restore_command(path)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Index an EPUB library")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("extract", help="Phase 1: extract chapters and passage chunks (no API calls)")
    p.add_argument("--limit", type=int, help="process at most N books")
    p.add_argument("--dry-run", action="store_true", help="print counts only; write nothing")
    p.add_argument("-v", "--verbose", action="store_true", help="show kept/skipped items and chapters")
    p.add_argument("--dump", metavar="DIR", help="write each chapter's Markdown and chunks to DIR")
    p.add_argument("--force", action="store_true",
                   help="re-extract unchanged books too (skips books that already have summaries or embeddings)")
    p.set_defaults(func=cmd_extract)

    p = sub.add_parser("status", help="stage and cost-to-finish for every book")
    p.set_defaults(func=cmd_status)

    p = sub.add_parser("book", help="finish indexing one book: summaries, then embeddings")
    p.add_argument("selector", help='book id, or words from the title/author, e.g. "morrie"')
    p.add_argument("--redo-summaries", action="store_true",
                   help="delete this book's summaries and generate them again")
    p.add_argument("--dry-run", action="store_true", help="print the cost estimate only")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.set_defaults(func=cmd_book)

    p = sub.add_parser("summaries", help="read a book's stored summaries (paged; no API calls)")
    p.add_argument("selector", help='book id, or words from the title/author, e.g. "wayward"')
    view = p.add_mutually_exclusive_group()
    view.add_argument("--chapter", type=int, metavar="N", help="only chapter N (the number `status`/search show)")
    view.add_argument("--book-only", action="store_true", help="only the book-level summary")
    p.add_argument("--out", metavar="FILE", help="write Markdown to FILE instead of paging")
    p.set_defaults(func=cmd_summaries)

    p = sub.add_parser("summarize", help="Phase 2: chapter and book summaries (Anthropic API)")
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--book-id", type=int, action="append", help="book id to summarize (repeatable)")
    sel.add_argument("--all", action="store_true", help="summarize every remaining book in the library")
    p.add_argument("--dry-run", action="store_true", help="print the cost estimate only")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("embed", help="Phase 3: embed chunks (Voyage/OpenAI)")
    sel = p.add_mutually_exclusive_group()
    sel.add_argument("--book-id", type=int, action="append", help="book id to embed (repeatable)")
    sel.add_argument("--all", action="store_true", help="embed every pending chunk in the library")
    p.add_argument("--dry-run", action="store_true", help="print the cost estimate only")
    p.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p.set_defaults(func=cmd_embed)

    p = sub.add_parser("search", help="raw vector search over embedded chunks (debugging)")
    p.add_argument("query")
    p.add_argument("--book-id", type=int)
    p.add_argument("--level", choices=["passage", "chapter_summary", "book_summary"])
    p.add_argument("-k", type=int, default=8)
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("backup", help="dated, verified pg_dump of the database (outside the project)")
    p.add_argument("--label", help='added to the file name, e.g. "before-redo"')
    p.add_argument("--keep", type=int, metavar="N", help="then delete all but the newest N backups")
    p.add_argument("--list", action="store_true", help="list existing backups instead of making one")
    p.set_defaults(func=cmd_backup)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
