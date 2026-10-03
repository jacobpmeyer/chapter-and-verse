"""Phase 2: hierarchical summaries (chapter -> book).

* Chapter summaries run in order within a book. Each call also sees the
  previous chapter's summary, so it can recognise returning characters and
  notice shifts in tone. Different books run in parallel.
* Book summary: if the whole book fits in the summary model's context (with
  headroom), it's written from the full text. Otherwise it's built from the
  chapter summaries. books.book_summary_method records which was used. Its
  length scales with the book: long books get longer book summaries.
* Nothing is sent to the model until the user confirms the estimated cost.
  Counting tokens (count_tokens) is free, and the estimate uses it together
  with averages from past calls (the api_calls table).
"""

from __future__ import annotations

import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import anthropic

import db
from config import FALLBACK_MODELS, PRICES, settings
from stage import Progress, StageResult, report
from tokens import estimate_tokens

# Bump when the prompts change, so the estimate stops averaging over calls made
# with the old prompts. Logged on every call (api_calls.prompt_version).
# v5: beliefs, predictions, rumors and lies are attributed, not stated as fact.
PROMPT_VERSION = "v5"

CHAPTER_MAX_TOKENS = 16_000
BOOK_MAX_TOKENS = 32_000
PROMPT_OVERHEAD_TOKENS = 700  # system prompt + instructions
TOKENS_PER_WORD = 1.35  # Markdown summary prose, Claude tokenizer (measured on Morrie)

# Fallback thinking-token allowance per call, used only until api_calls has
# history for the current model/effort/prompt. "medium" was measured on a real
# run (~350/call); the other levels are extrapolated guesses.
THINKING_ALLOWANCE = {"low": 150, "medium": 350, "high": 900, "xhigh": 1_800, "max": 3_500}
MIN_HISTORY_CALLS = 5  # calls needed before history replaces the fallback numbers
DEFAULT_CONDENSE_RATE = 0.1  # expected condense calls per summary, until history says otherwise
MAX_CONDENSE_ATTEMPTS = 2

SYSTEM = """You write reference summaries of books for a private reading index. \
Other tools retrieve these summaries to answer questions about plot, characters, \
themes, and arcs, so be specific: name characters, places, objects, and concepts \
exactly as the book does, and prefer concrete details over generalities. Write in \
present tense. Never invent anything that isn't in the text. Spoilers are expected; \
this index is for someone who has read the book. The book may be fiction or \
nonfiction; where the instructions give alternatives, use the one that fits.

Keep what the book establishes as true separate from what characters believe, \
predict, hope, suspect, hear, or claim, since those can turn out to be wrong or \
to be lies. Attribute each one to whoever holds it, and say how they came by it: \
write "a neighbor insists the house is haunted", not "the house is haunted". If \
the text confirms or contradicts one, say so. In nonfiction, likewise keep the \
author's own claims apart from views the author reports or argues against."""

# --------------------------------------------------------------------------- #
# Lengths
# --------------------------------------------------------------------------- #
# Section budgets are ranges: their minimums pull the model toward the upper
# half of the target (it otherwise treats "up to N" as a ceiling and lands
# low). Maximums add up to just under the hard cap. enforce_length() condenses
# anything that still comes back too long.

CHAPTER_WORDS = (350, 450)  # target range; 450 is the hard cap

CHAPTER_INSTRUCTIONS = """Summarize the chapter above in 350-450 words of Markdown. \
The 450-word maximum is strict; stay within each section's range. \
Use exactly these headings:

### Summary
(180-210 words) Fiction: what happens, in order. Nonfiction: the chapter's argument and key claims.

### Characters
(70-90 words) Fiction: who appears and what they do, want, or reveal. Nonfiction: rename this heading "Key concepts" and cover the ideas, terms, and examples introduced.

### Motifs and recurring images
(45-60 words) Images, symbols, phrases, or ideas that recur or echo earlier chapters.

### Unresolved threads
(35-50 words) Open questions, set-ups, and promises not yet paid off (nonfiction: questions deferred to later chapters).

### Tone
(30-40 words) The chapter's tone, and any shift in tone within it or compared with the previous chapter.

Use the previous chapter's summary, if given, only for continuity. Summarize only this chapter. \
Don't comment on the book's genre or on these instructions. Output only the Markdown."""

# Book summary length scales with the book: (exact tokens up to, target range, section scale).
BOOK_SIZE_TIERS = [
    (150_000, (600, 1000), 1.0),
    (300_000, (900, 1500), 1.5),
    (None, (1200, 2000), 2.0),
]
BOOK_SECTIONS = [  # (heading, base word range, guidance)
    ("Premise", (80, 120), ""),
    ("Arc", (220, 300), "How the book develops from beginning to end, including the ending "
                        "(nonfiction: how the argument builds)."),
    ("Major themes", (120, 180), ""),
    ("Tone", (50, 80), "The overall tone and how it changes across the book."),
    ("Motifs and recurring images", (70, 120), ""),
    ("Key characters", (100, 150), 'Nonfiction: rename this heading "Key concepts".'),
]

CONDENSE_INSTRUCTIONS = """The summary above is {words} words, but the limit is {limit} words. \
Rewrite it to at most {target} words. Keep exactly the same headings. Keep the most specific \
names, quotes, and details; cut repetition and generalities first. Don't change or add any facts, \
and keep every attribution: a belief, prediction, rumor, or claim stays attributed to whoever holds \
it, never shortened into a plain fact. Output only the Markdown."""


def book_words(exact_tokens: int) -> tuple[tuple[int, int], float]:
    for limit, words, scale in BOOK_SIZE_TIERS:
        if limit is None or exact_tokens <= limit:
            return words, scale
    raise AssertionError("unreachable")


def book_instructions(source: str, exact_tokens: int) -> str:
    (lo, hi), scale = book_words(exact_tokens)
    sections = []
    for heading, (a, b), guidance in BOOK_SECTIONS:
        line = f"### {heading}\n({round(a * scale)}-{round(b * scale)} words)"
        sections.append(f"{line} {guidance}".rstrip())
    return (
        f"Write a book-level summary of {source} in {lo}-{hi} words of Markdown. "
        f"The {hi}-word maximum is strict; stay within each section's range. Use exactly these headings:\n\n"
        + "\n".join(sections)
        + "\n\nDon't comment on the book's genre or on these instructions. Output only the Markdown."
    )


def word_count(markdown: str) -> int:
    """Words as a reader would count them: ignores Markdown symbols like ###, **, and list dashes."""
    return len(re.findall(r"[A-Za-z0-9][\w'’.-]*", markdown))


class SummaryError(RuntimeError):
    pass


# --------------------------------------------------------------------------- #
# API call
# --------------------------------------------------------------------------- #

@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    calls: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, input_tokens: int, output_tokens: int) -> None:
        with self.lock:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.calls += 1

    def cost(self, model: str) -> float:
        pin, pout = PRICES.get(model, (0.0, 0.0))
        return (self.input_tokens * pin + self.output_tokens * pout) / 1e6


@dataclass
class CallContext:
    """Where a call's usage goes: the run total and the api_calls log."""
    usage: Usage
    conn: object  # psycopg connection for this worker
    book_id: int


def _request_kwargs(user: str, max_tokens: int) -> dict:
    kwargs = dict(
        model=settings.summary_model,
        max_tokens=max_tokens,
        system=SYSTEM,
        thinking={"type": "adaptive"},
        output_config={"effort": settings.summary_effort},
        messages=[{"role": "user", "content": user}],
    )
    if settings.summary_model in FALLBACK_MODELS:
        # On a safety-classifier decline, the API re-runs the request on
        # Anthropic's recommended fallback model instead of returning a refusal.
        kwargs.update(betas=["server-side-fallback-2026-07-01"], fallbacks="default")
    return kwargs


def call_model(client: anthropic.Anthropic, user: str, max_tokens: int, ctx: CallContext,
               purpose: str) -> tuple[str, str]:
    """Returns (markdown, model_that_answered). Logs usage to api_calls."""
    kwargs = _request_kwargs(user, max_tokens)
    api = client.beta.messages if "betas" in kwargs else client.messages
    with api.stream(**kwargs) as stream:
        msg = stream.get_final_message()
    u = msg.usage
    input_tokens = (u.input_tokens or 0) + (getattr(u, "cache_read_input_tokens", 0) or 0) \
        + (getattr(u, "cache_creation_input_tokens", 0) or 0)
    ctx.usage.add(input_tokens, u.output_tokens or 0)
    pin, pout = PRICES.get(settings.summary_model, (0.0, 0.0))
    db.log_api_call(ctx.conn, ctx.book_id, purpose, settings.summary_model, settings.summary_effort,
                    PROMPT_VERSION, input_tokens, u.output_tokens or 0,
                    cost_usd=(input_tokens * pin + (u.output_tokens or 0) * pout) / 1e6)
    if msg.stop_reason == "refusal":
        details = getattr(msg, "stop_details", None)
        raise SummaryError(f"model declined (category: {getattr(details, 'category', None)})")
    if msg.stop_reason == "max_tokens":
        raise SummaryError(f"hit max_tokens={max_tokens} before finishing")
    text = "\n".join(b.text for b in msg.content if b.type == "text").strip()
    if not text:
        raise SummaryError("empty response")
    return text, msg.model


def count_tokens(client: anthropic.Anthropic, text: str) -> int:
    return client.messages.count_tokens(
        model=settings.summary_model,
        system=SYSTEM,
        messages=[{"role": "user", "content": text}],
    ).input_tokens


def enforce_length(client: anthropic.Anthropic, text: str, max_words: int, ctx: CallContext, label: str) -> str:
    """Condense `text` until it's within max_words. Each attempt is a cheap call
    (the draft is the only input). If it's still over after MAX_CONDENSE_ATTEMPTS,
    keep the shortest version and warn."""
    best = text
    for _ in range(MAX_CONDENSE_ATTEMPTS):
        words = word_count(best)
        if words <= max_words:
            return best
        prompt = f"<summary>\n{best}\n</summary>\n\n" + CONDENSE_INSTRUCTIONS.format(
            words=words, limit=max_words, target=int(max_words * 0.9)
        )
        condensed, _model = call_model(client, prompt, CHAPTER_MAX_TOKENS, ctx, "condense")
        if word_count(condensed) < words:
            best = condensed
    if word_count(best) > max_words:
        print(f"  warning: {label} is still {word_count(best)} words (limit {max_words})", flush=True)
    return best



# --------------------------------------------------------------------------- #
# Prompts
# --------------------------------------------------------------------------- #

def _book_header(book: dict) -> str:
    return f'<book title="{book["title"]}" authors="{", ".join(book["authors"])}">'


def chapter_prompt(book: dict, chapter: dict, previous_summary: str | None) -> str:
    parts = [_book_header(book) + "</book>"]
    if previous_summary:
        parts.append(f"<previous_chapter_summary>\n{previous_summary}\n</previous_chapter_summary>")
    parts.append(
        f'<chapter index="{chapter["chapter_index"]}" title="{chapter["title"]}">\n{chapter["content"]}\n</chapter>'
    )
    parts.append(CHAPTER_INSTRUCTIONS)
    return "\n\n".join(parts)


def full_text(chapters: list[dict]) -> str:
    return "\n\n".join(f'## Chapter {c["chapter_index"]}: {c["title"]}\n\n{c["content"]}' for c in chapters)


def book_prompt_full_text(book: dict, chapters: list[dict], exact_tokens: int) -> str:
    return f"{_book_header(book)}\n{full_text(chapters)}\n</book>\n\n" + book_instructions(
        "the book above", exact_tokens
    )


def book_prompt_from_summaries(book: dict, chapters: list[dict], summaries: dict[int, str],
                               exact_tokens: int) -> str:
    body = "\n\n".join(
        f'## Chapter {c["chapter_index"]}: {c["title"]}\n\n{summaries[c["chapter_index"]]}' for c in chapters
    )
    return (
        f"{_book_header(book)}\n<chapter_summaries>\n{body}\n</chapter_summaries>\n</book>\n\n"
        + book_instructions("the book from its chapter summaries above", exact_tokens)
    )


# --------------------------------------------------------------------------- #
# Planning / estimate
# --------------------------------------------------------------------------- #

@dataclass
class CallCosts:
    """Expected tokens for each kind of call, from history when there's enough."""
    chapter_output: float
    book_thinking: float
    condense_rate: float  # condense calls per summary
    condense_input: float
    condense_output: float
    source: str  # "history" or "defaults"

    @classmethod
    def load(cls, conn) -> "CallCosts":
        stats = db.api_call_stats(conn, settings.summary_model, settings.summary_effort, PROMPT_VERSION)
        thinking = THINKING_ALLOWANCE.get(settings.summary_effort, 350)
        chapter_summary_tokens = CHAPTER_WORDS[1] * 0.9 * TOKENS_PER_WORD
        costs = cls(
            chapter_output=chapter_summary_tokens + thinking,
            book_thinking=2 * thinking,
            condense_rate=DEFAULT_CONDENSE_RATE,
            condense_input=chapter_summary_tokens * 1.2 + PROMPT_OVERHEAD_TOKENS,
            condense_output=chapter_summary_tokens + thinking,
            source="defaults",
        )
        ch = stats.get("chapter_summary")
        if ch and ch["n"] >= MIN_HISTORY_CALLS:
            costs.chapter_output = float(ch["avg_out"])
            cond = stats.get("condense")
            n_summaries = ch["n"] + (stats["book_summary"]["n"] if "book_summary" in stats else 0)
            costs.condense_rate = (cond["n"] if cond else 0) / n_summaries
            if cond:
                costs.condense_input, costs.condense_output = float(cond["avg_in"]), float(cond["avg_out"])
            costs.source = f"history ({ch['n']} chapter calls)"
        return costs


@dataclass
class BookPlan:
    book: dict
    chapters: list[dict]
    pending_chapters: list[dict]
    needs_book_summary: bool
    method: str  # full_text | chapter_summaries
    exact_tokens: int
    est_input: int = 0
    est_output: int = 0

    @property
    def est_cost(self) -> float:
        pin, pout = PRICES.get(settings.summary_model, (0.0, 0.0))
        return (self.est_input * pin + self.est_output * pout) / 1e6


def plan(conn, client: anthropic.Anthropic, book_ids: list[int] | None) -> list[BookPlan]:
    """Plan the remaining summary work for the given books (None = all books)."""
    books = [b for b in db.book_stages(conn, book_ids) if b["stage"] in ("extracted", "summarizing")]
    costs = CallCosts.load(conn)
    chapter_summary_in = CHAPTER_WORDS[1] * 0.9 * TOKENS_PER_WORD  # previous summary, as input
    plans = []
    for book in books:
        chapters = db.get_chapters(conn, book["id"])
        done = db.get_chapter_summaries(conn, book["id"])
        exact = book["exact_token_count"]
        if exact is None:
            exact = count_tokens(client, full_text(chapters))
            db.set_exact_token_count(conn, book["id"], exact)
        ratio = exact / max(1, sum(c["token_count"] for c in chapters))  # estimate -> real tokenizer

        fits = exact + PROMPT_OVERHEAD_TOKENS + BOOK_MAX_TOKENS <= settings.summary_context_limit
        p = BookPlan(
            book=book,
            chapters=chapters,
            pending_chapters=[c for c in chapters if c["chapter_index"] not in done],
            needs_book_summary=not book["has_book_summary"],
            method="full_text" if fits else "chapter_summaries",
            exact_tokens=exact,
        )
        n_summaries = len(p.pending_chapters) + (1 if p.needs_book_summary else 0)
        for c in p.pending_chapters:
            p.est_input += PROMPT_OVERHEAD_TOKENS + chapter_summary_in + c["token_count"] * ratio
            p.est_output += costs.chapter_output
        if p.needs_book_summary:
            (_lo, hi), _scale = book_words(exact)
            source = exact if fits else len(chapters) * chapter_summary_in
            p.est_input += PROMPT_OVERHEAD_TOKENS + source
            p.est_output += hi * 0.8 * TOKENS_PER_WORD + costs.book_thinking
        p.est_input += n_summaries * costs.condense_rate * costs.condense_input
        p.est_output += n_summaries * costs.condense_rate * costs.condense_output
        p.est_input, p.est_output = round(p.est_input), round(p.est_output)
        plans.append(p)
    return plans


def print_estimate(conn, plans: list[BookPlan]) -> float:
    pin, pout = PRICES.get(settings.summary_model, (0.0, 0.0))
    print(f"\nSummary model: {settings.summary_model} (effort={settings.summary_effort}); "
          f"prices ${pin}/${pout} per MTok in/out; output estimate from {CallCosts.load(conn).source}")
    total_in = total_out = 0
    for p in plans:
        book_part = f"book summary via {p.method}" if p.needs_book_summary else "book summary done"
        print(f"  [{p.book['id']}] {p.book['title'][:50]:50} {p.exact_tokens:>8,} tok | "
              f"{len(p.pending_chapters):3} chapter calls, {book_part} | "
              f"~{p.est_input:,} in / ~{p.est_output:,} out  ≈ ${p.est_cost:.2f}")
        total_in += p.est_input
        total_out += p.est_output
    total = (total_in * pin + total_out * pout) / 1e6
    print(f"  TOTAL ≈ ${total:.2f}  (~{total_in:,} input, ~{total_out:,} output incl. thinking)")
    return total


# --------------------------------------------------------------------------- #
# Execution
# --------------------------------------------------------------------------- #

def summarize_book(client: anthropic.Anthropic, p: BookPlan, usage: Usage, progress: Progress | None = None) -> str:
    conn = db.connect()  # one connection per worker thread
    title = p.book["title"][:40]
    book_id = p.book["id"]
    ctx = CallContext(usage, conn, book_id)
    try:
        summaries = db.get_chapter_summaries(conn, book_id)
        for c in p.chapters:
            if c["chapter_index"] in summaries:
                continue
            prev = summaries.get(c["chapter_index"] - 1)
            text, model = call_model(client, chapter_prompt(p.book, c, prev), CHAPTER_MAX_TOKENS, ctx,
                                     "chapter_summary")
            text = enforce_length(client, text, CHAPTER_WORDS[1], ctx, f"chapter {c['chapter_index']}")
            db.save_chapter_summary(conn, book_id, c["chapter_index"], c["title"], text, estimate_tokens(text), model)
            summaries[c["chapter_index"]] = text
            report(progress, f"  {title}: chapter {c['chapter_index']}/{len(p.chapters)} summarized")

        if p.needs_book_summary:
            if p.method == "full_text":
                prompt = book_prompt_full_text(p.book, p.chapters, p.exact_tokens)
            else:
                prompt = book_prompt_from_summaries(p.book, p.chapters, summaries, p.exact_tokens)
            text, model = call_model(client, prompt, BOOK_MAX_TOKENS, ctx, "book_summary")
            (_lo, hi), _scale = book_words(p.exact_tokens)
            text = enforce_length(client, text, hi, ctx, "book summary")
            db.save_book_summary(conn, book_id, text, estimate_tokens(text), model, p.method)
            report(progress, f"  {title}: book summary done ({p.method}, {word_count(text)} words)")
        return f"{title}: ok"
    except (SummaryError, anthropic.APIError) as e:
        # Progress so far is saved; a re-run resumes from the first missing chapter.
        return f"{title}: FAILED ({e}); re-run to resume"
    finally:
        conn.close()


def estimate(conn, client: anthropic.Anthropic, book_ids: list[int] | None) -> tuple[list[BookPlan], float]:
    """Plan and price the remaining summary work. count_tokens calls are free."""
    plans = plan(conn, client, book_ids)
    return plans, (print_estimate(conn, plans) if plans else 0.0)


def execute(client: anthropic.Anthropic, plans: list[BookPlan], progress: Progress | None = None) -> StageResult:
    """Run the planned summaries. The result is truthy if every book finished."""
    usage = Usage()
    with ThreadPoolExecutor(max_workers=min(4, len(plans))) as pool:
        results = list(pool.map(lambda p: summarize_book(client, p, usage, progress), plans))
    print("\n" + "\n".join(results))
    estimated = sum(p.est_cost for p in plans)
    actual = usage.cost(settings.summary_model)
    print(f"Actual usage: {usage.calls} calls, {usage.input_tokens:,} input / {usage.output_tokens:,} output tokens "
          f"≈ ${actual:.2f} (estimated ${estimated:.2f}, {(actual - estimated) / estimated:+.0%})")
    return StageResult(all(r.endswith(": ok") for r in results), actual)


def confirm(yes: bool) -> bool:
    if yes:
        return True
    if not sys.stdin.isatty():
        sys.exit("Not a terminal; re-run with --yes to confirm.")
    return input("\nProceed? [y/N] ").strip().lower() in ("y", "yes")


def run(book_ids: list[int] | None, yes: bool, dry_run: bool) -> None:
    """book_ids=None means every book in the library (the CLI requires --all for that)."""
    client = anthropic.Anthropic()
    conn = db.connect()
    db.ensure_schema(conn)
    print("Counting tokens (count_tokens is free)...")
    plans, _ = estimate(conn, client, book_ids)
    if not plans:
        print("Nothing to summarize: selected books are already summarized.")
        return
    if dry_run:
        print("\nDRY RUN: no summaries generated.")
        return
    if not confirm(yes):
        print("Aborted.")
        return
    execute(client, plans)
