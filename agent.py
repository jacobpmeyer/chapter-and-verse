"""Phase 4: a hand-written tool-calling agent over the library.

    python agent.py

The loop, for each question you ask:

  1. Append your question to `messages` (the conversation history).
  2. Send system prompt + tools + messages to the model (streamed, so text
     prints as it's written).
  3. Append the model's reply to `messages` exactly as returned. That matters:
     the reply contains thinking blocks, and the API requires them to be sent
     back unchanged on the next request.
  4. If the model stopped to call tools (stop_reason == "tool_use"): run every
     tool call in the reply, then append ONE user message holding all the
     tool_result blocks, and go back to step 2.
  5. Otherwise it's done (end_turn), or something needs handling (max_tokens,
     refusal).

AGENT_MAX_TURNS caps the model calls per question. On the last allowed call,
tools are switched off (tool_choice "none") and the model is told to answer
with what it has, so the answer ends cleanly instead of being cut off.

Nothing in `messages` is ever edited or removed, only appended to (except
dropping a refused question entirely). That keeps prompt caching effective
and satisfies the API's rule that earlier thinking blocks stay unchanged.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from dataclasses import dataclass

import anthropic

from config import CACHE_READ_MULTIPLIER, CACHE_WRITE_MULTIPLIER, FALLBACK_MODELS, PRICES, settings
from tools import TOOLS, ToolError, ToolRunner

MAX_TOKENS = 32_000

SYSTEM_PROMPT = """You answer questions about the user's personal ebook library. You have \
tools to search it; your answers should be grounded in what those tools return.

Choosing tools:
- Specific scenes, events, quotes, or details ("what happens when...", "who says...", "how is X \
described") -> search_passages.
- Themes, character arcs, tone, motifs, how something develops across a book, which chapter \
something happens in, and comparisons between books -> search_summaries (chapter summaries for \
where and how things develop; book summaries for the whole-book view), plus get_book_summary.
- Deep analysis of one book that summaries and passages can't support -> load_full_book, and \
only then: it is large and stays in the conversation. It refuses books over the size limit; \
fall back to summaries and passages.
- Use list_books to find book ids, check whether a book is in the library, or see what's searchable.
- Combine tools when it helps: summaries to find where something happens, passages to get the \
actual text.

Citations:
- Cite the book and chapter for every claim that comes from a retrieved result, using the \
book title and the chapter title exactly as the tool gives them, e.g. (Witchcraft for Wayward \
Girls, "Chapter 10 (30 WEEKS)"). Don't use the result numbers or your own chapter numbering.
- For claims from a book summary, cite it as (Title, book summary).
- Quote the text when the exact wording matters.

When retrieval is weak:
- If the results are flagged WEAK RETRIEVAL, or don't actually address the question, try one \
or two differently worded searches. If it's still weak, say plainly that you couldn't find it \
in the library instead of guessing.
- Keep what the books say separate from your general knowledge. If you add background that \
didn't come from the tools, say so.

Books that aren't indexed:
- Only books at stage "embedded" are searchable. If a question needs a book that is in the \
library but not indexed, say so and give the user the command to index it (the tool error \
includes it). Indexing costs money, so the user runs it themselves.

Spoilers are fine: the user has read these books. Be direct and specific, and match the length \
of your answer to the question.

Your reply is shown in a plain terminal and is not rendered as Markdown, so don't use Markdown \
syntax (no **bold**, # headings, or tables). Write plain sentences; use numbered or dashed lists \
where they help."""


# --------------------------------------------------------------------------- #
# Terminal output
# --------------------------------------------------------------------------- #

DIM, ITALIC, BOLD, CYAN, RESET = "\033[2m", "\033[3m", "\033[1m", "\033[36m", "\033[0m"
if not sys.stdout.isatty():
    DIM = ITALIC = BOLD = CYAN = RESET = ""


class StreamPrinter:
    """Prints streamed text wrapped at word boundaries.

    Text arrives in arbitrary fragments, so a word can be split across two
    deltas: the trailing partial word is held back until whitespace arrives.
    Blank lines are held too, so a block never ends with stray empty lines.
    """

    def __init__(self):
        self.width = shutil.get_terminal_size((100, 24)).columns - 1
        self.col = 0  # characters on the current line
        self.partial = ""  # an unfinished word
        self.newlines = 0  # newlines seen but not yet printed
        self.space = False  # a space waiting for the next word
        self.style = ""

    def write(self, text: str, style: str = "") -> None:
        if style != self.style:
            self._flush_partial()
            self.style = style
        pieces = re.split(r"(\s+)", self.partial + text)
        self.partial = pieces.pop()  # may be an unfinished word ('' if text ended in whitespace)
        for piece in pieces:
            if piece.isspace():
                self._whitespace(piece)
            elif piece:
                self._word(piece)

    def end_block(self) -> None:
        """Finish a content block: print what's held, end the line, drop trailing blank lines."""
        self._flush_partial()
        self.newlines, self.space = 0, False
        if self.col:
            self._out("\n")
            self.col = 0

    def _flush_partial(self) -> None:
        if self.partial:
            self._word(self.partial)
            self.partial = ""

    def _whitespace(self, ws: str) -> None:
        n = ws.count("\n")
        if n:
            self.newlines += n
        elif self.col and not self.newlines:
            self.space = True  # printed only if the next word stays on this line

    def _word(self, word: str) -> None:
        if self.newlines:
            self._out("\n" * min(self.newlines, 2))  # at most one blank line
            self.col, self.newlines = 0, 0
        elif self.col and self.col + self.space + len(word) > self.width:
            self._out("\n")
            self.col = 0
        elif self.space:
            self._out(" ")
            self.col += 1
        self.space = False
        self._out(f"{self.style}{word}{RESET if self.style else ''}")
        self.col += len(word)

    @staticmethod
    def _out(text: str) -> None:
        print(text, end="", flush=True)


def format_call(name: str, args: dict) -> str:
    shown = ", ".join(f"{k}={json.dumps(v, ensure_ascii=False)}" for k, v in args.items())
    return f"{CYAN}→ {name}({shown}){RESET}"


# --------------------------------------------------------------------------- #
# Usage / cost
# --------------------------------------------------------------------------- #

@dataclass
class SessionUsage:
    calls: int = 0
    input: int = 0
    cache_write: int = 0
    cache_read: int = 0
    output: int = 0

    def add(self, u) -> None:
        self.calls += 1
        self.input += u.input_tokens or 0
        self.cache_write += getattr(u, "cache_creation_input_tokens", 0) or 0
        self.cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
        self.output += u.output_tokens or 0

    def cost(self) -> float:
        pin, pout = PRICES.get(settings.agent_model, (0.0, 0.0))
        return (self.input * pin + self.cache_write * pin * CACHE_WRITE_MULTIPLIER
                + self.cache_read * pin * CACHE_READ_MULTIPLIER + self.output * pout) / 1e6

    def report(self) -> str:
        return (f"{self.calls} model calls; input {self.input:,} + cache write {self.cache_write:,} + "
                f"cache read {self.cache_read:,}; output {self.output:,} ≈ ${self.cost():.3f}")


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #

class Agent:
    def __init__(self, client: anthropic.Anthropic | None = None, tools: ToolRunner | None = None):
        # Both can be passed in, which is how the tests substitute fakes.
        self.client = client or anthropic.Anthropic()
        self.tools = tools or ToolRunner()
        self.messages: list[dict] = []  # the whole session's history
        self.usage = SessionUsage()

    # -- one model call ----------------------------------------------------- #

    def _request_kwargs(self, final_turn: bool) -> dict:
        thinking: dict = {"type": "adaptive"}
        betas = []
        if settings.agent_thinking_display in ("summarized", "updates"):
            thinking["display"] = settings.agent_thinking_display
        if settings.agent_thinking_display == "updates":
            betas.append("thinking-display-updates-2026-08-18")
        kwargs = dict(
            model=settings.agent_model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=self.messages,
            thinking=thinking,
            output_config={"effort": settings.agent_effort},
            # Automatic prompt caching: caches everything up to the last block, so
            # each call only pays full price for what's new since the previous one.
            cache_control={"type": "ephemeral"},
        )
        if final_turn:
            kwargs["tool_choice"] = {"type": "none"}
        if settings.agent_model in FALLBACK_MODELS:
            # If a safety classifier declines, the API retries on a fallback model.
            betas.append("server-side-fallback-2026-07-01")
            kwargs["fallbacks"] = "default"
        if betas:
            kwargs["betas"] = betas
        return kwargs

    def _call_model(self, final_turn: bool):
        """Stream one response to the terminal and return the complete message."""
        kwargs = self._request_kwargs(final_turn)
        api = self.client.beta.messages if "betas" in kwargs else self.client.messages
        out = StreamPrinter()
        with api.stream(**kwargs) as stream:
            last_block = None
            for event in stream:
                if event.type == "content_block_start":
                    kind = event.content_block.type
                    if kind == "text" and last_block == "thinking":
                        print()  # a blank line between the reasoning and the answer
                    if kind in ("text", "thinking"):
                        last_block = kind
                elif event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta" and event.delta.thinking:
                        # Reasoning summaries (display=summarized) or progress notes (updates).
                        out.write(event.delta.thinking, style=f"{DIM}{ITALIC}")
                    elif event.delta.type == "text_delta":
                        out.write(event.delta.text)
                elif event.type == "content_block_stop":
                    out.end_block()
            message = stream.get_final_message()
        self.usage.add(message.usage)
        return message

    # -- tools ---------------------------------------------------------------- #

    def _run_tools(self, message) -> list[dict]:
        """Run every tool_use block in the reply; return the matching tool_result blocks."""
        results = []
        for block in message.content:
            if block.type != "tool_use":
                continue
            print(format_call(block.name, block.input), flush=True)
            try:
                result = self.tools.run(block.name, block.input)
                print(f"{DIM}  ← {result.summary}{RESET}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": result.text})
            except ToolError as e:
                print(f"{DIM}  ← error: {e}{RESET}")
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(e),
                                "is_error": True})
            except Exception as e:  # a bug or outage in our code: report it to the model, keep going
                print(f"{DIM}  ← failed: {type(e).__name__}: {e}{RESET}")
                results.append({"type": "tool_result", "tool_use_id": block.id,
                                "content": f"The tool failed unexpectedly ({type(e).__name__}). "
                                           "Try another approach or tell the user.",
                                "is_error": True})
        return results

    # -- one question --------------------------------------------------------- #

    def ask(self, question: str) -> None:
        start = len(self.messages)
        try:
            self._ask(question)
        except BaseException:
            # API error or Ctrl-C partway through: the history might now end with a
            # tool call that has no result, which the API would reject. Remove this
            # question's messages; earlier conversation is untouched.
            del self.messages[start:]
            raise

    def _ask(self, question: str) -> None:
        start = len(self.messages)
        self.messages.append({"role": "user", "content": question})

        for turn in range(1, settings.agent_max_turns + 1):
            final_turn = turn == settings.agent_max_turns
            message = self._call_model(final_turn)

            if message.stop_reason == "refusal":
                # Nothing usable came back (the fallback model declined too). Drop
                # this question from the history so the conversation stays valid.
                del self.messages[start:]
                category = getattr(getattr(message, "stop_details", None), "category", None)
                print(f"\n[The model declined to answer this (category: {category}). "
                      "It's been removed from the conversation; try rephrasing.]")
                return

            # Step 3: keep the reply exactly as returned, thinking blocks included.
            self.messages.append({"role": "assistant", "content": message.content})

            if message.stop_reason == "tool_use":
                results = self._run_tools(message)
                if turn + 1 == settings.agent_max_turns:
                    # Next call is the last one allowed. Say so in the same message
                    # as the results (appending, never editing, earlier turns).
                    results.append({"type": "text", "text": (
                        f"[Tool-call limit reached ({settings.agent_max_turns} steps). Tools are now "
                        "disabled: answer with what you've found, and say what you couldn't check.]")})
                self.messages.append({"role": "user", "content": results})
                continue

            if message.stop_reason == "max_tokens":
                # The reply was cut off. Any tool calls in it are incomplete, but the
                # API still requires a tool_result for each, so close them out.
                pending = [b for b in message.content if b.type == "tool_use"]
                if pending:
                    self.messages.append({"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": b.id, "is_error": True,
                         "content": "Not run: the response hit max_tokens."} for b in pending]})
                print(f"\n[Response cut off at max_tokens={MAX_TOKENS}.]")
            elif message.stop_reason == "pause_turn":
                continue  # the API paused a long turn; sending the history again resumes it
            return  # end_turn: the answer is complete

        print(f"\n[Stopped after {settings.agent_max_turns} steps (AGENT_MAX_TURNS).]")


# --------------------------------------------------------------------------- #
# REPL
# --------------------------------------------------------------------------- #

HELP = """Commands: /reset (start a new conversation), /usage (tokens and cost so far), /quit"""


def main() -> None:
    agent = Agent()
    books = agent.tools.tool_list_books().summary
    print(f"{BOLD}Library agent{RESET} ({settings.agent_model}, effort={settings.agent_effort}; {books})")
    print(HELP)
    while True:
        try:
            question = input(f"\n{BOLD}you>{RESET} ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not question:
            continue
        if question in ("/quit", "/exit"):
            break
        if question == "/reset":
            agent.messages.clear()
            print("[Conversation cleared.]")
            continue
        if question == "/usage":
            print(agent.usage.report())
            continue
        if question.startswith("/"):
            print(HELP)
            continue
        print()
        try:
            agent.ask(question)
        except KeyboardInterrupt:
            print("\n[Interrupted. That question was dropped; the rest of the conversation is kept.]")
        except anthropic.APIError as e:
            print(f"\n[API error: {e}. That question was dropped; try again.]")
    print(f"Session: {agent.usage.report()}")


if __name__ == "__main__":
    main()
