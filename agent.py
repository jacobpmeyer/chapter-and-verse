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
from typing import Callable

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


class TerminalRenderer:
    """Renders the agent's events in the terminal (the REPL's view of a question)."""

    def __init__(self):
        self.out = StreamPrinter()
        self.last_block = None

    def __call__(self, event: dict) -> None:
        kind = event["type"]
        if kind == "block_start":
            if event["block"] == "text" and self.last_block == "thinking":
                print()  # a blank line between the reasoning and the answer
            self.last_block = event["block"]
        elif kind == "thinking":
            self.out.write(event["text"], style=f"{DIM}{ITALIC}")
        elif kind == "text":
            self.out.write(event["text"])
        elif kind == "block_end":
            self.out.end_block()
        elif kind == "tool_call":
            print(format_call(event["name"], event["input"]), flush=True)
        elif kind == "tool_result":
            label = "error: " if event["is_error"] else ""
            print(f"{DIM}  ← {label}{event['summary']}{RESET}")
        elif kind == "notice":
            print(f"\n[{event['message']}]")


# --------------------------------------------------------------------------- #
# Usage / cost
# --------------------------------------------------------------------------- #

def usage_cost(u, model: str) -> float:
    """USD for one response's usage, with cache writes and reads at their own rates."""
    pin, pout = PRICES.get(model, (0.0, 0.0))
    return ((u.input_tokens or 0) * pin
            + (getattr(u, "cache_creation_input_tokens", 0) or 0) * pin * CACHE_WRITE_MULTIPLIER
            + (getattr(u, "cache_read_input_tokens", 0) or 0) * pin * CACHE_READ_MULTIPLIER
            + (u.output_tokens or 0) * pout) / 1e6


@dataclass
class SessionUsage:
    calls: int = 0
    input: int = 0
    cache_write: int = 0
    cache_read: int = 0
    output: int = 0
    dollars: float = 0.0

    def add(self, u) -> None:
        self.calls += 1
        self.input += u.input_tokens or 0
        self.cache_write += getattr(u, "cache_creation_input_tokens", 0) or 0
        self.cache_read += getattr(u, "cache_read_input_tokens", 0) or 0
        self.output += u.output_tokens or 0
        self.dollars += usage_cost(u, settings.agent_model)

    def cost(self) -> float:
        return self.dollars

    def as_dict(self) -> dict:
        return {"calls": self.calls, "input_tokens": self.input, "cache_write_tokens": self.cache_write,
                "cache_read_tokens": self.cache_read, "output_tokens": self.output,
                "cost_usd": round(self.dollars, 4)}

    def report(self) -> str:
        return (f"{self.calls} model calls; input {self.input:,} + cache write {self.cache_write:,} + "
                f"cache read {self.cache_read:,}; output {self.output:,} ≈ ${self.cost():.3f}")


# --------------------------------------------------------------------------- #
# The agent
# --------------------------------------------------------------------------- #

def to_param(block) -> dict:
    """A response content block as a plain, JSON-ready dict that can be sent back
    to the API unchanged (thinking blocks keep their signatures)."""
    if hasattr(block, "model_dump"):
        return block.model_dump(mode="json", exclude_none=True)
    return dict(vars(block))  # test doubles


@dataclass
class AskResult:
    answer: str  # the final reply's text ('' if declined)
    tool_calls: list[dict]  # [{"name", "input", "summary", "is_error"}]
    usage: SessionUsage  # this question only
    stop: str  # end_turn | declined | cut_off | step_limit


class Agent:
    """The hand-written tool loop.

    `messages` is the conversation so far, as JSON-ready dicts; pass a stored
    conversation to continue it. Each call to ask() appends to it (or leaves it
    untouched if the question fails). Events describing the work are passed to
    `on_event` as plain dicts, so the terminal and the HTTP API can render them
    their own way. `on_usage(usage, cost_usd)` is called after every model call.
    """

    def __init__(self, client: anthropic.Anthropic | None = None, tools: ToolRunner | None = None,
                 messages: list[dict] | None = None, on_usage: Callable | None = None):
        # Client and tools can be passed in, which is how the tests substitute fakes.
        self.client = client or anthropic.Anthropic()
        self.tools = tools or ToolRunner()
        self.messages: list[dict] = messages if messages is not None else []
        self.usage = SessionUsage()  # whole session
        self.on_usage = on_usage

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

    def _call_model(self, final_turn: bool, emit: Callable, usage: SessionUsage):
        """Stream one response, emitting events as it arrives; return the complete message."""
        kwargs = self._request_kwargs(final_turn)
        api = self.client.beta.messages if "betas" in kwargs else self.client.messages
        with api.stream(**kwargs) as stream:
            for event in stream:
                if event.type == "content_block_start":
                    emit({"type": "block_start", "block": event.content_block.type})
                elif event.type == "content_block_delta":
                    if event.delta.type == "thinking_delta" and event.delta.thinking:
                        # Reasoning summaries (display=summarized) or progress notes (updates).
                        emit({"type": "thinking", "text": event.delta.thinking})
                    elif event.delta.type == "text_delta":
                        emit({"type": "text", "text": event.delta.text})
                elif event.type == "content_block_stop":
                    emit({"type": "block_end"})
            message = stream.get_final_message()
        cost = usage_cost(message.usage, settings.agent_model)
        usage.add(message.usage)
        self.usage.add(message.usage)
        if self.on_usage:
            self.on_usage(message.usage, cost)
        return message

    # -- tools ---------------------------------------------------------------- #

    def _run_tools(self, message, emit: Callable, calls: list[dict]) -> list[dict]:
        """Run every tool_use block in the reply; return the matching tool_result blocks."""
        results = []
        for block in message.content:
            if block.type != "tool_use":
                continue
            emit({"type": "tool_call", "id": block.id, "name": block.name, "input": block.input})
            try:
                result = self.tools.run(block.name, block.input)
                content, summary, is_error = result.text, result.summary, False
            except ToolError as e:
                content = summary = str(e)
                is_error = True
            except Exception as e:  # a bug or outage in our code: report it to the model, keep going
                content = (f"The tool failed unexpectedly ({type(e).__name__}). "
                           "Try another approach or tell the user.")
                summary, is_error = f"failed: {type(e).__name__}: {e}", True
            emit({"type": "tool_result", "id": block.id, "name": block.name, "summary": summary,
                  "is_error": is_error})
            calls.append({"name": block.name, "input": block.input, "summary": summary, "is_error": is_error})
            item = {"type": "tool_result", "tool_use_id": block.id, "content": content}
            if is_error:
                item["is_error"] = True
            results.append(item)
        return results

    # -- one question --------------------------------------------------------- #

    def ask(self, question: str, on_event: Callable | None = None) -> AskResult:
        start = len(self.messages)
        try:
            return self._ask(question, on_event or (lambda event: None))
        except BaseException:
            # API error or Ctrl-C partway through: the history might now end with a
            # tool call that has no result, which the API would reject. Remove this
            # question's messages; earlier conversation is untouched.
            del self.messages[start:]
            raise

    def _ask(self, question: str, emit: Callable) -> AskResult:
        start = len(self.messages)
        usage = SessionUsage()  # this question
        calls: list[dict] = []
        self.messages.append({"role": "user", "content": question})

        def finish(stop: str, answer: str = "", notice: str | None = None) -> AskResult:
            if notice:
                emit({"type": "notice", "kind": stop, "message": notice})
            result = AskResult(answer, calls, usage, stop)
            emit({"type": "done", "stop": stop, "answer": answer, "usage": usage.as_dict()})
            return result

        for turn in range(1, settings.agent_max_turns + 1):
            final_turn = turn == settings.agent_max_turns
            message = self._call_model(final_turn, emit, usage)

            if message.stop_reason == "refusal":
                # Nothing usable came back (the fallback model declined too). Drop
                # this question from the history so the conversation stays valid.
                del self.messages[start:]
                category = getattr(getattr(message, "stop_details", None), "category", None)
                return finish("declined", notice=f"The model declined to answer this (category: {category}). "
                                                 "It's been removed from the conversation; try rephrasing.")

            # Keep the reply exactly as returned, thinking blocks included.
            self.messages.append({"role": "assistant", "content": [to_param(b) for b in message.content]})
            answer = "\n".join(b.text for b in message.content if b.type == "text").strip()

            if message.stop_reason == "tool_use":
                results = self._run_tools(message, emit, calls)
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
                return finish("cut_off", answer, f"Response cut off at max_tokens={MAX_TOKENS}.")
            if message.stop_reason == "pause_turn":
                continue  # the API paused a long turn; sending the history again resumes it
            return finish("end_turn", answer)  # the answer is complete

        return finish("step_limit", answer, f"Stopped after {settings.agent_max_turns} steps (AGENT_MAX_TURNS).")


# --------------------------------------------------------------------------- #
# REPL
# --------------------------------------------------------------------------- #

HELP = """Commands: /reset (start a new conversation), /usage (tokens and cost so far), /quit"""


def main() -> None:
    import db

    tools = ToolRunner()
    agent = Agent(tools=tools, on_usage=lambda u, cost: db.log_api_call(
        tools.conn, None, "agent", settings.agent_model, settings.agent_effort, "agent",
        u.input_tokens or 0, u.output_tokens or 0, cost_usd=cost,
        cache_read_tokens=getattr(u, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(u, "cache_creation_input_tokens", 0) or 0))
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
            agent.ask(question, on_event=TerminalRenderer())
        except KeyboardInterrupt:
            print("\n[Interrupted. That question was dropped; the rest of the conversation is kept.]")
        except anthropic.APIError as e:
            print(f"\n[API error: {e}. That question was dropped; try again.]")
    print(f"Session: {agent.usage.report()}")


if __name__ == "__main__":
    main()
