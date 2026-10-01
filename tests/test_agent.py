"""agent.py: the hand-written tool loop and the terminal printer.

A fake Anthropic client replays scripted responses and records every request,
so the tests can check exactly what the loop sends and keeps in its history.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import random
from types import SimpleNamespace as NS

import pytest

import agent as A
from tests.fakes import FakeClient, msg, text, thinking, tool
from tools import ToolError, ToolResult

# --------------------------------------------------------------------------- #
# Fakes (the Anthropic client fakes live in tests/fakes.py)
# --------------------------------------------------------------------------- #


class FakeTools:
    def __init__(self, behavior=None):
        self.calls = []
        self.behavior = behavior or {}

    def run(self, name, args):
        self.calls.append((name, args))
        outcome = self.behavior.get(name)
        if isinstance(outcome, BaseException):
            raise outcome
        return ToolResult(f"{name} result for {args}", f"{name} ok")


def make_agent(script, behavior=None):
    client = FakeClient(script)
    tools = FakeTools(behavior)
    return A.Agent(client=client, tools=tools), client.messages, tools


@pytest.fixture
def max_turns(monkeypatch):
    def set_max(n):
        monkeypatch.setattr(A, "settings", dataclasses.replace(A.settings, agent_max_turns=n))
    return set_max


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #

def test_answer_without_tools(capsys):
    agent, api, _ = make_agent([msg("end_turn", thinking(), text("Forty-two."))])
    agent.ask("What is the answer?")
    assert [m["role"] for m in agent.messages] == ["user", "assistant"]
    assert "Forty-two." in capsys.readouterr().out
    assert len(api.requests) == 1


def test_every_tool_call_runs_and_results_go_back_in_one_message():
    agent, api, tools = make_agent([
        msg("tool_use", thinking(), tool("t1", "list_books", query="hendrix"),
            tool("t2", "search_passages", query="the eels", book_id=10)),
        msg("end_turn", text("Chapter 22.")),
    ])
    agent.ask("Where are the eels?")

    assert tools.calls == [("list_books", {"query": "hendrix"}),
                           ("search_passages", {"query": "the eels", "book_id": 10})]
    results = agent.messages[2]
    assert results["role"] == "user"
    assert [r["tool_use_id"] for r in results["content"]] == ["t1", "t2"]
    assert all(r["type"] == "tool_result" and "is_error" not in r for r in results["content"])
    assert api.requests[1]["messages"][-1] is results  # sent back on the next call
    assert len(agent.messages) == 4


def test_replies_are_kept_exactly_as_returned():
    reply = msg("tool_use", thinking("signed reasoning"), tool("t1", "list_books"))
    agent, _, _ = make_agent([reply, msg("end_turn", text("ok"))])
    agent.ask("q")
    assert agent.messages[1]["content"] is reply.content  # thinking blocks untouched


def test_history_is_append_only_across_calls_and_questions():
    agent, api, _ = make_agent([
        msg("tool_use", tool("a", "list_books")), msg("end_turn", text("one")),
        msg("tool_use", tool("b", "list_books")), msg("tool_use", tool("c", "list_books")),
        msg("end_turn", text("two")),
    ])
    agent.ask("first")
    agent.ask("second")
    snapshots = [r["messages"] for r in api.requests]
    for earlier, later in zip(snapshots, snapshots[1:]):
        assert len(later) > len(earlier)
        assert all(x is y for x, y in zip(earlier, later)), "an earlier message was changed or replaced"


def test_tool_errors_are_reported_to_the_model():
    agent, _, _ = make_agent(
        [msg("tool_use", tool("t1", "load_full_book", book_id=10), tool("t2", "get_book_summary", book_id=7)),
         msg("end_turn", text("Used summaries instead."))],
        behavior={"load_full_book": ToolError("over the 150,000-token limit"),
                  "get_book_summary": RuntimeError("database went away")},
    )
    agent.ask("Load the whole book")
    r1, r2 = agent.messages[2]["content"]
    assert r1["is_error"] and "150,000-token limit" in r1["content"]
    assert r2["is_error"] and "failed unexpectedly (RuntimeError)" in r2["content"]


def test_last_allowed_step_turns_tools_off_and_says_so(max_turns):
    max_turns(2)
    agent, api, _ = make_agent([msg("tool_use", tool("t1", "list_books")),
                                msg("end_turn", text("Here's what I found."))])
    agent.ask("q")
    assert "tool_choice" not in api.requests[0]
    assert api.requests[1]["tool_choice"] == {"type": "none"}
    note = agent.messages[2]["content"][-1]
    assert note["type"] == "text" and "Tool-call limit reached" in note["text"]


def test_running_out_of_steps_is_reported(max_turns, capsys):
    max_turns(1)
    # The model shouldn't ask for tools when they're off, but if a reply still
    # isn't final the loop must stop rather than call again.
    agent, api, _ = make_agent([msg("pause_turn", text("partial"))])
    agent.ask("q")
    assert len(api.requests) == 1
    assert "Stopped after 1 steps" in capsys.readouterr().out


def test_max_tokens_closes_out_unfinished_tool_calls(capsys):
    agent, _, tools = make_agent([msg("max_tokens", text("Let me look"), tool("t1", "search_passages", query="x"))])
    agent.ask("q")
    assert tools.calls == []  # a cut-off tool call is never run
    closing = agent.messages[-1]
    assert closing["role"] == "user"
    assert closing["content"][0]["tool_use_id"] == "t1" and closing["content"][0]["is_error"]
    assert "cut off" in capsys.readouterr().out


def test_refusal_removes_only_that_question(capsys):
    agent, _, _ = make_agent([msg("end_turn", text("Fine.")), msg("refusal", category="cyber")])
    agent.ask("first")
    agent.ask("second")
    assert [m["role"] for m in agent.messages] == ["user", "assistant"]
    assert agent.messages[0]["content"] == "first"
    assert "declined" in capsys.readouterr().out


def test_an_error_mid_question_rolls_back_only_that_question():
    agent, _, _ = make_agent([msg("end_turn", text("Fine.")),
                              msg("tool_use", tool("t1", "list_books")), ConnectionError("network down")])
    agent.ask("first")
    with pytest.raises(ConnectionError):
        agent.ask("second")
    assert len(agent.messages) == 2  # no dangling tool call left behind


def test_pause_turn_resumes():
    agent, api, _ = make_agent([msg("pause_turn", text("thinking...")), msg("end_turn", text("done"))])
    agent.ask("q")
    assert len(api.requests) == 2
    assert agent.messages[-1]["content"][0].text == "done"


def test_request_shape():
    agent, api, _ = make_agent([msg("end_turn", text("ok"))])
    agent.ask("q")
    req = api.requests[0]
    assert req["model"] == A.settings.agent_model
    assert req["system"] == A.SYSTEM_PROMPT
    assert req["tools"] is A.TOOLS
    assert req["cache_control"] == {"type": "ephemeral"}
    assert req["thinking"]["type"] == "adaptive"
    assert req["output_config"] == {"effort": A.settings.agent_effort}


def test_session_usage_counts_cache_at_discounted_rates():
    u = A.SessionUsage()
    u.add(NS(input_tokens=1_000_000, output_tokens=0, cache_creation_input_tokens=0, cache_read_input_tokens=0))
    full = u.cost()
    u = A.SessionUsage()
    u.add(NS(input_tokens=0, output_tokens=0, cache_creation_input_tokens=0, cache_read_input_tokens=1_000_000))
    assert u.cost() == pytest.approx(full * A.CACHE_READ_MULTIPLIER)


# --------------------------------------------------------------------------- #
# Terminal printer
# --------------------------------------------------------------------------- #

SAMPLE = ("The book is about a 1970 home for unwed mothers, and nearly every theme comes back to who gets to "
          "control the girls' bodies, names and choices. The main ones:\n\n\n1. Institutions controlling "
          "girls' bodies. Wellwood House treats pregnancy as both a sin and an illness (Witchcraft for Wayward "
          "Girls, \"Chapter 5 (May 1970: 26 WEEKS)\").\n\n2. Shame and erasure.   The girls lose their names.\n\n")


def printed(fragments, width=60, style=""):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        p = A.StreamPrinter()
        p.width = width
        for f in fragments:
            p.write(f, style=style)
        p.end_block()
    return buf.getvalue()


@pytest.mark.parametrize("seed", range(100))
def test_printer_wraps_words_however_the_stream_is_split(seed):
    rng = random.Random(seed)
    cuts = sorted(rng.sample(range(1, len(SAMPLE)), 30))
    out = printed([SAMPLE[i:j] for i, j in zip([0, *cuts], [*cuts, len(SAMPLE)])])
    lines = out.split("\n")
    assert all(len(line) <= 60 for line in lines)
    assert not any(line.endswith(" ") for line in lines)
    assert out.split() == SAMPLE.split(), "no word added, lost, or split"
    assert "\n\n\n" not in out, "at most one blank line in a row"
    assert out.endswith("\n") and not out.endswith("\n\n"), "a block ends with exactly one newline"


def test_printer_never_breaks_a_long_word():
    out = printed(["short " + "x" * 80 + " end"], width=40)
    assert "x" * 80 in out
