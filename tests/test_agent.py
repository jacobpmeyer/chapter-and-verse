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
    result = agent.ask("What is the answer?", on_event=A.TerminalRenderer())
    assert (result.answer, result.stop, result.tool_calls) == ("Forty-two.", "end_turn", [])
    assert [m["role"] for m in agent.messages] == ["user", "assistant"]
    assert "Forty-two." in capsys.readouterr().out  # the terminal renderer prints the answer
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
    stored = agent.messages[1]["content"]
    assert stored == [A.to_param(b) for b in reply.content]  # every block, thinking included
    assert stored[0] == {"type": "thinking", "thinking": "signed reasoning"}


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
    result = agent.ask("q", on_event=A.TerminalRenderer())
    assert len(api.requests) == 1
    assert result.stop == "step_limit"
    assert "Stopped after 1 steps" in capsys.readouterr().out


def test_max_tokens_closes_out_unfinished_tool_calls(capsys):
    agent, _, tools = make_agent([msg("max_tokens", text("Let me look"), tool("t1", "search_passages", query="x"))])
    assert agent.ask("q", on_event=A.TerminalRenderer()).stop == "cut_off"
    assert tools.calls == []  # a cut-off tool call is never run
    closing = agent.messages[-1]
    assert closing["role"] == "user"
    assert closing["content"][0]["tool_use_id"] == "t1" and closing["content"][0]["is_error"]
    assert "cut off" in capsys.readouterr().out


def test_refusal_removes_only_that_question(capsys):
    agent, _, _ = make_agent([msg("end_turn", text("Fine.")), msg("refusal", category="cyber")])
    agent.ask("first")
    assert agent.ask("second", on_event=A.TerminalRenderer()).stop == "declined"
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
    assert agent.messages[-1]["content"][0]["text"] == "done"


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


def test_events_describe_the_work_in_order():
    agent, _, _ = make_agent([msg("tool_use", thinking("plan"), tool("t1", "list_books", query="x")),
                              msg("end_turn", text("Answer."))])
    events = []
    result = agent.ask("q", on_event=events.append)
    kinds = [e["type"] for e in events]
    assert kinds == ["block_start", "thinking", "block_end", "block_start", "block_end",  # first reply
                     "tool_call", "tool_result",
                     "block_start", "text", "block_end",  # second reply
                     "done"]
    assert events[5] == {"type": "tool_call", "id": "t1", "name": "list_books", "input": {"query": "x"}}
    assert events[6]["summary"] == "list_books ok" and events[6]["is_error"] is False
    assert events[-1]["answer"] == "Answer." and events[-1]["usage"]["calls"] == 2
    assert result.tool_calls == [{"name": "list_books", "input": {"query": "x"}, "summary": "list_books ok",
                                  "is_error": False}]


def test_usage_is_reported_after_every_model_call():
    seen = []
    client = FakeClient([msg("tool_use", tool("t1", "list_books")), msg("end_turn", text("ok"))])
    agent = A.Agent(client=client, tools=FakeTools(), on_usage=lambda u, cost: seen.append(cost))
    result = agent.ask("q")
    assert len(seen) == 2 and all(c > 0 for c in seen)
    assert result.usage.cost() == pytest.approx(sum(seen))


def test_stored_history_round_trips_through_json():
    import json

    agent, _, _ = make_agent([msg("tool_use", thinking("sig"), tool("t1", "list_books")), msg("end_turn", text("one"))])
    agent.ask("first")
    stored = json.loads(json.dumps(agent.messages))  # what the API saves and reloads
    assert stored == agent.messages

    client = FakeClient([msg("end_turn", text("two"))])
    resumed = A.Agent(client=client, tools=FakeTools(), messages=stored)
    resumed.ask("second")
    sent = client.messages.requests[0]["messages"]
    assert sent[:4] == agent.messages  # earlier turns go back exactly as stored
    assert sent[4] == {"role": "user", "content": "second"}


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
