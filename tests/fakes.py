"""A fake Anthropic client that replays scripted responses and records every
request. Used by the agent tests and by the end-to-end pipeline test."""

from __future__ import annotations

from types import SimpleNamespace as NS


def text(t):
    return NS(type="text", text=t)


def thinking(t=""):
    return NS(type="thinking", thinking=t)


def tool(id_, name, **args):
    return NS(type="tool_use", id=id_, name=name, input=args)


def msg(stop_reason, *blocks, category=None):
    usage = NS(input_tokens=10, output_tokens=5, cache_creation_input_tokens=100, cache_read_input_tokens=1000)
    details = NS(category=category) if stop_reason == "refusal" else None
    return NS(stop_reason=stop_reason, content=list(blocks), usage=usage, stop_details=details,
              model="fake-model")


class FakeStream:
    """Mimics the SDK's MessageStream: a context manager yielding events, then the final message."""

    def __init__(self, message):
        self.message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for block in self.message.content:
            yield NS(type="content_block_start", content_block=NS(type=block.type))
            if block.type == "text":
                yield NS(type="content_block_delta", delta=NS(type="text_delta", text=block.text))
            elif block.type == "thinking" and block.thinking:
                yield NS(type="content_block_delta", delta=NS(type="thinking_delta", thinking=block.thinking))
            yield NS(type="content_block_stop")

    def get_final_message(self):
        return self.message


class FakeMessages:
    def __init__(self, script):
        self.script = list(script)
        self.requests = []  # kwargs of every call, with a snapshot of `messages`

    def stream(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        nxt = self.script.pop(0)
        if isinstance(nxt, BaseException):
            raise nxt
        return FakeStream(nxt)


class FakeClient:
    def __init__(self, script):
        self.messages = FakeMessages(script)
        self.beta = NS(messages=self.messages)  # the loop uses beta when beta features are on
