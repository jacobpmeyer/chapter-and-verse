"""Local token estimation.

Used for chunk sizing and --dry-run estimates, where calling an API is not
allowed. Claude's tokenizer is not available offline, so this is a
character-based approximation (~3.8 chars/token for English prose). Phase 2
uses the API's count_tokens endpoint when an exact number matters.
"""

CHARS_PER_TOKEN = 3.8


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, round(len(text) / CHARS_PER_TOKEN))
