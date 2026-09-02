from __future__ import annotations

from src.reason.generate import ABSTAIN_TEXT
from src.schemas.answer import Answer

QUERY_TOO_LARGE_TEXT = "That question is outside what this system accepts — please ask something shorter."
UNGROUNDED_DECLINE_TEXT = "I can't confirm this answer is well-supported by the retrieved papers, so I'm declining rather than risk giving you an unsupported claim."


def query_too_large() -> Answer:
    return Answer(text=QUERY_TOO_LARGE_TEXT, citations=[], abstained=True)


def out_of_scope() -> Answer:
    return Answer(text=ABSTAIN_TEXT, citations=[], abstained=True)


def ungrounded() -> Answer:
    return Answer(text=UNGROUNDED_DECLINE_TEXT, citations=[], abstained=True)
