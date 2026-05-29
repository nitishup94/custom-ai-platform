"""Heuristic category labels for OpenOrca-style question/answer rows."""

from __future__ import annotations

import re
from typing import Final

_PROGRAMMING_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(python|php|javascript|typescript|sql|html|css|java|code|programming|"
    r"debug|compiler|function|class|import|npm|node|react|api|json|xml|"
    r"git|github|stack\s*overflow)\b",
    re.IGNORECASE,
)
_CONVERSATION_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"^\s*(hello|hi|hey|how\s+are\s+you|good\s+morning|good\s+evening|"
    r"thanks|thank\s+you|bye|goodbye)\b",
    re.IGNORECASE,
)
_MATH_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"[\d]+\s*[\+\-\*/^%]\s*[\d]+|"
    r"\b(math|calculate|calculus|algebra|equation|integral|derivative|"
    r"matrix|probability|statistics|geometry)\b",
    re.IGNORECASE,
)
_SCIENCE_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b(physics|biology|chemistry|molecule|atom|experiment|"
    r"newton|einstein|dna|cell|photosynthesis)\b",
    re.IGNORECASE,
)


def detect_category(question: str, response: str) -> str:
    """Return a coarse category slug derived from *question* and *response*."""
    blob = f"{question}\n{response}".strip()
    if not blob:
        return "general"
    head = blob.splitlines()[0][:400]
    if _CONVERSATION_PATTERN.search(head) and len(blob) < 200:
        return "conversation"
    if _PROGRAMMING_PATTERN.search(blob):
        return "programming"
    if _SCIENCE_PATTERN.search(blob):
        return "science"
    if _MATH_PATTERN.search(blob):
        return "math"
    return "general"
