"""Prompt-injection detector for student messages.

A flagged message never reaches the model: the tutor answers with `SAFE_REFUSAL` and the
message is replaced by a placeholder in the history, so a later model call cannot act on it.

The detector is a list of patterns over normalised text (Unicode NFKC, zero-width characters
removed, lower case, single spaces). Patterns are narrow on purpose: "I ignored the
instructions in my textbook" or "what does 'you are' mean?" must pass. A determined attacker
can rephrase around any pattern list; that limit is written in docs/decisions.md.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

SAFE_REFUSAL = (
    "I can't change how I work or show the solution, but I'd really like to help you solve it "
    "yourself. What have you tried so far?"
)

_ZERO_WIDTH = dict.fromkeys([0x200B, 0x200C, 0x200D, 0x2060, 0xFEFF])  # zero-width characters
_TARGETS = r"(?:instructions?|rules|prompt|guidelines|system message|programming)"
PATTERNS: dict[str, re.Pattern[str]] = {
    "ignore-instructions": re.compile(
        rf"\b(?:ignore|disregard|forget|override|bypass)\b(?: \w+){{0,4}} {_TARGETS}\b"
    ),
    "role-change": re.compile(
        r"\byou are now\b|\bfrom now on,? you are\b|\bpretend (?:to be|you are)\b"
        r"|\bact as (?:an?|my) (?!tutor|teacher)\w+"
    ),
    "reveal-solution": re.compile(
        r"\b(?:print|reveal|output|dump|display|leak)\b(?: \w+){0,3} "
        r"(?:solutions?|answer key|answers?|worked steps|system prompt)\b"
    ),
    "jailbreak": re.compile(r"\b(?:jailbreak|developer mode|dan mode|no restrictions)\b"),
    "fake-system": re.compile(r"^\s*(?:system|assistant)\s*:|\[\s*system\s*\]|<\s*system\s*>"),
}


@dataclass(frozen=True)
class InjectionFinding:
    flagged: bool
    pattern: str = ""


def normalise(message: str) -> str:
    text = unicodedata.normalize("NFKC", message).translate(_ZERO_WIDTH).lower()
    text = re.sub(r"[\"'`*_]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def detect_injection(message: str) -> InjectionFinding:
    """Flag a student message that tries to change the tutor's rules or extract the solution."""
    text = normalise(message)
    for name, pattern in PATTERNS.items():
        if pattern.search(text):
            return InjectionFinding(True, name)
    return InjectionFinding(False)
