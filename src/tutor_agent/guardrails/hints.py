"""Hint-level cap: hints must be earned with attempts and given one level at a time.

A student who types "give me the last hint" on turn one would otherwise get the step just
before the answer, which is a leak in slow motion. The rule:

    allowed = min(requested, levels, attempts + 1, highest_given + 1)

Level 1 is always free. Each attempt unlocks one more level, and no level can be skipped.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HintDecision:
    requested: int
    granted: int

    @property
    def capped(self) -> bool:
        return self.granted < self.requested


def allowed_hint_level(requested: int, attempts: int, highest_given: int, levels: int) -> int:
    """The highest hint level the student may see now (always at least 1)."""
    return max(1, min(requested, levels, attempts + 1, highest_given + 1))


def decide(requested: int, attempts: int, highest_given: int, levels: int) -> HintDecision:
    requested = max(1, min(requested, levels))
    return HintDecision(requested, allowed_hint_level(requested, attempts, highest_given, levels))
