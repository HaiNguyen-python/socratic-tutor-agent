"""Answer-leak detector: does a tutor reply give away the answer before the student solved it?

Three ways a reply can leak:

1. **numeric**: it contains a value equal to the answer in any form (4, 4.0, 8/2, four;
   1/2, 0.5, 2/4, \\frac{1}{2}; 0.33 for 1/3). See `numbers.py`.
2. **text**: for grammar items, the answer word appears as a whole word ("went").
3. **steps**: it repeats `min_steps` or more of the worked solution steps verbatim
   (case and spacing ignored), which hands over the method even without the final value.

The bank guarantees that answers never appear in questions or hints, so a hit here means the
reply added the answer itself. Once the student has solved the exercise nothing is a leak.

What it cannot see: an answer hidden in arithmetic ("subtract 3 from 7") or spelled out
letter by letter ("w-e-n-t"). The eval harness plants such leaks on purpose to measure this.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..exercises import Exercise, normalise_text
from ..numbers import contains_value


@dataclass(frozen=True)
class LeakFinding:
    leaked: bool
    reason: str = ""  # "numeric", "text" or "steps"
    evidence: str = ""


NO_LEAK = LeakFinding(False)


def _step_matches(reply: str, steps: tuple[str, ...]) -> list[str]:
    norm_reply = normalise_text(reply)
    return [s for s in steps if normalise_text(s) and normalise_text(s) in norm_reply]


def _text_leak(reply: str, answer: str) -> bool:
    pattern = r"\b" + re.escape(normalise_text(answer)) + r"\b"
    return re.search(pattern, normalise_text(reply)) is not None


def detect_leak(
    reply: str, exercise: Exercise, solved: bool = False, min_steps: int = 2
) -> LeakFinding:
    """Check one reply. `min_steps` is N in "N worked steps verbatim count as a leak"."""
    if solved:
        return NO_LEAK
    matched = _step_matches(reply, exercise.steps)
    if len(matched) >= min(min_steps, len(exercise.steps)):
        return LeakFinding(True, "steps", " | ".join(matched))
    if exercise.kind == "numeric":
        token = contains_value(reply, exercise.value)
        if token is not None:
            return LeakFinding(True, "numeric", token.text)
    elif _text_leak(reply, exercise.answer):
        return LeakFinding(True, "text", exercise.answer)
    return NO_LEAK
