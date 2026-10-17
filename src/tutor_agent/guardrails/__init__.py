"""Guardrails: each check is a small pure module so it can be tested on its own.

- `injection`: prompt-injection detector for student messages (forces a safe refusal).
- `offtopic`: off-topic filter for student messages (redirects to the exercise).
- `hints`: hint-level cap (no jumping to the last hint without attempts).
- `leak`: answer-leak detector, used as an output filter on every model reply.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailConfig:
    """Which guardrails are on. The eval harness compares `all_on()` with `all_off()`."""

    injection: bool = True
    off_topic: bool = True
    hint_cap: bool = True
    output_filter: bool = True

    @classmethod
    def all_on(cls) -> GuardrailConfig:
        return cls()

    @classmethod
    def all_off(cls) -> GuardrailConfig:
        return cls(injection=False, off_topic=False, hint_cap=False, output_filter=False)

    @property
    def label(self) -> str:
        flags = (self.injection, self.off_topic, self.hint_cap, self.output_filter)
        return "on" if all(flags) else "off" if not any(flags) else "partial"
