"""The four tools the tutor model can call, and the per-session state they read and write.

Tools are plain functions that return JSON-serialisable dicts. The model asks for a tool with
`{"tool": name, "args": {...}}`; `run_tool` validates the name and arguments and never raises,
so a confused model gets an error message back instead of crashing the session.

The model never receives the final answer or the worked steps from any tool. It can only
check an answer the student gave, or read a hint the hint cap allows.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .bm25 import BM25Index
from .exercises import Bank, Exercise, check
from .guardrails.hints import decide

TOOL_SPECS = """\
get_hint(exercise_id: str, level: int) -> {"level", "hint", "capped"}: hint ladder, 1 = gentlest
check_answer(exercise_id: str, answer: str) -> {"correct", "attempts"}: check the student's answer
make_quiz(topic: str, n: int) -> {"questions": [...]}: practice questions on a topic
search_notes(query: str) -> {"hits": [{"doc", "heading", "text"}]}: search the course notes"""


@dataclass
class SessionState:
    """What the tools know about one tutoring session on one exercise."""

    exercise: Exercise
    hint_cap: bool = True
    attempts: int = 0
    highest_hint: int = 0
    solved: bool = False


@dataclass
class ToolBox:
    bank: Bank
    notes: BM25Index

    def get_hint(self, state: SessionState, exercise_id: str, level: int) -> dict[str, Any]:
        ex = self._current(state, exercise_id)
        level = int(level)  # models sometimes send "2"
        if state.hint_cap:
            decision = decide(level, state.attempts, state.highest_hint, ex.n_hints)
            granted, capped = decision.granted, decision.capped
        else:
            granted, capped = max(1, min(level, ex.n_hints)), False
        state.highest_hint = max(state.highest_hint, granted)
        return {
            "level": granted,
            "levels": ex.n_hints,
            "hint": ex.hints[granted - 1],
            "capped": capped,
        }

    def check_answer(self, state: SessionState, exercise_id: str, answer: str) -> dict[str, Any]:
        ex = self._current(state, exercise_id)
        correct = check(ex, answer)
        if not state.solved:
            state.attempts += 1
        state.solved = state.solved or correct
        return {"correct": correct, "attempts": state.attempts}

    def make_quiz(self, state: SessionState, topic: str, n: int = 3) -> dict[str, Any]:
        """Other exercises on the topic (never the current one), in bank order."""
        pool = [e for e in self.bank.by_topic(topic) if e.id != state.exercise.id]
        if not pool:
            return {"error": f"no exercises for topic {topic!r}"}
        n = max(1, min(int(n), 5))
        return {"questions": [{"id": e.id, "question": e.question} for e in pool[:n]]}

    def search_notes(self, state: SessionState, query: str) -> dict[str, Any]:
        hits = self.notes.search(query, k=2)
        return {
            "hits": [
                {"doc": h.section.doc, "heading": h.section.heading, "text": h.section.text}
                for h in hits
            ]
        }

    def _current(self, state: SessionState, exercise_id: str) -> Exercise:
        # Only the current exercise: the model must not fetch hints for other items.
        if exercise_id != state.exercise.id:
            raise ValueError(f"only the current exercise ({state.exercise.id}) is available")
        return state.exercise


def run_tool(box: ToolBox, state: SessionState, name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Dispatch one tool call; bad names or arguments come back as {"error": ...}."""
    handlers: dict[str, Callable[..., dict[str, Any]]] = {
        "get_hint": box.get_hint,
        "check_answer": box.check_answer,
        "make_quiz": box.make_quiz,
        "search_notes": box.search_notes,
    }
    if name not in handlers:
        return {"error": f"unknown tool {name!r}; available: {', '.join(handlers)}"}
    try:
        return handlers[name](state, **args)
    except (TypeError, ValueError) as exc:
        return {"error": str(exc)}
