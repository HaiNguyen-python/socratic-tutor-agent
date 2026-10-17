"""The exercise bank and the answer checker.

The bank is a YAML file shipped inside the package. Each exercise has a final answer, worked
steps and a hint ladder. The tutor never shows the answer or the steps; they exist so the tools
can check answers and the guardrails can recognise a leak.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from fractions import Fraction
from importlib import resources
from pathlib import Path

import yaml

from .numbers import equivalent, find_numbers, parse_value

TOPICS = ("linear-equations", "fractions", "word-problems", "grammar")
# Words students put around a short answer: "is it went?", "I think it's 'written'".
_FILLER = re.compile(
    r"^(?:(?:is it|it is|its|i think|i guess|maybe|the answer is|answer|is the answer)\s+)+"
)


@dataclass(frozen=True)
class Exercise:
    id: str
    topic: str
    kind: str  # "numeric" or "text"
    question: str
    answer: str
    steps: tuple[str, ...]
    hints: tuple[str, ...]
    errors: tuple[str, ...] = ()

    @property
    def value(self) -> Fraction:
        """The numeric answer as an exact fraction (numeric exercises only)."""
        return parse_value(self.answer)

    @property
    def n_hints(self) -> int:
        return len(self.hints)

    def completed_sentence(self) -> str:
        """The gap-fill question with the answer written in (text exercises)."""
        sentence = self.question.split(":", 1)[-1].strip()
        return sentence.replace("___", self.answer)


def normalise_text(text: str) -> str:
    """Lower-case, straight quotes, no punctuation, single spaces."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"[^\w\s'-]", " ", text)
    text = text.replace("'", "")  # "it's" -> "its", "'went'" -> "went"
    return re.sub(r"\s+", " ", text).strip()


def check(exercise: Exercise, attempt: str) -> bool:
    """Is the student's attempt correct?

    Numeric: the *last* number in the attempt must equal the answer ("2x = 8 so x = 4" is
    right, "4 or 5?" is not). Text: after removing filler words the attempt must be the answer
    itself, or the whole sentence with the gap filled in.
    """
    if exercise.kind == "numeric":
        tokens = find_numbers(attempt)
        return bool(tokens) and equivalent(tokens[-1], exercise.value)
    cleaned = _FILLER.sub("", normalise_text(attempt))
    accepted = {normalise_text(exercise.answer), normalise_text(exercise.completed_sentence())}
    return cleaned in accepted


def _from_dict(row: dict) -> Exercise:
    return Exercise(
        id=row["id"],
        topic=row["topic"],
        kind=row["kind"],
        question=" ".join(str(row["question"]).split()),
        answer=str(row["answer"]),
        steps=tuple(row["steps"]),
        hints=tuple(row["hints"]),
        errors=tuple(row.get("errors", ())),
    )


class Bank:
    """Exercises by id, in file order."""

    def __init__(self, exercises: list[Exercise]):
        self.exercises = exercises
        self._by_id = {e.id: e for e in exercises}

    def __len__(self) -> int:
        return len(self.exercises)

    def __contains__(self, exercise_id: object) -> bool:
        return exercise_id in self._by_id

    def get(self, exercise_id: str) -> Exercise:
        try:
            return self._by_id[exercise_id]
        except KeyError:
            raise KeyError(f"unknown exercise {exercise_id!r}") from None

    def by_topic(self, topic: str) -> list[Exercise]:
        return [e for e in self.exercises if e.topic == topic]


def load_bank(path: Path | None = None) -> Bank:
    """Load the packaged bank, or a YAML file with the same shape."""
    if path is None:
        text = resources.files("tutor_agent").joinpath("data/exercises.yaml").read_text("utf-8")
    else:
        text = path.read_text("utf-8")
    return Bank([_from_dict(row) for row in yaml.safe_load(text)])
