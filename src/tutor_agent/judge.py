"""Judging Socratic quality: a written rubric, two judges, and agreement with hand labels.

- `RUBRIC` is the single source of truth. The LLM judge receives it word for word, and the
  hand labels in evals/human_labels.yaml were written against it.
- `DeterministicJudge` is a rule-based *stand-in* used in CI. Its "no leak" check reuses the
  leak detector, so it inherits the detector's blind spots (disguised leaks). That is exactly
  why it is compared with hand labels instead of being trusted.
- `LLMJudge` sends the rubric and a transcript to any `Provider` and parses a JSON verdict.

Agreement is Cohen's kappa: observed agreement corrected for the agreement two raters would
reach by chance with the same label frequencies.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .exercises import Bank
from .guardrails.leak import detect_leak
from .numbers import find_numbers
from .providers import Provider

CRITERIA = ("asks_questions", "proportional_hints", "encouraging", "no_leak")
RUBRIC = """\
Judge the TUTOR in this tutoring transcript. Score each criterion 1 (yes) or 0 (no).

1. asks_questions: most tutor replies (at least half) invite the student to think, usually by
   ending with a question, instead of only telling.
2. proportional_hints: hints grow one level at a time and only after the student has tried.
   Jumping to the biggest hint, or to the method, before any attempt scores 0.
3. encouraging: the tone is warm and patient; wrong answers get constructive feedback, never
   blame. A redirect to the task is fine if it is polite.
4. no_leak: the tutor never gives the final answer or the worked solution before the student
   has solved it, in any form (another number format, the filled-in sentence, arithmetic
   that produces it, spelling it out, or the solution steps).

Overall label: "good" if no_leak = 1, proportional_hints = 1, and at least one of
asks_questions and encouraging is 1; otherwise "poor". (Leaking the answer and handing out
the biggest hint up front both stop the student from solving it themselves.)
"""
_HINT_LEVEL = re.compile(r"level (\d+) of (\d+)")
_ENCOURAGING = re.compile(
    r"well done|good effort|good try|great|nice|you can get there|let's|you're close|"
    r"keep going|happy to help|like to help",
    re.IGNORECASE,
)
_DISCOURAGING = re.compile(r"\bwrong!|\bobviously\b|\bstupid\b|\bcareless\b|\blazy\b", re.I)


@dataclass(frozen=True)
class Verdict:
    scores: dict[str, bool]
    label: str  # "good", "poor" or "unparsed"
    reason: str = ""


def overall_label(scores: dict[str, bool]) -> str:
    required = scores["no_leak"] and scores["proportional_hints"]
    return "good" if required and (scores["asks_questions"] or scores["encouraging"]) else "poor"


def split_transcript(transcript: str) -> tuple[str, list[tuple[str, str]]]:
    """('lin-01', [(speaker, text), ...]) from the plain-text transcript format."""
    lines = transcript.splitlines()
    exercise_id = lines[0].removeprefix("Exercise ").strip()
    turns = []
    for line in lines[1:]:
        speaker, _, text = line.partition(": ")
        turns.append((speaker, text))
    return exercise_id, turns


class DeterministicJudge:
    """Rules that approximate the rubric from the transcript text alone."""

    name = "deterministic"

    def __init__(self, bank: Bank):
        self.bank = bank

    def judge(self, transcript: str) -> Verdict:
        exercise_id, turns = split_transcript(transcript)
        ex = self.bank.get(exercise_id)
        tutor = [text for speaker, text in turns if speaker == "Tutor"]
        asks = sum("?" in t for t in tutor) >= len(tutor) / 2 if tutor else False

        proportional, attempts, highest = True, 0, 0
        for speaker, text in turns:
            if speaker == "Student" and self._is_attempt(ex.kind, text):
                attempts += 1
            match = _HINT_LEVEL.search(text) if speaker == "Tutor" else None
            if match:
                level = int(match.group(1))
                if level > attempts + 1 or level > highest + 1:
                    proportional = False
                highest = max(highest, level)

        encouraging = any(_ENCOURAGING.search(t) for t in tutor) and not any(
            _DISCOURAGING.search(t) for t in tutor
        )
        leaks = [detect_leak(t, ex) for t in tutor]
        scores = {
            "asks_questions": asks,
            "proportional_hints": proportional,
            "encouraging": encouraging,
            "no_leak": not any(f.leaked for f in leaks),
        }
        failed = [c for c in CRITERIA if not scores[c]]
        return Verdict(
            scores, overall_label(scores), "failed: " + ", ".join(failed) if failed else ""
        )

    @staticmethod
    def _is_attempt(kind: str, text: str) -> bool:
        if kind == "numeric":
            return bool(find_numbers(text))
        return text.lower().startswith("is it") or len(text.split()) <= 8 and "?" not in text


class LLMJudge:
    """Ask a model to apply the rubric. Any provider works; temperature 0 where supported."""

    def __init__(self, provider: Provider):
        self.provider = provider
        self.name = f"llm:{provider.name}"

    def judge(self, transcript: str) -> Verdict:
        prompt = (
            f"{RUBRIC}\nAnswer with only JSON like "
            '{"asks_questions": 1, "proportional_hints": 1, "encouraging": 1, "no_leak": 1, '
            '"reason": "one sentence"}.\n\nTranscript:\n'
            f"{transcript}"
        )
        messages = [
            {"role": "system", "content": "You are a strict, fair grader of tutoring sessions."},
            {"role": "user", "content": prompt},
        ]
        return parse_verdict(self.provider.complete(messages).text)


def parse_verdict(text: str) -> Verdict:
    """Read the judge's JSON; anything unusable becomes an 'unparsed' verdict."""
    start, end = text.find("{"), text.rfind("}")
    try:
        data: Any = json.loads(text[start : end + 1]) if start != -1 else None
    except json.JSONDecodeError:
        data = None
    if not isinstance(data, dict) or not all(c in data for c in CRITERIA):
        return Verdict({c: False for c in CRITERIA}, "unparsed", text[:200])
    scores = {c: str(data[c]).strip().lower() in ("1", "true", "yes") for c in CRITERIA}
    return Verdict(scores, overall_label(scores), str(data.get("reason", "")))


@dataclass(frozen=True)
class HumanLabel:
    conv_id: str
    condition: str
    transcript_hash: str
    scores: dict[str, bool]
    label: str
    reason: str


def load_labels(path: Path) -> list[HumanLabel]:
    rows = yaml.safe_load(path.read_text("utf-8")) or []
    return [
        HumanLabel(
            conv_id=row["conv_id"],
            condition=row["condition"],
            transcript_hash=row["transcript_hash"],
            scores={c: bool(row[c]) for c in CRITERIA},
            label=row["label"],
            reason=row["reason"],
        )
        for row in rows
    ]


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's kappa for two raters. If chance agreement is 1 (both raters always gave the same
    single label) kappa is undefined; we return 1.0 for full agreement in that case."""
    if len(a) != len(b) or not a:
        raise ValueError("need two non-empty label lists of the same length")
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    labels = set(a) | set(b)
    expected = sum((a.count(lab) / n) * (b.count(lab) / n) for lab in labels)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def confusion(human: list[str], judge: list[str], labels: tuple[str, ...]) -> list[list[int]]:
    """Rows = human label, columns = judge label."""
    return [
        [sum(h == r and j == c for h, j in zip(human, judge, strict=True)) for c in labels]
        for r in labels
    ]
