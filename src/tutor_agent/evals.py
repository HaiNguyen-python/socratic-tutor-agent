"""Eval harness: simulated students talk to the tutor; metrics per persona and overall.

A *condition* is a provider plus a guardrail setting. Every condition meets the same students
(same persona, exercise and seed), so differences between conditions come from the tutor, not
from luck. All metric functions are pure functions of `ConversationRecord`s so they can be
tested on hand-made records.

Ground truth for leaks: the scripted and leaky providers label each reply they meant to leak
(`oracle_leak`). A *delivered* leak is one the student saw (not replaced by the output filter).
Cost and latency are *modelled* from estimated tokens with the example rates in `CostModel`;
the real wall-clock time of each model call is kept in the SQLite trace.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from statistics import mean

from .agent import Event, TurnResult, Tutor
from .bm25 import BM25Index
from .exercises import Bank
from .guardrails import GuardrailConfig
from .providers import Provider, make_provider, stable_hash
from .students import PERSONAS, SimulatedStudent, TutorFeedback
from .tools import ToolBox

MAX_TURNS = 8
CONVERSATIONS_PER_PERSONA = 14
CONDITIONS = (("scripted", False), ("scripted", True), ("leaky", False), ("leaky", True))
LEGIT_INTENTS = frozenset({"attempt", "hint", "concept", "demand", "copied"})


@dataclass(frozen=True)
class CostModel:
    """Example rates for estimates (USD per million tokens, milliseconds). Edit for your model."""

    price_in_per_m: float = 1.00
    price_out_per_m: float = 5.00
    call_overhead_ms: float = 300.0
    ms_per_input_token: float = 0.1
    ms_per_output_token: float = 15.0

    def cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.price_in_per_m + output_tokens * self.price_out_per_m) / 1e6

    def latency_ms(self, model_calls: int, input_tokens: int, output_tokens: int) -> float:
        return (
            model_calls * self.call_overhead_ms
            + input_tokens * self.ms_per_input_token
            + output_tokens * self.ms_per_output_token
        )


@dataclass
class TurnRecord:
    student: str
    intent: str
    result: TurnResult

    @property
    def reply(self) -> str:
        return self.result.reply


@dataclass
class ConversationRecord:
    conv_id: str
    persona: str
    exercise_id: str
    provider: str
    guardrails: bool
    turns: list[TurnRecord] = field(default_factory=list)
    solved: bool = False

    @property
    def condition(self) -> str:
        return condition_label(self.provider, self.guardrails)

    @property
    def leaked(self) -> bool:
        return any(t.result.delivered_leak for t in self.turns)

    @property
    def solved_by_student(self) -> bool:
        # Conversations stop at the solve, so any delivered leak came before it.
        return self.solved and not self.leaked

    @property
    def turns_to_solve(self) -> int | None:
        return len(self.turns) if self.solved_by_student else None

    @property
    def input_tokens(self) -> int:
        return sum(t.result.input_tokens for t in self.turns)

    @property
    def output_tokens(self) -> int:
        return sum(t.result.output_tokens for t in self.turns)

    @property
    def tool_calls(self) -> int:
        return sum(t.result.tool_calls for t in self.turns)

    def transcript(self) -> str:
        lines = [f"Exercise {self.exercise_id}"]
        for t in self.turns:
            lines += [f"Student: {t.student}", f"Tutor: {t.reply}"]
        return "\n".join(lines)

    def transcript_hash(self) -> str:
        """Short content hash: hand labels are tied to the exact transcript they judged."""
        return hashlib.sha256(self.transcript().encode()).hexdigest()[:12]


def condition_label(provider: str, guardrails: bool) -> str:
    return f"{provider} / guardrails {'on' if guardrails else 'off'}"


@dataclass(frozen=True)
class PlannedConversation:
    conv_id: str
    persona: str
    exercise_id: str
    seed: int


def plan_conversations(
    bank: Bank, per_persona: int = CONVERSATIONS_PER_PERSONA
) -> list[PlannedConversation]:
    """Personas x exercises; each persona starts at a different offset in the bank."""
    plan = []
    for p, persona in enumerate(PERSONAS):
        for i in range(per_persona):
            ex = bank.exercises[(p * per_persona + i) % len(bank)]
            conv_id = f"{persona}-{i:02d}-{ex.id}"
            plan.append(PlannedConversation(conv_id, persona, ex.id, stable_hash(conv_id)))
    return plan


def run_conversation(
    tutor: Tutor, planned: PlannedConversation, max_turns: int = MAX_TURNS
) -> ConversationRecord:
    session = tutor.start(planned.exercise_id)
    student = SimulatedStudent(planned.persona, session.exercise, planned.seed)
    record = ConversationRecord(
        planned.conv_id,
        planned.persona,
        planned.exercise_id,
        tutor.provider.name,
        tutor.guardrails.label == "on",
    )
    feedback = TutorFeedback()
    for _ in range(max_turns):
        message = student.next_message(feedback)
        result = tutor.respond(session, message.text)
        record.turns.append(TurnRecord(message.text, message.intent, result))
        if session.solved:
            record.solved = True
            break
        feedback = TutorFeedback(result.hint_level, result.delivered_leak)
    return record


ProviderFactory = Callable[[str, Bank], Provider]


def run_condition(
    bank: Bank,
    notes: BM25Index,
    provider: str,
    guardrails: bool,
    plan: list[PlannedConversation],
    factory: ProviderFactory = make_provider,
) -> list[ConversationRecord]:
    config = GuardrailConfig.all_on() if guardrails else GuardrailConfig.all_off()
    tutor = Tutor(factory(provider, bank), ToolBox(bank, notes), config)
    return [run_conversation(tutor, p) for p in plan]


def run_eval(
    bank: Bank,
    notes: BM25Index,
    conditions: Iterable[tuple[str, bool]] = CONDITIONS,
    per_persona: int = CONVERSATIONS_PER_PERSONA,
    factory: ProviderFactory = make_provider,
) -> list[ConversationRecord]:
    plan = plan_conversations(bank, per_persona)
    records: list[ConversationRecord] = []
    for provider, guardrails in conditions:
        records += run_condition(bank, notes, provider, guardrails, plan, factory)
    return records


def _rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else float("nan")


def _turn_results(records: list[ConversationRecord]) -> list[tuple[str, TurnResult]]:
    return [(t.intent, t.result) for r in records for t in r.turns]


@dataclass(frozen=True)
class Metrics:
    conversations: int
    leak_rate: float
    solved_by_student_rate: float
    copied_rate: float
    mean_turns_to_solve: float
    tool_calls_per_conversation: float
    tokens_per_conversation: float
    cost_per_conversation: float
    latency_per_turn_ms: float
    leak_attempts: int
    leaks_caught: int
    false_alarms: int
    output_blocks: int
    catch_rate: float
    false_positive_rate: float
    input_blocks: int
    input_false_positives: int


def compute_metrics(records: list[ConversationRecord], costs: CostModel | None = None) -> Metrics:
    """All headline metrics for a group of conversations.

    - leak rate: share of conversations where the student saw a leak
    - solved-by-student: solved with no leak seen; copied: solved after a leak
    - mean turns to solve: student turns, over conversations solved by the student
    - catch rate: leak attempts replaced by the output filter / all leak attempts
    - false-positive rate: replies blocked that were not leaks / all non-leak model replies
    - input false positives: student messages blocked whose intent was legitimate
    """
    costs = costs or CostModel()
    turns = _turn_results(records)
    model_turns = [r for _, r in turns if r.model_calls > 0]
    attempts = [r for r in model_turns if r.oracle_leak]
    clean = [r for r in model_turns if not r.oracle_leak]
    caught = sum(r.blocked_output for r in attempts)
    false_alarms = sum(r.blocked_output for r in clean)
    solve_turns = [r.turns_to_solve for r in records if r.turns_to_solve is not None]
    tokens = [r.input_tokens + r.output_tokens for r in records]
    latencies = [costs.latency_ms(r.model_calls, r.input_tokens, r.output_tokens) for _, r in turns]
    return Metrics(
        conversations=len(records),
        leak_rate=_rate([r.leaked for r in records]),
        solved_by_student_rate=_rate([r.solved_by_student for r in records]),
        copied_rate=_rate([r.solved and r.leaked for r in records]),
        mean_turns_to_solve=mean(solve_turns) if solve_turns else float("nan"),
        tool_calls_per_conversation=mean(r.tool_calls for r in records) if records else 0.0,
        tokens_per_conversation=mean(tokens) if tokens else 0.0,
        cost_per_conversation=(
            mean(costs.cost(r.input_tokens, r.output_tokens) for r in records) if records else 0.0
        ),
        latency_per_turn_ms=mean(latencies) if latencies else 0.0,
        leak_attempts=len(attempts),
        leaks_caught=caught,
        false_alarms=false_alarms,
        output_blocks=caught + false_alarms,
        catch_rate=_rate([r.blocked_output for r in attempts]),
        false_positive_rate=_rate([r.blocked_output for r in clean]),
        input_blocks=sum(bool(r.blocked_input) for _, r in turns),
        input_false_positives=sum(
            bool(r.blocked_input) and intent in LEGIT_INTENTS for intent, r in turns
        ),
    )


def group_metrics(
    records: list[ConversationRecord], by_persona: bool = False, costs: CostModel | None = None
) -> dict[tuple[str, str], Metrics]:
    """Metrics keyed by (condition, persona); persona is "all" unless `by_persona`."""
    groups: dict[tuple[str, str], list[ConversationRecord]] = {}
    for r in records:
        groups.setdefault((r.condition, r.persona if by_persona else "all"), []).append(r)
    return {key: compute_metrics(group, costs) for key, group in groups.items()}


def event_rows(record: ConversationRecord) -> Iterable[tuple[int, int, Event]]:
    """(turn, step, event) for every event, in order, for the trace store."""
    for turn, t in enumerate(record.turns, start=1):
        for step, event in enumerate(t.result.events):
            yield turn, step, event
