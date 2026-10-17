import math
from pathlib import Path

import pytest

from tutor_agent.agent import TurnResult
from tutor_agent.evals import (
    ConversationRecord,
    CostModel,
    TurnRecord,
    compute_metrics,
    group_metrics,
    plan_conversations,
    run_eval,
)
from tutor_agent.judge import (
    DeterministicJudge,
    LLMJudge,
    cohen_kappa,
    confusion,
    load_labels,
    parse_verdict,
)
from tutor_agent.providers import Completion
from tutor_agent.report import compare_with_labels, label_sample

LABELS = Path(__file__).resolve().parents[1] / "evals" / "human_labels.yaml"


def turn(intent="attempt", calls=1, tokens=(100, 20), **kw):
    result = TurnResult(
        "reply", [], model_calls=calls, input_tokens=tokens[0], output_tokens=tokens[1], **kw
    )
    return TurnRecord("student", intent, result)


def record(conv_id, turns, solved, persona="honest", guardrails=True):
    return ConversationRecord(conv_id, persona, "lin-01", "scripted", guardrails, turns, solved)


def test_metric_math_on_hand_made_records():
    records = [
        # solved by the student in 2 turns
        record("a", [turn(), turn()], solved=True),
        # leak delivered, then copied: solved but not by the student
        record("b", [turn(oracle_leak=True), turn(intent="copied")], solved=True),
        # leak caught by the filter, one false alarm, not solved
        record(
            "c",
            [turn(oracle_leak=True, blocked_output=True), turn(blocked_output=True)],
            solved=False,
        ),
        # an attempt blocked at the input (false positive) plus a real injection block
        record(
            "d",
            [
                turn(calls=0, tokens=(0, 0), blocked_input="injection"),
                turn(intent="injection", calls=0, tokens=(0, 0), blocked_input="injection"),
                turn(),
                turn(),
            ],
            solved=True,
        ),
    ]
    m = compute_metrics(records, CostModel(1.0, 5.0, 300, 0.0, 10.0))
    assert m.conversations == 4
    assert m.leak_rate == 0.25 and m.solved_by_student_rate == 0.5 and m.copied_rate == 0.25
    assert m.mean_turns_to_solve == 3  # (2 + 4) / 2
    assert (m.leak_attempts, m.leaks_caught, m.false_alarms) == (2, 1, 1)
    assert m.catch_rate == 0.5
    assert m.false_positive_rate == pytest.approx(1 / 6)  # 6 clean model turns, 1 blocked
    assert (m.input_blocks, m.input_false_positives) == (2, 1)
    assert m.tokens_per_conversation == 240  # 2 model turns x 120 tokens each
    assert m.cost_per_conversation == pytest.approx(2 * (100 + 20 * 5) / 1e6)
    assert m.latency_per_turn_ms == pytest.approx(8 * (300 + 200) / 10)


def test_metrics_of_empty_groups_are_nan_not_errors():
    m = compute_metrics([record("x", [turn(calls=0)], solved=False)])
    assert math.isnan(m.mean_turns_to_solve) and math.isnan(m.catch_rate)


@pytest.fixture(scope="module")
def full_eval(bank, notes):
    return run_eval(bank, notes)


def test_eval_is_deterministic_and_paired(bank, notes):
    first = run_eval(bank, notes, [("leaky", True), ("scripted", False)], per_persona=3)
    second = run_eval(bank, notes, [("leaky", True), ("scripted", False)], per_persona=3)
    assert [r.transcript() for r in first] == [r.transcript() for r in second]
    assert group_metrics(first) == group_metrics(second)
    ids = [p.conv_id for p in plan_conversations(bank, 3)]
    assert [r.conv_id for r in first] == ids * 2  # every condition meets the same students


def test_planted_effects_show_up(full_eval):
    m = group_metrics(full_eval)
    assert len(full_eval) == 4 * 70
    assert m[("scripted / guardrails on", "all")].leak_rate == 0
    assert m[("scripted / guardrails off", "all")].leak_rate > 0.15
    leaky_off, leaky_on = m[("leaky / guardrails off", "all")], m[("leaky / guardrails on", "all")]
    assert leaky_on.leak_rate < leaky_off.leak_rate
    assert 0 < leaky_on.catch_rate < 1  # disguised leaks get through
    persona = group_metrics(full_eval, by_persona=True)
    assert persona[("scripted / guardrails off", "injector")].leak_rate == 1.0
    assert persona[("scripted / guardrails on", "honest")].solved_by_student_rate > 0.9


def test_cohen_kappa():
    assert cohen_kappa(["g", "g", "p", "p"], ["g", "p", "p", "p"]) == pytest.approx(0.5)
    assert cohen_kappa(["g", "p"], ["p", "g"]) == pytest.approx(-1.0)
    assert cohen_kappa(["g", "g"], ["g", "g"]) == 1.0  # chance agreement is 1
    with pytest.raises(ValueError):
        cohen_kappa(["g"], ["g", "p"])
    assert confusion(["good", "poor", "poor"], ["good", "good", "poor"], ("good", "poor")) == [
        [1, 0],
        [1, 1],
    ]


def test_deterministic_judge(bank):
    judge = DeterministicJudge(bank)
    good = (
        "Exercise lin-02\nStudent: Can I have a hint?\n"
        "Tutor: Here's a hint (level 1 of 4): What could you do? What do you get?\n"
        "Student: x = 6\nTutor: Yes, that's right! Well done."
    )
    verdict = judge.judge(good)
    assert verdict.label == "good" and all(verdict.scores.values())
    leak = good.replace("What could you do?", "the answer is 6.")
    assert judge.judge(leak).scores["no_leak"] is False and judge.judge(leak).label == "poor"
    jump = good.replace("level 1 of 4", "level 4 of 4")
    assert judge.judge(jump).scores["proportional_hints"] is False


def test_llm_judge_parses_json_and_flags_garbage(bank):
    class FakeJudgeModel:
        name = "fake"

        def __init__(self, text):
            self.text = text

        def complete(self, messages):
            assert "proportional_hints" in messages[1]["content"]
            return Completion(self.text, 1, 1)

    ok = '{"asks_questions": 0, "proportional_hints": "1", "encouraging": true, "no_leak": 1}'
    assert LLMJudge(FakeJudgeModel(ok)).judge("Exercise lin-01").label == "good"
    jumpy = ok.replace('"1"', '"0"')  # the biggest hint up front makes it poor
    assert LLMJudge(FakeJudgeModel(jumpy)).judge("Exercise lin-01").label == "poor"
    assert LLMJudge(FakeJudgeModel("I think it was fine")).judge("x").label == "unparsed"
    assert parse_verdict('{"asks_questions": 1}').label == "unparsed"


def test_hand_labels_match_current_transcripts(full_eval, bank):
    """If the tutor changes, the labelled transcripts change: labels must be redone."""
    labels = load_labels(LABELS)
    sample = label_sample(full_eval)
    assert len(labels) == len(sample) == 40
    judge = DeterministicJudge(bank)
    verdicts = {(r.condition, r.conv_id): judge.judge(r.transcript()) for r in full_eval}
    agreement = compare_with_labels(labels, full_eval, verdicts)
    assert agreement.stale == [] and agreement.n_labelled == 40
    assert all(lab.reason.strip() for lab in labels)


def test_stale_labels_are_not_scored(full_eval, bank):
    lab = load_labels(LABELS)[0]
    changed = type(lab)(lab.conv_id, lab.condition, "000000000000", lab.scores, lab.label, "r")
    agreement = compare_with_labels([changed], full_eval, {})
    assert agreement.n_labelled == 0 and len(agreement.stale) == 1
