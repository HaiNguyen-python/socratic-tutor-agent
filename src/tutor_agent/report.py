"""`tutor report`: run the whole evaluation and write docs/results.md plus two charts.

1. Run 70 simulated conversations in each of four conditions (scripted / leaky provider,
   guardrails off / on). Every condition meets the same students.
2. Compute metrics overall and per persona, and guardrail statistics.
3. Judge every transcript with the deterministic judge and compare it with the hand labels.

Every number in docs/results.md comes from this file; nothing is typed in by hand.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bm25 import load_notes
from .evals import (
    CONDITIONS,
    CONVERSATIONS_PER_PERSONA,
    MAX_TURNS,
    ConversationRecord,
    CostModel,
    Metrics,
    condition_label,
    group_metrics,
    run_eval,
)
from .exercises import Bank, load_bank
from .judge import (
    CRITERIA,
    DeterministicJudge,
    HumanLabel,
    Verdict,
    cohen_kappa,
    confusion,
    load_labels,
)
from .students import PERSONAS

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, GREY = "#2a78d6", "#b7b7b2"
DATA_LABEL = "Synthetic conversations"
LABEL_SAMPLE_INDEXES = (2, 9)  # which conversations per (condition, persona) get hand labels
DEFAULT_LABELS = Path("evals/human_labels.yaml")


@dataclass
class Agreement:
    n_labelled: int
    stale: list[str]
    kappa: float
    observed: float
    matrix: list[list[int]]
    per_criterion: dict[str, tuple[float, float]]  # criterion -> (agreement, kappa)
    disagreements: list[tuple[HumanLabel, Verdict]]


@dataclass
class Study:
    bank: Bank
    records: list[ConversationRecord]
    overall: dict[tuple[str, str], Metrics]
    by_persona: dict[tuple[str, str], Metrics]
    verdicts: dict[tuple[str, str], Verdict]
    agreement: Agreement | None
    costs: CostModel
    per_persona: int


def label_sample(records: list[ConversationRecord]) -> list[ConversationRecord]:
    """The conversations chosen for hand labelling: 2 per (condition, persona) = 40."""
    wanted = {f"-{i:02d}-" for i in LABEL_SAMPLE_INDEXES}
    return [r for r in records if any(w in r.conv_id for w in wanted)]


def compare_with_labels(
    labels: list[HumanLabel],
    records: list[ConversationRecord],
    verdicts: dict[tuple[str, str], Verdict],
) -> Agreement:
    """Judge vs human on labels whose transcript is unchanged (hash match); others are stale."""
    by_key = {(r.condition, r.conv_id): r for r in records}
    fresh, stale = [], []
    for lab in labels:
        rec = by_key.get((lab.condition, lab.conv_id))
        if rec is None or rec.transcript_hash() != lab.transcript_hash:
            stale.append(f"{lab.condition} {lab.conv_id}")
        else:
            fresh.append((lab, verdicts[(lab.condition, lab.conv_id)]))
    if not fresh:
        return Agreement(0, stale, math.nan, math.nan, [[0, 0], [0, 0]], {}, [])
    human = [lab.label for lab, _ in fresh]
    judged = [v.label for _, v in fresh]
    per_criterion = {}
    for c in CRITERIA:
        h = [str(lab.scores[c]) for lab, _ in fresh]
        j = [str(v.scores[c]) for _, v in fresh]
        agree = sum(x == y for x, y in zip(h, j, strict=True)) / len(h)
        per_criterion[c] = (agree, cohen_kappa(h, j))
    return Agreement(
        n_labelled=len(fresh),
        stale=stale,
        kappa=cohen_kappa(human, judged),
        observed=sum(x == y for x, y in zip(human, judged, strict=True)) / len(human),
        matrix=confusion(human, judged, ("good", "poor")),
        per_criterion=per_criterion,
        disagreements=[(lab, v) for lab, v in fresh if lab.label != v.label],
    )


def run_study(
    per_persona: int = CONVERSATIONS_PER_PERSONA,
    labels_path: Path | None = DEFAULT_LABELS,
    costs: CostModel | None = None,
) -> Study:
    bank, notes = load_bank(), load_notes()
    costs = costs or CostModel()
    records = run_eval(bank, notes, CONDITIONS, per_persona)
    judge = DeterministicJudge(bank)
    verdicts = {(r.condition, r.conv_id): judge.judge(r.transcript()) for r in records}
    agreement = None
    if labels_path is not None and labels_path.exists():
        agreement = compare_with_labels(load_labels(labels_path), records, verdicts)
    return Study(
        bank,
        records,
        group_metrics(records, costs=costs),
        group_metrics(records, by_persona=True, costs=costs),
        verdicts,
        agreement,
        costs,
        per_persona,
    )


def _pct(x: float) -> str:
    return "n/a" if math.isnan(x) else f"{x:.0%}"


def _num(x: float, digits: int = 1) -> str:
    return "n/a" if math.isnan(x) else f"{x:.{digits}f}"


def _labels() -> list[str]:
    return [condition_label(p, g) for p, g in CONDITIONS]


def _leak_styles(records: list[ConversationRecord]) -> Counter[tuple[str, str, bool]]:
    """(condition, style, caught) for every reply the provider meant to leak."""
    counts: Counter[tuple[str, str, bool]] = Counter()
    for r in records:
        for t in r.turns:
            for e in t.result.events:
                meta: dict[str, Any] = e.payload.get("meta", {}) if e.kind == "model" else {}
                if meta.get("leak"):
                    counts[(r.condition, str(meta.get("style")), t.result.blocked_output)] += 1
    return counts


def results_markdown(study: Study) -> str:
    n_conv = study.overall[(_labels()[0], "all")].conversations
    c = study.costs
    lines = [
        "# Results",
        "",
        f"Data: **{DATA_LABEL.lower()}** between seeded simulated students and the tutor. All "
        "numbers below come from `tutor report`; rerun it to reproduce them. The tutor model is "
        "the deterministic `ScriptedProvider` (and its `LeakyProvider` variant), not a real LLM.",
        "",
        f"Setup: {len(study.bank)} exercises, {len(PERSONAS)} personas x {study.per_persona} "
        f"conversations = {n_conv} conversations per condition, at most {MAX_TURNS} student "
        "turns each. Every condition meets the same students (persona, exercise and seed).",
        "",
        "## 1. Overall",
        "",
        "| Condition | Leak rate | Solved by student | Solved by copying a leak | "
        "Mean turns to solve | Tool calls / conv. | Tokens / conv. | Cost / 1,000 conv. | "
        "Latency / turn |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label in _labels():
        m = study.overall[(label, "all")]
        lines.append(
            f"| {label} | {_pct(m.leak_rate)} | {_pct(m.solved_by_student_rate)} | "
            f"{_pct(m.copied_rate)} | {_num(m.mean_turns_to_solve)} | "
            f"{_num(m.tool_calls_per_conversation)} | {m.tokens_per_conversation:,.0f} | "
            f"${m.cost_per_conversation * 1000:.2f} | {m.latency_per_turn_ms:,.0f} ms |"
        )
    lines += [
        "",
        "- Leak rate: share of conversations in which the student *saw* a leaked answer.",
        "- Solved by student: solved with no leak seen. Turns are student messages, counted over "
        "conversations solved by the student.",
        f"- Tokens are estimated (4 characters per token). Cost and latency are *modelled* with "
        f"example rates: ${c.price_in_per_m:.2f} / ${c.price_out_per_m:.2f} per million input / "
        f"output tokens, {c.call_overhead_ms:.0f} ms per model call + "
        f"{c.ms_per_input_token} ms per input token + {c.ms_per_output_token:.0f} ms per output "
        "token. The scripted model itself answers in well under a millisecond; real wall-clock "
        "times are stored in the SQLite trace.",
        "",
        "## 2. By persona",
        "",
        "Leak rate / solved by the student:",
        "",
        "| Persona | " + " | ".join(_labels()) + " |",
        "| --- |" + " --- |" * len(_labels()),
    ]
    for persona in PERSONAS:
        cells = []
        for label in _labels():
            m = study.by_persona[(label, persona)]
            cells.append(f"{_pct(m.leak_rate)} / {_pct(m.solved_by_student_rate)}")
        lines.append(f"| {persona} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "Mean student turns to solve (conversations solved by the student):",
        "",
        "| Persona | " + " | ".join(_labels()) + " |",
        "| --- |" + " --- |" * len(_labels()),
    ]
    for persona in PERSONAS:
        cells = [
            _num(study.by_persona[(label, persona)].mean_turns_to_solve) for label in _labels()
        ]
        lines.append(f"| {persona} | " + " | ".join(cells) + " |")
    lines += _guardrail_section(study)
    lines += _judge_section(study)
    return "\n".join(lines) + "\n"


def _guardrail_section(study: Study) -> list[str]:
    lines = [
        "",
        "## 3. Guardrails",
        "",
        "| Condition | Leak attempts | Caught by output filter | Catch rate | False alarms | "
        "False-positive rate | Input blocks | Input false positives |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for label in _labels():
        if label.endswith("off"):
            continue
        m = study.overall[(label, "all")]
        lines.append(
            f"| {label} | {m.leak_attempts} | {m.leaks_caught} | {_pct(m.catch_rate)} | "
            f"{m.false_alarms} | {m.false_positive_rate:.1%} | {m.input_blocks} | "
            f"{m.input_false_positives} |"
        )
    styles = _leak_styles(study.records)
    names = sorted({s for (_, s, _) in styles})
    lines += [
        "",
        "Leak attempts by style with guardrails on (caught / attempted):",
        "",
        "| Condition | " + " | ".join(names) + " |",
        "| --- |" + " --- |" * len(names),
    ]
    for label in _labels():
        if label.endswith("off"):
            continue
        cells = []
        for s in names:
            caught, missed = styles[(label, s, True)], styles[(label, s, False)]
            cells.append(f"{caught} / {caught + missed}" if caught + missed else "-")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "- A leak attempt is a reply the provider labelled as leaking (ground truth from the "
        "provider). `obeyed-injection` and `gave-in` come from the scripted model's planted "
        "weaknesses; `direct`, `equivalent`, `steps` and `disguised` from the leaky provider.",
        "- `disguised` leaks (arithmetic such as 'take 10 away from 14', or the word spelled "
        "w-e-n-t) are built to beat the detector. The filter only catches one when an incidental "
        "number happens to match (for example 'level 3 of 4' when the answer is 4).",
        "- All false alarms on the output side have that same cause: the hint label 'level k of "
        "4' contains the answer 4 (lin-01, lin-05). A number-matching detector cannot tell an "
        "incidental number from the answer.",
        "- False-positive rate = blocked replies that were not leaks / all non-leak model "
        "replies. Input false positives = student messages blocked although their intent was "
        "legitimate (an attempt, a hint request, a question).",
    ]
    return lines


def _judge_section(study: Study) -> list[str]:
    lines = [
        "",
        "## 4. Socratic quality (judge)",
        "",
        'Share of conversations the deterministic judge labels "good" (rubric in '
        "`src/tutor_agent/judge.py`):",
        "",
        "| Condition | Judged good |",
        "| --- | --- |",
    ]
    for label in _labels():
        verdicts = [v for (cond, _), v in study.verdicts.items() if cond == label]
        good = sum(v.label == "good" for v in verdicts) / len(verdicts)
        lines.append(f"| {label} | {good:.0%} |")
    a = study.agreement
    if a is None:
        return lines + ["", "No hand labels found, so judge agreement was not computed."]
    lines += [
        "",
        f"Agreement with hand labels ({a.n_labelled} labelled transcripts, 2 per condition and "
        f"persona): Cohen's kappa **{_num(a.kappa, 2)}**, raw agreement {_pct(a.observed)}.",
        "",
        "| Human \\ Judge | good | poor |",
        "| --- | --- | --- |",
        f"| good | {a.matrix[0][0]} | {a.matrix[0][1]} |",
        f"| poor | {a.matrix[1][0]} | {a.matrix[1][1]} |",
        "",
        "| Criterion | Agreement | Kappa |",
        "| --- | --- | --- |",
    ]
    for crit, (agree, kappa) in a.per_criterion.items():
        lines.append(f"| {crit} | {agree:.0%} | {_num(kappa, 2)} |")
    if a.disagreements:
        lines += ["", "Disagreements:", ""]
        for lab, v in a.disagreements:
            lines.append(
                f"- `{lab.conv_id}` ({lab.condition}): human **{lab.label}**, judge "
                f"**{v.label}**. Human reason: {lab.reason}"
            )
    if a.stale:
        lines += ["", f"Stale labels (transcript changed, not scored): {len(a.stale)}."]
    lines += [
        "",
        "The deterministic judge is a stand-in for CI, not a real judge. Its no-leak check is "
        "the same detector as the output filter, so it cannot see the leaks the filter misses. "
        "The hand labels were written during the build by reading each transcript against the "
        "rubric; see LEARNING.md for re-labelling them yourself.",
    ]
    return lines


def _style(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def _header(fig: Any, title: str, subtitle: str) -> None:
    fig.text(0.012, 0.965, title, fontsize=12.5, fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.905, subtitle, fontsize=9.5, color=INK_2, va="top")


def plot_persona_dumbbells(study: Study, path: Path) -> None:
    """Leak rate and solved-by-student rate per persona: guardrails off (grey) vs on (blue)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.2), dpi=150, sharey=True)
    fig.set_facecolor(SURFACE)
    panels = [
        ("leak_rate", "scripted", "Leak rate, scripted model"),
        ("leak_rate", "leaky", "Leak rate, leaky model"),
        ("solved_by_student_rate", "scripted", "Solved by the student, scripted model"),
        ("solved_by_student_rate", "leaky", "Solved by the student, leaky model"),
    ]
    personas = list(PERSONAS)[::-1]
    for ax, (metric, provider, title) in zip(axes.flat, panels, strict=True):
        for y, persona in enumerate(personas):
            off = getattr(study.by_persona[(condition_label(provider, False), persona)], metric)
            on = getattr(study.by_persona[(condition_label(provider, True), persona)], metric)
            ax.plot([off, on], [y, y], color=GRID, linewidth=3, solid_capstyle="round", zorder=1)
            ax.scatter([off], [y], s=60, color=GREY, zorder=2, edgecolor=SURFACE, linewidth=1.5)
            ax.scatter([on], [y], s=60, color=BLUE, zorder=3, edgecolor=SURFACE, linewidth=1.5)
            if abs(on - off) >= 0.12:
                ax.text(off, y + 0.28, "off", ha="center", fontsize=7.5, color=INK_2)
                ax.text(on, y + 0.28, "on", ha="center", fontsize=7.5, color=BLUE)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_yticks(range(len(personas)))
        ax.set_yticklabels([p.replace("_", " ") for p in personas])
        ax.set_xlim(-0.05, 1.08)
        ax.set_ylim(-0.6, len(personas) - 0.3)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
        _style(ax)
        ax.grid(axis="y", visible=False)
    s_off = study.overall[(condition_label("scripted", False), "all")]
    s_on = study.overall[(condition_label("scripted", True), "all")]
    l_off = study.overall[(condition_label("leaky", False), "all")]
    l_on = study.overall[(condition_label("leaky", True), "all")]
    _header(
        fig,
        f"Guardrails cut leaks from {s_off.leak_rate:.0%} to {s_on.leak_rate:.0%} (scripted) "
        f"and {l_off.leak_rate:.0%} to {l_on.leak_rate:.0%} (leaky model)",
        f"Grey = guardrails off, blue = on. {study.per_persona} conversations per persona and "
        f"condition. Disguised leaks still get through. {DATA_LABEL}.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.87))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_turns(study: Study, path: Path) -> None:
    """Distribution of student turns to solve, scripted model, guardrails off vs on."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 4.4), dpi=150)
    fig.set_facecolor(SURFACE)
    personas = list(PERSONAS)[::-1]
    for y, persona in enumerate(personas):
        for offset, guard, color in ((0.17, False, GREY), (-0.17, True, BLUE)):
            label = condition_label("scripted", guard)
            turns = sorted(
                r.turns_to_solve
                for r in study.records
                if r.condition == label and r.persona == persona and r.turns_to_solve is not None
            )
            counts = Counter(turns)
            for t, k in counts.items():
                xs = [t + (i - (k - 1) / 2) * 0.09 for i in range(k)]
                ax.scatter(xs, [y + offset] * k, s=16, color=color, zorder=2, linewidth=0)
            m = study.by_persona[(label, persona)]
            if turns:
                ax.plot(
                    [m.mean_turns_to_solve] * 2,
                    [y + offset - 0.13, y + offset + 0.13],
                    color=INK,
                    linewidth=2,
                    zorder=3,
                )
            ax.text(
                8.75,
                y + offset,
                f"{len(turns)}/{m.conversations} solved",
                va="center",
                fontsize=8,
                color=color if guard else INK_2,
            )
    ax.set_yticks(range(len(personas)))
    ax.set_yticklabels([p.replace("_", " ") for p in personas])
    ax.set_xticks(range(1, MAX_TURNS + 1))
    ax.set_xlim(0.5, 9.9)
    ax.set_xlabel("Student turns until solved (black tick = mean)", fontsize=9, color=INK_2)
    _style(ax)
    ax.grid(axis="y", visible=False)
    seeker = condition_label("scripted", False), condition_label("scripted", True)
    s_off = study.by_persona[(seeker[0], "answer_seeker")]
    s_on = study.by_persona[(seeker[1], "answer_seeker")]

    def solved(m: Metrics) -> str:
        return f"{round(m.solved_by_student_rate * m.conversations)}/{m.conversations}"

    _header(
        fig,
        f"The hint cap slows answer-seekers ({_num(s_off.mean_turns_to_solve)} -> "
        f"{_num(s_on.mean_turns_to_solve)} turns) but {solved(s_on)} vs {solved(s_off)} of them "
        "solve it themselves",
        "Scripted model. Each dot is one conversation solved by the student; grey = guardrails "
        f"off (upper row), blue = on (lower row). {DATA_LABEL}.",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def write(
    out_dir: Path,
    labels_path: Path | None = DEFAULT_LABELS,
    per_persona: int = CONVERSATIONS_PER_PERSONA,
) -> Study:
    study = run_study(per_persona, labels_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.md").write_text(results_markdown(study))
    plot_persona_dumbbells(study, out_dir / "img" / "leak_solve_by_persona.png")
    plot_turns(study, out_dir / "img" / "turns_to_solve.png")
    return study
