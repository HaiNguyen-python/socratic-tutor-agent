"""Command line: tutor chat | eval | report | sample | judge | serve."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

from .exercises import Bank
from .guardrails import GuardrailConfig
from .providers import Provider

log = logging.getLogger("tutor")
DEFAULT_TRACES = Path("reports/traces.db")
PROVIDERS = ["scripted", "leaky", "ollama", "anthropic"]


def _guardrails(on: bool) -> GuardrailConfig:
    return GuardrailConfig.all_on() if on else GuardrailConfig.all_off()


def cmd_chat(args: argparse.Namespace) -> None:
    """A terminal tutoring session. Commands: /next, /quit."""
    from .agent import Tutor
    from .bm25 import load_notes
    from .exercises import load_bank
    from .providers import make_provider
    from .tools import ToolBox
    from .tracing import Tracer

    bank = load_bank()
    tutor = Tutor(
        make_provider(args.provider, bank, args.model),
        ToolBox(bank, load_notes()),
        _guardrails(not args.no_guardrails),
    )
    tracer = Tracer(args.traces)
    run_id = f"chat:{uuid.uuid4().hex[:8]}"
    tracer.start_run(run_id, "chat", tutor.provider.name, tutor.guardrails.label)
    ids = [e.id for e in bank.exercises]
    index = ids.index(args.exercise) if args.exercise else 0
    session = tutor.start(ids[index])
    turn = 0
    print(f"Exercise {session.exercise.id}: {session.exercise.question}")
    print("(type /next for another exercise, /quit to stop)")
    for line in sys.stdin:
        message = line.strip()
        if not message:
            continue
        if message == "/quit":
            break
        if message == "/next" or session.solved and message.lower() in ("yes", "next"):
            index = (index + 1) % len(ids)
            session, turn = tutor.start(ids[index]), 0
            print(f"Exercise {session.exercise.id}: {session.exercise.question}")
            continue
        turn += 1
        result = tutor.respond(session, message)
        tracer.log_turn(run_id, session.exercise.id, turn, result.events)
        flag = (
            f"  [guardrail: {result.blocked_input or 'leak blocked'}]"
            if result.stopped == "guardrail"
            else ""
        )
        print(f"Tutor: {result.reply}{flag}")
        if session.solved:
            print("(solved: type /next for another exercise)")
    tracer.close()


def cmd_eval(args: argparse.Namespace) -> None:
    from .bm25 import load_notes
    from .evals import CONDITIONS, group_metrics, run_eval
    from .exercises import load_bank
    from .providers import make_provider
    from .tracing import Tracer

    bank = load_bank()
    if args.provider:
        conditions = [(args.provider, g) for g in (False, True)]
    else:
        conditions = list(CONDITIONS)

    def factory(name: str, b: Bank) -> Provider:
        return make_provider(name, b, args.model)

    records = run_eval(bank, load_notes(), conditions, args.per_persona, factory)
    tracer = Tracer(args.traces)
    tracer.log_eval(records)
    tracer.close()
    table = {
        f"{cond} [{persona}]": {
            "leak_rate": round(m.leak_rate, 3),
            "solved_by_student": round(m.solved_by_student_rate, 3),
            "mean_turns_to_solve": round(m.mean_turns_to_solve, 2),
            "tool_calls": round(m.tool_calls_per_conversation, 2),
            "output_blocks": m.output_blocks,
            "input_blocks": m.input_blocks,
        }
        for (cond, persona), m in group_metrics(records, by_persona=args.by_persona).items()
    }
    print(json.dumps(table, indent=2))
    log.info("wrote %d conversations to %s", len(records), args.traces)


def cmd_report(args: argparse.Namespace) -> None:
    from .report import write

    study = write(args.out, args.labels)
    agreement = study.agreement
    kappa = None if agreement is None else round(agreement.kappa, 3)
    print(json.dumps({"conversations": len(study.records), "judge_kappa": kappa}, indent=2))
    log.info("wrote %s and charts in %s", args.out / "results.md", args.out / "img")


def cmd_sample(args: argparse.Namespace) -> None:
    """Print the transcripts chosen for hand labelling, with the hash a label must carry."""
    from .bm25 import load_notes
    from .evals import run_eval
    from .exercises import load_bank
    from .report import label_sample

    for r in label_sample(run_eval(load_bank(), load_notes())):
        print(f"### {r.conv_id} | {r.condition} | hash {r.transcript_hash()}")
        print(r.transcript() + "\n")


def cmd_judge(args: argparse.Namespace) -> None:
    """Run an LLM judge (any provider) on the hand-labelled sample and report kappa."""
    from .bm25 import load_notes
    from .evals import run_eval
    from .exercises import load_bank
    from .judge import LLMJudge, cohen_kappa, load_labels
    from .providers import make_provider
    from .report import label_sample

    bank = load_bank()
    judge = LLMJudge(make_provider(args.provider, bank, args.model))
    labels = {(lab.condition, lab.conv_id): lab for lab in load_labels(args.labels)}
    human, judged = [], []
    for r in label_sample(run_eval(bank, load_notes())):
        lab = labels.get((r.condition, r.conv_id))
        if lab is None or lab.transcript_hash != r.transcript_hash():
            continue
        human.append(lab.label)
        judged.append(judge.judge(r.transcript()).label)
    kappa = cohen_kappa(human, judged) if human else float("nan")
    print(json.dumps({"judge": judge.name, "n": len(human), "kappa": kappa}, indent=2))


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    from .api import create_app

    uvicorn.run(create_app(args.provider), host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tutor", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    chat = sub.add_parser("chat", help="tutoring session in the terminal")
    chat.add_argument("--exercise", help="exercise id, e.g. lin-01 (default: the first)")
    chat.add_argument("--no-guardrails", action="store_true")
    chat.add_argument("--traces", type=Path, default=DEFAULT_TRACES)

    ev = sub.add_parser("eval", help="simulated-student evaluation, traces to SQLite")
    ev.add_argument("--per-persona", type=int, default=14)
    ev.add_argument("--by-persona", action="store_true")
    ev.add_argument("--traces", type=Path, default=DEFAULT_TRACES)

    report = sub.add_parser("report", help="write docs/results.md and docs/img/*.png")
    report.add_argument("--out", type=Path, default=Path("docs"))

    sub.add_parser("sample", help="print the transcripts chosen for hand labelling")

    judge = sub.add_parser("judge", help="LLM judge on the labelled sample, kappa vs hand labels")

    serve = sub.add_parser("serve", help="run the FastAPI /chat endpoint")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    for p in (chat, serve):
        p.add_argument("--provider", choices=PROVIDERS, default="scripted")
    ev.add_argument("--provider", choices=PROVIDERS, help="one provider, guardrails off and on")
    judge.add_argument("--provider", choices=PROVIDERS, default="ollama")
    for p in (chat, ev, judge):
        p.add_argument("--model", help="model name for ollama / anthropic")
    for p in (report, judge):
        p.add_argument("--labels", type=Path, default=Path("evals/human_labels.yaml"))
    return parser


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    handlers = {
        "chat": cmd_chat,
        "eval": cmd_eval,
        "report": cmd_report,
        "sample": cmd_sample,
        "judge": cmd_judge,
        "serve": cmd_serve,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
