"""Structured traces in SQLite: one row per run, per conversation and per agent step.

Every model call, tool call and guardrail decision is an event with its JSON payload, token
counts and measured latency. That is enough to replay a conversation, to find every blocked
reply with one SQL query, or to compare two runs.

    SELECT conv_id, turn, json_extract(payload, '$.reason')
    FROM events WHERE kind = 'output_guard';
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from .agent import Event
from .evals import ConversationRecord, event_rows

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    kind TEXT,
    provider TEXT,
    guardrails TEXT
);
CREATE TABLE IF NOT EXISTS conversations (
    run_id TEXT,
    conv_id TEXT,
    persona TEXT,
    exercise_id TEXT,
    solved INTEGER,
    solved_by_student INTEGER,
    leaked INTEGER,
    turns INTEGER,
    transcript_hash TEXT,
    PRIMARY KEY (run_id, conv_id)
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    conv_id TEXT,
    turn INTEGER,
    step INTEGER,
    kind TEXT,
    payload TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    latency_ms REAL
);
"""


class Tracer:
    def __init__(self, path: Path | str):
        if isinstance(path, Path):
            path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def start_run(self, run_id: str, kind: str, provider: str, guardrails: str) -> None:
        """Register a run; starting the same run id again replaces its old rows."""
        for table in ("events", "conversations", "runs"):
            self.conn.execute(f"DELETE FROM {table} WHERE run_id = ?", (run_id,))
        self.conn.execute(
            "INSERT INTO runs (run_id, kind, provider, guardrails) VALUES (?, ?, ?, ?)",
            (run_id, kind, provider, guardrails),
        )
        self.conn.commit()

    def log_event(self, run_id: str, conv_id: str, turn: int, step: int, event: Event) -> None:
        self.conn.execute(
            "INSERT INTO events (run_id, conv_id, turn, step, kind, payload, input_tokens, "
            "output_tokens, latency_ms) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                conv_id,
                turn,
                step,
                event.kind,
                json.dumps(event.payload, default=str),
                event.input_tokens,
                event.output_tokens,
                event.latency_ms,
            ),
        )

    def log_turn(self, run_id: str, conv_id: str, turn: int, events: list[Event]) -> None:
        for step, event in enumerate(events):
            self.log_event(run_id, conv_id, turn, step, event)
        self.conn.commit()

    def log_conversation(self, run_id: str, record: ConversationRecord) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO conversations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                record.conv_id,
                record.persona,
                record.exercise_id,
                int(record.solved),
                int(record.solved_by_student),
                int(record.leaked),
                len(record.turns),
                record.transcript_hash(),
            ),
        )
        for turn, step, event in event_rows(record):
            self.log_event(run_id, record.conv_id, turn, step, event)

    def log_eval(self, records: list[ConversationRecord], prefix: str = "eval") -> list[str]:
        """Write every condition as its own run; returns the run ids."""
        run_ids: list[str] = []
        for r in records:
            run_id = f"{prefix}:{r.provider}:{'on' if r.guardrails else 'off'}"
            if run_id not in run_ids:
                self.start_run(run_id, "eval", r.provider, "on" if r.guardrails else "off")
                run_ids.append(run_id)
            self.log_conversation(run_id, r)
        self.conn.commit()
        return run_ids

    def summary(self, run_id: str) -> dict[str, Any]:
        convs = self.conn.execute(
            "SELECT count(*), coalesce(avg(leaked), 0), coalesce(avg(solved_by_student), 0) "
            "FROM conversations WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        events = self.conn.execute(
            "SELECT sum(kind = 'tool'), sum(kind = 'output_guard'), sum(kind = 'input_guard'), "
            "coalesce(sum(input_tokens + output_tokens), 0), coalesce(avg(CASE WHEN kind = "
            "'model' THEN latency_ms END), 0) FROM events WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return {
            "conversations": convs[0],
            "leak_rate": convs[1],
            "solved_by_student_rate": convs[2],
            "tool_calls": events[0] or 0,
            "output_blocks": events[1] or 0,
            "input_blocks": events[2] or 0,
            "tokens": events[3],
            "mean_model_latency_ms": events[4],
        }

    def close(self) -> None:
        self.conn.close()
