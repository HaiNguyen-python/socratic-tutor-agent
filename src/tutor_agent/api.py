"""FastAPI service: `POST /chat` keeps a tutoring session per `session_id`.

The first call may omit `session_id`; the response returns a new one. Sessions live in
memory, which is fine for a demo and a single worker. A real deployment would store the
message history (it is plain JSON) in Redis or Postgres keyed by session id.
"""

from __future__ import annotations

import uuid

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .agent import Tutor, TutorSession
from .bm25 import load_notes
from .exercises import load_bank
from .guardrails import GuardrailConfig
from .providers import make_provider
from .tools import ToolBox


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    session_id: str | None = None
    exercise_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    exercise_id: str
    question: str
    reply: str
    solved: bool
    guardrail: str | None  # "injection", "off_topic", "leak" or None
    tool_calls: int


def create_app(provider: str = "scripted", guardrails: GuardrailConfig | None = None) -> FastAPI:
    bank = load_bank()
    tutor = Tutor(make_provider(provider, bank), ToolBox(bank, load_notes()), guardrails)
    sessions: dict[str, TutorSession] = {}
    app = FastAPI(title="Socratic tutor agent", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "provider": tutor.provider.name}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest) -> ChatResponse:
        if req.session_id is None:
            exercise_id = req.exercise_id or bank.exercises[0].id
            if exercise_id not in bank:
                raise HTTPException(422, f"unknown exercise {exercise_id!r}")
            session_id = uuid.uuid4().hex
            sessions[session_id] = tutor.start(exercise_id)
        else:
            session_id = req.session_id
            if session_id not in sessions:
                raise HTTPException(404, "unknown session_id")
        session = sessions[session_id]
        result = tutor.respond(session, req.message)
        guardrail = result.blocked_input or ("leak" if result.blocked_output else None)
        return ChatResponse(
            session_id=session_id,
            exercise_id=session.exercise.id,
            question=session.exercise.question,
            reply=result.reply,
            solved=session.solved,
            guardrail=guardrail,
            tool_calls=result.tool_calls,
        )

    return app
