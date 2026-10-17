"""The agent loop: one student message in, one tutor reply out, with guardrails around it.

    student message
      -> input guardrails (injection -> safe refusal, off-topic -> redirect)   [no model call]
      -> model step: reply, or a JSON tool call -> run tool -> feed result back -> next step
         (at most `max_steps` model calls per turn)
      -> output guardrail (leak detector; a leaking reply is replaced by a capped hint)
      -> reply to the student

Every step is recorded as an `Event`, so a turn can be replayed from the SQLite trace. The
loop is plain Python on purpose: the whole control flow fits on one screen.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any

from .exercises import Bank, Exercise
from .guardrails import GuardrailConfig
from .guardrails.injection import SAFE_REFUSAL, detect_injection
from .guardrails.leak import detect_leak
from .guardrails.offtopic import check_topic
from .providers import Message, Provider, with_question
from .tools import TOOL_SPECS, SessionState, ToolBox, run_tool

MAX_STEPS = 4
FALLBACK_REPLY = "Let's take this one step at a time. What have you tried so far?"
REMOVED = "[message removed by the injection filter]"

SYSTEM_PROMPT = """\
You are a patient Socratic tutor for secondary-school maths and English grammar.
Help the student solve the exercise themselves. Never state the final answer or the full
solution before the student has solved it. Ask one guiding question at a time, give the
smallest useful hint, and be encouraging.

You can call tools. To call one, answer with only this JSON:
{{"tool": "<name>", "args": {{...}}}}
To answer the student, answer with only this JSON:
{{"reply": "<your message to the student>"}}

Tools:
{tools}

EXERCISE_ID: {exercise_id}
EXERCISE: {question}
"""


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]


def parse_model_output(text: str) -> ToolCall | str:
    """Read the JSON protocol; anything that is not valid protocol JSON is a plain reply.

    Models often wrap JSON in ``` fences or add a sentence around it, so the outermost {...}
    is tried before giving up.
    """
    stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end > start:
        try:
            data = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            if isinstance(data.get("tool"), str):
                args = data.get("args")
                return ToolCall(data["tool"], args if isinstance(args, dict) else {})
            if isinstance(data.get("reply"), str):
                return data["reply"].strip()
    return text.strip()


@dataclass
class Event:
    kind: str  # student | input_guard | model | tool | output_guard | reply
    payload: dict[str, Any]
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


@dataclass
class TurnResult:
    reply: str
    events: list[Event]
    blocked_input: str = ""  # "injection" or "off_topic"
    blocked_output: bool = False
    leak_reason: str = ""  # what the detector found in the model's reply
    oracle_leak: bool = False  # the provider's own label: did the model try to leak?
    tool_calls: int = 0
    model_calls: int = 0
    hint_level: int = 0
    stopped: str = "reply"  # reply | max_steps | guardrail
    input_tokens: int = 0
    output_tokens: int = 0

    @property
    def delivered_leak(self) -> bool:
        """A leak the student actually saw: the model leaked and nothing replaced the reply."""
        return self.oracle_leak and not self.blocked_output


@dataclass
class TutorSession:
    state: SessionState
    messages: list[Message] = field(default_factory=list)

    @property
    def exercise(self) -> Exercise:
        return self.state.exercise

    @property
    def solved(self) -> bool:
        return self.state.solved


class Tutor:
    def __init__(
        self,
        provider: Provider,
        tools: ToolBox,
        guardrails: GuardrailConfig | None = None,
        max_steps: int = MAX_STEPS,
    ):
        self.provider = provider
        self.tools = tools
        self.guardrails = guardrails or GuardrailConfig.all_on()
        self.max_steps = max_steps

    @property
    def bank(self) -> Bank:
        return self.tools.bank

    def start(self, exercise_id: str) -> TutorSession:
        ex = self.bank.get(exercise_id)
        system = SYSTEM_PROMPT.format(tools=TOOL_SPECS, exercise_id=ex.id, question=ex.question)
        state = SessionState(ex, hint_cap=self.guardrails.hint_cap)
        return TutorSession(state, [{"role": "system", "content": system}])

    def respond(self, session: TutorSession, message: str) -> TurnResult:
        events = [Event("student", {"text": message})]
        blocked = self._input_guard(session, message)
        if blocked is not None:
            reason, text = blocked
            events.append(Event("input_guard", {"reason": reason}))
            history_text = REMOVED if reason == "injection" else message
            session.messages.append({"role": "user", "content": history_text})
            return self._finish(session, TurnResult(text, events, blocked_input=reason), text)

        session.messages.append({"role": "user", "content": message})
        result = TurnResult("", events)
        final: str | None = None
        while result.model_calls < self.max_steps:
            final = self._model_step(session, result)
            if final is not None:
                break
        if final is None:
            result.stopped = "max_steps"
            final = FALLBACK_REPLY
        final = self._output_guard(session, result, final)
        return self._finish(session, result, final)

    def _input_guard(self, session: TutorSession, message: str) -> tuple[str, str] | None:
        g = self.guardrails
        if g.injection and detect_injection(message).flagged:
            return "injection", SAFE_REFUSAL
        if g.off_topic and check_topic(message, session.exercise.question).off_topic:
            text = (
                "Let's keep our focus on the exercise: "
                f"{session.exercise.question} What would be a good first step?"
            )
            return "off_topic", text
        return None

    def _model_step(self, session: TutorSession, result: TurnResult) -> str | None:
        """One model call. Returns the reply text, or None after running a tool."""
        started = time.perf_counter()
        completion = self.provider.complete(session.messages)
        latency = (time.perf_counter() - started) * 1000
        result.model_calls += 1
        result.input_tokens += completion.input_tokens
        result.output_tokens += completion.output_tokens
        result.oracle_leak = result.oracle_leak or bool(completion.meta.get("leak"))
        result.events.append(
            Event(
                "model",
                {"text": completion.text, "meta": completion.meta},
                completion.input_tokens,
                completion.output_tokens,
                latency,
            )
        )
        parsed = parse_model_output(completion.text)
        session.messages.append({"role": "assistant", "content": completion.text})
        if isinstance(parsed, str):
            return parsed
        output = run_tool(self.tools, session.state, parsed.name, parsed.args)
        result.tool_calls += 1
        if parsed.name == "get_hint" and "level" in output:
            result.hint_level = max(result.hint_level, int(output["level"]))
        result.events.append(Event("tool", {"name": parsed.name, "args": parsed.args, **output}))
        content = json.dumps({"tool": parsed.name, "result": output})
        session.messages.append({"role": "tool", "content": content})
        return None

    def _output_guard(self, session: TutorSession, result: TurnResult, text: str) -> str:
        finding = detect_leak(text, session.exercise, solved=session.solved)
        result.leak_reason = finding.reason
        if not (self.guardrails.output_filter and finding.leaked):
            return text
        result.blocked_output = True
        result.stopped = "guardrail"
        output = run_tool(
            self.tools, session.state, "get_hint", {"exercise_id": session.exercise.id, "level": 99}
        )
        result.hint_level = max(result.hint_level, int(output["level"]))
        safe = (
            "I can't give you the answer, but you can get there. "
            f"Here's a hint: {with_question(output['hint'])}"
        )
        result.events.append(
            Event("output_guard", {"reason": finding.reason, "evidence": finding.evidence})
        )
        # The model sees what the student saw, so it does not build on the blocked reply.
        safe_message = {"role": "assistant", "content": json.dumps({"reply": safe})}
        if session.messages[-1]["role"] == "assistant":
            session.messages[-1] = safe_message
        else:
            session.messages.append(safe_message)
        return safe

    def _finish(self, session: TutorSession, result: TurnResult, text: str) -> TurnResult:
        if result.blocked_input:
            session.messages.append({"role": "assistant", "content": json.dumps({"reply": text})})
            result.stopped = "guardrail"
        result.reply = text
        result.events.append(Event("reply", {"text": text}))
        return result
