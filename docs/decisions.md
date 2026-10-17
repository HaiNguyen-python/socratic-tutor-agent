# Design decisions

Short records of the main choices: what was decided, why, and what it costs. Format inspired by
Architecture Decision Records (ADRs).

## 1. A plain-Python agent loop, no framework

- **Decision:** the loop is about 60 lines in `agent.py`: call the model, parse JSON, run a
  tool, repeat up to `max_steps`.
- **Why:** the guardrails need to sit at exact points (before the model, around tools, after
  the reply). In plain code those points are visible and testable. A framework would hide them
  behind callbacks.
- **Cost:** no streaming, no parallel tool calls, no retries on HTTP errors. Fine for a tutor
  that makes one or two tool calls per turn.

## 2. Tool calls as JSON in text, not native function calling

- **Decision:** the model answers with `{"tool": ...}` or `{"reply": ...}` in plain text.
- **Why:** it works with every provider, including small Ollama models that have no native tool
  calling, and the scripted provider speaks the same protocol, so tests exercise the real parser.
- **Cost:** models sometimes break the format. The parser is lenient (fences, surrounding text),
  and anything unparsable becomes a plain reply, which still goes through the output filter.

## 3. A deterministic scripted model for CI and for the headline numbers

- **Decision:** `ScriptedProvider` is the "model" in tests and in `docs/results.md`; Ollama and
  Anthropic providers exist but are not run here.
- **Why:** CI has no GPU, no network and no keys, and results must be reproducible. A scripted
  model also lets me plant known weaknesses and know the ground truth for every leak.
- **Cost:** the numbers describe the *guardrails and harness* on a simple policy, not how well a
  real LLM tutors. A real model will leak in ways I did not script. The README says this next to
  the results and shows how to rerun with Ollama.

## 4. Planted weaknesses, labelled by the provider

- **Decision:** the scripted model obeys injections and gives in after three demands; the leaky
  model leaks in four styles; both report `meta["leak"]`.
- **Why:** without real failures, a guardrail test shows nothing. Planting them gives a
  measurable before/after and a ground truth that is independent of the detector. Otherwise the
  detector would grade itself.
- **Cost:** the leak rates measure my planted failure mix. The `disguised` style is built to beat
  the detector, so the catch rate (62%) is a property of that mix, not a general number.

## 5. The bank never contains the answer outside `answer` and `steps`

- **Decision:** no question or hint contains the answer in any form, checked by a test.
- **Why:** then the leak detector can be a simple "is the value in the reply?" check, and a hit
  means the tutor added it. It also shapes how hints are written: "simplify 6/15" for the answer
  2/5 would already be a leak, because 6/15 equals 2/5.
- **Cost:** some hints are less natural than a teacher would write ("divide 18 by 4" instead of
  "18/4").

## 6. Exact fractions and a rounding rule for numeric equivalence

- **Decision:** parse values to `Fraction`, compare exactly, and accept decimals that are the
  answer rounded to at least two places.
- **Why:** floats make `0.1 + 0.2` problems and cannot say whether "0.33" means 1/3. Exact
  fractions make 0.5 = 1/2 = 2/4 = one half trivially true; the two-place rule matches how people
  write repeating decimals without accepting "0.3" for 1/3.
- **Cost:** a numeric detector flags *any* number equal to the answer. In the eval every output
  false alarm comes from the hint label "level k of 4" when the answer is 4. I kept the label and
  report the false alarms instead of whitelisting "level k of n", because a whitelist is a hole
  ("level 4" is then never checked).

## 7. Injected messages never reach the model

- **Decision:** a flagged message gets a fixed refusal and is replaced in the history by a
  placeholder.
- **Why:** refusing once is not enough if the attack stays in the context and the next model call
  reads it. Removing it is the simplest guarantee.
- **Cost:** a false positive also loses the student's real content. In the eval, 5 (scripted) and
  3 (leaky) legitimate messages were blocked, all the planted "my teacher says we can ignore those
  instructions" case, including correct answers. Pattern lists are also easy to rephrase around;
  a learned classifier is listed under next steps.

## 8. Conservative off-topic filter

- **Decision:** block only when there is an off-topic word *and* no on-topic signal.
- **Why:** blocking a real question is worse than one chatty turn. Shared words with the question
  keep "football" legal for the football exercise.
- **Cost:** chatter without a listed word passes ("tell me about your day"). That is acceptable
  for a tutor; the model can still redirect.

## 9. The hint cap counts attempts, not time

- **Decision:** `allowed = min(requested, levels, attempts + 1, highest + 1)`.
- **Why:** the strongest hint is one step from the answer. Earning it with attempts keeps the
  student working; "no skipping" keeps hints proportional.
- **Cost:** students who ask for hints without trying get stuck at level 1. In the eval this
  slows answer-seekers (4.1 -> 5.8 turns) and lowers the off-topic persona's solve rate within
  8 turns (57% -> 43% with the scripted model). A real product might relax the cap after N turns.

## 10. The output filter replaces, it does not just delete

- **Decision:** a leaking reply becomes "I can't give you the answer, but you can get there.
  Here's a hint: ..." with the highest allowed hint, and the model's history gets that text.
- **Why:** an empty or generic reply wastes the student's turn; a hint keeps the session moving.
  Updating the history keeps the model consistent with what the student saw.
- **Cost:** the replacement can repeat a hint the student has already seen.

## 11. Same students in every condition

- **Decision:** each conversation has a seed from a stable hash of its id; all four conditions
  replay the same plan.
- **Why:** paired comparison: differences come from the tutor, not from luck.
- **Cost:** once the tutor's replies differ, the student's random stream diverges, so pairing is
  exact only up to the first different reply. With 70 conversations per condition the
  per-persona numbers (14 each) are noisy; I report counts, not confidence intervals.

## 12. Cost and latency are modelled, not measured

- **Decision:** estimate tokens (4 characters per token) and apply example rates in `CostModel`.
- **Why:** the scripted model costs nothing and answers in microseconds, which says nothing about
  a real deployment. Token counts are real properties of the prompts (the system prompt and
  history dominate), so the *relative* cost of guardrails is meaningful.
- **Cost:** absolute dollars and milliseconds depend on the model. Measured wall-clock latency of
  every model call is in the SQLite trace for real providers.

## 13. A deterministic judge plus hand labels, and an honest kappa

- **Decision:** a rule-based judge runs in CI; it is scored against 40 hand labels with Cohen's
  kappa; an `LLMJudge` exists for real models.
- **Why:** an LLM judge in CI would need a key and would not be reproducible. Comparing the
  stand-in with hand labels shows exactly where it fails: all three disagreements are disguised
  leaks that its leak check (the same detector) cannot see.
- **Cost:** the labels were written during the build with an AI assistant reading the
  transcripts, so they are one rater's view and not independent of the person who wrote the
  rubric. Labels carry a transcript hash: if the tutor changes, they are reported as stale.

## 14. The rubric requires proportional hints

- **Decision:** "good" needs no leak *and* proportional hints, plus one of the softer criteria.
- **Why:** my first rubric allowed "good" with any two of three soft criteria, which rated a tutor
  that hands out the last hint before any attempt as good. That contradicts the project's
  question, so I changed the rule before labelling.
- **Cost:** stricter labels; fewer conversations count as good without guardrails.
