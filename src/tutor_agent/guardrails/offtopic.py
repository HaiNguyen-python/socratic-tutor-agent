"""Off-topic filter for student messages.

A message is off-topic only when it has an off-topic signal (football, games, films...) and
*no* on-topic signal. On-topic signals: any digit, maths or grammar vocabulary, a learning
word ("hint", "stuck", "answer"), or a content word shared with the current question. The
shared-word rule matters: "is it about football?" is fine when the exercise is about a brother
who watches football. Short replies ("ok", "thanks", "I don't know") always pass.

Being conservative is deliberate: blocking a real question costs more than one chatty turn.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

OFF_TOPIC_WORDS = frozenset(
    "football soccer hockey basketball match goal game games gaming minecraft fortnite roblox "
    "movie movies film films netflix tiktok instagram youtube youtuber snapchat music song songs "
    "band concert weather pizza burger party weekend holiday birthday dog cat pet celebrity "
    "anime marvel girlfriend boyfriend crush phone".split()
)
ON_TOPIC_WORDS = frozenset(
    "x equation solve solving side sides both subtract add plus minus times multiply divide "
    "divided fraction fractions numerator denominator simplify simplest common reciprocal "
    "percent per cent decimal mixed improper total sum difference product equals equal value "
    "unknown variable bracket brackets expand perimeter width length speed distance time cost "
    "verb verbs tense past present perfect participle irregular regular plural singular "
    "comparative superlative adjective noun sentence gap word grammar spell spelling form "
    "hint hints answer answers stuck help understand explain mean step steps try tried check "
    "wrong right correct question exercise problem quiz practice maths math".split()
)
_WORD_RE = re.compile(r"[a-z]+")


@dataclass(frozen=True)
class TopicFinding:
    off_topic: bool
    trigger: str = ""


def _content_words(text: str) -> set[str]:
    return {w for w in _WORD_RE.findall(text.lower()) if len(w) >= 4}


def check_topic(message: str, question: str) -> TopicFinding:
    """Is this student message off-topic for the current exercise?"""
    text = message.lower()
    words = set(_WORD_RE.findall(text))
    if re.search(r"\d", text) or words & ON_TOPIC_WORDS:
        return TopicFinding(False)
    shared = (words & _content_words(question)) - OFF_TOPIC_WORDS
    off = sorted((words & OFF_TOPIC_WORDS) - _content_words(question))
    if off and not shared:
        return TopicFinding(True, off[0])
    return TopicFinding(False)
