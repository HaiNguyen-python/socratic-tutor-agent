"""BM25 search over the course notes, written from scratch (no search library).

The notes are split into sections at each `## ` heading, so a hit is a short, focused passage
the tutor can quote. BM25 is enough here: the notes are small and students ask with the same
words the notes use ("denominator", "reciprocal", "past participle").
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

STOPWORDS = frozenset(
    "a an and are as at be by do does for from how i in is it me my of on or so the this to "
    "what when why with you your can don t understand mean means explain".split()
)


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens without stopwords; a trailing plural 's' is dropped."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words if w not in STOPWORDS]


@dataclass(frozen=True)
class Section:
    doc: str
    heading: str
    text: str


@dataclass(frozen=True)
class Hit:
    section: Section
    score: float


def split_sections(doc: str, markdown: str) -> list[Section]:
    sections: list[Section] = []
    heading, lines = "", list[str]()
    for line in markdown.splitlines():
        if line.startswith("## "):
            if heading:
                sections.append(Section(doc, heading, " ".join(" ".join(lines).split())))
            heading, lines = line[3:].strip(), []
        elif heading:
            lines.append(line)
    if heading:
        sections.append(Section(doc, heading, " ".join(" ".join(lines).split())))
    return sections


class BM25Index:
    """Okapi BM25 with the usual k1 = 1.5 and b = 0.75."""

    def __init__(self, sections: list[Section], k1: float = 1.5, b: float = 0.75):
        self.sections = sections
        self.k1, self.b = k1, b
        # The heading is counted twice: a section titled "Numerator and denominator" should
        # beat a section that only uses the word in passing.
        self._docs = [Counter(tokenize(f"{s.heading} {s.heading} {s.text}")) for s in sections]
        self._lengths = [sum(d.values()) for d in self._docs]
        self._avg_len = sum(self._lengths) / max(len(self._docs), 1)
        df: Counter[str] = Counter()
        for doc in self._docs:
            df.update(doc.keys())
        n = len(self._docs)
        # The "+1" form of IDF keeps every weight positive, even for very common terms.
        self._idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def score(self, query: str, i: int) -> float:
        doc, length = self._docs[i], self._lengths[i]
        total = 0.0
        for term in tokenize(query):
            tf = doc.get(term, 0)
            if tf == 0:
                continue
            norm = tf + self.k1 * (1 - self.b + self.b * length / self._avg_len)
            total += self._idf[term] * tf * (self.k1 + 1) / norm
        return total

    def search(self, query: str, k: int = 2) -> list[Hit]:
        """Top-k sections with a positive score, best first (ties keep file order)."""
        scored = [Hit(s, self.score(query, i)) for i, s in enumerate(self.sections)]
        ranked = sorted((h for h in scored if h.score > 0), key=lambda h: -h.score)
        return ranked[:k]


def load_notes(directory: Path | None = None) -> BM25Index:
    """Index the packaged notes, or every `*.md` file in `directory`."""
    sections: list[Section] = []
    if directory is None:
        root = resources.files("tutor_agent").joinpath("data/notes")
        files = sorted(
            (p.name, p.read_text("utf-8")) for p in root.iterdir() if p.name.endswith(".md")
        )
    else:
        files = sorted((p.name, p.read_text("utf-8")) for p in directory.glob("*.md"))
    for name, text in files:
        sections.extend(split_sections(name.removesuffix(".md"), text))
    return BM25Index(sections)
