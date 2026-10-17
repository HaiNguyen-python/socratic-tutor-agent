"""Find numbers in free text and compare them with an answer.

Both the answer checker and the leak detector need the same question answered: "does this
text contain a value equal to the answer?" Students and models write the same value in many
ways (4, 4.0, 8/2, x = 4, four, 0.5, 1/2, 0,5, \\frac{1}{2}), so values are parsed into exact
`Fraction`s and compared exactly. A decimal that is the answer *rounded* to at least two
places (0.33 for 1/3) also counts, because that is how people write repeating decimals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from fractions import Fraction

UNITS = (
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen "
    "fifteen sixteen seventeen eighteen nineteen"
).split()
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70}
WORD_VALUES: dict[str, int] = {w: i for i, w in enumerate(UNITS)} | TENS
DENOMINATORS = {
    "half": 2,
    "halves": 2,
    "third": 3,
    "thirds": 3,
    "quarter": 4,
    "quarters": 4,
    "fourth": 4,
    "fourths": 4,
    "fifth": 5,
    "fifths": 5,
    "sixth": 6,
    "sixths": 6,
    "eighth": 8,
    "eighths": 8,
    "tenth": 10,
    "tenths": 10,
}

# A minus sign only counts when it is not a subtraction: "12-8" is 12 and 8, "x = -2" is -2.
_SIGN = r"(?:(?<![\w)])-)?"
_NUMBER_RE = re.compile(
    rf"""
    \\frac\{{\s*(?P<lnum>{_SIGN}\d+)\s*\}}\{{\s*(?P<lden>\d+)\s*\}}
    | (?P<whole>{_SIGN}\d+)\s+(?P<mnum>\d+)\s*/\s*(?P<mden>\d+)(?!\d|\.\d)
    | (?P<num>{_SIGN}\d+)\s*/\s*(?P<den>\d+)(?!\d|\.\d)
    | (?P<thousands>{_SIGN}\d{{1,3}}(?:,\d{{3}})+)(?!\d|\.\d)
    | (?P<comma>{_SIGN}\d+,\d{{1,2}})(?!\d)
    | (?P<dec>{_SIGN}(?:\d+(?:\.\d+)?|\.\d+))
    """,
    re.VERBOSE,
)
_WORD_NUMBER = r"(?:{tens})(?:-(?:{units}))?|{units}|a|an".format(
    tens="|".join(TENS), units="|".join(UNITS)
)
_WORD_RE = re.compile(
    rf"\b(?P<word>{_WORD_NUMBER})(?:\s+(?P<denom>{'|'.join(DENOMINATORS)}))?\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class NumberToken:
    """A value found in text; `decimals` is set for decimals so rounding can be judged."""

    value: Fraction
    text: str
    decimals: int | None = None


def _word_value(word: str) -> int | None:
    word = word.lower()
    if word in ("a", "an"):
        return 1
    if "-" in word:
        tens, unit = word.split("-")
        return TENS[tens] + WORD_VALUES[unit]
    return WORD_VALUES.get(word)


def _digit_tokens(text: str) -> list[tuple[int, NumberToken]]:
    found: list[tuple[int, NumberToken]] = []
    for m in _NUMBER_RE.finditer(text):
        g = m.groupdict()
        if g["lnum"] is not None:
            value = Fraction(int(g["lnum"]), int(g["lden"])) if int(g["lden"]) else None
            token = NumberToken(value, m.group(0)) if value is not None else None
        elif g["whole"] is not None:
            whole, num, den = int(g["whole"]), int(g["mnum"]), int(g["mden"])
            sign = -1 if g["whole"].startswith("-") else 1
            token = (
                NumberToken(whole + sign * Fraction(num, den), m.group(0))
                if den and num < den
                else None
            )
            if token is None:  # "2 6/3" is not a mixed number: read the two parts separately
                found.append((m.start("whole"), NumberToken(Fraction(whole), g["whole"])))
                token = NumberToken(Fraction(num, den), f"{num}/{den}") if den else None
        elif g["num"] is not None:
            den = int(g["den"])
            token = NumberToken(Fraction(int(g["num"]), den), m.group(0)) if den else None
        elif g["thousands"] is not None:
            token = NumberToken(Fraction(g["thousands"].replace(",", "")), m.group(0))
        elif g["comma"] is not None:
            raw = g["comma"].replace(",", ".")
            token = NumberToken(Fraction(raw), m.group(0), len(raw.split(".")[1]))
        else:
            raw = g["dec"]
            decimals = len(raw.split(".")[1]) if "." in raw else None
            token = NumberToken(Fraction(raw), m.group(0), decimals)
        if token is not None:
            found.append((m.start(), token))
    return found


def _word_tokens(text: str) -> list[tuple[int, NumberToken]]:
    found: list[tuple[int, NumberToken]] = []
    for m in _WORD_RE.finditer(text):
        value = _word_value(m.group("word"))
        denom = m.group("denom")
        if value is None:
            continue
        if denom:
            found.append(
                (m.start(), NumberToken(Fraction(value, DENOMINATORS[denom.lower()]), m[0]))
            )
        elif m.group("word").lower() not in ("a", "an"):
            found.append((m.start(), NumberToken(Fraction(value), m.group(0))))
    return found


def find_numbers(text: str) -> list[NumberToken]:
    """All numeric values in `text`, in reading order (digits, fractions and number words)."""
    tokens = _digit_tokens(text) + _word_tokens(text)
    return [token for _, token in sorted(tokens, key=lambda pair: pair[0])]


def parse_value(text: str) -> Fraction:
    """Parse a single answer such as '3/4', '-2', '1 1/2' or '4.5' (used for the bank)."""
    tokens = find_numbers(text)
    if len(tokens) != 1:
        raise ValueError(f"expected exactly one number in {text!r}, found {len(tokens)}")
    return tokens[0].value


def equivalent(token: NumberToken, answer: Fraction) -> bool:
    """True if `token` is the answer exactly, or the answer rounded to >= 2 decimal places.

    Two places minimum: '0.3' is not accepted for 1/3, but '0.33' is.
    """
    if token.value == answer:
        return True
    if token.decimals is None or token.decimals < 2:
        return False
    return round(answer, token.decimals) == token.value


def contains_value(text: str, answer: Fraction) -> NumberToken | None:
    """The first number in `text` equivalent to `answer`, or None."""
    return next((t for t in find_numbers(text) if equivalent(t, answer)), None)


def format_value(value: Fraction) -> str:
    """Write a value the way the bank does: '4', '-2', '3/4'."""
    return str(value.numerator) if value.denominator == 1 else f"{value}"
