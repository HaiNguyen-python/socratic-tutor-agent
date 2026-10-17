import pytest

from tutor_agent.bm25 import BM25Index, Section, tokenize
from tutor_agent.exercises import check
from tutor_agent.tools import SessionState, run_tool


@pytest.mark.parametrize(
    ("exercise_id", "attempt", "correct"),
    [
        ("lin-01", "x = 4", True),
        ("lin-01", "4.0", True),
        ("lin-01", "2x = 8 so x = 4", True),  # the last number counts
        ("lin-01", "4 or 5?", False),
        ("lin-08", "x = 0.5", True),
        ("fra-01", "0.75", True),
        ("fra-07", "0.83", True),  # rounded repeating decimal
        ("fra-07", "0.8", False),
        ("wor-05", "4 1/2 cups", True),
        ("lin-01", "no idea", False),
        ("gra-01", "Is it went?", True),
        ("gra-01", "I think it's 'went'", True),
        ("gra-01", "Yesterday I went to the library.", True),
        ("gra-01", "goed", False),
        ("gra-03", "more easy", False),
    ],
)
def test_check_answer_forms(bank, exercise_id, attempt, correct):
    assert check(bank.get(exercise_id), attempt) is correct


def test_check_answer_tool_counts_attempts_until_solved(box, state):
    assert run_tool(box, state, "check_answer", {"exercise_id": "lin-01", "answer": "5"}) == {
        "correct": False,
        "attempts": 1,
    }
    assert run_tool(box, state, "check_answer", {"exercise_id": "lin-01", "answer": "x=4"})[
        "correct"
    ]
    assert state.solved and state.attempts == 2
    run_tool(box, state, "check_answer", {"exercise_id": "lin-01", "answer": "7"})
    assert state.solved and state.attempts == 2  # a solved exercise stays solved


def test_get_hint_tool_applies_the_cap_and_can_be_switched_off(box, bank):
    capped = SessionState(bank.get("lin-01"), hint_cap=True)
    out = run_tool(box, capped, "get_hint", {"exercise_id": "lin-01", "level": 4})
    assert out["level"] == 1 and out["capped"] and out["hint"] == bank.get("lin-01").hints[0]
    free = SessionState(bank.get("lin-01"), hint_cap=False)
    assert run_tool(box, free, "get_hint", {"exercise_id": "lin-01", "level": "4"})["level"] == 4


@pytest.mark.parametrize(
    ("name", "args", "error_part"),
    [
        ("get_hint", {"exercise_id": "fra-01", "level": 1}, "only the current exercise"),
        ("delete_bank", {}, "unknown tool"),
        ("get_hint", {"exercise_id": "lin-01"}, "level"),
        ("get_hint", {"exercise_id": "lin-01", "level": "two"}, "invalid literal"),
    ],
)
def test_bad_tool_calls_return_errors_instead_of_raising(box, state, name, args, error_part):
    assert error_part in run_tool(box, state, name, args)["error"]


def test_make_quiz_excludes_the_current_exercise(box, state):
    out = run_tool(box, state, "make_quiz", {"topic": "linear-equations", "n": 3})
    ids = [q["id"] for q in out["questions"]]
    assert len(ids) == 3 and "lin-01" not in ids
    assert "error" in run_tool(box, state, "make_quiz", {"topic": "chemistry", "n": 2})


def test_search_notes_finds_the_right_section(box, state):
    hits = run_tool(box, state, "search_notes", {"query": "What is a denominator?"})["hits"]
    assert hits[0]["doc"] == "fractions" and hits[0]["heading"] == "Numerator and denominator"
    hits = run_tool(box, state, "search_notes", {"query": "past participle of irregular verbs"})
    assert hits["hits"][0]["doc"] == "english-grammar"


def test_bm25_scores_rare_terms_higher_and_ignores_unknown_words():
    sections = [
        Section("a", "one", "apple apple banana"),
        Section("b", "two", "banana banana banana cherry"),
        Section("c", "three", "banana"),
    ]
    index = BM25Index(sections)
    assert [h.section.doc for h in index.search("cherry")] == ["b"]
    assert index.search("apple")[0].section.doc == "a"
    assert index.search("zebra") == []
    assert tokenize("What are the Fractions?") == ["fraction"]
