import pytest

from tutor_agent.bm25 import load_notes
from tutor_agent.exercises import load_bank
from tutor_agent.tools import SessionState, ToolBox


@pytest.fixture(scope="session")
def bank():
    return load_bank()


@pytest.fixture(scope="session")
def notes():
    return load_notes()


@pytest.fixture()
def box(bank, notes):
    return ToolBox(bank, notes)


@pytest.fixture()
def state(bank):
    return SessionState(bank.get("lin-01"))
