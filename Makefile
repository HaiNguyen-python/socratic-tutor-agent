.PHONY: install test lint typecheck format check eval report chat serve

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

typecheck:
	mypy src

format:
	ruff format .
	ruff check --fix .

# The same gate as CI
check: lint typecheck
	pytest --cov-fail-under=80

# 280 simulated conversations (4 conditions), traces in reports/traces.db
eval:
	tutor eval --by-persona

# docs/results.md and docs/img/*.png (a few seconds)
report:
	tutor report

chat:
	tutor chat

serve:
	tutor serve
