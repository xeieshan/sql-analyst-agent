.PHONY: install test lint data eval report

MODEL ?= oracle

install:
	pip install -e ".[dev]"

test:
	pytest

lint:
	ruff check .
	ruff format --check .

data:
	sql-agent download

