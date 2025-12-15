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

eval:
	sql-agent eval --model $(MODEL)

report:
	sql-agent report

# Maintenance pass: 2025-12-15 10:48
