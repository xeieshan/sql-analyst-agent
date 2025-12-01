.PHONY: install test lint data eval report

MODEL ?= oracle

install:
	pip install -e ".[dev]"

