.PHONY: help install serve chat test lint fmt check docker demo clean

PY ?= python3
VENV ?= .venv
BIN := $(VENV)/bin

help:
	@echo "FRIDAY — make targets"
	@echo "  install   create .venv and install the package with dev deps"
	@echo "  serve     run the HUD on http://localhost:8000"
	@echo "  chat      interactive terminal session"
	@echo "  demo      one-shot demo against the simulation core"
	@echo "  test      run the test suite"
	@echo "  lint      ruff check"
	@echo "  fmt       ruff format"
	@echo "  check     lint + tests"
	@echo "  docker    build the container image"
	@echo "  clean     remove caches (keeps .friday data)"

install:
	$(PY) -m venv $(VENV)
	$(BIN)/pip install -U pip
	$(BIN)/pip install -e ".[dev]"

serve:
	$(BIN)/friday serve

chat:
	$(BIN)/friday chat

demo:
	FRIDAY_SIM_PACE=0.015 $(BIN)/friday run "system status and what is 17 * 42 + 9"

test:
	$(BIN)/pytest -q

lint:
	$(BIN)/ruff check friday tests

fmt:
	$(BIN)/ruff format friday tests

check: lint test

docker:
	docker build -t friday:latest .

clean:
	rm -rf .pytest_cache .ruff_cache build dist *.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
