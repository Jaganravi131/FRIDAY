PYTHON ?= python3
VENV   ?= .venv
BIN    := $(VENV)/bin

.PHONY: help install dev test lint wheel serve docker-build docker-run clean

help:
	@echo "FRIDAY"
	@echo "  make install       core only (zero dependencies)"
	@echo "  make dev           core + dev tools + the web HUD"
	@echo "  make test          full suite"
	@echo "  make lint          ruff (pyflakes + import order)"
	@echo "  make wheel         build a wheel and smoke-test it in a scratch venv"
	@echo "  make serve         run the HUD on 127.0.0.1:8642"
	@echo "  make docker-build  build the container"
	@echo "  make docker-run    run it with a named volume"

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/python -m pip install --upgrade pip

install: $(BIN)/python
	$(BIN)/pip install -e .

dev: $(BIN)/python
	$(BIN)/pip install -e ".[dev,serve]"

test: dev
	$(BIN)/python -m pytest

lint: dev
	$(BIN)/ruff check friday tests

wheel: dev
	$(BIN)/python -m build --wheel
	$(BIN)/python scripts/check_wheel.py dist

serve: dev
	$(BIN)/friday serve --host 127.0.0.1 --port 8642

docker-build:
	docker build -t friday:local .

docker-run:
	docker run --rm -p 8642:8642 -v friday-data:/data friday:local

clean:
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
