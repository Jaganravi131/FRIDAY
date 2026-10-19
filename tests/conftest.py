"""Shared fixtures: a disposable workspace, an isolated DB and a zero-latency simulator."""

from __future__ import annotations

import pytest

from friday.config import Settings


@pytest.fixture(autouse=True)
def _no_sim_delay(monkeypatch):
    monkeypatch.setenv("FRIDAY_SIM_PACE", "0")


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "workspace"
    root.mkdir()
    (root / "README.md").write_text("# Demo project\n\nFRIDAY lives here.\n", encoding="utf-8")
    (root / "notes.txt").write_text("alpha\nbeta\nneedle in a haystack\n", encoding="utf-8")
    (root / "src").mkdir()
    (root / "src" / "app.py").write_text("def hello():\n    return 'world'\n", encoding="utf-8")
    return root


@pytest.fixture
def settings(tmp_path, workspace):
    return Settings(
        provider="mock",
        workspace=workspace,
        data_dir=tmp_path / "data",
        approval="dangerous",
    )
