from __future__ import annotations

import pytest

from friday.config import Settings


def test_defaults_produce_a_working_simulation(settings):
    assert settings.provider == "mock"
    assert settings.is_simulation is True
    assert settings.model == "friday-sim-1"
    assert settings.approval == "dangerous"


def test_load_respects_env_overrides(monkeypatch, tmp_path):
    monkeypatch.setenv("FRIDAY_PROVIDER", "groq")
    monkeypatch.setenv("FRIDAY_MODEL", "mixtral-8x7b")
    monkeypatch.setenv("FRIDAY_GROQ_API_KEY_SHIM", "x")
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("FRIDAY_DATA_DIR", str(tmp_path / "d"))
    monkeypatch.setenv("FRIDAY_APPROVAL", "always")

    loaded = Settings.load(tmp_path / "absent.yaml")
    assert loaded.provider == "groq"
    assert loaded.model == "mixtral-8x7b"
    assert loaded.api_key == "gsk_test"
    assert loaded.approval == "always"
    assert loaded.base_url.endswith("api.groq.com/openai/v1")


def test_keyed_provider_without_key_falls_back_to_simulation(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("FRIDAY_PROVIDER", "openai")
    loaded = Settings.load("/nonexistent/friday.yaml")
    assert loaded.provider == "mock"
    assert loaded.is_simulation is True


def test_explicit_base_url_may_run_without_a_key(monkeypatch):
    monkeypatch.setenv("FRIDAY_PROVIDER", "openai")
    monkeypatch.setenv("FRIDAY_BASE_URL", "http://localhost:1234/v1")
    loaded = Settings.load("/nonexistent/friday.yaml")
    assert loaded.provider == "openai"
    assert loaded.is_simulation is False


def test_yaml_file_is_loaded(tmp_path):
    config = tmp_path / "friday.yaml"
    config.write_text(
        "provider: mock\napproval: always\nmax_steps: 3\ntemperature: 0.1\npersona: stark\n",
        encoding="utf-8",
    )
    loaded = Settings.load(config)
    assert loaded.approval == "always"
    assert loaded.max_steps == 3
    assert loaded.temperature == 0.1
    assert loaded.persona == "stark"


def test_invalid_approval_is_rejected(settings):
    with pytest.raises(ValueError):
        Settings(approval="sometimes", workspace=settings.workspace, data_dir=settings.data_dir)


def test_public_dict_never_leaks_the_key(settings):
    settings.api_key = "sk-super-secret"
    public = settings.public_dict()
    assert "api_key" not in public
    assert "sk-super-secret" not in str(public)


def test_ensure_dirs_creates_the_database_parent(settings):
    settings.ensure_dirs()
    assert settings.data_dir.is_dir()
    assert settings.db_file.parent.is_dir()
