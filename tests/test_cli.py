"""CLI and plugin-loader coverage: the surfaces a user touches first."""

from __future__ import annotations

import subprocess
import sys

import pytest

from friday.cli import build_parser, main
from friday.config import Settings
from friday.tools import build_registry


def test_parser_exposes_the_documented_commands():
    parser = build_parser()
    for command in ("serve", "chat", "run", "doctor", "tools", "sessions"):
        args = parser.parse_args([command] if command != "run" else [command, "hi"])
        assert callable(args.func)


def test_version_and_help_exit_cleanly(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert "FRIDAY" in capsys.readouterr().out


def test_no_command_prints_help(capsys):
    assert main([]) == 0
    assert "usage: friday" in capsys.readouterr().out.lower()


def test_doctor_reports_healthy(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("FRIDAY_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("FRIDAY_PROVIDER", "mock")
    monkeypatch.chdir(tmp_path)
    code = main(["doctor"])
    output = capsys.readouterr().out
    assert code == 0
    assert "PASS" in output
    assert "simulation core" in output


def test_tools_command_lists_the_toolkit(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("FRIDAY_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)
    assert main(["tools"]) == 0
    output = capsys.readouterr().out
    assert "read_file" in output
    assert "HIGH" in output


def test_plugin_loader_registers_tools(settings, tmp_path):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "echo.py").write_text(
        "from friday.tools.base import Tool\n"
        "from friday.types import ToolResult\n"
        "class EchoTool(Tool):\n"
        "    name = 'echo'\n"
        "    description = 'echoes'\n"
        "    parameters = {'type': 'object', 'properties': {'text': {'type': 'string'}},"
        " 'required': ['text']}\n"
        "    async def run(self, args, ctx):\n"
        "        return ToolResult(ok=True, output=args['text'])\n"
        "TOOLS = [EchoTool()]\n",
        encoding="utf-8",
    )
    settings.plugin_dirs = [str(plugin_dir)]
    registry, loaded = build_registry(settings)
    assert loaded == ["echo"]
    assert registry.get("echo") is not None
    assert "echo" in [schema.name for schema in registry.schemas()]


def test_broken_plugin_is_skipped_not_fatal(settings, tmp_path, capsys):
    plugin_dir = tmp_path / "plugins"
    plugin_dir.mkdir()
    (plugin_dir / "broken.py").write_text("raise RuntimeError('boom')\n", encoding="utf-8")
    settings.plugin_dirs = [str(plugin_dir)]
    registry, loaded = build_registry(settings)
    assert loaded == []
    assert len(registry.all()) >= 16


def test_entry_point_installed():
    """`friday` must be a real console script after `pip install -e .`."""
    result = subprocess.run(
        [sys.executable, "-m", "friday", "--version"], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0
    assert "FRIDAY" in result.stdout


def test_settings_from_yaml_and_env(tmp_path, monkeypatch):
    config = tmp_path / "friday.yaml"
    config.write_text("approval: always\nmax_steps: 4\n", encoding="utf-8")
    monkeypatch.setenv("FRIDAY_MAX_STEPS", "7")
    loaded = Settings.load(config)
    assert loaded.approval == "always"
    assert loaded.max_steps == 7  # environment wins over YAML
