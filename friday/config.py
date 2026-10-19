"""Configuration loading: sane defaults, an optional YAML file, then environment overrides.

Precedence (lowest to highest): built-in defaults < friday.yaml < FRIDAY_* environment.
Secrets are never written to disk by FRIDAY and never exposed through the API.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

PROVIDER_DEFAULT_MODELS: dict[str, str] = {
    "openai": "gpt-4o-mini",
    "anthropic": "claude-sonnet-4-20250514",
    "openrouter": "openai/gpt-4o-mini",
    "groq": "llama-3.3-70b-versatile",
    "ollama": "llama3.2",
    "mock": "friday-sim-1",
}

PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "ollama": "http://localhost:11434/v1",
}

ANTHROPIC_PROVIDERS = {"anthropic"}

APPROVAL_MODES = {"never", "dangerous", "always"}


def _env(name: str) -> str | None:
    value = os.environ.get(f"FRIDAY_{name}")
    return value if value not in (None, "") else None


def _as_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_list(value: str | list[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [item.strip() for item in value.replace(";", ",").split(",") if item.strip()]


class Settings(BaseModel):
    """Runtime configuration for a FRIDAY instance."""

    model_config = ConfigDict(extra="ignore")

    # Brain
    provider: str = "auto"
    model: str = ""
    base_url: str = ""
    api_key: str = ""
    temperature: float = 0.4
    max_tokens: int = 2048

    # Loop discipline
    max_steps: int = 12
    max_tool_calls: int = 24
    max_history_messages: int = 80
    approval: str = "dangerous"

    # Filesystem
    workspace: Path = Field(default_factory=Path.cwd)
    data_dir: Path = Field(default_factory=lambda: Path.cwd() / ".friday")
    db_path: Path | None = None

    # Interface
    host: str = "0.0.0.0"
    port: int = 8000
    persona: str = "friday"

    # Extensibility / safety
    disabled_tools: list[str] = Field(default_factory=list)
    plugin_dirs: list[str] = Field(default_factory=list)
    web_search_provider: str = "auto"
    tavily_api_key: str = ""
    request_timeout: float = 90.0

    @field_validator("approval")
    @classmethod
    def _valid_approval(cls, value: str) -> str:
        value = (value or "dangerous").lower()
        if value not in APPROVAL_MODES:
            raise ValueError(f"approval must be one of {sorted(APPROVAL_MODES)}")
        return value

    @field_validator("provider")
    @classmethod
    def _valid_provider(cls, value: str) -> str:
        return (value or "auto").lower().strip()

    def model_post_init(self, __context: Any) -> None:
        self.workspace = Path(self.workspace).expanduser().resolve()
        self.data_dir = Path(self.data_dir).expanduser().resolve()
        if self.db_path is None:
            self.db_path = self.data_dir / "friday.db"
        else:
            self.db_path = Path(self.db_path).expanduser().resolve()
        self._resolve_provider()

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, config_path: str | Path | None = None) -> Settings:
        raw: dict[str, Any] = {}

        candidate = Path(config_path) if config_path else Path(_env("CONFIG") or "friday.yaml")
        if candidate.is_file():
            loaded = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
            if not isinstance(loaded, dict):
                raise ValueError(f"{candidate} must contain a YAML mapping")
            raw.update(loaded)

        # ---- environment overrides ----
        env_map: dict[str, tuple[str, Any]] = {
            "provider": ("PROVIDER", str),
            "model": ("MODEL", str),
            "base_url": ("BASE_URL", str),
            "temperature": ("TEMPERATURE", float),
            "max_tokens": ("MAX_TOKENS", int),
            "max_steps": ("MAX_STEPS", int),
            "max_tool_calls": ("MAX_TOOL_CALLS", int),
            "max_history_messages": ("MAX_HISTORY_MESSAGES", int),
            "approval": ("APPROVAL", str),
            "workspace": ("WORKSPACE", str),
            "data_dir": ("DATA_DIR", str),
            "db_path": ("DB_PATH", str),
            "host": ("HOST", str),
            "port": ("PORT", int),
            "persona": ("PERSONA", str),
            "web_search_provider": ("WEB_SEARCH_PROVIDER", str),
            "request_timeout": ("REQUEST_TIMEOUT", float),
        }
        tavily_key = _env("TAVILY_API_KEY") or os.environ.get("TAVILY_API_KEY")
        if tavily_key:
            raw["tavily_api_key"] = tavily_key
        for field_name, (env_name, caster) in env_map.items():
            value = _env(env_name)
            if value is not None:
                try:
                    raw[field_name] = caster(value)
                except ValueError as exc:
                    raise ValueError(f"invalid FRIDAY_{env_name}: {value!r}") from exc

        disabled = _env("DISABLED_TOOLS")
        if disabled is not None:
            raw["disabled_tools"] = _as_list(disabled)
        plugins = _env("PLUGIN_DIRS")
        if plugins is not None:
            raw["plugin_dirs"] = _as_list(plugins)

        return cls(**raw)

    def _resolve_provider(self) -> None:
        """Pick a provider (and its key/model defaults) when set to auto."""
        explicit_key = self.api_key or _env("API_KEY") or ""
        explicit_base = bool(self.base_url)
        self.api_key = explicit_key

        if self.provider == "auto":
            if explicit_key and self.base_url or os.environ.get("OPENAI_API_KEY"):
                self.provider = "openai"
            elif os.environ.get("ANTHROPIC_API_KEY"):
                self.provider = "anthropic"
            elif os.environ.get("OPENROUTER_API_KEY"):
                self.provider = "openrouter"
            elif os.environ.get("GROQ_API_KEY"):
                self.provider = "groq"
            elif os.environ.get("OLLAMA_HOST"):
                self.provider = "ollama"
            else:
                self.provider = "mock"

        if not self.api_key:
            key_env = {
                "openai": "OPENAI_API_KEY",
                "anthropic": "ANTHROPIC_API_KEY",
                "openrouter": "OPENROUTER_API_KEY",
                "groq": "GROQ_API_KEY",
            }.get(self.provider)
            if key_env:
                self.api_key = os.environ.get(key_env, "")

        if not self.base_url:
            self.base_url = PROVIDER_BASE_URLS.get(self.provider, "")

        keyed = {"openai", "anthropic", "openrouter", "groq"}
        if self.provider in keyed and not self.api_key and not explicit_base:
            # A key was requested but none exists; be honest about running the demo core.
            self.provider = "mock"

        if not self.model:
            self.model = PROVIDER_DEFAULT_MODELS.get(self.provider, "")

        if self.provider == "mock":
            self.api_key = ""
            self.base_url = ""

    # ------------------------------------------------------------- accessors
    @property
    def is_simulation(self) -> bool:
        """True when no real model is configured; the deterministic demo core runs."""
        return self.provider == "mock"

    @property
    def db_file(self) -> Path:
        return Path(self.db_path or self.data_dir / "friday.db")

    @property
    def static_dir(self) -> Path:
        return Path(__file__).parent / "server" / "static"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.db_file.parent.mkdir(parents=True, exist_ok=True)

    def public_dict(self) -> dict[str, Any]:
        """Configuration safe to ship to a browser — never includes the API key."""
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "max_steps": self.max_steps,
            "max_tool_calls": self.max_tool_calls,
            "approval": self.approval,
            "workspace": str(self.workspace),
            "simulation": self.is_simulation,
            "persona": self.persona,
            "disabled_tools": self.disabled_tools,
        }
