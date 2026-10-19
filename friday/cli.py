"""Command-line interface: serve the HUD, chat in the terminal, inspect tools.

friday serve          # launch the web HUD
friday chat           # interactive terminal session
friday run "…"        # one-shot task
friday tools          # list the toolkit
friday sessions       # list saved sessions
friday doctor         # environment + provider diagnostics
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import textwrap
from typing import Any

from friday import __version__
from friday.config import Settings
from friday.engine import Agent, ApprovalRequest, PolicyApprover
from friday.llm import build_provider
from friday.memory import MemoryStore
from friday.tools import build_registry

CYAN = "\033[38;5;45m"
DIM = "\033[2m"
BOLD = "\033[1m"
AMBER = "\033[38;5;214m"
RED = "\033[38;5;203m"
GREEN = "\033[38;5;78m"
RESET = "\033[0m"


def _supports_color() -> bool:
    return sys.stdout.isatty()


def _c(text: str, color: str) -> str:
    return f"{color}{text}{RESET}" if _supports_color() else text


class TerminalApprover:
    async def decide(self, request: ApprovalRequest) -> tuple[bool, str]:
        prompt = (
            f"\n{_c('APPROVAL REQUIRED', AMBER)} · {_c(request.risk.upper(), RED)}\n"
            f"  {request.summary}\n  Allow? [y/N] "
        )
        try:
            answer = await asyncio.to_thread(input, prompt)
        except (EOFError, KeyboardInterrupt):
            return False, "denied at prompt"
        if answer.strip().lower() in {"y", "yes"}:
            return True, "approved at prompt"
        return False, "denied at prompt"


def _print_header(settings: Settings) -> None:
    mode = "SIMULATION" if settings.is_simulation else settings.provider.upper()
    print(
        _c("\n  F.R.I.D.A.Y. ", CYAN)
        + _c(f"v{__version__}", DIM)
        + _c(f"  ·  {mode}  ·  {settings.model or 'no model'}", DIM)
    )
    print(_c(f"  workspace: {settings.workspace}", DIM))
    if settings.is_simulation:
        print(_c("  simulation core · set OPENAI_API_KEY / ANTHROPIC_API_KEY for a real model", AMBER))
    print(_c("  /new  /sessions  /tools  /approval <mode>  /quit\n", DIM))


async def _one_turn(agent: Agent, session_id: str, text: str) -> str:
    """Render the event stream in the terminal; returns the final answer text."""
    answer: list[str] = []
    streaming = False
    async for event in agent.run(session_id, text):
        kind = event["type"]
        if kind == "token":
            if not streaming:
                print(_c("\n  FRIDAY ", CYAN), end="")
                streaming = True
            print(event["text"], end="", flush=True)
            answer.append(event["text"])
        elif kind == "tool_call":
            print(
                _c(f"\n  ⚙ {event['name']}", AMBER)
                + _c(f" {json.dumps(event['arguments'], ensure_ascii=False)[:100]}", DIM)
            )
        elif kind == "tool_result":
            status = _c("ok", GREEN) if event["ok"] else _c("failed", RED)
            preview = " ".join(event["output"].split())[:140]
            print(_c(f"    ↳ {status} {event['duration_ms']}ms · {preview}", DIM))
        elif kind == "approval_required":
            pass  # TerminalApprover prompts
        elif kind == "notice":
            print(_c(f"\n  ! {event['message']}", AMBER))
        elif kind == "error":
            print(_c(f"\n  ✖ {event['message']}", RED))
        elif kind == "done":
            usage = event.get("usage", {})
            if streaming:
                print()
            print(
                _c(
                    f"  {event['stop_reason']} · {event['steps']} step(s) · "
                    f"{usage.get('total_tokens', 0)} tokens · {event['duration_ms']}ms",
                    DIM,
                )
            )
    return "".join(answer)


def _make_agent(settings: Settings) -> tuple[Agent, MemoryStore]:
    memory = MemoryStore(settings.db_file)
    registry, _plugins = build_registry(settings)
    provider = build_provider(settings)
    approver = None if settings.approval == "never" else TerminalApprover()
    if approver is None:
        approver = PolicyApprover(True)
    return Agent(settings, provider, registry, memory, approver=approver), memory


async def _chat(settings: Settings, session_id: str | None) -> int:
    settings.ensure_dirs()
    agent, memory = _make_agent(settings)
    session_id = memory.ensure_session(session_id)
    _print_header(settings)
    if session_id:
        existing = memory.get_messages(session_id, limit=200)
        if existing:
            print(_c(f"  resuming session {session_id} ({len(existing)} messages)\n", DIM))

    try:
        while True:
            try:
                text = (await asyncio.to_thread(input, _c("  you › ", CYAN))).strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not text:
                continue
            if text in {"/quit", "/exit", "/q"}:
                break
            if text == "/new":
                session_id = memory.create_session()
                print(_c(f"  started session {session_id}", DIM))
                continue
            if text == "/sessions":
                for item in memory.list_sessions(limit=10):
                    print(_c(f"  {item['id']}  {item['updated_at']}  {item['title']}", DIM))
                continue
            if text == "/tools":
                for tool in agent.tools.all():
                    color = AMBER if tool.is_dangerous() else DIM
                    print(f"  {tool.name:<16} {_c(tool.risk, color)}  {tool.description[:80]}")
                continue
            if text.startswith("/approval"):
                parts = text.split()
                if len(parts) == 2 and parts[1] in {"never", "dangerous", "always"}:
                    agent.settings.approval = parts[1]
                    print(_c(f"  approval mode → {parts[1]}", DIM))
                else:
                    print(_c("  usage: /approval never|dangerous|always", DIM))
                continue
            await _one_turn(agent, session_id, text)
    finally:
        memory.close()
    print(_c("\n  offline.\n", DIM))
    return 0


async def _run_once(settings: Settings, prompt: str) -> int:
    settings.ensure_dirs()
    agent, memory = _make_agent(settings)
    session_id = memory.create_session(title=prompt[:60])
    try:
        await _one_turn(agent, session_id, prompt)
    finally:
        memory.close()
    return 0


def _serve(args: argparse.Namespace) -> int:
    import uvicorn

    settings = Settings.load(args.config)
    if args.host:
        settings.host = args.host
    if args.port:
        settings.port = args.port
    settings.ensure_dirs()
    mode = "SIMULATION" if settings.is_simulation else settings.provider
    print(_c(f"\n  F.R.I.D.A.Y. online · {mode} · http://{settings.host}:{settings.port}\n", CYAN))
    uvicorn.run("friday.server.app:app", host=settings.host, port=settings.port, log_level="info")
    return 0


def _doctor(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    checks: list[tuple[str, bool, str]] = []
    checks.append(("workspace exists", settings.workspace.exists(), str(settings.workspace)))
    checks.append(("data dir writable", _writable(settings.data_dir), str(settings.data_dir)))
    checks.append(("python >= 3.11", sys.version_info >= (3, 11), sys.version.split()[0]))
    checks.append(
        (
            "model configured",
            not settings.is_simulation,
            f"{settings.provider} · {settings.model}"
            if not settings.is_simulation
            else "simulation core (no API key)",
        )
    )
    registry, plugins = build_registry(settings)
    checks.append(("tools registered", len(registry.all()) > 0, f"{len(registry.all())} tools"))
    if plugins:
        checks.append(("plugins loaded", True, ", ".join(plugins)))
    memory = MemoryStore(settings.db_file)
    stats = memory.stats()
    memory.close()
    checks.append(
        (
            "memory store",
            True,
            f"{stats['sessions']} sessions · {stats['memories']} facts · fts={stats['fts']}",
        )
    )

    print(_c(f"\n  FRIDAY doctor · v{__version__}\n", CYAN))
    failures = 0
    for label, ok, detail in checks:
        mark = _c("PASS", GREEN) if ok else _c("WARN", AMBER)
        if not ok and label in {"workspace exists", "python >= 3.11"}:
            failures += 1
        print(f"  [{mark}] {label:<20} {_c(detail, DIM)}")
    if settings.is_simulation:
        print(
            _c(
                "\n  Running on the simulation core. Set OPENAI_API_KEY, ANTHROPIC_API_KEY,\n"
                "  OPENROUTER_API_KEY or GROQ_API_KEY — or FRIDAY_PROVIDER=ollama — for a real model.\n",
                DIM,
            )
        )
    print()
    return 1 if failures else 0


def _writable(path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".write-test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        return True
    except OSError:
        return False


def _list_tools(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    registry, plugins = build_registry(settings)
    print(_c(f"\n  Toolkit · {len(registry.all())} tools\n", CYAN))
    for tool in registry.all():
        risk = tool.risk.upper()
        color = RED if risk == "HIGH" else (AMBER if risk == "MEDIUM" else GREEN)
        print(f"  {_c(tool.name.ljust(16), BOLD)} {_c(risk.ljust(8), color)} {tool.description}")
        print(_c(f"{'':<26}{_schema_hint(tool.parameters)}", DIM))
    if plugins:
        print(_c(f"\n  plugins: {', '.join(plugins)}", DIM))
    print()
    return 0


def _schema_hint(parameters: dict[str, Any]) -> str:
    required = set(parameters.get("required") or [])
    props = parameters.get("properties") or {}
    bits = [name + ("*" if name in required else "") for name in props]
    return ("args: " + ", ".join(bits)) if bits else "no arguments"


def _list_sessions(args: argparse.Namespace) -> int:
    settings = Settings.load(args.config)
    settings.ensure_dirs()
    memory = MemoryStore(settings.db_file)
    rows = memory.list_sessions(limit=30)
    memory.close()
    if not rows:
        print(_c("  no sessions yet.", DIM))
        return 0
    print(_c("\n  Sessions\n", CYAN))
    for row in rows:
        print(
            f"  {_c(row['id'], BOLD)}  {_c(row['updated_at'][:16], DIM)}  "
            f"{row['title'][:60]:<60} {_c(str(row['message_count']) + ' msgs', DIM)}"
        )
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="friday",
        description="F.R.I.D.A.Y. — local-first AI agent with tools, memory and a HUD.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """\
            examples:
              friday serve --port 8000
              friday chat
              friday run "summarise README.md and list the files in friday/"
              friday doctor
            """
        ),
    )
    parser.add_argument("--version", action="version", version=f"FRIDAY {__version__}")
    parser.add_argument("--config", help="path to friday.yaml", default=None)
    sub = parser.add_subparsers(dest="command")

    serve = sub.add_parser("serve", help="launch the web HUD")
    serve.add_argument("--host", default=None)
    serve.add_argument("--port", type=int, default=None)
    serve.set_defaults(func=_serve)

    chat = sub.add_parser("chat", help="interactive terminal session")
    chat.add_argument("--session", default=None, help="resume a session id")
    chat.set_defaults(func=lambda a: asyncio.run(_chat(Settings.load(a.config), a.session)))

    run = sub.add_parser("run", help="one-shot task")
    run.add_argument("prompt", help="what should FRIDAY do?")
    run.set_defaults(func=lambda a: asyncio.run(_run_once(Settings.load(a.config), a.prompt)))

    doctor = sub.add_parser("doctor", help="diagnose configuration and environment")
    doctor.set_defaults(func=_doctor)

    tools = sub.add_parser("tools", help="list available tools")
    tools.set_defaults(func=_list_tools)

    sessions = sub.add_parser("sessions", help="list saved sessions")
    sessions.set_defaults(func=_list_sessions)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print(_c("\n  interrupted.", DIM))
        return 130
    except Exception as exc:  # noqa: BLE001 - CLI surfaces a clean error
        print(_c(f"\n  error: {exc}\n", RED), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
