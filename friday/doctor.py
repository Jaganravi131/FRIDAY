"""`friday doctor` — answers "will this run on MY machine?" in one command.

WHY THIS EXISTS
---------------
The deployment target is a specific ASUS VivoBook: Windows 11, Ryzen 5, Radeon
integrated graphics, 16 GB RAM, 512 GB SSD, no CUDA, ₹0 cloud budget. Development
happens on Linux. Almost every way this project can fail to deliver is an environment
difference nobody noticed until the user tried it — Python 3.9 instead of 3.11, sqlite
too old for the schema, a model server that was never started, 3 GB of free disk when a
GGUF needs 4, a FRIDAY_ROOT that is not writable, or a credential pasted into memory
before redaction existed.

Each of those is checkable in milliseconds. A doctor that prints a pass/fail report
with the fix is worth more than any amount of README, because the README describes the
machine the author had and the doctor describes the machine the user has.

Design rules:
  * STDLIB ONLY. A doctor that needs pip to run cannot diagnose a broken pip.
  * NEVER RAISE. Every check is wrapped; a check that crashes reports itself as a
    failure with the exception, because "doctor crashed" is the least useful possible
    answer to "does this work on my computer?".
  * Every finding carries a FIX, not just a verdict.
  * Read-only by default. `--fix` is opt-in and only does things that cannot lose data.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

MIN_PYTHON = (3, 11)
MIN_SQLITE = (3, 35)          # RETURNING, and the FTS5 behaviour the schema assumes
MIN_FREE_GB = 4.0             # one quantized model plus headroom
IDEAL_FREE_GB = 12.0
MIN_RAM_GB = 8.0
IDEAL_RAM_GB = 16.0
HARD_RAM_FLOOR_GB = 2.0       # below this it is a container limit or a real problem

#: Where a local model server is expected. llama.cpp's `llama-server` defaults to 8080;
#: Ollama to 11434. Both are probed because the user may have installed either.
MODEL_ENDPOINTS = (
    ("llama.cpp", "http://127.0.0.1:8080/v1/models"),
    ("Ollama", "http://127.0.0.1:11434/api/tags"),
    ("LM Studio", "http://127.0.0.1:1234/v1/models"),
)


@dataclass
class Finding:
    name: str
    status: str                       # ok | warn | fail | skip
    detail: str = ""
    fix: str = ""

    @property
    def icon(self) -> str:
        return {"ok": "✅", "warn": "⚠️ ", "fail": "❌", "skip": "⏭️ "}[self.status]

    def render(self) -> str:
        out = f"{self.icon} {self.name}\n     {self.detail}"
        if self.fix and self.status in ("warn", "fail"):
            out += f"\n     → {self.fix}"
        return out


@dataclass
class Report:
    findings: list[Finding] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "", fix: str = "") -> Finding:
        f = Finding(name, status, detail, fix)
        self.findings.append(f)
        return f

    @property
    def failed(self) -> bool:
        return any(f.status == "fail" for f in self.findings)

    @property
    def counts(self) -> dict[str, int]:
        out = {"ok": 0, "warn": 0, "fail": 0, "skip": 0}
        for f in self.findings:
            out[f.status] += 1
        return out

    def verdict(self) -> str:
        """`ok` | `warn` | `fail` — exposed because callers (CLI, tests, `/api/status`)
        need the judgement without parsing the rendered banner."""
        return "fail" if self.failed else ("warn" if self.counts["warn"] else "ok")

    def render(self) -> str:
        lines = ["FRIDAY doctor", "=" * 62]
        for f in self.findings:
            lines.append(f.render())
        c = self.counts
        lines.append("=" * 62)
        banner = {
            "ok":   "READY — FRIDAY will run on this machine.",
            "warn": "READY WITH WARNINGS — it will run; read the ⚠️  lines.",
            "fail": "NOT READY — fix the ❌ lines above and run this again.",
        }[self.verdict()]
        lines.append(
            f"{c['ok']} ok · {c['warn']} warn · {c['fail']} fail · {c['skip']} skip\n{banner}")
        return "\n".join(lines)

    def as_dict(self) -> dict:
        return {
            "ready": not self.failed,
            "verdict": self.verdict(),
            "counts": self.counts,
            "findings": [f.__dict__ for f in self.findings],
            "platform": {"system": platform.system(), "release": platform.release(),
                         "machine": platform.machine(),
                         "python": platform.python_version()},
        }


# ── platform probes (stdlib only, and they must work on Windows) ───────────────

def total_ram_gb() -> float | None:
    """Physical RAM. No psutil: a doctor that needs a dependency is not a doctor."""
    try:
        if sys.platform == "win32":
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                return stat.ullTotalPhys / (1024 ** 3)
            return None
        pages = os.sysconf("SC_PHYS_PAGES")
        size = os.sysconf("SC_PAGE_SIZE")
        return (pages * size) / (1024 ** 3)
    except (OSError, ValueError, AttributeError):
        return None


def has_cuda() -> bool:
    return shutil.which("nvidia-smi") is not None


#: DLLs / shared objects whose presence means a Vulkan loader is installed. AMD and
#: Intel ship these with every graphics driver on Windows, which is why the Vulkan path
#: needs no extra install — and why "no CUDA" does NOT mean "no GPU acceleration".
_VULKAN_LOADER = (
    r"C:\Windows\System32\vulkan-1.dll",
    r"C:\Windows\SysWOW64\vulkan-1.dll",
    "/usr/lib/x86_64-linux-gnu/libvulkan.so.1",
    "/usr/lib64/libvulkan.so.1",
)


def gpu_info() -> dict:
    """Best-effort GPU identification. Stdlib only, Windows-first.

    ⚠️ This used to be `has_cuda()` alone, which on the actual target machine — a Ryzen
    5 with a Radeon iGPU — reported "no CUDA, expected" and stopped. That was not
    merely incomplete, it was actively misleading: llama.cpp's Vulkan backend runs on
    AMD iGPUs on Windows with no ROCm and no extra drivers, and field reports put a 26B
    Q4 model at ~25 tok/s on a Radeon 780M that way. Telling the user their iGPU was
    irrelevant cost them roughly a 10x larger model.
    """
    import subprocess

    names: list[str] = []
    system = platform.system()
    try:
        if system == "Windows":
            # `wmic` is removed in Windows 11 24H2+, so go through PowerShell CIM.
            out = subprocess.run(
                ["powershell", "-NoProfile", "-NonInteractive", "-Command",
                 "(Get-CimInstance Win32_VideoController).Name"],
                capture_output=True, text=True, timeout=15)
            names = [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        elif system == "Linux":
            for card in sorted(Path("/sys/class/drm").glob("card*/device/vendor")) \
                    if Path("/sys/class/drm").exists() else []:
                vend = card.read_text().strip()
                names.append({"0x1002": "AMD/ATI", "0x8086": "Intel",
                              "0x10de": "NVIDIA"}.get(vend, f"vendor {vend}"))
            if not names and shutil.which("lspci"):
                out = subprocess.run(["lspci"], capture_output=True, text=True, timeout=15)
                names = [ln.split(":", 2)[-1].strip() for ln in out.stdout.splitlines()
                         if "VGA" in ln or "3D controller" in ln]
        elif system == "Darwin":
            out = subprocess.run(
                ["system_profiler", "SPDisplaysDataType", "-json"],
                capture_output=True, text=True, timeout=20)
            for g in json.loads(out.stdout or "{}").get("SPDisplaysDataType", []):
                names.append(g.get("sppci_model", "Apple GPU"))
    except Exception:
        pass                                   # identification is best-effort, never fatal

    blob = " ".join(names).lower()
    return {
        "names": names,
        "cuda": has_cuda(),
        "vulkan_loader": any(Path(p).exists() for p in _VULKAN_LOADER)
                         or shutil.which("vulkaninfo") is not None,
        "amd": any(k in blob for k in ("radeon", "amd", "ati")),
        "intel": "intel" in blob or "arc" in blob,
    }


def _probe(url: str, timeout: float = 1.5) -> str | None:
    """GET a URL with the stdlib. Returns the body, or None if unreachable."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:   # noqa: S310 (localhost)
            return r.read(4096).decode("utf-8", "replace")
    except Exception:
        return None


# ── the checks ─────────────────────────────────────────────────────────────────

def check_runtime(rep: Report) -> None:
    v = sys.version_info
    if v[:2] >= MIN_PYTHON:
        rep.add("Python", "ok", f"{platform.python_version()} at {sys.executable}")
    else:
        rep.add("Python", "fail",
                f"{platform.python_version()} — FRIDAY needs {'.'.join(map(str, MIN_PYTHON))}+",
                "Install Python 3.12 from python.org and tick 'Add to PATH' in the installer.")

    sv = tuple(int(x) for x in sqlite3.sqlite_version.split(".")[:2])
    if sv >= MIN_SQLITE:
        rep.add("SQLite", "ok", f"{sqlite3.sqlite_version}")
    else:
        rep.add("SQLite", "fail", f"{sqlite3.sqlite_version} — needs "
                                   f"{'.'.join(map(str, MIN_SQLITE))}+",
                "This ships inside Python on Windows; upgrading Python upgrades SQLite.")

    # ⚠️ Low RAM is a WARN, not a FAIL, below the hard floor. It constrains which model
    # you can load; it does not stop FRIDAY running. Marking it `fail` made the doctor
    # print "NOT READY" and exit nonzero on a machine that would have worked fine — and
    # a diagnostic that cries wolf gets ignored, which is worse than no diagnostic.
    # Only genuinely blocking conditions are allowed to be `fail`.
    ram = total_ram_gb()
    if ram is None:
        rep.add("RAM", "skip", "could not read total physical memory on this platform")
    elif ram >= IDEAL_RAM_GB:
        rep.add("RAM", "ok", f"{ram:.1f} GB — the design target")
    elif ram >= MIN_RAM_GB:
        rep.add("RAM", "warn", f"{ram:.1f} GB — workable, but keep the model small",
                "Use a 1.2B-class quantized model (Q4). Do not run a browser and a "
                "7B model at the same time.")
    elif ram >= HARD_RAM_FLOOR_GB:
        rep.add("RAM", "warn", f"{ram:.1f} GB — FRIDAY will run, a local model may not",
                "The memory half works at any size. For conversation, point FRIDAY at a "
                "model server on another machine, or use a 0.5B-1B Q4 model.")
    else:
        rep.add("RAM", "fail", f"{ram:.1f} GB is below the {HARD_RAM_FLOOR_GB:.0f} GB floor",
                "This is likely a container limit rather than the real machine — check "
                "with your OS. On real hardware this low, Python itself will struggle.")

    g = gpu_info()
    label = ", ".join(dict.fromkeys(g["names"])) or "none identified"
    if g["cuda"]:
        rep.add("GPU", "ok", f"{label} — CUDA available")
    elif g["vulkan_loader"] and (g["amd"] or g["intel"]):
        # ⭐ The case this machine actually is. Not a skip, and not a consolation.
        rep.add("GPU", "ok",
                f"{label} — no CUDA, but a Vulkan loader is present",
                "Use llama.cpp's Vulkan backend (`--n-gpu-layers 999`): it runs on AMD "
                "and Intel iGPUs on Windows with no ROCm and no extra drivers. This "
                "raises your realistic ceiling from a 1.2B model to roughly Qwen3 14B "
                "Q4 (~8.5 GB) or Gemma 3 12B (~6.7 GB). See INSTALL.md §3 — and note "
                "that Vulkan needs the llama.cpp *vulkan* build, not the cpu build.")
    elif g["names"]:
        rep.add("GPU", "warn", f"{label} — but no Vulkan loader found",
                "Update your graphics driver; AMD and Intel ship the Vulkan loader with "
                "it. Without it you are CPU-only and should stay at 1.2B–3B Q4.")
    else:
        rep.add("GPU", "skip",
                "no GPU identified. CPU-only: stay at 1.2B–3B Q4. Local fine-tuning is "
                "not part of the plan either way — free Kaggle T4 hours cover it.")


def check_disk(rep: Report, root: Path) -> None:
    try:
        target = root if root.exists() else Path.home()
        free = shutil.disk_usage(str(target)).free / (1024 ** 3)
    except OSError as e:
        rep.add("Disk", "skip", f"could not stat {target}: {e}")
        return
    if free >= IDEAL_FREE_GB:
        rep.add("Disk", "ok", f"{free:.1f} GB free on {target.drive or target}")
    elif free >= MIN_FREE_GB:
        rep.add("Disk", "warn", f"{free:.1f} GB free — enough for one model",
                f"Keep {IDEAL_FREE_GB:.0f} GB free if you want to compare two models.")
    else:
        rep.add("Disk", "fail", f"{free:.1f} GB free, needs {MIN_FREE_GB:.0f} GB",
                "A quantized 1.2B model is ~1 GB; the artifacts and traces grow slowly.")


def _rel(d: Path, root: Path) -> str:
    """`d` relative to `root`, falling back to the absolute path when it is not inside it.

    `Path.relative_to` RAISES `ValueError` rather than returning something usable, so a
    report line that only exists to say "this directory is missing" used to be able to
    crash the whole check. `run()` now rebinds paths for `--root`, which makes the two
    agree, but `check_layout` is also a plain function other code can call with any
    root — and a diagnostic must not depend on its caller having set globals correctly
    first. This is the difference between "soul/ is missing" and a stack trace.
    """
    try:
        return str(d.relative_to(root))
    except ValueError:
        return str(d)


def check_layout(rep: Report, root: Path) -> None:
    from . import paths

    if not root.exists():
        rep.add("FRIDAY_ROOT", "fail", f"{root} does not exist",
                "Run `python -m friday seed` — it creates the layout and starter files.")
        return
    try:
        probe = root / ".doctor-write-probe"
        probe.write_text("x", encoding="utf-8")
        probe.unlink()
        rep.add("FRIDAY_ROOT", "ok", f"{root} (writable)")
    except OSError as e:
        rep.add("FRIDAY_ROOT", "fail", f"{root} is not writable: {e}",
                "Pick a directory you own, or unset FRIDAY_ROOT to use the default.")

    missing = [_rel(d, root) for d in (paths.SOUL, paths.FACTS, paths.TRACES)
               if not d.exists()]
    if missing:
        rep.add("Directory layout", "warn", f"missing: {', '.join(missing)}",
                "Run `python -m friday seed` then `python -m friday build`.")
    else:
        rep.add("Directory layout", "ok", "soul/, memory/facts/, memory/traces/ present")

    souls = sorted(p.name for p in paths.SOUL.glob("*.md")) if paths.SOUL.exists() else []
    if souls:
        rep.add("Soul files", "ok", f"{len(souls)}: {', '.join(souls)}")
    else:
        rep.add("Soul files", "warn", "no soul/*.md — FRIDAY has no identity or rules",
                "Run `python -m friday seed`.")


def check_store(rep: Report, root: Path) -> None:
    from . import paths

    if not paths.DB_PATH.exists():
        rep.add("Derived index", "warn", f"no database at {paths.DB_PATH.name}",
                "Run `python -m friday build`. Markdown is truth; this is rebuildable.")
        return
    try:
        # ⭐ `Path.as_uri()`, not an f-string. SQLite URI filenames are URIs, so an
        # absolute path must be `file:///...` with forward slashes; on Windows the
        # naive form produced `file:C:\Users\...\friday.db?mode=ro`, which SQLite
        # cannot resolve to an absolute path — and Windows is the primary deployment
        # platform, so the machine that most needs this check was the one it broke on.
        # as_uri() also percent-encodes, which makes "C:\Users\John Doe\..." work.
        conn = sqlite3.connect(f"{paths.DB_PATH.as_uri()}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        facts = conn.execute("SELECT count(*) AS n FROM facts").fetchone()["n"]
        live = conn.execute(
            "SELECT count(*) AS n FROM facts WHERE retracted_at IS NULL").fetchone()["n"]
        tables = {r["name"] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")}
        conn.close()
    except sqlite3.Error as e:
        rep.add("Derived index", "fail", f"cannot read the database: {e}",
                "Delete artifacts/ and run `python -m friday rebuild` — it is derived.")
        return

    required = {"facts", "turns", "audit", "senses", "origin_state"}
    absent = sorted(required - tables)
    if absent:
        rep.add("Schema", "fail", f"tables missing: {', '.join(absent)}",
                "This database predates the current schema. "
                "`python -m friday rebuild` recreates it from your Markdown.")
    else:
        rep.add("Schema", "ok", f"{len(tables)} tables, all required ones present")
    rep.add("Memory", "ok" if live else "warn",
            f"{live} live facts ({facts - live} retracted, kept as history)",
            "" if live else "Nothing in memory yet. Talk to it, or `python -m friday seed`.")


def check_model(rep: Report) -> None:
    """Is there a model to talk to? The mock is a fallback, not a deployment."""
    reached = []
    for label, url in MODEL_ENDPOINTS:
        if _probe(url) is not None:
            reached.append(label)
    if reached:
        rep.add("Model server", "ok", f"reachable: {', '.join(reached)}")
        return

    rep.add("Model server", "warn",
            "nothing listening on 8080 (llama.cpp), 11434 (Ollama) or 1234 (LM Studio)",
            "FRIDAY will run on its mock client, which proves the memory half but "
            "cannot converse. See INSTALL.md §3 — the short version is "
            "`ollama pull qwen2.5:1.5b` or llama-server with an LFM2.5-1.2B GGUF.")

    from .llm import get_client

    client = get_client("auto")
    rep.add("Client selection", "ok" if "mock" in type(client).__name__.lower() else "ok",
            f"`get_client('auto')` resolves to {type(client).__name__} "
            f"({getattr(client, 'name', '?')})")


def check_phone_access(rep: Report) -> None:
    """Exit test #12. A laptop-only agent is a toy; this is the Presence Fabric seed."""
    if shutil.which("tailscale"):
        try:
            import subprocess

            out = subprocess.run(["tailscale", "status"], capture_output=True, text=True,
                                 timeout=10)
            running = out.returncode == 0
        except Exception:
            running = False
        rep.add("Tailscale", "ok" if running else "warn",
                "installed and connected" if running else "installed but not connected",
                "" if running else "Run `tailscale up`, then `friday serve --host tailscale`.")
        return
    rep.add("Tailscale", "skip",
            "not installed — you cannot reach FRIDAY from your phone yet",
            "Free for personal use: tailscale.com/download. This is how exit test #12 "
            "passes, and it keeps the port off the public internet.")


def check_secrets(rep: Report) -> None:
    """⚠️ Scan the user's Markdown for credentials that predate write-time redaction.

    Redaction now runs inside `assert_fact` and the trace writer, so nothing new leaks.
    But a file hand-edited before that, or imported from elsewhere, is already on disk —
    and Markdown is truth, which means it gets backed up, synced, and possibly
    committed. This is the one check that looks at CONTENT rather than environment, and
    it exists because "the repo was accidentally made public" is a severe risk in the
    doc 09 threat model with a cheap defence.
    """
    from . import paths
    from .security import scan

    if not paths.MEMORY.exists():
        rep.add("Secret scan", "skip", "no memory/ directory yet")
        return
    hits: list[str] = []
    scanned = 0
    for p in list(paths.MEMORY.rglob("*.md")) + list(paths.SOUL.rglob("*.md")):
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        scanned += 1
        r = scan(text)
        if r.count:
            hits.append(f"{p.name} ({r.summary()})")
    if hits:
        rep.add("Secret scan", "fail",
                f"credentials found in {len(hits)} file(s): " + "; ".join(hits[:5]),
                "Remove them by hand, then `python -m friday rebuild`. New writes are "
                "redacted automatically; these predate that.")
    else:
        rep.add("Secret scan", "ok", f"{scanned} Markdown file(s), no credentials found")


def check_backup(rep: Report, root: Path) -> None:
    """The git remote IS the backup (doc 09 §10: laptop loss = FRIDAY death)."""
    # ⚠️ Ask whether FRIDAY_ROOT is inside a repo — NOT whether the installed package
    # happens to be one. Falling back to the package directory reported "git remote
    # configured (origin)" for a memory folder that was in no repository at all, which
    # is the one false "ok" that could cost the user everything: doc 09 lists laptop
    # loss as a severe risk whose only defence is that the git remote IS the backup.
    import subprocess

    try:
        top = subprocess.run(["git", "-C", str(root), "rev-parse", "--show-toplevel"],
                             capture_output=True, text=True, timeout=10)
        in_repo = top.returncode == 0 and top.stdout.strip()
    except Exception:
        in_repo = ""
    if not in_repo:
        rep.add("Backup", "warn", f"{root} is not inside a git repository",
                "Your Markdown is the only copy of your memory. Put FRIDAY_ROOT in a "
                "PRIVATE git repo — never a public one.")
        return
    try:
        out = subprocess.run(["git", "-C", str(root), "remote"], capture_output=True,
                             text=True, timeout=10)
        remotes = out.stdout.split()
    except Exception:
        remotes = []
    if remotes:
        rep.add("Backup", "ok",
                f"{root} is in a repo at {in_repo} with remote(s): {', '.join(remotes)}")
    else:
        rep.add("Backup", "warn",
                f"in a repo at {in_repo}, but no remote — nothing is off this disk",
                "Add a PRIVATE remote. `artifacts/` is rebuildable; `memory/` and "
                "`soul/` are not.")


def check_optional(rep: Report) -> None:
    """Optional dependencies. Absent is fine — the core must not need them."""
    mods = []
    for name, purpose in (("numpy", "RSC prototypes"), ("torch", "RSC training"),
                          ("sqlite_vec", "real vector search")):
        try:
            __import__(name)
            mods.append(f"{name} ✓")
        except ImportError:
            mods.append(f"{name} —")
    rep.add("Optional deps", "ok",
            " ".join(mods) + "  (none are required by the core; the retrieval pipeline "
            "falls back to a hashing embedder and brute-force cosine)")


def run(*, root: Path | None = None) -> Report:
    """Every check, in the order you would debug them.

    ⭐ `--root` MEANS IT. Several checks read the module-level `paths.*` constants
    rather than the `root` argument — `check_store` looks at `paths.DB_PATH`,
    `check_secrets` walks `paths.MEMORY`. Passing a different root therefore used to
    produce a report about the WRONG directory while claiming to be about the one you
    asked for, and `check_layout` crashed outright because `paths.SOUL.relative_to(root)`
    raised when the two disagreed. That matters disproportionately here: `friday doctor`
    is the command the deployment target runs first, on a machine that is not the
    author's, and a silent wrong answer is worse than the crash that revealed it.

    So an explicit root rebinds `paths` for the duration of the run and restores it
    afterwards. Restoring is not optional politeness — `run()` is a library function,
    and a caller that inspects one root and then writes memory would otherwise write it
    into the inspected tree.
    """
    from functools import partial

    from . import paths

    rep = Report()
    previous = paths.ROOT
    requested = Path(root) if root is not None else None
    rebound = False
    try:
        if requested is not None and requested.resolve() != Path(previous).resolve():
            paths.rebind(requested)
            rebound = True
        root = Path(requested or paths.ROOT)

        # ⭐ (label, callable) rather than a bare list of callables. The label is only
        # used when the check itself crashes, which is exactly the moment a name matters
        # and exactly the moment `fn.__name__` is least likely to be one: the root-bound
        # checks were wrapped in lambdas, so a crash reported `❌ <lambda>` with a
        # ValueError about paths — a diagnostic that cannot name the thing that broke.
        checks = (
            ("check_runtime", check_runtime),
            ("check_disk", partial(check_disk, root=root)),
            ("check_layout", partial(check_layout, root=root)),
            ("check_store", partial(check_store, root=root)),
            ("check_model", check_model),
            ("check_phone_access", check_phone_access),
            ("check_secrets", check_secrets),
            ("check_backup", partial(check_backup, root=root)),
            ("check_optional", check_optional),
        )
        for name, fn in checks:
            try:
                fn(rep)
            except Exception as e:      # a crashed check is a finding, not a stack trace
                rep.add(name, "fail", f"the check itself crashed: {type(e).__name__}: {e}",
                        "Report this — the doctor should never be the thing that breaks.")
    finally:
        if rebound:
            paths.rebind(previous)
    return rep
