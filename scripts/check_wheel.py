#!/usr/bin/env python3
"""Smoke-test the built wheel, not the source tree (stdlib only).

Run `python -m build` first, then `python scripts/check_wheel.py dist`.
A fresh venv, an unrelated working directory, and no PYTHONPATH keep checkout
imports and optional dependencies from hiding missing distribution files. pip
installs only the supplied wheel, offline. CI runs this on Windows and Linux.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path
from zipfile import ZipFile


def check(dist: Path) -> None:
    wheels = list(dist.resolve().glob("friday-*.whl"))
    if len(wheels) != 1:
        raise RuntimeError(f"expected exactly one FRIDAY wheel in {dist}, found {len(wheels)}")
    wheel = wheels[0]
    with ZipFile(wheel) as archive:
        required = {"friday/agent/loop.py", "friday/memory/compiler.py",
                    "friday/store/db.py", "friday/store/schema.sql"}
        missing = required - set(archive.namelist())
        if missing:
            raise RuntimeError(f"wheel is missing runtime files: {sorted(missing)}")

    with tempfile.TemporaryDirectory(prefix="friday-wheel-") as scratch:
        root = Path(scratch)
        env_dir = root / "venv"
        venv.EnvBuilder(with_pip=True).create(env_dir)
        bin_dir = env_dir / ("Scripts" if os.name == "nt" else "bin")
        python = bin_dir / ("python.exe" if os.name == "nt" else "python")
        entrypoint = bin_dir / ("friday.exe" if os.name == "nt" else "friday")
        env = {k: v for k, v in os.environ.items()
               if not k.startswith(("PYTHON", "FRIDAY_"))}
        env.update(FRIDAY_ROOT=str(root / "data"), PYTHONUTF8="1", PYTHONIOENCODING="utf-8")

        def run(*args: str) -> str:
            result = subprocess.run(args, cwd=root, env=env, capture_output=True,
                                    text=True, encoding="utf-8", timeout=120, check=False)
            if result.returncode:
                raise RuntimeError(f"{args} exited {result.returncode}\n"
                                   f"{result.stdout}\n{result.stderr}")
            return result.stdout

        run(str(python), "-m", "pip", "install", "--no-index", "--no-deps", str(wheel))
        run(str(entrypoint), "--help")
        for args in (
            ("seed",), ("build",),
            ("ask", "what is my monthly rent", "--mock"),
            ("needle", "--length", "128", "--trials", "1", "--depths", "32", "--dataset-only"),
            ("ablation", "--quick"),
            ("heartbeat",),
        ):
            print(f"installed wheel: {' '.join(args)}", flush=True)
            output = run(str(python), "-m", "friday", *args)
            if args[0] == "ask" and "28000" not in output.replace(",", ""):
                raise RuntimeError(f"mock recall did not return the seeded rent:\n{output}")
        # The blind-holdout runner must ship too, and its leak rule must hold in
        # the installed package: aggregates printed, query text never echoed.
        holdout_file = root / "holdout-smoke.json"
        holdout_file.write_text(json.dumps({
            "suite": "wheel-smoke",
            "author": "scripts/check_wheel.py — a packaging smoke, not a blind set",
            "cases": [
                {"query": "who is my manager", "expected": "manager_name"},
                {"query": "when is my dentist appointment", "expected": None},
            ],
        }, indent=1), encoding="utf-8")
        print("installed wheel: holdout (aggregates only)", flush=True)
        output = run(str(python), "-m", "friday", "holdout", str(holdout_file))
        if "recall@5" not in output or "who is my manager" in output:
            raise RuntimeError(f"holdout smoke malformed or leaked a query:\n{output}")
        if not (root / "data" / "artifacts" / "friday.db").exists():
            raise RuntimeError("build did not create the database in FRIDAY_ROOT")
    print("installed-wheel smoke test passed", flush=True)


if __name__ == "__main__":
    check(Path(sys.argv[1] if len(sys.argv) > 1 else "dist"))
