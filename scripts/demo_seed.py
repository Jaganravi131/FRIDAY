"""Seed a throwaway FRIDAY root with enough memory to demonstrate the system.

Why this exists
---------------
`friday seed` writes starter files that describe the *author*. To show someone what
FRIDAY actually does — provenance, the ledger, bi-temporal history, a refusal — you need
a root with a few facts in it that have quotes attached. Doing that by hand is a dozen
commands, and in a sandbox that gets wiped it is a dozen commands *again*.

So: one command, deterministic, idempotent, and safe to point at a real root only if you
mean to (it writes demo facts, and says so).

    python scripts/demo_seed.py --root /tmp/friday-demo
    python -m friday serve --host 0.0.0.0        # with FRIDAY_ROOT=/tmp/friday-demo

This is a demonstration fixture, not user data. It is deliberately NOT wired into
`friday seed`, because seeding a real installation with someone else's fake memories
would be exactly the confusion Law 7 exists to prevent.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Invoked as `python scripts/demo_seed.py`, which puts scripts/ — not the repository
# root — on sys.path[0]. Without this the import of `friday` fails unless the package
# happens to be installed, and "works only if you pip installed it" is a bad contract
# for a script whose job is to make the project easy to try.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

#: predicate, value, and the quote that authorises it. The quote matters: without it the
#: provenance gate treats the write as unauthorised and asks for confirmation, which is
#: correct behaviour but makes for a confusing demo.
DEMO_FACTS = [
    ("lease_amount_monthly", "18000 INR", "my rent is 18000 a month"),
    ("manager_name", "Priya Raman", "my manager is Priya Raman"),
    ("laptop_model", "ASUS VivoBook, Ryzen 5, 16 GB",
     "I have an ASUS VivoBook with a Ryzen 5 and 16 gigs of RAM"),
    ("cloud_budget", "zero rupees, local and free tiers only",
     "my cloud budget is zero, local and free tiers only"),
    ("standup_day", "Tuesday 10:00 IST", "my standup is Tuesday at ten"),
]

#: A contradiction, so `history` and the bi-temporal trail have something to show.
DEMO_RETRACTION = ("standup_day", "Tuesday 09:30 IST",
                   "actually standup moved to half past nine on Tuesday")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None, help="FRIDAY_ROOT to populate")
    ap.add_argument("--wipe", action="store_true",
                    help="delete the root first (default: reuse and top up)")
    args = ap.parse_args(argv)

    root = Path(args.root or os.environ.get("FRIDAY_ROOT") or "").expanduser()
    if not str(root):
        print("need --root or $FRIDAY_ROOT", file=sys.stderr)
        return 2
    if args.wipe and root.exists():
        import shutil
        shutil.rmtree(root)
    # Must be set BEFORE friday.paths is imported: it reads the env at import time and
    # rebinding it afterwards leaves every module pointing at the old root.
    os.environ["FRIDAY_ROOT"] = str(root)
    root.mkdir(parents=True, exist_ok=True)

    from friday.cli import main as cli
    from friday.store import db
    from friday import paths

    if cli(["seed"]) != 0:
        return 1
    if cli(["build"]) != 0:
        return 1

    written = skipped = 0
    for predicate, value, quote in DEMO_FACTS + [DEMO_RETRACTION]:
        argv_ = ["write", predicate, value, "--user-edit", "--quote", quote]
        rc = cli(argv_)
        if rc == 0:
            written += 1
        else:
            skipped += 1

    conn = db.connect(paths.DB_PATH)
    # There is no `state` column: liveness is expressed by the two time axes, which is
    # the whole point of the bi-temporal design. `retracted_at IS NULL` is BELIEF-time
    # liveness (FRIDAY still believes it) and `valid_to IS NULL` is WORLD-time liveness
    # (it is still true). A retracted fact is still present and still queryable — it is
    # just no longer current. Counting belief-time is what a demo wants.
    live = conn.execute(
        "SELECT count(*) FROM facts WHERE retracted_at IS NULL").fetchone()[0]
    total = conn.execute("SELECT count(*) FROM facts").fetchone()[0]
    print(f"\ndemo root ready at {root}")
    print(f"  {written} writes accepted, {skipped} skipped, "
          f"{live} live of {total} total (the rest are history, not deleted)")
    print("\ntry:")
    for q in ("my rent", "who is my manager", "my standup", "what is my laptop"):
        print(f'  python -m friday ask "{q}"')
    print("  python -m friday history standup_day      # the bi-temporal trail")
    print("  python -m friday why \"my standup\"         # the literal source quote")
    print("  python -m friday serve --host tailscale   # then open it on your phone")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
