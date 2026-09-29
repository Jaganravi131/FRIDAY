#!/usr/bin/env python3
"""Run the RSC ablation ladder against the pure-Python operators.

docs/architecture/13 §5.2. This is a CORRECTNESS check and a shape-of-the-curve
preview — not the 1.3B/100B-token result. It answers "does my implementation of the
update rule behave the way the paper says it should?", which you must know before
burning Kaggle hours on the real thing.

    python scripts/rsc_ablation.py                     # full ladder, 2 seeds
    python scripts/rsc_ablation.py --rungs kda,gdn2_e  # just the decisive pair
    python scripts/rsc_ablation.py --quick             # 1 seed, short needles
    python scripts/rsc_ablation.py --md docs/architecture/BENCHMARKS-rsc.md

The three probes:

    associative   can the state return the value stored at a key?    rung 0 vs 1+
    needle        does a fact planted at depth d survive?            rung 2 vs 3
    interference  does writing at key B corrupt the value at key A?  rung 4 vs 5

`interference` is FRIDAY's probe and the one nobody's benchmark sheet reports: it is
the bi-temporal correction case in miniature. A write-anchored erase (GDN-2) cannot
suppress a stale association at a DIFFERENT address; EDA's independently addressed
e_t can. If rung 5 doesn't move that number, the address-level coupling isn't
binding on your data and you can ship rung 3 or 4.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from friday.rsc.ablation import format_report, run_ladder  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rungs", default="",
                    help="comma-separated subset, e.g. kda,gdn2_e,gdn2,eda")
    # 32, not 16: the associative probe writes n_pairs + distractors = 18 distinct
    # addresses, and at d_k=16 that EXCEEDS the state's capacity, so every rung
    # collapses to 0.00 and the ladder reports a tie it did not measure. Keep
    # d_k comfortably above n_pairs + distractors.
    ap.add_argument("--dk", type=int, default=32)
    ap.add_argument("--dv", type=int, default=32)
    ap.add_argument("--length", type=int, default=384, help="needle haystack steps")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--md", default=None)
    args = ap.parse_args(argv)

    if args.quick:
        args.length, args.seeds = 128, "0"

    rungs = tuple(r.strip() for r in args.rungs.split(",") if r.strip()) or None
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())

    if args.dk < 24 or args.dv < 24:
        print(f"\n⚠️  d_k={args.dk} is likely BELOW state capacity for "
              f"n_pairs + distractors = 18 addresses.\n"
              f"    Every rung will collapse to 0.00 on the associative probe and "
              f"the ladder will\n    report a tie it never measured. Use --dk 32 "
              f"--dv 32 unless you know why you are not.\n",
              file=sys.stderr)
    print(f"RSC ablation ladder — d_k={args.dk} d_v={args.dv} "
          f"needle_length={args.length} seeds={seeds}")
    print("(pure Python; expect minutes, not seconds)\n")

    t0 = time.perf_counter()
    kw: dict = {"d_k": args.dk, "d_v": args.dv, "needle_length": args.length, "seeds": seeds}
    if rungs:
        kw["rungs"] = rungs
    result = run_ladder(**kw)
    elapsed = time.perf_counter() - t0

    md = format_report(result)
    print(md)
    print(f"\n({elapsed:.1f}s wall clock)")

    verdict = _verdict(result)
    print("\n" + verdict)

    if args.json:
        p = Path(args.json)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({**result, "seconds": round(elapsed, 1)},
                                indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {p}")
    if args.md:
        p = Path(args.md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md + "\n## Automatic checks\n\n" + verdict + "\n", encoding="utf-8")
        print(f"wrote {p}")
    return 0


def _verdict(result: dict) -> str:
    """Machine-checkable expectations. If these fail, the operators are wrong —
    not merely underperforming."""
    rungs = result.get("rungs", {})
    lines = ["## Automatic checks", ""]
    ok = True

    def get(name: str, probe: str) -> float | None:
        r = rungs.get(name)
        return r["scores"].get(probe) if r else None

    a0, a2 = get("retnet", "associative"), get("kda", "associative")
    if a0 is not None and a2 is not None:
        passed = a0 < a2
        ok &= passed
        lines.append(f"- [{'PASS' if passed else 'FAIL'}] rung 0 (`retnet`) loses associative "
                     f"recall to rung 2 (`kda`): {a0} vs {a2}. "
                     + ("" if passed else "**A fixed-decay additive write cannot do selective "
                                          "associative recall — if this passes, the probe is broken.**"))

    n2, n3 = get("kda", "needle"), get("gdn2_e", "needle")
    if n2 is not None and n3 is not None:
        passed = n3 >= n2
        lines.append(f"- [{'PASS' if passed else 'note'}] rung 2 -> 3 needle: {n2} -> {n3}. "
                     f"NVlabs reports the erase gate carries most of the gain.")
        ok &= passed

    i4, i5 = get("gdn2", "interference"), get("eda", "interference")
    if i4 is not None and i5 is not None:
        lines.append(f"- [info] rung 4 -> 5 interference (retain): {i4} -> {i5}. "
                     f"EDA's win should show up here, not in needle.")

    x4 = x5 = None
    for name, target in (("gdn2", 4), ("eda", 5)):
        r = rungs.get(name)
        if r:
            for d in r["detail"]:
                if d.get("probe") == "interference":
                    v = d.get("crosstalk_new_into_old")
                    if v is not None:
                        if target == 4:
                            x4 = v
                        else:
                            x5 = v
    if x4 is not None and x5 is not None:
        passed = x5 <= x4 + 1e-9
        lines.append(f"- [{'PASS' if passed else 'note'}] crosstalk new->old: GDN-2 {x4} vs EDA {x5}. "
                     f"Lower is better; this is the address-level decoupling being measured.")

    lines.append("")
    lines.append("**Overall: " + ("checks pass — the operators behave as the papers describe."
                                 if ok else "at least one check FAILED — fix the operators before "
                                            "spending Kaggle hours.**"))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
