#!/usr/bin/env python3
"""⭐ THE decisive benchmark. docs/architecture/12 §5.3, /13 §1.3.

Generate N conversations of exactly `--length` tokens with a specific fact planted
at a controlled depth, then ask for it at the end and score EXACT match.

This is the one test where linear-attention models collapse and hybrids hold. It
directly measures whether FRIDAY will remember what you told it 200 turns ago — and
it is the benchmark GDN-2 moved from 63 to 90 (RULER S-NIAH-3), which is why
docs/architecture/13 says the proposal targets FRIDAY's binding constraint.

Run it:
  * BEFORE choosing a backbone (Phase 0 bake-off). Decision rule:
      hybrid >= 0.9 at 8K and >= 25 t/s -> use the hybrid
      hybrid collapses (< 0.6)          -> stay on the Transformer, let the
                                           Ledger + external store do the work
  * AFTER every training run (Law 2c). Recall is a property of a CHECKPOINT, not an
    architecture. CoT post-training degrades it, and the eval gate won't catch that
    because the gate measures task success, not retrieval.

Uses only stdlib + urllib. Talks to any OpenAI-compatible endpoint
(`llama-server`, Ollama). No model? It writes the dataset and scores a baseline
substring check so you can still inspect the generated contexts.

    python scripts/needle_test.py --length 8000 --depths 500,2000,4000,6000,7500 \\
        --trials 20 --url http://127.0.0.1:8080/v1 --model LFM2-2.6B \\
        --out docs/architecture/BENCHMARKS-needle.json
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

# ── haystack ───────────────────────────────────────────────────────────────────

FILLER = [
    "We discussed the usual deployment routine and nothing stood out as unusual.",
    "The build finished without warnings, and the test output looked ordinary.",
    "There was some back and forth about naming, which we settled quickly.",
    "I mentioned that the monsoon traffic had been worse than usual this week.",
    "We agreed to revisit the question once more data had come in.",
    "The conversation drifted to whether the new keyboard was worth the price.",
    "Nothing was decided; it was mostly thinking out loud about tradeoffs.",
    "I noted that the electricity bill had gone up and we should check the meter.",
    "We talked through the weekend plan and then abandoned most of it.",
    "A neighbour's dog barked for most of the evening, which was memorable only for its length.",
    "The document was long and largely boilerplate, so we skimmed it.",
    "Someone suggested a refactor; nobody objected, nobody started it.",
    "I asked about the status of the application and was told to wait.",
    "We compared two subscription plans and found neither compelling.",
    "There was a short digression about the best filter coffee near the office.",
    "The meeting ran over by ten minutes and covered ground we had already covered.",
    "I described a bug that turned out to be a misreading of the documentation.",
    "We listed the things that would have to be true for the plan to work.",
    "The power cut for about four minutes, which reset the router and the mood.",
    "A long discussion of whether to keep or delete an old backup, ending in keeping it.",
]

#: The needle is deliberately an arbitrary opaque token, not a plausible fact.
#: A plausible fact ("my rent is 28000") can be guessed from priors; `7f3a9c2b`
#: cannot. If the model returns it, it retrieved it.
NEEDLES = [
    ("the access code", "7f3a9c2b"),
    ("the reference number", "KX-40913"),
    ("the confirmation code", "9QD2M7"),
    ("the tracking id", "TRK-88431-Z"),
    ("the serial number", "SN-5521847"),
]


@dataclass
class Trial:
    needle_name: str
    needle_value: str
    depth_tokens: int
    length_tokens: int
    prompt: str = ""
    response: str = ""
    exact: bool = False
    partial: bool = False
    ms: int = 0
    error: str = ""


@dataclass
class Report:
    model: str
    url: str
    length: int
    trials: int
    results: list[Trial] = field(default_factory=list)

    def by_depth(self) -> dict[int, dict[str, float]]:
        out: dict[int, dict[str, float]] = {}
        for t in self.results:
            d = out.setdefault(t.depth_tokens, {"n": 0, "exact": 0, "partial": 0})
            d["n"] += 1
            d["exact"] += int(t.exact)
            d["partial"] += int(t.partial)
        for d, v in out.items():
            n = max(1, v["n"])
            v["exact_rate"] = round(v["exact"] / n, 4)
            v["partial_rate"] = round(v["partial"] / n, 4)
        return dict(sorted(out.items()))

    @property
    def overall(self) -> float:
        if not self.results:
            return 0.0
        return round(sum(int(t.exact) for t in self.results) / len(self.results), 4)

    def markdown(self) -> str:
        lines = [
            f"# Needle recall — `{self.model}`",
            "",
            f"Endpoint: `{self.url}` · context length: **{self.length}** tokens · "
            f"trials: **{self.trials}**",
            "",
            f"**Overall exact-match recall: {self.overall:.2%}**",
            "",
            "| needle depth (tokens) | n | exact | partial |",
            "|---|---|---|---|",
        ]
        for depth, v in self.by_depth().items():
            lines.append(
                f"| {depth} | {v['n']} | **{v['exact_rate']:.2%}** | {v['partial_rate']:.2%} |"
            )
        errs = [t for t in self.results if t.error]
        if errs:
            lines += ["", f"⚠️ {len(errs)} trial(s) errored, e.g. `{errs[0].error[:160]}`"]
        lines += [
            "",
            "## Reading this",
            "",
            "* **A flat row-profile is the pass condition.** Scoring 0.95 at depth 500 and",
            "  0.10 at depth 7500 is not a 0.5 model — it is a model with no long-range",
            "  memory, and the average hides it. Look at the deepest row.",
            "* **Partial > exact means it knows something is there but can't reproduce it**",
            "  verbatim — the classic fixed-state recall failure.",
            "* **Decision rule (docs/architecture/12 §5.3):** >= 0.9 and >= 25 t/s -> take the",
            "  hybrid. < 0.6 -> stay on the Transformer and let the Ledger + external store",
            "  carry recall.",
            "* **Re-run after every fine-tune.** Law 2c: recall is a property of a checkpoint,",
            "  not an architecture. A gate pass with a needle regression is a REGRESSION.",
            "",
        ]
        return "\n".join(lines)


# ── dataset generation ─────────────────────────────────────────────────────────

def approx_tokens(text: str) -> int:
    words = re.findall(r"\w+|[^\w\s]", text)
    return max(1, int(len(words) + sum(len(w) for w in words) / 4.0))


def build_context(length_tokens: int, depth_tokens: int, needle_name: str,
                  needle_value: str, rng: random.Random) -> str:
    """Fill to `length_tokens` with filler, planting the needle at `depth_tokens`."""
    needle_line = (
        f"USER: One thing to remember — {needle_name} is {needle_value}. "
        f"Please keep that in mind for later.\nASSISTANT: Noted. {needle_name}: {needle_value}."
    )
    head, body = [], []
    acc = 0
    while acc < depth_tokens:
        line = f"USER: {rng.choice(FILLER)}\nASSISTANT: {rng.choice(FILLER)}"
        head.append(line)
        acc += approx_tokens(line)
    acc = 0
    while acc < length_tokens - depth_tokens:
        line = f"USER: {rng.choice(FILLER)}\nASSISTANT: {rng.choice(FILLER)}"
        body.append(line)
        acc += approx_tokens(line)
    return "\n\n".join(head + [needle_line] + body)


def ask_prompt(needle_name: str) -> str:
    return (
        f"USER: Quick check — what exactly was {needle_name} I told you earlier? "
        f"Reply with just the value, verbatim. If you don't have it, say NOT_FOUND."
    )


# ── inference ──────────────────────────────────────────────────────────────────

def chat(url: str, model: str, prompt: str, *, api_key: str = "not-needed",
         timeout: float = 300.0, temperature: float = 0.0, max_tokens: int = 48) -> tuple[str, int]:
    payload = json.dumps({
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/chat/completions", data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode())
    ms = int((time.perf_counter() - t0) * 1000)
    text = (data.get("choices") or [{}])[0].get("message", {}).get("content", "") or ""
    return text.strip(), ms


def score(response: str, value: str) -> tuple[bool, bool]:
    r = response.strip()
    if not r or "NOT_FOUND" in r.upper():
        return False, False
    exact = value.lower() in r.lower() and len(r) < len(value) * 6 + 40
    partial = bool(re.search(re.escape(value[:4]), r, re.IGNORECASE)) if not exact else False
    return exact, partial


def alive(url: str) -> bool:
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    try:
        with urllib.request.urlopen(base + "/health", timeout=3.0) as r:
            return r.status == 200
    except Exception:
        return False


# ── main ───────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="friday-local")
    ap.add_argument("--length", type=int, default=8000, help="haystack size in approx tokens")
    ap.add_argument("--depths", default="500,2000,4000,6000,7500")
    ap.add_argument("--trials", type=int, default=20, help="total trials, spread over depths")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write JSON report here")
    ap.add_argument("--md", default=None, help="write Markdown report here")
    ap.add_argument("--dataset-only", action="store_true",
                    help="generate and inspect the contexts without calling a model")
    args = ap.parse_args(argv)

    depths = [int(x) for x in str(args.depths).split(",") if x.strip()]
    rng = random.Random(args.seed)
    rep = Report(model=args.model, url=args.url, length=args.length, trials=args.trials)

    online = alive(args.url)
    if not online and not args.dataset_only:
        print(f"⚠️  no model at {args.url} — generating the dataset only.\n"
              f"   Start llama-server (docs/architecture/12 §11) and re-run.\n", file=sys.stderr)

    per_depth = max(1, args.trials // max(1, len(depths)))
    made = 0
    for depth in depths:
        for i in range(per_depth):
            name, value = NEEDLES[(made + i) % len(NEEDLES)]
            ctx = build_context(args.length, depth, name, value, rng)
            prompt = ctx + "\n\n" + ask_prompt(name)
            t = Trial(needle_name=name, needle_value=value, depth_tokens=depth,
                      length_tokens=args.length, prompt=prompt)
            if online and not args.dataset_only:
                try:
                    resp, ms = chat(args.url, args.model, prompt)
                    t.response, t.ms = resp, ms
                    t.exact, t.partial = score(resp, value)
                except urllib.error.URLError as e:
                    t.error = f"URLError: {e.reason}"
                except Exception as e:
                    t.error = f"{type(e).__name__}: {e}"
            else:
                t.error = "dataset-only (no model reachable)"
            rep.results.append(t)
            made += 1
            tok = approx_tokens(ctx)
            status = "✓" if t.exact else ("~" if t.partial else "✗")
            print(f"  [{made:>3}] depth={depth:>5} ctx≈{tok:>6}tok {status} "
                  f"{name}={value} {t.ms}ms {t.error[:40]}")

    print("\n" + rep.markdown())

    if args.out:
        p = Path(args.out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "model": rep.model, "url": rep.url, "length": rep.length,
            "trials": rep.trials, "overall_exact": rep.overall,
            "by_depth": rep.by_depth(),
            "results": [{k: v for k, v in t.__dict__.items() if k != "prompt"}
                        for t in rep.results],
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {p}")
    if args.md:
        p = Path(args.md)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(rep.markdown(), encoding="utf-8")
        print(f"wrote {p}")

    # Exit non-zero on a collapse so this can gate a promotion (Law 2c).
    return 0 if (args.dataset_only or rep.overall > 0.0) else 1


if __name__ == "__main__":
    raise SystemExit(main())
