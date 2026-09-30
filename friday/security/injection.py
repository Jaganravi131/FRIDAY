"""L0 prompt-injection detection — the cheap classifier doc 09 §5.2 asks for.

WHERE THIS SITS IN THE DEFENCE
------------------------------
Third of three layers, and the least important one:

  1. `trust.neutralize` — structural. An attacker cannot emit `</recalled>` because
     the characters never reach the prompt. This cannot be evaded by clever wording.
  2. Provenance-gated write authority — below-USER text cannot authorise a
     destructive or externally-visible action, whatever it says.
  3. **This module** — heuristic. Catches the *intent* layer: content that reads like
     it is trying to issue instructions, including spellings that are not on the
     protected-tag list.

Layer 3 is a pattern matcher and will be evaded by anyone determined. It is worth
having anyway because it does something the other two cannot: it makes an attempted
injection VISIBLE. A neutralized fence escape is silent from the user's point of
view; a flagged one shows up in `/why`, in `audit --today`, and in the answer. For a
personal agent the user has to trust with their memory, "tell me when someone tried
this" is worth more than "block 90% of attempts".

FALSE POSITIVES ARE THE DESIGN CONSTRAINT
-----------------------------------------
A personal memory legitimately contains "my API key lives in .env", "ignore those
reminder emails", and "the system prompt for my class project". Flagging those would
train the user to ignore the warning, which is worse than not warning. So no single
signal escalates anything: a verdict requires either one strong signal (an explicit
override phrase) or a combination across categories (an imperative plus a target plus
a channel). Weights below are tuned so ordinary prose scores 0-1.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum


class Verdict(str, Enum):
    CLEAN = "clean"            # ordinary content
    SUSPICIOUS = "suspicious"  # reads like instructions; fence hard and flag it
    HOSTILE = "hostile"        # withhold from context, audit, and tell the user


#: (category, weight, pattern). Grouped by category so the combination rule below can
#: count DISTINCT categories rather than raw hits — three synonyms for "ignore" in one
#: sentence are one signal, not three.
_SIGNALS: tuple[tuple[str, int, re.Pattern], ...] = (
    # ── explicit override attempts: strong on their own ────────────────────────
    ("override", 4, re.compile(
        r"\b(?:ignore|disregard|forget|override|bypass|discard)\s+"
        r"(?:all\s+|any\s+|the\s+|your\s+|every\s+)*"
        r"(?:previous|prior|above|earlier|preceding|system|safety|initial)\s+"
        r"(?:instructions?|prompts?|rules?|directives?|guidance|constraints?)", re.I)),
    ("override", 4, re.compile(
        r"\byou\s+are\s+now\s+(?:unrestricted|unfiltered|uncensored|free|in\s+"
        r"(?:developer|admin|god|dan)\s+mode)|\bnew\s+(?:system\s+)?instructions?\s*:", re.I)),
    ("override", 3, re.compile(
        r"\b(?:developer|debug|maintenance|admin|god)\s+mode\s+(?:enabled|activated|on)\b"
        r"|\bjailbreak\b|\bDAN\s+mode\b", re.I)),

    # ── role spoofing ─────────────────────────────────────────────────────────
    ("role", 3, re.compile(
        r"^\s*(?:system|assistant|developer|tool|admin|root)\s*:\s*\S", re.I | re.M)),
    ("role", 3, re.compile(
        r"</?\s*(?:system|instructions?|identity|soul|agents|developer|override)\s*>", re.I)),
    ("role", 2, re.compile(
        r"\b(?:act|respond|behave|pretend)\s+(?:as|like)\s+(?:the\s+)?"
        r"(?:system|admin|developer|root|an?\s+unrestricted)", re.I)),

    # ── imperative tool use: telling FRIDAY to call something ─────────────────
    ("imperative", 2, re.compile(
        r"\b(?:you\s+must|you\s+should|always|immediately|first)\s+"
        r"(?:call|invoke|use|run|execute)\s+(?:the\s+)?[`']?"
        r"(?:memory_write|memory_search|read_artifact|tool)\b", re.I)),
    ("imperative", 2, re.compile(
        r"^\s*(?:call|invoke|run|execute|delete|drop|overwrite|set)\s+"
        r"[`']?(?:memory_write|memory_search|read_artifact|facts|every\s+fact)\b",
        re.I | re.M)),

    # ── exfiltration: a target AND a channel ──────────────────────────────────
    ("exfil_target", 3, re.compile(
        r"(?:~/\.ssh|id_rsa|id_ed25519|\.netrc|\.git-credentials|\.env\b"
        r"|credentials?\.json|\.aws/credentials|private\s+key|api[_\s-]?key"
        r"|secret[_\s-]?key|password(?:s|list)?\b|\.npmrc|token\.json)", re.I)),
    ("exfil_channel", 3, re.compile(
        r"\b(?:post|send|upload|exfiltrate|transmit|leak|forward|paste)\b[^\n]{0,40}"
        r"\b(?:to|via)\s+(?:https?://|a\s+webhook|the\s+webhook|an?\s+endpoint"
        r"|\w+@\w+\.\w+)", re.I)),
    ("exfil_channel", 2, re.compile(
        r"\b(?:curl|wget|Invoke-WebRequest|requests\.post|fetch\()\s*[^\n]{0,60}https?://", re.I)),

    # ── persistence / self-modification ───────────────────────────────────────
    ("persist", 2, re.compile(
        r"\b(?:from\s+now\s+on|henceforth|permanently|for\s+all\s+future\s+"
        r"(?:turns|queries|requests))\b[^\n]{0,60}\b(?:always|never|must|will)\b", re.I)),
    ("persist", 2, re.compile(
        r"\b(?:write|store|save|record|add)\s+(?:this|these|the\s+following)\s+"
        r"(?:to|into|in)\s+(?:your\s+)?(?:memory|facts?|soul|SOUL\.md|AGENTS\.md)\b", re.I)),

    # ── concealment ───────────────────────────────────────────────────────────
    ("conceal", 2, re.compile(
        r"\b(?:do\s+not|don'?t|never)\s+(?:tell|mention|reveal|inform|show|let\s+know"
        r"|surface|report)\s+(?:the\s+user|them|anyone|this)\b", re.I)),
    ("conceal", 2, re.compile(
        r"\b(?:silently|secretly|without\s+(?:telling|asking|informing)\s+the\s+user)\b", re.I)),
)


#: Categories that are adversarial in DATA even on their own. Contrast with a secret
#: path or an egress channel, which appear constantly in legitimate memory.
STRONG_ALONE = frozenset({"override", "role", "imperative"})


@dataclass
class InjectionReport:
    """The verdict, and the evidence for it — the evidence is the point."""

    text_preview: str = ""
    score: int = 0
    hits: list[tuple[str, str]] = field(default_factory=list)   # (category, matched)
    verdict: Verdict = Verdict.CLEAN

    @property
    def categories(self) -> set[str]:
        return {c for c, _ in self.hits}

    def summary(self) -> str:
        if self.verdict is Verdict.CLEAN:
            return ""
        cats = ", ".join(sorted(self.categories))
        return f"{self.verdict.value} (score {self.score}: {cats})"

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict.value,
            "score": self.score,
            "categories": sorted(self.categories),
            "matched": [m for _, m in self.hits][:12],
        }


def detect(text: str) -> InjectionReport:
    """Score a block of text for injection intent. Regex only; microseconds.

    Cheap enough to run on every retrieved candidate on every turn, which is the only
    way this is useful — a check that runs on a schedule is a check that runs after
    the payload is already in the prompt.
    """
    rep = InjectionReport(text_preview=(text or "")[:160])
    if not text:
        return rep

    # Count DISTINCT categories, and take the strongest hit per category. Three
    # synonyms for "ignore previous instructions" are one signal, not three; but
    # "ignore previous instructions" plus "post ~/.ssh to a webhook" is two.
    best: dict[str, tuple[int, str]] = {}
    for category, weight, pattern in _SIGNALS:
        m = pattern.search(text)
        if m and (category not in best or weight > best[category][0]):
            best[category] = (weight, m.group(0).strip()[:80])

    for category, (weight, matched) in sorted(best.items()):
        rep.score += weight
        rep.hits.append((category, matched))

    n = len(best)

    # ⭐ Not every category means the same thing on its own, and treating them alike
    # is what produces both false positives and false negatives:
    #
    #   STRONG_ALONE — an override phrase, a spoofed role tag, or an imperative naming
    #   one of our own tools. Ordinary memory content does not contain these, so one
    #   is enough to flag.
    #   WEAK_ALONE — a secret path or an egress channel. "my API key lives in .env" and
    #   "curl https://api.example.com returns 404" are completely normal things for a
    #   personal memory to contain. Either one alone must stay CLEAN; it takes a second
    #   category — a target AND a channel, or an imperative AND a destination — to
    #   become an exfiltration attempt rather than a sentence about one.
    #
    # Getting this wrong in either direction is expensive: flag benign content and the
    # user learns to ignore the warning, which is worse than never warning.
    strong = bool(set(best) & STRONG_ALONE)
    if rep.score >= 7 or n >= 3 or (strong and n >= 2):
        rep.verdict = Verdict.HOSTILE
    elif strong or n >= 2:
        rep.verdict = Verdict.SUSPICIOUS
    return rep


def is_hostile(text: str) -> bool:
    return detect(text).verdict is Verdict.HOSTILE


#: What to tell the user when content was withheld. Specific on purpose: "content
#: filtered" is the message of a system that cannot be trusted to explain itself, and
#: the whole product thesis here is that FRIDAY shows its work.
WITHHELD_NOTICE = (
    "⚠ I withheld {n} retrieved item(s) that read like instructions rather than "
    "memory ({reasons}). Memory is data — it cannot tell me what to do. If one of "
    "those is legitimate, tell me and I will treat your word as the authority."
)


def withheld_notice(reports: list[InjectionReport]) -> str:
    reasons = sorted({c for r in reports for c in r.categories})
    return WITHHELD_NOTICE.format(n=len(reports), reasons=", ".join(reasons) or "flagged")
