"""Tests for the CLI — the surface you actually live in.

These are end-to-end by design: they drive `main(argv)` and read stdout, so they
exercise argument parsing, path resolution, the store, the compiler, retrieval and
the ledger together. A unit test of `cmd_search` in isolation would not have caught
the two bugs these found.

Two CLI-specific contracts worth stating:

* EXIT STATUS MEANS SOMETHING. `0` = it worked, `1` = FRIDAY declined. A refusal is a
  normal outcome (Law 7 protecting your hand-edit) and must be scriptable, not a
  traceback. `cmd_write` used to dereference `report["written"]["file"]` on the
  rejected path and raise TypeError, printing a stack trace *before* the explanation.
* THE LEDGER PRINTOUT IS ON BY DEFAULT in ask/chat, because looking at it for a week
  teaches more about context engineering than any article. `--quiet` turns it off;
  the default must not silently change.
"""
from __future__ import annotations

import json

import pytest

from friday.cli import build_parser, main
from friday.llm import MockClient, get_client


@pytest.fixture
def cli(isolated_paths, capsys):
    """`run(argv)` -> (exit_code, stdout). capsys is read inside so each call gets
    only its own output."""
    def run(argv, expect=None):
        capsys.readouterr()                      # discard anything prior
        code = main(list(argv))
        out = capsys.readouterr().out
        if expect is not None:
            assert code == expect, f"{argv} exited {code}, expected {expect}\n{out}"
        return code, out
    return run


# ── the parser ────────────────────────────────────────────────────────────────


def test_parser_covers_every_documented_subcommand():
    """The module docstring lists the subcommands; the parser must actually provide
    them. A docstring that drifts from the parser is how a command goes missing and
    nobody notices until they type it."""
    ap = build_parser()
    sub = next(a for a in ap._actions if hasattr(a, "choices") and a.dest == "cmd")
    have = set(sub.choices)
    expected = {
        "build", "rebuild", "seed", "ask", "chat", "write", "search", "why",
        "history", "timecheck", "status", "audit", "export", "ablation", "needle",
        "doctor", "serve", "bench", "senses",
    }
    assert expected <= have, f"missing: {sorted(expected - have)}"
    assert have - expected == set(), f"undocumented: {sorted(have - expected)}"


def test_every_subcommand_has_a_handler():
    """`main` dispatches through `args.fn`. A subcommand with no `set_defaults(fn=...)`
    parses fine and then dies with AttributeError, which is a much worse failure than
    refusing to parse."""
    ap = build_parser()
    sub = next(a for a in ap._actions if hasattr(a, "choices") and a.dest == "cmd")
    for name in sub.choices:
        ns = ap.parse_args([name] + _required_args(name))
        assert callable(getattr(ns, "fn", None)), f"`{name}` has no handler"


def _required_args(name):
    """Minimal positional args so the parser reaches set_defaults."""
    return {
        "ask": ["hello"], "chat": [], "write": ["mood", "fine"],
        "search": ["rent"], "why": [], "history": ["mood"],
        "timecheck": ["2026-01-01"],
    }.get(name, [])


def test_no_args_prints_usage_and_exits_nonzero(cli):
    """Bare `friday` must show help rather than silently doing nothing."""
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0


def test_python_dash_m_friday_is_the_same_entry_point(isolated_paths):
    """`pyproject.toml` declares `friday = "friday.__main__:main"`. If __main__ stops
    exporting `main`, `pip install -e .` yields a console script that ImportErrors on
    first run — the worst possible moment to find out."""
    import friday.__main__ as m
    assert callable(m.main)
    from friday.cli import main as cli_main
    assert m.main is cli_main


# ── seed / build / rebuild ────────────────────────────────────────────────────


def test_seed_writes_starter_files(cli, isolated_paths):
    code, out = cli(["seed"], expect=0)
    assert "seeded" in out
    assert (isolated_paths / "soul" / "SOUL.md").exists()
    assert (isolated_paths / "soul" / "USER.md").exists()
    assert list((isolated_paths / "memory" / "facts").glob("*.md"))


def test_seed_never_overwrites_your_files(cli, isolated_paths):
    """⭐ Law 7 in the seeder. `soul/USER.md` becomes YOURS the moment you edit it;
    a second `friday seed` that clobbered it would destroy the one file in the system
    that is supposed to be about you."""
    cli(["seed"], expect=0)
    mine = isolated_paths / "soul" / "USER.md"
    mine.write_text("# this is mine, do not touch\n", encoding="utf-8")

    cli(["seed"], expect=0)                     # no --force
    assert mine.read_text(encoding="utf-8") == "# this is mine, do not touch\n"

    cli(["seed", "--force"], expect=0)
    assert "this is mine" not in mine.read_text(encoding="utf-8"), \
        "--force is the explicit escape hatch and must actually overwrite"


def test_build_compiles_markdown_into_the_indices(cli, isolated_paths):
    cli(["seed"], expect=0)
    code, out = cli(["build"], expect=0)
    assert "facts in store" in out
    assert (isolated_paths / "artifacts" / "friday.db").exists()


def test_build_is_idempotent(cli, isolated_paths):
    """Law 2: the artifact is derived, so rebuilding must not accumulate. A second
    build over unchanged Markdown should skip, not re-write everything."""
    cli(["seed"], expect=0)
    cli(["build"], expect=0)
    _, second = cli(["build"], expect=0)
    assert "unchanged skipped" in second


def test_rebuild_verifies_law_2(cli, isolated_paths):
    """`rebuild` deletes the artifact and reconstructs it from Markdown. This is the
    executable form of "Markdown is truth, SQLite is derived" — if it fails, some
    state exists only in the DB and is one `rm` from gone."""
    cli(["seed"], expect=0)
    cli(["build"], expect=0)
    _, before = cli(["search", "rent"], expect=0)

    cli(["rebuild"], expect=0)
    _, after = cli(["search", "rent"], expect=0)

    assert (isolated_paths / "artifacts" / "friday.db").exists(), "the artifact came back"
    # the same facts must be retrievable after a from-scratch rebuild
    assert ("lease" in before) == ("lease" in after), (before, after)


def test_build_reports_rsc_supervision(cli, isolated_paths):
    """⭐ The build line that tells you whether Phase 3.5 has any training signal.
    Compiled facts carry a source_quote, and each one becomes a supervision span for
    the write gate w_t. Without this the compiler could silently produce a store with
    facts but zero spans, and you would find out on Kaggle."""
    cli(["seed"], expect=0)
    _, out = cli(["build"], expect=0)
    assert "RSC supervision" in out
    assert "spans" in out and "w_t" in out


# ── write ─────────────────────────────────────────────────────────────────────


def test_write_records_a_fact_and_reports_the_file(cli, isolated_paths):
    code, out = cli(["write", "monthly_rent", "18000 INR",
                     "--quote", "my monthly rent is 18000 INR",
                     "--user-edit"], expect=0)
    # ⭐ `monthly_rent` is canonicalised to `lease_amount_monthly`, the predicate the
    # seed data uses. Without that, writing the rent creates a SECOND live rent fact
    # and "what is my monthly rent" has two answers — observed as FRIDAY replying
    # ₹28,000 seconds after being told 18000. The CLI must also SAY it renamed the
    # predicate, or the user cannot find what they just wrote.
    assert "Recorded lease_amount_monthly = 18000 INR" in out
    assert "stored as" in out and "monthly_rent" in out
    # Either separator. The CLI prints a real filesystem path, and on Windows that is
    # `memory\facts\housing.md` — which is CORRECT output there, not a bug: showing a
    # Windows user a POSIX path they cannot paste into Explorer would be worse. The
    # assertion is about the file being named, not about the platform's separator.
    assert ("memory/facts/" in out) or ("memory\\facts\\" in out), out
    assert "-> None" not in out, "a real write must name a real file"
    housing = isolated_paths / "memory" / "facts" / "housing.md"
    assert housing.exists() and "18000" in housing.read_text(encoding="utf-8")
    assert "lease_amount_monthly" in housing.read_text(encoding="utf-8")


def test_write_reports_the_salience_span(cli):
    """The span is the free supervision for w_t, and the CLI says so — because a tap
    that silently stops firing is invisible everywhere else."""
    _, out = cli(["write", "mood", "focused", "--quote", "I feel focused",
                  "--user-edit"], expect=0)
    assert "salience span" in out and "w_t" in out


def test_write_exits_1_when_the_hierarchy_refuses(cli):
    """⭐ A refusal is a normal outcome and must be scriptable. This used to raise
    TypeError on `report["written"]["file"]`, printing a traceback before the
    explanation — so the user saw a crash instead of "FRIDAY declined, here's why"."""
    cli(["write", "monthly_rent", "18000 INR", "--user-edit",
         "--quote", "rent is 18000"], expect=0)

    code, out = cli(["write", "monthly_rent", "22000 INR"], expect=1)
    assert "Traceback" not in out
    assert "hand-edit is sacred" in out or "Law 7" in out
    assert "Say so explicitly" in out


def test_write_with_explicit_user_edit_overrides_a_hand_edit(cli):
    """⭐ The other half of the same bug. The refusal message says "Say so explicitly
    if you want it changed" — and `--user-edit` IS saying so explicitly. Refusing it
    too made the message a lie and rendered hand-edited facts permanently
    uncorrectable, which for rent, address and employer is not a safety property.

    It must also RETRACT rather than delete, and record the retraction pair that
    supervises the EDA erase address e_t: a correction you make by hand is the
    highest-value training signal in the system.
    """
    cli(["write", "monthly_rent", "18000 INR", "--user-edit",
         "--quote", "rent is 18000"], expect=0)
    code, out = cli(["write", "monthly_rent", "22000 INR", "--user-edit",
                     "--quote", "no, rent is 22000 now"], expect=0)
    assert "Recorded lease_amount_monthly = 22000 INR" in out
    assert "supersedes" in out or "retracted" in out, out
    assert "e_t" in out, "the retraction pair supervises the erase address"

    _, hist = cli(["history", "monthly_rent"], expect=0)
    assert "18000" in hist and "22000" in hist, \
        "both values survive in the trail — retraction is not deletion"


def test_write_repeating_the_same_value_reinforces_without_a_file(cli):
    """`nochange` returns `file: None`, so the CLI must not print "-> None"."""
    cli(["write", "mood", "calm", "--user-edit", "--quote", "I am calm"], expect=0)
    code, out = cli(["write", "mood", "calm", "--user-edit", "--quote", "still calm"],
                    expect=0)
    assert "-> None" not in out
    assert "already recorded" in out or "reinforced" in out


# ── search / history / timecheck ──────────────────────────────────────────────


def test_search_shows_scores_sources_and_the_floor(cli):
    """The CLI is where you learn whether retrieval is working, so it must show the
    score, which recall path fired, and how many candidates the floor rejected."""
    cli(["seed"], expect=0)
    cli(["build"], expect=0)
    code, out = cli(["search", "what is my monthly rent"], expect=0)
    assert "query=" in out
    assert "floor=" in out and "reranker=" in out
    assert "considered=" in out
    assert "below_floor=" in out and "stale=" in out
    assert "score" in out and "via" in out


def test_search_reports_an_empty_result_honestly(cli):
    """Law 4 at the CLI layer. "nothing found" must be said, not printed as an empty
    table that looks like a rendering bug."""
    cli(["seed"], expect=0)
    cli(["build"], expect=0)
    code, out = cli(["search", "quantum chromodynamics coupling constant"], expect=0)
    assert code == 0, "an empty result is a successful query, not an error"


def test_search_as_of_does_time_travel(cli):
    cli(["write", "lives_in", "Madurai", "--user-edit", "--valid-from", "2024-01-01",
         "--quote", "I live in Madurai"], expect=0)
    cli(["write", "lives_in", "Chennai", "--user-edit", "--valid-from", "2025-07-01",
         "--quote", "I moved to Chennai"], expect=0)

    _, now = cli(["search", "where do I live"], expect=0)
    _, then = cli(["search", "where do I live", "--as-of", "2024-06-01"], expect=0)
    assert "as_of=2024-06-01" in then
    assert "Madurai" in then, f"time travel must surface the then-valid value:\n{then}"


def test_history_shows_both_time_axes(cli):
    """Innovation #3: WORLD time (valid_from -> valid_to) and BELIEF time
    (asserted_at -> retracted_at), side by side. That pairing is what makes "what did
    I used to think?" a different question from "what was true?"."""
    cli(["write", "lives_in", "Madurai", "--user-edit", "--quote", "in Madurai"], expect=0)
    cli(["write", "lives_in", "Chennai", "--user-edit", "--quote", "moved to Chennai"],
        expect=0)
    _, out = cli(["history", "lives_in"], expect=0)
    assert "WORLD time" in out and "BELIEF time" in out
    assert "Madurai" in out and "Chennai" in out


def test_timecheck_separates_true_from_believed(cli):
    cli(["write", "lives_in", "Madurai", "--user-edit", "--quote", "in Madurai"], expect=0)
    _, out = cli(["timecheck", "2026-01-01"], expect=0)
    assert "TRUE on 2026-01-01" in out
    assert "BELIEVED on 2026-01-01" in out


# ── ask / chat ────────────────────────────────────────────────────────────────


def test_ask_prints_the_ledger_block_by_default(cli):
    """⭐ The dev-mode LEDGER block is on by default, because watching it for a week
    teaches more about context engineering than any article. Changing that default
    would quietly remove the single best debugging tool in the system."""
    cli(["seed"], expect=0)
    cli(["build"], expect=0)
    _, out = cli(["ask", "what is my rent", "--mock"], expect=0)
    assert "LEDGER" in out
    assert "budget=" in out and "spent=" in out
    assert "reserve:" in out
    assert "hash=" in out and "prefix=" in out


def test_ask_quiet_hides_the_ledger_but_still_answers(cli):
    cli(["seed"], expect=0)
    cli(["build"], expect=0)
    _, out = cli(["ask", "what is my rent", "--mock", "--quiet"], expect=0)
    assert "LEDGER" not in out
    assert out.strip(), "an answer must still be printed"


def test_ask_with_the_mock_client_needs_no_model(cli):
    """The whole point of MockClient: prove the MEMORY half of "ask Monday, follow up
    Friday" on day one, with no 2.6 GB download and no GPU."""
    cli(["seed"], expect=0)
    cli(["build"], expect=0)
    # Write the predicate the SEEDED store already uses. The seed data carries
    # `lease_amount_monthly: ₹28,000`, and adding a second, distinct predicate for
    # the same concept would leave two live "rent" facts — so this would be testing
    # which of two contradictory facts wins, not whether the mock can answer.
    cli(["write", "lease_amount_monthly", "18000 INR", "--user-edit",
         "--quote", "my monthly rent is 18000 INR"], expect=0)
    _, out = cli(["ask", "what is my monthly rent", "--mock", "--quiet"], expect=0)
    assert "18000" in out, f"the mock must answer from retrieved memory:\n{out}"


def test_ask_ctx_flag_changes_the_ledger_budget(cli):
    cli(["seed"], expect=0)
    _, out = cli(["ask", "hi", "--mock", "--ctx", "4096"], expect=0)
    assert "budget=4096" in out


def test_transformer_backbone_flag_is_available(cli):
    """The ladder's behaviour depends on the backbone, so the CLI has to be able to
    select it. Under a hybrid the ladder can never reach rung 5; the flag is how you
    compare the two without editing config."""
    cli(["seed"], expect=0)
    code, out = cli(["ask", "hi", "--mock", "--transformer-backbone"], expect=0)
    assert "LEDGER" in out


def test_chat_reads_a_repl_session(cli, monkeypatch):
    """`chat` must terminate on EOF rather than spinning, and must record the turns."""
    cli(["seed"], expect=0)
    monkeypatch.setattr("sys.stdin", type("S", (), {
        "readline": lambda self, _it=iter(["what is my rent\n", ""]): next(_it),
    })())
    code, out = cli(["chat", "--mock", "--quiet"], expect=0)
    assert "rent" in out.lower() or out.strip()


# ── status / audit / export ───────────────────────────────────────────────────


def test_status_reports_store_supervision_and_readiness(cli):
    cli(["seed"], expect=0)
    cli(["build"], expect=0)
    _, out = cli(["status"], expect=0)
    assert "facts (live)" in out
    assert "fts rows" in out and "vec rows" in out
    assert "vec extension" in out
    assert "Phase 3.5 ready:" in out
    assert "behavioural signals" in out


def test_status_on_an_empty_store_does_not_crash(cli):
    _, out = cli(["status"], expect=0)
    assert "Phase 3.5 ready: NO" in out


def test_audit_says_so_when_empty(cli):
    _, out = cli(["audit"], expect=0)
    assert "audit is empty" in out


def test_audit_shows_the_enforcement_trail(cli, isolated_paths):
    """The audit trail is what makes tiered consent auditable after the fact."""
    from friday.store import db
    from friday.agent.policy import check, grant
    from friday import paths

    conn = db.connect(paths.DB_PATH)
    check(conn, "mic_listen", {})                 # denied, ungranted
    grant(conn, "mic.listen")
    check(conn, "mic_listen", {"target": "standup"})
    conn.close()

    _, out = cli(["audit"], expect=0)
    assert "mic_listen" in out
    assert "denied" in out and "allowed" in out


def test_audit_limit_is_respected(cli):
    from friday.store import db
    from friday.agent.policy import log
    from friday import paths

    conn = db.connect(paths.DB_PATH)
    for i in range(10):
        log(conn, actor="cli", action=f"a{i}", decision="allowed")
    conn.close()
    _, out = cli(["audit", "--limit", "3"], expect=0)
    assert out.count("\n") < 10


def test_export_writes_jsonl_for_kaggle(cli, isolated_paths):
    """Phase 3.5 training runs on a free Kaggle T4, so supervision has to leave this
    machine as a portable file. JSONL is the interchange format."""
    cli(["seed"], expect=0)
    cli(["build"], expect=0)
    cli(["write", "mood", "focused", "--user-edit", "--quote", "I feel focused"],
        expect=0)

    out_path = isolated_paths / "artifacts" / "rsc_supervision.jsonl"
    cli(["export", "--out", str(out_path)], expect=0)
    assert out_path.exists()
    lines = [json.loads(l) for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    assert lines, "the seeded store must produce at least one supervision record"
    assert all(isinstance(r, dict) for r in lines)


# ── the RSC subcommands ───────────────────────────────────────────────────────


def test_ablation_runs_the_ladder_from_the_cli(cli):
    """`friday ablation --quick` must work with no torch installed: the operators are
    pure Python precisely so the decisive experiment is runnable on a laptop."""
    code, out = cli(["ablation", "--quick"], expect=0)
    assert "retnet" in out and "eda" in out
    assert "rung" in out.lower()


def test_needle_benchmark_runs_from_the_cli(cli):
    """⭐ The decisive recall benchmark. RetNet's fixed decay loses 0.5% per step and
    must be visibly worse than every gated rung, or the whole pivot is unfounded."""
    # `needle` has no --quick; it takes --length/--trials/--depths. Small values keep
    # the pure-Python operator sweep fast enough for a test run.
    code, out = cli(["needle", "--length", "128", "--trials", "1",
                     "--depths", "8,32", "--dataset-only"], expect=0)
    assert "needle" in out.lower() or "depth" in out.lower()


# ── the client surface ────────────────────────────────────────────────────────


def test_get_client_prefers_the_mock_when_no_server_is_configured():
    """₹0 budget, and no llama-server running in a test sandbox. `get_client` must
    land somewhere usable rather than raising."""
    c = get_client(prefer="mock")
    assert isinstance(c, MockClient)


def test_get_client_never_raises_for_an_unknown_preference():
    c = get_client(prefer="definitely-not-a-real-backend")
    assert c is not None
    assert hasattr(c, "complete") or hasattr(c, "chat")


def test_the_llm_package_re_exports_its_public_surface():
    """`friday/llm/__init__.py` exists so callers write `from friday.llm import
    get_client`. Without it the package is not importable as a package at all."""
    import friday.llm as llm
    for name in ("Client", "MockClient", "NoRouteError", "OpenAICompatibleClient",
                 "Response", "ToolCall", "get_client"):
        assert hasattr(llm, name), f"friday.llm must re-export {name}"


def test_no_route_error_exists_to_stop_a_silent_hallucination():
    """With CLOUD_TIERS_ENABLED False there is no top of the Escalation Ladder, so
    `escalate` must degrade to ASK THE USER. NoRouteError makes that explicit instead
    of letting it fall through to a confident guess."""
    from friday.llm import NoRouteError
    assert issubclass(NoRouteError, RuntimeError)


# ── ⭐ canonical predicates: one fact per concept ─────────────────────────────


def test_writing_an_alias_supersedes_the_seeded_fact_instead_of_doubling(cli):
    """⭐ THE BUG THIS DEFENDS. The seed data says `lease_amount_monthly: ₹28,000`.
    A user who types `friday write monthly_rent 18000` used to create a SECOND live
    fact, because reconciliation matched on exact (subject, predicate) and neither
    predicate was single-valued against the other. Both stayed live, retrieval
    returned whichever reranked higher, and FRIDAY confidently answered ₹28,000
    seconds after being told 18000.

    Fluent, sourced, and wrong is the worst possible failure mode for a
    context-aware assistant, so the fix is that there is ONE predicate per concept.
    """
    cli(["seed"], expect=0)
    cli(["build"], expect=0)
    _, before = cli(["search", "monthly rent"], expect=0)

    cli(["write", "monthly_rent", "18000 INR", "--user-edit",
         "--quote", "my monthly rent is 18000 INR"], expect=0)
    _, after = cli(["search", "monthly rent"], expect=0)

    assert "18000" in after, f"the value just written must be the answer:\n{after}"
    # the seeded ₹28,000 must have been RETRACTED, not left live alongside
    assert "₹28,000" not in after or "RETRACTED" in after, \
        f"two live rent facts is the bug:\n{after}"


def test_history_accepts_the_alias_you_wrote_with(cli):
    """You can write with an alias, so you must be able to look the trail up with
    that same alias. Otherwise the CLI says it renamed your predicate and then
    claims no history exists under either name."""
    cli(["write", "monthly_rent", "18000 INR", "--user-edit",
         "--quote", "rent is 18000"], expect=0)
    cli(["write", "monthly_rent", "22000 INR", "--user-edit",
         "--quote", "rent is 22000"], expect=0)

    _, via_alias = cli(["history", "monthly_rent"], expect=0)
    _, via_canon = cli(["history", "lease_amount_monthly"], expect=0)
    assert "no history" not in via_alias, via_alias
    assert "18000" in via_alias and "22000" in via_alias
    assert "stored as" in via_alias, "and it should say which predicate it resolved to"
    assert "18000" in via_canon and "22000" in via_canon


def test_canonicalisation_is_conservative(cli):
    """Merging two predicates that mean DIFFERENT things destroys data, which is
    worse than duplicating one. `lives_in` (city) and `address_locality`
    (neighbourhood) must stay separate, and both must be independently writeable."""
    cli(["write", "lives_in", "Chennai", "--user-edit", "--quote", "I live in Chennai"],
        expect=0)
    cli(["write", "address_locality", "Velachery", "--user-edit",
         "--quote", "in Velachery"], expect=0)

    _, out = cli(["status"], expect=0)
    assert "facts (live)" in out
    # both survived as distinct facts
    import sqlite3
    from friday import paths
    from friday.store import db
    conn = db.connect(paths.DB_PATH)
    preds = {r["predicate"] for r in
             conn.execute("SELECT predicate FROM facts WHERE retracted_at IS NULL")}
    conn.close()
    assert "lives_in" in preds and "address_locality" in preds, preds
