"""Tests for the fine-tuning pipeline (docs/architecture/18 §4).

The sandbox cannot download models — huggingface egress is blocked, and that is a
measured fact, not an assumption. What MUST work here is the part that decides
whether training is worth anything: the dataset is honest (provenance applied,
secrets redacted), the notebook is a valid Colab document that runs the real run,
and the model table's arithmetic does not lie about the laptop.
"""
from __future__ import annotations

import json

import pytest

from friday.memory.facts import Fact, assert_fact
from friday.store import db
from friday.tune import dataset, models, notebook


@pytest.fixture
def lived_memory(root):
    conn = db.connect(root / "artifacts" / "friday.db")
    facts = [
        Fact(subject="user", predicate="standup_day", object="Tuesday 10:00 IST",
             domain="schedule", source_kind="stated", source_quote="my standup is Tuesday ten"),
        Fact(subject="user", predicate="lease_amount_monthly", object="18000 INR",
             domain="housing", source_kind="user_edit", source_quote="my rent is 18000"),
        Fact(subject="user", predicate="guessed_favorite_color", object="teal",
             domain="preference", source_kind="inferred",
             source_quote="the word teal appeared"),
        Fact(subject="user", predicate="old_job", object="Bogus Corp",
             domain="work", source_kind="stated", source_quote="never said"),
    ]
    for f in facts:
        assert_fact(conn, f)
    # Retract the stale fact; it must not train anything.
    row = conn.execute("SELECT id FROM facts WHERE predicate='old_job'").fetchone()
    conn.execute("UPDATE facts SET retracted_at='2026-10-01T00:00:00+00:00' WHERE id=?",
                 (row["id"],))
    conn.commit()
    yield conn
    conn.close()


def test_user_authority_facts_become_training_rows(lived_memory):
    rows, stats = dataset.export_dataset(lived_memory)
    bodies = json.dumps(rows, ensure_ascii=False)
    assert "standup day" in bodies and "Tuesday 10:00 IST" in bodies
    assert "rent" in bodies.lower() or "lease" in bodies.lower()
    assert stats.persona_rows >= 1 and stats.routine_rows >= 1


def test_inferred_facts_never_become_ground_truth(lived_memory):
    rows, stats = dataset.export_dataset(lived_memory)
    bodies = json.dumps(rows, ensure_ascii=False)
    assert "teal" not in bodies, \
        "an inferred fact in the weights would be FRIDAY believing its own guess"
    assert stats.skipped.get("inferred_or_external", 0) >= 1


def test_retracted_facts_are_excluded(lived_memory):
    rows, _ = dataset.export_dataset(lived_memory)
    assert "Bogus Corp" not in json.dumps(rows, ensure_ascii=False)


def test_secrets_are_redacted_before_export(root):
    """The export is a network edge (the file goes to Colab), so redaction must
    happen in export_dataset itself — not rely on the write path having done it.
    A hand-written markdown fact reaches the store via the compiler, NOT via
    assert_fact's redactor, which is exactly the path this test exercises."""
    conn = db.connect(root / "artifacts" / "friday.db")
    token = "ghp_" + "A" * 36

    # First: a normal write. The WRITE path redacts (facts.py), so this fact lands
    # already-markered — proof the layer below works, and that export staying clean
    # is not legible as "export redacted anything".
    assert_fact(conn, Fact(subject="user", predicate="build_key", object=token,
                           source_kind="stated", source_quote="key ghp AAA"))
    conn.commit()
    stored = conn.execute("SELECT object FROM facts WHERE predicate='build_key'"
                          ).fetchone()["object"]
    assert token not in stored, "the write path is the first redaction layer"

    # Second: the compile path — hand-edit a markdown facts file and compile it.
    # This is exactly how a secret escapes assert_fact's redactor in real life: the
    # user (or an imported file) writes it into truth, and the compiler stores it.
    md = root / "memory" / "facts" / "handnotes.md"
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("- deploy_token: **" + token + "** [" +
                  "f_manual1 · user_edit · conf 1.0]\n", encoding="utf-8")
    from friday.memory import compiler
    n = compiler.compile_file(md, conn)
    conn.commit()
    assert n >= 1
    stored2 = conn.execute("SELECT object FROM facts WHERE predicate='deploy_token'"
                           ).fetchone()["object"]
    assert token in stored2, "the compile path stores verbatim — THAT is the hole"

    try:
        rows, stats = dataset.export_dataset(conn)
        bodies = json.dumps(rows)
        assert token not in bodies, "a GitHub token must not leave the machine"
        assert stats.redacted_rows >= 1, "export must redact what the compiler stored verbatim"
    finally:
        conn.close()


def test_jsonl_roundtrip_is_valid_chat_format(tmp_path, lived_memory):
    rows, _ = dataset.export_dataset(lived_memory)
    out = tmp_path / "d.jsonl"
    dataset.write_jsonl(rows, out)
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert lines and len(lines) == len(rows)
    for line in lines:
        r = json.loads(line)
        roles = [m["role"] for m in r["messages"]]
        assert roles[0] == "system" and "user" in roles and "assistant" in roles
        assert all(m["content"] for m in r["messages"])


def test_notebook_is_a_valid_colab_document(tmp_path):
    p = tmp_path / "run.ipynb"
    notebook.write_notebook(p, base_repo="unsloth/Qwen3-4B-Instruct-2507", epochs=2)
    nb = json.loads(p.read_text(encoding="utf-8"))
    assert nb["nbformat"] == 4
    code = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    for anchor in ("FastLanguageModel.from_pretrained", "SFTTrainer",
                   "save_pretrained_gguf", "q4_k_m", "files.download",
                   "unsloth/Qwen3-4B-Instruct-2507"):
        assert anchor in code, f"notebook missing: {anchor}"
    md = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "markdown")
    assert "bench --gate" in md, "Law 10's gate must be in the runbook, not implied"
    assert "T4" in md or "accelerator" in json.dumps(nb)


def test_model_table_stays_honest_about_16gb():
    for c in models.CHOICES:
        if c.local_16gb_ok:
            # Only the smallest may claim CPU viability, and even that is "~3 GB"
            assert c.params_b <= 2.0
        if c.colab_free_ok:
            assert c.qlora_train_gb <= 12.0, \
                "a free T4 is 15 GB VRAM; a row over ~12 has no headroom for context"
    assert models.default_choice().colab_free_ok
