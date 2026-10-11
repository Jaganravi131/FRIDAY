"""Guard tests for scripts/tune_verify.py — the fine-tune chain prover.

The verifier's job (docs/architecture/18 §4.3) is to prove training links on
machines that cannot download base weights. Its torch/peft/transformers imports
are deliberately lazy (inside functions) so the module loads on stdlib alone —
that matters because CI does, and because a broken stdlib path would fail HERE
at collection time instead of on the user's machine inside Act 3.

Only the dependency-free functions are exercised: the byte-level vocab mapping
(inlined because transformers 5.x moved it), the ByteTokenizer contract the GGUF
side relies on byte-for-byte, and load_rows — including the guard that ABORTS
rather than truncates an answer, added after observing that a 308-byte row in
a 256-block window silently taught the model a chopped-off fact.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
SCRIPT = REPO / "scripts" / "tune_verify.py"


@pytest.fixture(scope="module")
def tv():
    """Load the script as a module without making scripts/ a package."""
    spec = importlib.util.spec_from_file_location("tune_verify", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_loads_without_heavy_deps(tv):
    """The whole point of lazy imports: collection must not need torch."""
    assert "torch" not in sys.modules or "tune_verify" in sys.modules
    # Weak form that holds regardless of test order: the import above already
    # succeeded — a top-level torch import would have raised on stdlib-only CI.


def test_bytes_to_unicode_is_a_bijection_over_all_256(tv):
    b2u = tv._bytes_to_unicode()
    assert len(b2u) == 256, "every byte value must get a code point"
    assert len(set(b2u.values())) == 256, "mapping must be reversible"
    for code in b2u.values():
        assert code.isprintable(), f"non-printable mapped char would break vocab.json: {code!r}"


def test_byte_tokenizer_is_utf8_verbatim(tv):
    tok = tv.ByteTokenizer()
    text = "budget ₹0 — lease 18000 INR é漢"
    assert tok.encode(text) == list(text.encode("utf-8"))
    assert (tok.eos, tok.im_start, tok.im_end) == (256, 257, 258)
    # 259 = the inert merge target; 260 = vocab size. llama.cpp refuses a GGUF
    # with no merges array, so ids 0..259 must all exist exactly once.
    assert tok.vocab_size == 260


def _chat_row(system: str, user: str, assistant: str) -> str:
    return json.dumps({"messages": [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]})


def test_load_rows_masks_prompt_and_closes_with_im_end_eos(tv, tmp_path):
    ds = tmp_path / "rows.jsonl"
    ds.write_text(_chat_row("You are FRIDAY.", "Q?", "The answer is 42.") + "\n",
                  encoding="utf-8")
    rows_x, rows_y = tv.load_rows(ds, tv.ByteTokenizer(), block=512)
    assert len(rows_x) == 1
    ids, labels = rows_x[0], rows_y[0]
    answer_len = len("The answer is 42.".encode()) + 2   # + im_end, eos
    prompt_len = len(ids) - answer_len
    assert prompt_len > 0
    assert all(lab == -100 for lab in labels[:prompt_len]), \
        "loss must not fire on the prompt — the model never writes it"
    assert labels[-2:] == [tv.ByteTokenizer().im_end, tv.ByteTokenizer().eos]
    assert labels[prompt_len:-2] == list("The answer is 42.".encode())


def test_load_rows_aborts_instead_of_truncating_an_answer(tv, tmp_path):
    """Regression for the observed wrong-facts bug: prompt(260)+answer > 256
    block used to amputate the answer tail and train on the stump. The guard
    must refuse loudly rather than teach a chopped fact."""
    ds = tmp_path / "rows.jsonl"
    ds.write_text(_chat_row("s" * 200, "q" * 50, "a" * 60) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="truncat"):
        tv.load_rows(ds, tv.ByteTokenizer(), block=256)


def test_load_rows_skips_blank_lines_and_keeps_order(tv, tmp_path):
    ds = tmp_path / "rows.jsonl"
    ds.write_text(
        _chat_row("s1", "first?", "answer one.") + "\n\n" +
        _chat_row("s2", "second?", "answer two.") + "\n",
        encoding="utf-8")
    rows_x, _ = tv.load_rows(ds, tv.ByteTokenizer(), block=512)
    assert len(rows_x) == 2, "blank lines are whitespace, not data"
    # chat scaffold alone is 21 bytes, so the len(ids)<4 degenerate-guard is
    # unreachable through load_rows' own prompt template; order is the
    # observable contract worth pinning instead.
    assert b"first?" in bytes(i for i in rows_x[0] if i < 256)
    assert b"second?" in bytes(i for i in rows_x[1] if i < 256)
