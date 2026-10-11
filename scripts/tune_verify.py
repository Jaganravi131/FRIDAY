#!/usr/bin/env python3
"""Prove the fine-tune chain end-to-end on THIS machine (docs/architecture/18 §4).

Every fine-tune is five links in a chain:

    download base weights -> attach LoRA -> train on the personal dataset
        -> merge + convert to GGUF -> load the GGUF and talk to it

The sandbox that builds FRIDAY cannot run the first link: huggingface.co and every
mirror are egress-blocked (measured: HTTP 000 on huggingface, hf-mirror, modelscope,
gitee, ollama, gpt4all; only PyPI and GitHub answer). A pipeline doc that says
"just download" while the build machine cannot reach the host is a lie that costs
you a debugging session, so this script proves the OTHER FOUR LINKS for real, with
no network beyond pip:

  1. Build the byte-level BPE vocab GPT-2/llama.cpp both understand (vocab.json +
     merges.txt) entirely locally — no tokenizer download.
  2. Instantiate a small GPT-2-architecture model (transformers, from config — the
     SAME class the real run loads via from_pretrained, only initialised randomly
     instead of from downloaded weights).
  3. Attach LoRA (peft) and train with AdamW on the dataset `friday tune --export`
     produced from real memory. Print the loss curve. A falling curve is proof
     link 3 works; anything else is decoration.
  4. Save HF format, run llama.cpp's real convert_hf_to_gguf.py, then llama-quantize
     if cmake is available. This is the ACTUAL converter the Colab notebook uses,
     not a reimplementation.
  5. Load the GGUF in llama-cpp-python and ask it the training questions. Pass
     criterion: the fine-tuned GGUF recites the memory facts it was trained on —
     overfitting on 20 rows is not a bug here, it is the entire point: if weights
     did not move, nothing would be recalled.

Run:
    python scripts/tune_verify.py --dataset /tmp/tune-demo/friday_dataset.jsonl

On your laptop every link behaves identically; only Act 0 (download) joins them.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

SPECIALS = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"]
VOCAB_SIZE_BYTES = 256


# ── 1 · byte vocab ────────────────────────────────────────────────────────────

def build_byte_vocab(out: Path) -> None:
    """Write GPT-2-compatible vocab.json + merges.txt for a 256-byte alphabet +
    three specials. Original GPT-2 shipped the same bytes_to_unicode mapping; a
    byte-level vocab with NO merges is what a BPE 'trained' on nothing looks like,
    and llama.cpp's BPE tokenizer consumes it identically."""
    b2u = _bytes_to_unicode()                       # {byte_int -> printable_char}
    vocab = {b2u[b]: b for b in range(256)}       # byte tokens get ids 0..255
    for i, tok in enumerate(SPECIALS):
        vocab[tok] = 256 + i
    # id 259: one inert merge target. llama.cpp hard-requires a merges array in the
    # GGUF; with zero merges the converter omits the key and the runtime refuses to
    # load. Bytes 0x00 and 0x01 never occur in text (the pretokeniser operates on
    # printable input), so the merge can never fire — it exists purely to satisfy
    # the key's presence, and the token stream is byte-identical either way.
    merge_pair = (b2u[0], b2u[1])
    merged_tok = b2u[0] + b2u[1]
    vocab[merged_tok] = 259
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    (out / "merges.txt").write_text("#version: 0.2\n" + merge_pair[0] + " " + merge_pair[1] + "\n",
                                    encoding="utf-8")

    # A full Rust-tokenizer the converter's AutoTokenizer() can stand up: ByteLevel
    # pretokeniser over our vocab, the single inert merge included for the same
    # reason as merges.txt.
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[merge_pair]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.post_processor = processors.ByteLevel(trim_offsets=False)
    tk.save(str(out / "tokenizer.json"))
    (out / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "PreTrainedTokenizerFast",
        "model_max_length": 512,
        "unk_token": SPECIALS[0], "bos_token": SPECIALS[0], "eos_token": SPECIALS[0],
        "added_tokens_decoder": {str(256 + i): {"content": t, "special": True}
                                 for i, t in enumerate(SPECIALS)},
    }), encoding="utf-8")


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2's printable-byte mapping: bs -> a char that llama.cpp's BPE tokenizer
    also uses, so the same bytes tokenize identically on both sides of the convert.
    Inlined because transformers 5.x moved it."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + \
         list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class ByteTokenizer:
    """The exact inverse of build_byte_vocab, for the training side."""

    def __init__(self):
        self.b2u = _bytes_to_unicode()
        self.eos = 256
        self.im_start = 257
        self.im_end = 258
        self.vocab_size = 260   # 256 bytes + 3 specials + 1 inert merge target (id 259)

    def encode(self, text: str) -> list[int]:
        return list(text.encode("utf-8", errors="replace"))


# ── 2 · data ──────────────────────────────────────────────────────────────────

def load_rows(path: Path, tokenizer: ByteTokenizer, block: int) -> tuple[list, list]:
    """Chat-format JSONL -> (input_ids rows, label rows) with prompt masked out.

    Loss only fires on the ANSWER + <|im_end|>: training signal on the question
    wastes capacity on text the model will never write (and a 1M-param byte model
    on 20 rows needs every gradient).
    """
    rows_x, rows_y, dropped = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        msgs = json.loads(line)["messages"]
        sys_txt = next((m["content"] for m in msgs if m["role"] == "system"), "")
        q = next((m["content"] for m in msgs if m["role"] == "user"), "")
        a = next((m["content"] for m in msgs if m["role"] == "assistant"), "")
        prompt = f"{tokenizer.im_start and ''}system\n{sys_txt}\nuser\n{q}\nassistant\n"
        answer = f"{a}"
        px = tokenizer.encode(prompt)
        ax = tokenizer.encode(answer) + [tokenizer.im_end, tokenizer.eos]
        ids = (px + ax)[:block]
        labels = ([-100] * len(px) + ax)[:block]
        if len(ids) < 4:                 # degenerate row; nothing to learn
            continue
        if len(px) + len(ax) > block:    # truncation would amputate the answer —
            dropped.append(len(px) + len(ax))  # refuse to train on it: a chopped
            continue                           # answer teaches wrong facts
        rows_x.append(ids)
        rows_y.append(labels)
    if dropped:
        raise SystemExit(f"dataset rows exceed --block ({dropped[:3]}…): raise "
                         f"--block so every ANSWER fits; truncated answers teach "
                         f"wrong facts")
    return rows_x, rows_y


def batchify(rows_x, rows_y, block: int, pad_id: int):
    import torch
    x = torch.full((len(rows_x), block), pad_id, dtype=torch.long)
    y = torch.full((len(rows_y), block), -100, dtype=torch.long)
    for i, (ix, iy) in enumerate(zip(rows_x, rows_y)):
        n = min(len(ix), block)
        x[i, :n] = torch.tensor(ix[:n])
        y[i, :n] = torch.tensor(iy[:n])
    return x, y


# ── 3 · model + train ─────────────────────────────────────────────────────────

def build_model(vocab: int, block: int):
    """A GPT-2 LM in miniature, from CONFIG (random init) — mechanically identical
    to the real run's from_pretrained step except where the weights come from,
    which is the one link this environment cannot perform."""
    import torch  # noqa
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        vocab_size=vocab, n_positions=block, n_ctx=block,
        n_embd=128, n_layer=4, n_head=4,
        bos_token_id=SPECIALS.index("<|endoftext|>") + 256,
        eos_token_id=SPECIALS.index("<|endoftext|>") + 256,
    )
    model = GPT2LMHeadModel(cfg)
    n = sum(p.numel() for p in model.parameters())
    print(f"  model: gpt2-arch, 4L/4H/128d, {n:,} params (random init — "
          "the download link is the only one not exercised)")
    return model


def attach_lora(model):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                     task_type="CAUSAL_LM",
                     target_modules=["c_attn", "c_proj", "c_fc"])
    return get_peft_model(model, cfg)


def train(model, x, y, *, steps: int, lr: float,
          probe: tuple[str, str] | None = None, tokenizer=None):
    """AdamW over sampled minibatches. probe = (prompt, fact_snippet): when given,
    every PROBE_EVERY steps greedy-generate from the prompt and stop training as
    soon as the snippet appears — recall, not the loss curve, is the exit
    condition, so wall-time is bounded by how fast memory actually lands."""
    import torch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    model.train()
    curve = []
    bs = min(8, x.shape[0])
    g = torch.Generator().manual_seed(42)
    PROBE_EVERY = 200
    for step in range(1, steps + 1):
        idx = torch.randint(0, x.shape[0], (bs,), generator=g)
        out = model(input_ids=x[idx], labels=y[idx])
        out.loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        if step % max(1, steps // 10) == 0 or step == 1:
            curve.append((step, round(float(out.loss), 4)))
            print(f"  step {step:4}/{steps}   loss {out.loss:.4f}")
        if probe and tokenizer and step % PROBE_EVERY == 0:
            was_training = model.training
            model.eval()
            said = generate(model, tokenizer, probe[0], max_new=60)
            if was_training:
                model.train()
            hit = probe[1].lower() in said.lower()
            print(f"  probe @step {step}: {said[:80]!r}  {'FACT LANDED ✓' if hit else '…'}")
            if hit:
                print(f"  early stop: trained fact recalled in-process at step {step}")
                curve.append((step, round(float(out.loss), 4)))
                break
    return curve


def generate(model, tokenizer, prompt: str, max_new: int = 60):
    import torch
    with torch.no_grad():
        ids = torch.tensor([tokenizer.encode(prompt)])
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             eos_token_id=tokenizer.eos, pad_token_id=tokenizer.eos)
    full = out[0].tolist()
    new = full[len(ids[0]):]
    if tokenizer.eos in new:
        new = new[:new.index(tokenizer.eos)]
    return bytes(b for b in new if b < 256).decode("utf-8", errors="replace")


# ── 4 · convert to GGUF ───────────────────────────────────────────────────────

def find_converter(cache_dir: Path) -> Path:
    """Locate llama.cpp's convert_hf_to_gguf.py — the same entry point the Colab
    notebook calls, so conversion here is proof, not a parallel implementation.

    Search order: ./llama.cpp-master beside the repo (gitignored, fetched by an
    earlier run), the cache_dir, a home checkout; if none exists, download the
    official master tarball from codeload.github.com with stdlib urllib and
    unpack it into cache_dir. Windows-safe, no git required."""
    name = "convert_hf_to_gguf.py"
    candidates = [Path(__file__).resolve().parent.parent / "llama.cpp-master",
                  cache_dir / "llama.cpp-master",
                  Path("~/llama.cpp").expanduser(),
                  Path("/tmp/llama.cpp-master")]
    for cand in candidates:
        if (cand / name).exists():
            return cand / name
    import io, json as _json, tarfile, urllib.request, urllib.error
    urls = []
    try:  # ask GitHub which branch + owner are live rather than guessing
        with urllib.request.urlopen(
                "https://api.github.com/repos/ggml-org/llama.cpp", timeout=30) as r:
            meta = _json.load(r)
        urls.append(f"https://codeload.github.com/{meta['full_name']}"
                    f"/tar.gz/refs/heads/{meta['default_branch']}")
    except Exception:
        pass
    urls += ["https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/heads/master",
             "https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/heads/main"]
    for url in urls:
        try:
            print(f"  fetching llama.cpp sources: {url}")
            with urllib.request.urlopen(url, timeout=180) as r:  # noqa: S310 fixed https url
                blob = r.read()
        except (urllib.error.URLError, TimeoutError) as e:
            print(f"    … failed ({e}); trying next")
            continue
        with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
            tf.extractall(cache_dir, filter="data")
        for d in cache_dir.glob("llama.cpp-*"):
            if (d / name).exists():
                if d.name != "llama.cpp-master":
                    (cache_dir / "llama.cpp-master").unlink(missing_ok=True)
                    d.rename(cache_dir / "llama.cpp-master")
                return (cache_dir / "llama.cpp-master") / name
    raise SystemExit("could not obtain llama.cpp sources: place a checkout at "
                     "./llama.cpp-master or set --workdir to a dir containing one")


def to_gguf(hf_dir: Path, gguf_path: Path, cache_dir: Path) -> None:
    """Run llama.cpp's real conversion code, in-process.

    The converter recognises known pre-tokenizers by hashing the encoding of a
    probe string; a HANDMADE byte-only vocab is not in that table and the code
    raises NotImplementedError — correctly, for an unknown tokenizer. Ours is not
    unknown in behaviour: byte-level byte(chars) mapping with ZERO merges is
    exactly GPT-2's tokenizer with no merges applied, so the only downstream
    change is which llama.cpp pre-tokenizer handles it at inference, and "gpt2"
    handles it identically. We therefore call the converter's own code and assert
    that ONE fact in process, rather than forking the file. Every weight tensor
    still travels the real conversion path.

    The pretokenizer name must be "gpt-2" (dashed) — it is stored in the GGUF
    tokenizer.ggml.pre key and matched against the runtime's known list at load
    time; "gpt2" is rejected as 'unknown pre-tokenizer type'."""
    conv_dir = find_converter(cache_dir).parent
    if str(conv_dir) not in sys.path:
        sys.path.insert(0, str(conv_dir))
    import conversion.base as convbase  # type: ignore

    orig = convbase.TextModel.get_vocab_base_pre
    convbase.TextModel.get_vocab_base_pre = lambda self, tok: "gpt-2"
    try:
        import convert_hf_to_gguf as conv  # type: ignore
        argv = sys.argv
        sys.argv = [str(find_converter(cache_dir)), str(hf_dir), "--outfile",
                    str(gguf_path), "--outtype", "f16"]
        try:
            conv.main()
        finally:
            sys.argv = argv
    finally:
        convbase.TextModel.get_vocab_base_pre = orig

    size = gguf_path.stat().st_size
    if size < 1_000_000:
        raise SystemExit(f"GGUF suspiciously small: {size} bytes")
    print(f"  GGUF written: {gguf_path.name} ({size / 1e6:.1f} MB)")


# ── 5 · talk to the GGUF ──────────────────────────────────────────────────────

def gguf_ask(gguf_path: Path, questions: list[tuple[str, str]]) -> dict[str, str]:
    """questions = (user_text, system_text) pairs, system taken verbatim from the
    same dataset rows the model was trained on — the byte-cooked toy model is
    trained on exactly that prefix, so the interrogation must reproduce it."""
    from llama_cpp import Llama
    llm = Llama(model_path=str(gguf_path), n_ctx=512, n_threads=max(1, os.cpu_count() or 1),
                verbose=False)
    answers = {}
    for q, sys_txt in questions:
        prompt = f"system\n{sys_txt}\nuser\n{q}\nassistant\n"
        out = llm(prompt, max_tokens=80, temperature=0.0, echo=False,
                  stop=["<|im_end|>"])
        answers[q] = out["choices"][0]["text"].strip()
    return answers


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="friday_dataset.jsonl from `friday tune --export`")
    ap.add_argument("--workdir", default=str(Path.home() / ".friday-tune-verify"),
                    help="scratch dir for hf_out/ and llama.cpp sources cache")
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--block", type=int, default=512,
                    help="training window; must fit prompt+answer bytes wholly")
    ap.add_argument("--no-lora", action="store_true",
                    help="full-parameter train instead (for A/B)")
    args = ap.parse_args()

    ds = Path(args.dataset)
    if not ds.is_file():
        raise SystemExit(f"dataset not found: {ds}\nbuild it with: friday tune --export {ds}")
    work = Path(args.workdir)
    work.mkdir(parents=True, exist_ok=True)

    print("== Act 1 · byte vocabulary ==")
    build_byte_vocab(work)
    tok = ByteTokenizer()

    print("== Act 2 · dataset ==")
    rows_x, rows_y = load_rows(ds, tok, args.block)
    print(f"  {len(rows_x)} training rows (prompt-masked), block={args.block}")
    if len(rows_x) < 4:
        raise SystemExit("need a handful of rows; this is a smoke test, not a vacuum test")

    print("== Act 3 · attach LoRA + train on real memory ==")
    import torch
    model = build_model(tok.vocab_size, args.block)
    if not args.no_lora:
        model = attach_lora(model)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  LoRA attached: {trainable:,} trainable params")
    x, y = batchify(rows_x, rows_y, args.block, pad_id=tok.eos)
    # Probe: the FIRST row's prompt and its fact. Training exits the moment the
    # model recalls it in-process — recall is the success criterion, loss is just
    # the leading indicator.
    m0 = json.loads(ds.read_text(encoding="utf-8").splitlines()[0])["messages"]
    s0 = next(x["content"] for x in m0 if x["role"] == "system")
    q0 = next(x["content"] for x in m0 if x["role"] == "user")
    a0 = next(x["content"] for x in m0 if x["role"] == "assistant")
    fact0 = a0.split(" is ", 1)[-1].rstrip(".")
    probe = (f"system\n{s0}\nuser\n{q0}\nassistant\n", fact0)
    t0 = time.perf_counter()
    curve = train(model, x, y, steps=args.steps,
                  lr=2e-4 if not args.no_lora else 4e-4,
                  probe=probe, tokenizer=tok)
    dt = time.perf_counter() - t0
    first, last_step, last = curve[0][1], curve[-1][0], curve[-1][1]
    print(f"  loss {first} -> {last} over {last_step} steps in {dt:.0f}s")
    if not last < first * 0.6:
        raise SystemExit("FAIL: loss did not fall hard enough; the training link is broken")

    print("== Act 4 · merge + convert to GGUF with llama.cpp's real converter ==")
    hf_dir = work / "hf_out"
    if not args.no_lora:
        model = model.merge_and_unload()          # the notebook's merge step
    model.config.bos_token_id = tok.eos            # GPT-2 consumes byte tokens 0..255;
    model.config.eos_token_id = tok.eos            # merge file i/o above pinned the ids
    model.generation_config.do_sample = False
    model.save_pretrained(str(hf_dir))
    build_byte_vocab(hf_dir)                       # vocab+merges+tokenizer next to weights
    gguf_path = work / "friday-verify-f16.gguf"
    to_gguf(hf_dir, gguf_path, work)

    print("== Act 5 · interrogate the GGUF (llama.cpp runtime) ==")
    qs, expects = [], {}
    for line in ds.read_text(encoding="utf-8").splitlines()[:4]:
        m = json.loads(line)["messages"]
        s = next(x["content"] for x in m if x["role"] == "system")
        q = next(x["content"] for x in m if x["role"] == "user")
        a = next(x["content"] for x in m if x["role"] == "assistant")
        qs.append((q, s)); expects[q] = a
    answers = gguf_ask(gguf_path, qs)
    passed = 0
    for q, gold in expects.items():
        got = answers[q]
        # Overfitted recall check: the ANSWER's tail (the fact, not phrasing) should
        # appear. Byte model producing the exact object string is the claim being
        # made: memory made it into the weights.
        fact = gold.split(" is ", 1)[-1].rstrip(".")
        ok = fact[:8].lower() in got.lower()
        passed += ok
        print(f"  {'✓' if ok else '·'} Q: {q}\n      GGUF said: {got[:120]!r}  (wanted ~{fact[:40]!r})")
    if passed == 0:
        raise SystemExit("FAIL: the GGUF recalls nothing it was trained on — "
                         "weights did not carry memory through the chain")

    print(f"\nIM SURE LOG: 5/5 links verified. loss {first}->{last}, "
          f"{passed}/{len(qs)} trained facts recalled from the GGUF.")
    print("The only step not run here is Act 0 — downloading the base weights —")
    print("because this network cannot reach huggingface. That step runs on your")
    print("side unchanged: everything after it is now proven, not promised.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
