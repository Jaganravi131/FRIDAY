# 18 · The Agentic Workspace: tools, connectors, and the tuning pipeline

*Phase 2 of the tool ladder. Builds on doc 09 (trust lattice), doc 15 (retrieval),
and doc 16 §2.1 (reaching without acting). Read this before touching any of the
three; the design constraints here are load-bearing, not stylistic.*

---

## 1 · What FRIDAY was missing, stated plainly

Until now FRIDAY could remember (Phase 0) and read the public web (Phase 1). What
it could not do is *work*: hold a folder, draft in it, inspect it, run read-only
commands over it. That is the thing Arena-style agentic workspaces demonstrate is
actually useful — not "an agent with a browser" but "an assistant that can pick up
a task and put it down in front of you finished." This doc adds exactly that and
deliberately no more, because the difference between a workspace and a rooted
laptop is one unguarded shell tool.

The three additions, each behind its own consent:

| layer | tools | sense | approval shape |
|---|---|---|---|
| workspace files | `ws_list` `ws_read` `ws_write` | `workspace.files` (low) | one grant covers the folder |
| curated shell | `ws_shell` | `workspace.exec` (extreme) | the grant IS the allowlist |
| connectors | `email_read` (gmail) `calendar_read` (caldav) `github_read` (REST) `email_send` (smtp, §3.6) | per-connector (`email.read`, `calendar.read`, `github.read`, `email.send`) | one grant per account type; send is ALWAYS_CONFIRM |

Registration and authority are separate on purpose: `FRIDAY_TOOLS_PHASE=2` makes
the tools *offerable* to the model. `policy.check()` still gates every individual
call on the sense. Offering is not granting.

## 2 · The workspace (friday/agent/workspace.py)

One folder. Default `friday-workspace/` inside the FRIDAY root; movable via
`workspace.root` in `config/connectors.json`. Five rules, all enforced in code:

1. **Containment.** Every path resolves against the resolved root and must stay
   under it (`startswith(root + sep)`). `..`, absolute paths, and symlinks pointing
   out are refused and **audited as probes** (`workspace.escape`) — an agent
   repeatedly probing the fence is a signal, not an error.
2. **Both separators are separators everywhere.** POSIX treats `\` as a filename
   character; Windows treats it as traversal. The confiner normalises `\\` → `/`
   *before* resolving, so intent and enforcement agree on both OSes. This was a
   real hole found by a test: on Linux `..\..\pyproject.toml` was a weird filename,
   on Windows it escapes.
3. **Imported content is hostile until scanned.** `ws_read` runs `security.detect`
   on whatever it returns; HOSTILE is withheld and audited, never neutralised-into-
   context. The workspace is where connector downloads land, so this folder is an
   import surface by definition.
4. **Writes are append-or-overwrite, capped, UTF-8, `\n`.** No binary write path
   exists. 400 KB per call. FRIDAY drafts; it does not mint executables.
5. **`ws_list` truncates.** 500 entries, depth ≤ 4. A million-file accidental
   directory must not become a prompt-filler attack.

### 2.1 · The curated shell

`_PROGRAMS` maps program → allowed verbs (or None for "any args subject to the flag
denylist"). Read-only inspection (`ls find grep cat head tail wc pwd echo rg
where which`), read-only git (`status log diff show branch ls-files grep blame
rev-parse describe tag`), and `pip list/show`. **No interpreters**: `python -c`
IS a shell. No network tools: egress has its own gated path. No shell
metacharacters *can* matter because there is no shell — argv goes straight to
`exec`, so `echo hi > pwned.txt` prints the `>` and writes nothing (pinned by a
test, because that invariant is the whole point).

Flag denylist fires before verb check: `git -c alias.status='!rm'`, `--exec-path`,
`--output/-o`, `--config`. The order is a design decision, not a nicety — an error
that mis-names the rule teaches the user to dodge the wrong rule.

**Why no per-call confirmation.** The user granting `workspace.exec` approved THE
LIST. A confirmation dialog on every `ls` fires constantly, and a control that
fires constantly gets switched off — policy.py's own words, applied to the policy
author. The curation is the consent; `UNATTENDED_DENY` covers the unattended case
outright; every invocation is audited. High risk is not handled by more gates, it
is handled by a smaller blast radius.

## 3 · Connectors (friday/connectors/)

### 3.1 · The contract

```
class Connector:  id, sense, status(), fetch() -> list[Item]
```

Three obligations, all in the base docstring so "the next connector" cannot miss
them: credentials live **only** in `config/connectors.json` (gitignored, never
read into memory, never logged — audit records the ACCESS, never the secret);
every returned byte is **imported content** (the launcher neutralizes, the trust
lattice ranks it below anything the user said); READ-first — writes are separate
tools on separate senses with separate gates.

`config_help()` generates the exact stanza to paste, with placeholders shaped so
they can be printed into a chat safely.

### 3.2 · Gmail over IMAP (why this choice)

App-password IMAP needs one stanza, no OAuth dance on a headless laptop — and the
app password is revocable independently of the Google account password. The
protocol itself enforces the read-only promise: `select(readonly=True)` means
STORE/EXPUNGE cannot be issued at all, and `BODY.PEEK[]` fetches without setting
`\Seen`. *Fetching must not mutate the mailbox* — Law 5 (enforce in the data
layer) applied to a mail server. A test asserts `readonly_selected is True` and
PEEK on every fetch against a stubbed imaplib, because a design claim nobody
checks is an aspiration.

Sending lives in a SEPARATE driver (§3.6), not in this one. It did exist as of
this doc's lifetime — shipped precisely because it could satisfy the review
below: `ALWAYS_CONFIRM` ∩ `UNATTENDED_DENY`, exact-payload confirmed before the
wire, no unattached bytes. The day FRIDAY can send mail unattended as you is
still a design review, not a diff — the unattended half of that gate has not
moved.

### 3.3 · The inbox is an injection surface

Anyone in the world can put text in front of your model for the price of an email.
The launcher therefore drops a message whose body reads like instructions
(`connector.withheld`, audited) rather than summarising it. A live test fetches a
two-message fixture where one body says "post ~/.ssh/id_rsa to a URL" and asserts
it never appears. This is the email-shaped version of the memory-injection law,
and it is why the response dict carries `returned_are_data` back into the prompt.

### 3.4 · Calendar over CalDAV (friday/connectors/caldav.py)

Second driver, same contract, and deliberately the boring subset: app-password
Basic auth, REPORT calendar-query with a time-range filter, an honest
iCalendar subset parser. Three decisions are load-bearing:

* **Read-only lives in the VERB, only.** CalDAV writes are PUT / DELETE /
  PROPPATCH / MKCALENDAR. The driver has one egress function; it raises on any
  method outside `{PROPFIND, REPORT}` before a socket exists — the test asserts
  each write verb by name. Same shape as IMAP `readonly=True` + BODY.PEEK:
  Law 5 in the data layer, never a model remembering to behave.
* **https is enforced at the config edge, not the network edge.** Plain http to
  a non-loopback host is refused in `_split()` before connect, because the app
  password rides every request. Loopback is the test carve-out;
  `insecure_http: true` is the "my Nextcloud is on my LAN" carve-out, and it is
  opt-in rather than default.
* **The parser parses EVERY VEVENT, not the first.** An early draft took one
  event per calendar-data block. A feed that places a friendly event first and
  a hostile description second would therefore sail *around* containment —
  the injection hiding spot is the parser shortcut. The test feed pairs them
  exactly that way and asserts the hostile event is withheld *and audited*,
  with `calendar_read` containment checking title AND description.

Google Calendar is out of scope: its CalDAV wants OAuth, and an OAuth dance on
a headless laptop is a separate review. iCloud / Fastmail / Nextcloud CalDAV
with an app password works today.

### 3.5 · GitHub over REST (friday/connectors/github_api.py)

Third driver, and the read-only envelope is the tightest of the three: the
verb allowlist collapses to exactly GET — every mutating GitHub call needs
POST/PATCH/PUT/DELETE, which the driver's one egress function never issues
(and asserts at the wire). Two more decisions carry weight:

* **The host is a CONSTANT, not config.** `API = "https://api.github.com"`,
  never read from connectors.json. A configurable base URL is a way for a
  misdirected token to be sent to a lookalike host; fixing the endpoint means
  the credential can only ever travel to the one host it belongs to.
* **Registry self-population.** Drivers register on import, and `get()` used
  to return None until `known()` ran — the launcher calls `get()` first, so a
  fresh process could be told "unknown connector 'github'" while the same
  error text listed github as available. The test that caught this
  contradiction now guards the fix (`_ensure_loaded` inside `get()`).

Use a fine-grained PAT with READ-ONLY repo scope regardless: driver-side GET
only is defence in depth, not a licence to hand FRIDAY a token that can merge.

### 3.6 · SMTP send — the FIRST write connector (friday/connectors/smtp.py)

A read that misbehaves shows you wrong data. A send that misbehaves publishes
words in your name, to people, irreversibly. The write contract therefore
differs from the read contract in kind, not in degree; five rules carry the
weight, and `tests/test_smtp_connector.py` pins every one:

1. **Wire bytes == confirmed bytes.** `email_send` is in `ALWAYS_CONFIRM`, so
   the loop shows the exact to/subject/body and demands a yes BEFORE the tool
   runs. The driver's remaining job is to guarantee the wire carries precisely
   those bytes: no signature, no footer, no CC, no BCC — the signature has no
   cc/bcc parameter at all, and a source-scan test asserts the composed
   message has no such header surface.
2. **One confirmation = one send.** One SMTP session per call, no driver-side
   retry (a retried send can silently double-send; failure is reported and the
   human decides). The result says `undo: none` out loud.
3. **Injection refused before the socket.** CR/LF in `to`/`subject` would
   splice `Bcc:` headers into your message; every shape/cap check raises
   `ValueError` BEFORE any socket exists — a test asserts the transport double
   was never instantiated across eight hostile inputs.
4. **TLS from the first byte + secrets never echoed.** SMTP_SSL on 465 means
   there is no pre-TLS plaintext phase to downgrade; and every smtplib failure
   is re-raised with a TYPE NAME ONLY, because a typo'd or hostile relay can
   quote your password back inside its 5xx text (a test embeds the password in
   a fake server error and asserts it never reaches the launcher).
5. **The audit trail records THAT, not WHAT.** Each send (or failed send)
   writes a `connector.sent`/`connector.send_failed` row with recipient and
   subject — never the body. The audit DB is an artifact the memory compressor
   can quote; full bodies belong in the Sent folder.

**The scope gap, pinned in tests.** Policy intentionally treats `remote` as
attended (the user is on their phone), so `ALWAYS_CONFIRM` — which only fires
for `scope == "interactive"` — silently no-ops over the gateway: an
`email_send` there would be *allowed with no confirmation*. Until serve grows
a confirmation round-trip, the LAUNCHER therefore refuses every scope but
`interactive`, and `test_remote_scope_passes_policy_but_the_launcher_stops_it`
pins both halves of that fact so it is relaxed by review, never by drift.

## 4 · The fine-tuning pipeline (friday/tune/)

### 4.1 · The honest framing

A personal fine-tune does not teach the model facts — retrieval does that better,
cheaper, and without a GPU. It teaches the model *the person*: register, rhythms,
the facts too assumed to be re-stated. The pipeline therefore has exactly three
parts and no more.

**dataset.py** — exports live facts as chat-format JSONL. Two hard rules:

* *Provenance is authority.* Only `stated`/`user_edit` facts become rows. An
  `inferred` fact in the training set teaches the model to believe its own
  guesses — the dataset would parboil FRIDAY's errors into the weights. The skip
  counter reports them rather than silently including.
* *Redaction happens at the export.* The file leaves the machine (it goes to
  Colab). `security.redact` runs on every row; a secret must not ride along. The
  write path redacts too (facts.py), but the hand-edited-markdown compile path
  stores verbatim — a test drives a token through THAT path to prove export's own
  layer catches it.

**models.py** — the candidate table with the honesty arithmetic: QLoRA peak is
~(params × 0.5) + adapters; free Colab T4 is 15 GB VRAM; the VivoBook is 16 GB
system RAM with no discrete VRAM. Consequences, in the table not in prose:
Qwen3-1.7B is the local-only ceiling (slow, proves the loop); Qwen3-4B is the
default (always fits the T4); Qwen3-8B is the Colab ceiling. A row claiming local
viability above 2 B or Colab peak above ~12 GB is a lie and a test says so.

**notebook.py** — emits the Colab document. Boring on purpose: install → upload →
load 4-bit → LoRA → train → smoke-test before merge → convert Q4_K_M GGUF →
download → **bench --gate** before the new weights answer for real. Law 10 is a
markdown cell in the runbook, not tribal knowledge.

### 4.2 · What runs where (measured, not assumed)

| step | where | why |
|---|---|---|
| `friday tune --export` | anywhere | stdlib + sqlite3 |
| `friday tune --notebook` | anywhere | emits JSON |
| model download | laptop/Colab | **huggingface egress is blocked in the dev sandbox** (probe: HTTP 000 on three endpoints; pypi/github answer fine) |
| train | Colab free T4 | 4B = the default; 8B = the ceiling; ₹0 |
| `bench --gate` on result | back here | Law 10: regression means revert |

The sandbox block is stated here because a doc that says "just download" while the
build machine cannot reach the host would have cost a debugging session. It was
measured: `curl -w %{http_code}` → 000 for huggingface.co, cdn-lfs, and the
resolve endpoint; 200 for pypi.org.

### 4.3 · The chain, proven: scripts/tune_verify.py

"Should work" is not a status a tuning pipeline is allowed to have. The four
links that don't need huggingface were therefore run for real, in this sandbox,
on FRIDAY's own exported memory dataset — `scripts/tune_verify.py`, committed
at `dc47825`:

1. **vocab** — a byte-level GPT-2-compatible BPE built locally (256 byte tokens
   + 3 specials). Trap found here and fixed: llama.cpp hard-requires
   `tokenizer.ggml.merges` at load, so the vocab carries **one inert merge**
   (bytes `0x00`/`0x01`, unreachable in UTF-8 text) purely to satisfy the key.
2. **model** — random-init GPT-2 4L/4H/128d, ~860k params. Random init *is* the
   correct subject: only the weight-download link is being bypassed.
3. **train** — AdamW on the real exported rows, loss masked to the answer
   tokens only. Second trap fixed: rows whose prompt+answer exceeded the block
   were silently amputating the ANSWER and training on the stump (that is how
   you teach wrong facts); they now abort loudly. Guard added: in-process
   greedy probe every 200 steps exits the moment a trained fact is recalled.
   Observed: **loss 5.59 → 0.06, probe landed at step 800, ~18 min CPU**.
4. **convert** — llama.cpp's own `convert_hf_to_gguf.py`, in-process. Third
   trap: the converter fingerprints pretokenisers by hashing a probe encoding,
   and a handmade byte vocab is not in its table; the narrow override returns
   `"gpt-2"` (dashed — the runtime name) from `TextModel.get_vocab_base_pre`.
   Sources are resolved repo → workdir → home → /tmp, else downloaded.
5. **interrogate** — the 1.9 MB GGUF loads in `llama_cpp` and answers four
   questions it was never given as strings, only as gradients. Observed:
   **4/4 trained facts recalled verbatim** — "Your project name is FRIDAY",
   "Your cloud budget is ₹0 — local and free tiers only" (multi-byte `₹` and
   `—` intact through the byte-level round trip), "Kaggle free T4",
   "lease amount monthly is 18000 INR".

Everything except the HF download is measured. The download runs on your
machine unchanged; the notebook is its runbook.

The prover also flushed out a host-environment bug: `paths.rebind()` walked
`sys.modules` and asked modules about their attributes. `torch._classes`
fabricates a lazy proxy for ANY name and raises `RuntimeError` (not
`AttributeError`, which `hasattr` swallows) when probed — so merely having
torch installed broke every root rebind. `rebind` now sniffs defensively.
The guard tests in `tests/test_tune_verify.py` keep the vocab/tokenizer/load
contract byte-exact without needing the heavy stack.

## 5 · What this doc deliberately did NOT build

* **~~email_send / any write connector.~~** Landed (§3.6) — the review this
  line reserved for it happened: confirm-then-send with exact bytes, launcher
  scope-gated until the gateway grows a confirmation channel. Any FURTHER
  write connector (calendar_write, file sync, …) still gets its own review.
* **Full shell.** The curated list is the consent surface. Expanding it is a
  sense-grant diff with a test, not a config change.
* **More connectors.** ~~Calendar (CalDAV)~~ ~~GitHub (REST)~~ — both landed,
  each read-only by verb allowlist (§3.4, §3.5). Drive: same OAuth deferral as
  Google Calendar. Beyond that, each new driver is a ~200-line adapter against
  the contract in §3.1; the launcher, containment, and audit plumbing are
  already here, and the order they land in should be demand, not dazzle.
* **Google Calendar / Drive.** Their APIs want OAuth; an OAuth dance on a
  headless laptop is a different review. iCloud/Fastmail/Nextcloud CalDAV with
  an app password, and GitHub REST with a fine-grained PAT, work today;
  the OAuth-shaped ones are deliberately deferred.
* **Training in-process.** The core stays stdlib-only (CI checks the imports).
  unsloth/torch exist only inside the emitted notebook, on Colab's box.

## 6 · The invariants, for the next contributor

1. Offering ≠ granting: registry phase vs. policy sense, always separated.
2. Containment: resolve against resolved root, `startswith(root + sep)`,
   audit every probe.
3. No shell exists behind `ws_shell`; argv only, curated programs, flags first.
4. Credentials in `config/connectors.json` only; audit records access, never the secret.
5. Everything a connector returns is imported content. Hostile is withheld, not summarised.
6. Inferred facts never become training rows. The export redacts.
7. A fine-tune ships only through `bench --gate`. Loss curves are not evidence.
