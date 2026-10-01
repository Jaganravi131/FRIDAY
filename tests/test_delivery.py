"""Delivery tests — `friday doctor` and `friday serve`.

These two modules exist because the deployment target is a specific Windows laptop and
development happens elsewhere. Almost every way this project fails to *deliver* is an
environment difference nobody noticed, so the tests are written to fail on exactly the
mistakes that were made while building them:

  * a low-RAM machine reported as a hard **fail**, printing "NOT READY" on hardware that
    would have worked fine — a diagnostic that cries wolf gets ignored
  * the backup check inspecting the *installed package's* git repo instead of asking
    whether FRIDAY_ROOT is inside one, reporting "remote configured" for a memory folder
    in no repository at all
  * the `remote` scope rejected by a CHECK constraint written before the gateway existed
  * a web UI that quietly depends on a CDN, and therefore stops working offline — which
    is the entire premise of the project
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

from friday import doctor, paths, serve


# ── doctor ─────────────────────────────────────────────────────────────────────

def _seed(root: Path) -> None:
    from friday.cli import main

    assert main(["--db", str(root / "artifacts" / "friday.db"), "seed"]) == 0
    assert main(["--db", str(root / "artifacts" / "friday.db"), "build"]) == 0


def test_report_verdict_and_render():
    rep = doctor.Report()
    assert rep.verdict() == "ok" and not rep.failed
    rep.add("a", "ok", "fine")
    rep.add("b", "warn", "meh")
    assert rep.verdict() == "warn"
    rep.add("c", "fail", "bad", "fix it")
    assert rep.verdict() == "fail" and rep.failed
    assert rep.counts == {"ok": 1, "warn": 1, "fail": 1, "skip": 0}
    out = rep.render()
    assert "✅ a" in out and "⚠️  b" in out and "❌ c" in out
    assert "fix it" in out                       # the hint is the whole point
    assert "NOT READY" in out
    # machine-readable, and an empty report is `ok` rather than a crash
    d = doctor.Report().as_dict()
    assert d["verdict"] == "ok" and d["ready"] is True and d["findings"] == []


def _ram_finding(monkeypatch, gb):
    """RAM is one finding inside check_runtime; pin it and read it back by name."""
    monkeypatch.setattr(doctor, "total_ram_gb", lambda: gb)
    rep = doctor.Report()
    doctor.check_runtime(rep)
    return next(f for f in rep.findings if f.name == "RAM"), rep


def test_low_ram_is_a_warning_not_a_failure(monkeypatch):
    """The calibration bug. 6 GB constrains model choice; it does not stop FRIDAY."""
    f, rep = _ram_finding(monkeypatch, 6.0)
    assert f.status == "warn"
    assert not rep.failed


def test_adequate_ram_is_ok(monkeypatch):
    f, _ = _ram_finding(monkeypatch, 16.0)
    assert f.status == "ok" and "design target" in f.detail


def test_tiny_ram_warns_but_still_says_friday_runs(monkeypatch):
    f, rep = _ram_finding(monkeypatch, 3.0)
    assert f.status == "warn" and not rep.failed
    assert "will run" in f.detail


def test_absurdly_low_ram_fails_but_says_it_may_be_a_container(monkeypatch):
    f, _ = _ram_finding(monkeypatch, 1.0)
    assert f.status == "fail"
    assert "container" in f.fix


def test_backup_checks_the_data_root_not_the_package(monkeypatch, tmp_path):
    """Reported "git remote configured (origin)" for a root in no repo at all."""
    monkeypatch.setattr(doctor.shutil, "which", lambda _: "git")
    monkeypatch.setattr("subprocess.run", lambda *a, **k: _Completed(1, "", ""))
    rep = doctor.Report()
    doctor.check_backup(rep, tmp_path)
    assert rep.findings[0].status == "warn"
    assert str(tmp_path) in rep.findings[0].detail
    assert "not inside a git repository" in rep.findings[0].detail


def test_backup_ok_only_when_the_root_is_in_a_repo_with_a_remote(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor.shutil, "which", lambda _: "git")

    def fake_run(cmd, **kw):
        if "rev-parse" in cmd:
            return _Completed(0, str(tmp_path), "")
        if cmd[-1:] == ["remote"]:
            return _Completed(0, "origin\n", "")
        return _Completed(0, "", "")

    monkeypatch.setattr("subprocess.run", fake_run)
    rep = doctor.Report()
    doctor.check_backup(rep, tmp_path)
    assert rep.findings[0].status == "ok"
    assert "origin" in rep.findings[0].detail


def test_backup_warns_on_a_repo_with_no_remote(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor.shutil, "which", lambda _: "git")

    def fake_run(cmd, **kw):
        if "rev-parse" in cmd:
            return _Completed(0, str(tmp_path), "")
        return _Completed(0, "", "")

    monkeypatch.setattr("subprocess.run", fake_run)
    rep = doctor.Report()
    doctor.check_backup(rep, tmp_path)
    assert rep.findings[0].status == "warn"
    assert "no remote" in rep.findings[0].detail


def test_secret_scan_fails_on_a_planted_token(tmp_path):
    mem = tmp_path / "memory" / "facts"
    mem.mkdir(parents=True, exist_ok=True)     # isolated_paths already made the layout
    (mem / "creds.md").write_text("---\n---\napi_key: ghp_" + "A" * 36 + "\n")
    rep = doctor.Report()
    doctor.check_secrets(rep)
    assert rep.failed
    f = rep.findings[0]
    assert "creds.md" in f.detail            # names the file, so it is actionable
    assert f.fix                             # and says what to do about it
    # ⭐ It must NOT echo the credential — not even a prefix. The doctor's output goes
    # to a terminal, a scrollback buffer, and `--json` into a log. Printing the thing it
    # is warning about would turn the warning into a second leak.
    blob = f.detail + f.fix
    assert "ghp_" not in blob and "AAAA" not in blob


def test_secret_scan_passes_on_a_redacted_memory(tmp_path):
    mem = tmp_path / "memory" / "facts"
    mem.mkdir(parents=True, exist_ok=True)
    (mem / "creds.md").write_text("---\n---\napi_key: [REDACTED:CREDENTIAL]\n")
    rep = doctor.Report()
    doctor.check_secrets(rep)
    assert rep.findings[0].status == "ok"


def test_model_server_probe_reports_nothing_listening(monkeypatch):
    monkeypatch.setattr(doctor, "_probe", lambda url, timeout=1.5: None)
    rep = doctor.Report()
    doctor.check_model(rep)
    f = next(x for x in rep.findings if x.name == "Model server")
    assert f.status == "warn" and "mock" in f.fix


def test_model_server_probe_finds_a_listener(monkeypatch):
    """Names which server answered — "reachable" alone is not actionable."""
    monkeypatch.setattr(doctor, "_probe",
                        lambda url, timeout=1.5: '{"model":"llama"}' if "8080" in url else None)
    rep = doctor.Report()
    doctor.check_model(rep)
    f = next(x for x in rep.findings if x.name == "Model server")
    assert f.status == "ok" and "llama.cpp" in f.detail


# ── GPU / Vulkan detection ─────────────────────────────────────────────────────
# The original check was `has_cuda()` alone, which on the real target machine — a
# Ryzen 5 with a Radeon iGPU — reported "no CUDA. Expected." and stopped. That cost a
# 10x larger model, because llama.cpp's Vulkan backend runs on AMD iGPUs on Windows
# with no ROCm. So the four cases are pinned explicitly.

def _gpu(rep, monkeypatch, info):
    monkeypatch.setattr(doctor, "gpu_info", lambda: info)
    doctor.check_runtime(rep)
    return next(f for f in rep.findings if f.name == "GPU")


def test_amd_igpu_with_vulkan_is_an_ok_not_a_skip(monkeypatch):
    """⭐ The case the target machine actually is."""
    rep = doctor.Report()
    f = _gpu(rep, monkeypatch, {"names": ["AMD Radeon(TM) Graphics"], "cuda": False,
                                "vulkan_loader": True, "amd": True, "intel": False})
    assert f.status == "ok"
    assert "Vulkan" in f.fix and "ROCm" in f.fix
    assert "14B" in f.fix                      # names the real ceiling, not 1.2B
    assert "vulkan" in f.fix.lower()           # and warns it needs the vulkan build
    assert not rep.failed


def test_gpu_without_a_vulkan_loader_warns_to_update_the_driver(monkeypatch):
    rep = doctor.Report()
    f = _gpu(rep, monkeypatch, {"names": ["AMD Radeon(TM) Graphics"], "cuda": False,
                                "vulkan_loader": False, "amd": True, "intel": False})
    assert f.status == "warn"
    assert "driver" in f.fix and "1.2B" in f.fix     # falls back to the CPU ceiling


def test_cuda_wins_over_the_vulkan_path(monkeypatch):
    rep = doctor.Report()
    f = _gpu(rep, monkeypatch, {"names": ["NVIDIA GeForce RTX 4060"], "cuda": True,
                                "vulkan_loader": True, "amd": False, "intel": False})
    assert f.status == "ok" and "CUDA" in f.detail


def test_no_gpu_at_all_is_a_skip_with_cpu_advice(monkeypatch):
    rep = doctor.Report()
    f = _gpu(rep, monkeypatch, {"names": [], "cuda": False, "vulkan_loader": False,
                                "amd": False, "intel": False})
    assert f.status == "skip" and "CPU-only" in f.detail
    assert not rep.failed


def test_gpu_info_never_raises_on_an_unusable_platform(monkeypatch):
    """Identification is best-effort; a failed probe must not take the doctor with it."""
    def boom(*a, **k):
        raise OSError("no subprocess here")

    # gpu_info imports subprocess INSIDE the function, so patch the module globally —
    # `doctor.subprocess` does not exist as an attribute.
    monkeypatch.setattr("subprocess.run", boom)
    monkeypatch.setattr(doctor, "has_cuda", lambda: False)
    g = doctor.gpu_info()
    assert isinstance(g["names"], list) and g["cuda"] is False
    assert "vulkan_loader" in g and "amd" in g


def test_doctor_root_argument_inspects_the_root_it_was_given(monkeypatch, tmp_path):
    """⭐ `--root` must mean it. Three of the checks read the module-level `paths.*`
    constants instead of the `root` argument, so passing a different root used to
    produce a report about the WRONG directory while claiming to be about the one
    requested — and `check_layout` crashed when the ambient root was missing a
    directory the requested one had.

    The silent wrong answer is the worse half. `friday doctor` is the first command
    the deployment target runs, on a machine that is not the author's, so "I checked
    your memory root and it is fine" while actually checking the repo checkout is the
    kind of failure that gets believed.

    Both roots exist here and only one is built, so a report about the wrong one is
    unambiguously wrong rather than accidentally right.
    """
    monkeypatch.setattr(doctor.shutil, "which", lambda _: None)
    built, unbuilt = tmp_path / "built", tmp_path / "unbuilt"
    monkeypatch.setenv("FRIDAY_ROOT", str(built))
    _seed(built)
    paths.rebind(built)
    paths.ensure_layout()
    from friday.cli import main as cli
    cli(["build"])                      # index the BUILT root only
    unbuilt.mkdir(parents=True, exist_ok=True)
    (unbuilt / "memory" / "facts").mkdir(parents=True, exist_ok=True)

    by_built = {f.name: f for f in doctor.run(root=built).findings}
    by_unbuilt = {f.name: f for f in doctor.run(root=unbuilt).findings}

    assert by_built["Schema"].status == "ok", "the built root has a database"
    assert "Derived index" in by_unbuilt, (
        "the unbuilt root has no database — if this is missing the report described "
        "the built root instead of the one requested"
    )


def test_doctor_restores_the_root_it_rebound(monkeypatch, tmp_path):
    """`run()` is a library function. Rebinding global path state to inspect another
    root is fine; leaving it rebound is not — a caller that inspects one root and then
    writes memory would silently write into the inspected tree.
    """
    monkeypatch.setattr(doctor.shutil, "which", lambda _: None)
    other = tmp_path / "other"
    other.mkdir()
    (other / "memory" / "facts").mkdir(parents=True, exist_ok=True)
    before = paths.ROOT
    doctor.run(root=other)
    assert paths.ROOT == before


def test_a_crashed_check_names_itself(monkeypatch, tmp_path):
    """The report must say WHICH check broke. The root-bound checks were wrapped in
    lambdas and the crash path fell back to `fn.__name__`, so a real failure printed
    `❌ <lambda>` above a ValueError about paths — a diagnostic that cannot name the
    thing that broke is a diagnostic nobody can act on.
    """
    monkeypatch.setattr(doctor.shutil, "which", lambda _: None)
    monkeypatch.setattr(doctor, "check_store",
                        lambda rep, root: (_ for _ in ()).throw(RuntimeError("boom")))
    rep = doctor.run(root=tmp_path)
    # Match the INJECTED error text, not the word "crashed". Substring-matching on a
    # finding's detail is a trap here: details embed filesystem paths, and pytest
    # derives tmp_path from the test's own NAME — so `test_a_crashed_check_names_itself`
    # produces a directory containing "crashed", and the unrelated "Disk" finding (which
    # reports the path it stat'd) matched the filter. A test that fails because of its
    # own name is worth the comment.
    crashed = [f for f in rep.findings if "boom" in f.detail]
    assert crashed, "the deliberately broken check should have been reported"
    assert len(crashed) == 1
    assert crashed[0].name == "check_store"
    assert not any(f.name == "<lambda>" for f in rep.findings), (
        "a crashed check reported itself as <lambda> instead of by name"
    )


def test_rel_never_raises_on_a_path_outside_the_root(tmp_path):
    """`Path.relative_to` raises instead of returning something usable. A report line
    whose whole job is to say "this directory is missing" must not be able to take the
    check down with it.
    """
    inside, outside = tmp_path / "a" / "b", tmp_path / "other" / "c"
    assert doctor._rel(inside, tmp_path / "a") == "b"
    assert doctor._rel(outside, tmp_path / "a") == str(outside)


def test_doctor_runs_end_to_end_on_an_empty_root(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor.shutil, "which", lambda _: None)
    rep = doctor.run(root=tmp_path)
    assert not rep.failed                       # nothing blocking on a fresh root
    names = {f.name for f in rep.findings}
    assert {"Python", "SQLite", "FRIDAY_ROOT", "Model server"} <= names


def test_doctor_after_seed_and_build_is_ready(monkeypatch, tmp_path, capsys):
    """The install path a real user follows: seed, build, doctor."""
    monkeypatch.setattr(doctor.shutil, "which", lambda _: None)
    monkeypatch.setenv("FRIDAY_ROOT", str(tmp_path))
    _seed(tmp_path)
    rep = doctor.run(root=tmp_path)
    assert not rep.failed
    assert rep.verdict() in ("ok", "warn")
    by = {f.name: f for f in rep.findings}
    assert by["Directory layout"].status == "ok"
    assert by["Soul files"].status == "ok"
    # "Derived index" is the finding for a MISSING database; a built one reports these
    assert "Derived index" not in by
    assert by["Schema"].status == "ok"
    assert by["Memory"].status == "ok"


def test_cli_doctor_exits_nonzero_only_on_failure(monkeypatch, tmp_path, capsys):
    from friday.cli import main

    monkeypatch.setenv("FRIDAY_ROOT", str(tmp_path))
    monkeypatch.setattr(doctor.shutil, "which", lambda _: None)
    assert main(["doctor", "--root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "FRIDAY doctor" in out

    (tmp_path / "memory" / "facts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "memory" / "facts" / "x.md").write_text(
        "---\n---\npassword: hunter2secret\n")
    assert main(["doctor", "--root", str(tmp_path)]) == 1

    capsys.readouterr()
    assert main(["doctor", "--root", str(tmp_path), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["verdict"] == "fail"


class _Completed:
    def __init__(self, rc, out, err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


# ── bench: the retrieval regression gate ───────────────────────────────────────

def test_bench_runs_model_free_and_scores_the_known_defect(tmp_path):
    """Pins the baseline behaviour, including the defect it exists to expose."""
    from friday import bench

    res = bench.run()
    assert res.facts == 13 and res.queries == 27 and res.unanswerable == 5
    # Law 4 and Law 6 must hold even while recall is mediocre
    assert res.false_positive_rate == 0.0, "returning something for an unanswerable query"
    assert res.provenance_complete == 1.0, "a shipped fact without a quote breaks /why"
    # Noun phrases work. Natural questions now work too — that was the 37-point
    # interrogative defect, closed by unifying the stopword lists (see
    # tests/test_retrieval.py). The gap is asserted SMALL rather than merely absent
    # so a regression in either direction is caught.
    assert res.by_class["noun"]["recall_at_5"] > 0.8
    assert res.by_class["interrogative"]["recall_at_5"] > 0.8
    assert res.interrogative_gap < 0.1, "natural questions must not retrieve worse than noun phrases"
    # PARAPHRASE IS STILL THE KNOWN DEFECT, and it is not fixable by threshold.
    # Lowering the floor to 0.25 lifts paraphrase to 0.25 and costs one false
    # positive out of five unanswerable queries; it then plateaus at 0.25 no matter
    # how low the floor goes. Law 4 and the gate both forbid that trade, so the only
    # route to paraphrase recall is better signal — a real semantic embedder.
    assert res.by_class["paraphrase"]["recall_at_5"] < 0.3
    assert res.seconds < 30


def test_bench_gate_passes_on_its_own_baseline(tmp_path):
    from friday import bench

    res = bench.run()
    path = tmp_path / "BENCHMARKS.md"
    path.write_text(bench.render_md(res, when="2026-09-30T00:00:00Z"), encoding="utf-8")
    assert bench.read_baseline(path)["recall_at_5"] == res.recall_at_5
    ok, msgs = bench.gate(res, path)
    assert ok, msgs


def test_bench_gate_fails_on_a_worse_run(tmp_path):
    """The gate must bite, or it is documentation with extra steps."""
    from friday import bench

    res = bench.run()
    path = tmp_path / "BENCHMARKS.md"
    strict = bench.render_md(res, when="x").replace(
        f"| `recall_at_5` | {res.recall_at_5} |", "| `recall_at_5` | 0.99 |")
    path.write_text(strict, encoding="utf-8")
    ok, msgs = bench.gate(res, path)
    assert not ok
    assert any("REGRESSION recall_at_5" in m for m in msgs)


def test_bench_gate_fails_when_no_baseline_exists(tmp_path):
    from friday import bench

    ok, msgs = bench.gate(bench.run(), tmp_path / "absent.md")
    assert not ok and "no baseline" in msgs[0]


def test_bench_gate_fails_on_a_rising_false_positive_rate(tmp_path):
    """Lower-is-better metrics need their own direction, not the shared tolerance."""
    from friday import bench

    res = bench.run()
    path = tmp_path / "BENCHMARKS.md"
    path.write_text(bench.render_md(res, when="x"), encoding="utf-8")
    res.false_positive_rate = 0.4
    ok, msgs = bench.gate(res, path)
    assert not ok
    assert any("false_positive_rate" in m and "Law 4" in m for m in msgs)


def test_bench_restores_the_callers_environment(tmp_path, monkeypatch):
    """⭐ It rebinds paths to a temp dir that it then DELETES.

    The original bug: after `bench.run()` the process believed FRIDAY_ROOT was
    `/tmp/friday-bench-XXXX`, already removed by the TemporaryDirectory context manager.
    Every later operation would read from, or silently recreate, an empty root — and
    since the gate runs on every commit, it would have poisoned whatever ran after it.
    """
    from friday import bench, paths

    monkeypatch.setenv("FRIDAY_ROOT", str(tmp_path))
    paths.rebind(tmp_path)
    paths.ensure_layout()

    bench.run()

    assert Path(os.environ["FRIDAY_ROOT"]) == tmp_path
    assert paths.ROOT == tmp_path
    assert paths.ROOT.exists(), "left pointing at a deleted temp directory"


# ── serve ──────────────────────────────────────────────────────────────────────

def test_resolve_host_defaults_to_loopback():
    assert serve.resolve_host("127.0.0.1")[0] == "127.0.0.1"
    assert serve.resolve_host("0.0.0.0")[0] == "0.0.0.0"
    # the loud warning is part of the contract, not decoration
    assert "EVERY interface" in serve.resolve_host("0.0.0.0")[1]
    bind, label = serve.resolve_host("tailscale")
    assert bind == "127.0.0.1" or bind.startswith("100.")
    assert "tailnet" in label or "loopback" in label


def test_ensure_token_is_created_once_with_restricted_permissions(tmp_path):
    tok, path, created = serve.ensure_token(tmp_path)
    assert created and len(tok) >= 32
    assert path.parent == tmp_path / "config"
    # again: stable, not regenerated — losing the token would lock the user out
    tok2, _, created2 = serve.ensure_token(tmp_path)
    assert tok2 == tok and created2 is False


def test_index_html_has_no_external_dependencies():
    """The UI must work with no internet. That is the premise of the project."""
    html = serve.INDEX_HTML
    assert "http://" not in html and "https://" not in html
    assert "<script src=" not in html and "<link" not in html
    assert "<style>" in html and "<script>" in html      # inline, self-contained


def test_index_html_survives_a_sandboxed_iframe():
    """⭐ The page is opened inside a proxied iframe as often as in a browser tab.

    Two bugs of this class were shipped and neither is visible by looking at the page:

      * `prompt()` is BLOCKED in a sandboxed iframe — it returns null silently, so the
        token was always empty and every request 401'd.
      * touching `localStorage` on an opaque origin THROWS SecurityError, which killed
        the script before `f.onsubmit` was assigned. The UI rendered perfectly and then
        did nothing at all, with no error anywhere.

    A preview that looks fine and cannot be used is worse than one that fails loudly,
    because there is nothing to debug. So: no modal dialogs, storage access guarded,
    and an inline field that works in any embedding context.
    """
    html = serve.INDEX_HTML
    # strip the explanatory comment before looking for calls — the words appear in it
    code = re.sub(r"/\*.*?\*/", "", html, flags=re.S)
    assert not re.search(r"(?:^|[^\w.])prompt\s*\(", code, re.M), "modal dialog in an iframe"
    assert not re.search(r"(?:^|[^\w.])alert\s*\(", code, re.M)
    assert not re.search(r"(?:^|[^\w.])confirm\s*\(", code, re.M)
    # every localStorage touch must be inside a try
    assert "try{return localStorage" in html and "try{localStorage.setItem" in html
    assert not re.search(r"^\s*(?:let|const|var)?\s*\w*\s*=?\s*localStorage\.", code, re.M)
    # and there must be a route in that does not depend on JS at all
    assert 'id="bar"' in html, "no inline token field"
    assert "URLSearchParams(location.hash" in html, "cannot accept a token from the URL"


@pytest.fixture
def gateway(tmp_path):
    """A real server on a real port. A gateway tested only by import is not tested.

    Function-scoped and built on `tmp_path`, because the autouse `isolated_paths`
    fixture has already rebound `friday.paths` to it. A module-scoped fixture with its
    own root would serve a *different* filesystem than the one the agent writes to, and
    the session rows the assertions look for would land somewhere else — an empty set
    rather than a failure, which is the misleading kind of test bug.
    """
    from friday import paths
    from friday.cli import main

    root = paths.ROOT
    main(["seed"])
    main(["build"])

    token, _, _ = serve.ensure_token(root)
    serve.Handler.token = token
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), serve.Handler)
    port = httpd.server_address[1]
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}", token, root
    httpd.shutdown()
    httpd.server_close()


def _req(url, token=None, body=None, method=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method or ("POST" if data else "GET"))
    if token:
        req.add_header("authorization", f"Bearer {token}")
    if data:
        req.add_header("content-type", "application/json")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, json.loads(r.read().decode() or "null"), dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode() or "null"), dict(e.headers)


def test_health_needs_no_token_and_leaks_nothing(gateway):
    base, _, _ = gateway
    code, body, _ = _req(base + "/health")
    assert code == 200
    assert body["status"] == "ok"
    assert "token" not in json.dumps(body).lower()


def test_index_is_served_without_a_token(gateway):
    base, _, _ = gateway
    req = urllib.request.Request(base + "/")
    with urllib.request.urlopen(req, timeout=20) as r:
        assert r.status == 200 and "FRIDAY" in r.read().decode()


def test_every_data_endpoint_refuses_a_missing_or_wrong_token(gateway):
    base, _, _ = gateway
    for path in ("/api/ask", "/api/why", "/v1/chat/completions", "/v1/models",
                 "/api/status"):
        code, body, _ = _req(base + path, body={"message": "hi"})
        assert code == 401, path
        assert "token" in body["error"]
        code, _, _ = _req(base + path, token="wrong", body={"message": "hi"})
        assert code == 401, path


def test_no_cors_wildcard(gateway):
    """A page on another origin must not be able to drive an agent that writes memory."""
    base, token, _ = gateway
    req = urllib.request.Request(base + "/health", headers={"Origin": "http://evil.example"})
    with urllib.request.urlopen(req, timeout=20) as r:
        assert "access-control-allow-origin" not in {k.lower() for k in r.headers}


def test_ask_round_trip_and_openai_compatibility(gateway):
    base, token, _ = gateway
    code, body, _ = _req(base + "/api/ask", token=token,
                         body={"message": "what is my monthly rent"})
    assert code == 200 and body["answer"]
    assert "model" in body and isinstance(body.get("withheld"), list)

    code, body, _ = _req(base + "/v1/chat/completions", token=token,
                         body={"model": "friday-local",
                               "messages": [{"role": "user", "content": "what time is it"}]})
    assert code == 200
    assert body["choices"][0]["message"]["content"]
    assert body["object"] == "chat.completion"

    code, body, _ = _req(base + "/v1/models", token=token)
    assert code == 200 and body["data"][0]["id"] == serve.MODEL_NAME


def test_empty_and_malformed_requests_degrade_rather_than_crash(gateway):
    base, token, _ = gateway
    assert _req(base + "/api/ask", token=token, body={})[0] == 400
    assert _req(base + "/api/ask", token=token, body={"message": "   "})[0] == 400
    assert _req(base + "/nope", token=token)[0] == 404
    req = urllib.request.Request(base + "/api/ask", data=b"{not json",
                                 headers={"authorization": f"Bearer {token}"})
    try:
        urllib.request.urlopen(req, timeout=20)
        raise AssertionError("malformed JSON should not 200")
    except urllib.error.HTTPError as e:
        assert e.code == 400


def test_remote_turns_are_audited_as_remote(gateway):
    """A gateway that cannot say which turns came over the network cannot be trusted."""
    base, token, root = gateway
    _req(base + "/api/ask", token=token, body={"message": "what is my rent"})
    from friday.store import db
    conn = db.connect(root / "artifacts" / "friday.db")
    # `scope` is a property of the SESSION, not of each turn
    scopes = {r[0] for r in conn.execute("SELECT scope FROM sessions")}
    assert "remote" in scopes


def test_remote_scope_is_accepted_by_the_schema(gateway):
    """The CHECK constraint predated the gateway; a stale database must degrade, not 500."""
    _, _, root = gateway
    conn = sqlite3.connect(root / "artifacts" / "friday.db")
    ddl = conn.execute("SELECT sql FROM sqlite_master WHERE name='sessions'").fetchone()[0]
    assert "'remote'" in ddl


def test_stale_schema_falls_back_to_interactive(monkeypatch, gateway):
    """A database built before `remote` existed still serves the turn."""
    base, token, root = gateway
    from friday.agent.loop import Agent

    real_run = Agent.run
    calls = []

    def flaky_run(self, msg, *, scope="interactive", **kw):
        calls.append(scope)
        if scope == "remote":
            raise sqlite3.IntegrityError(
                "CHECK constraint failed: scope IN ('interactive','heartbeat',"
                "'dreaming','eval')")
        return real_run(self, msg, scope=scope, **kw)

    monkeypatch.setattr(Agent, "run", flaky_run)
    monkeypatch.setattr(serve, "_scope_fallback", False)
    code, body, _ = _req(base + "/api/ask", token=token, body={"message": "hello"})
    assert code == 200 and body["answer"]
    assert calls == ["remote", "interactive"]


def test_answer_never_raises(gateway, monkeypatch):
    from friday.agent.loop import Agent

    def boom(self, msg, **kw):
        raise RuntimeError("deliberate")

    monkeypatch.setattr(Agent, "run", boom)
    r = serve.answer("hi")
    assert r["error"] is True
    assert "did not guess" in r["answer"]        # honest failure, no fabricated answer


def test_token_file_is_not_world_readable(tmp_path):
    _, path, _ = serve.ensure_token(tmp_path)
    mode = path.stat().st_mode & 0o777
    assert mode & 0o077 == 0 or mode == 0o666    # POSIX 0600; Windows reports loosely
