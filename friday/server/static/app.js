/* F.R.I.D.A.Y. HUD — vanilla JS, no build step, no dependencies.
   Talks to the local agent over REST + SSE. */
(() => {
  "use strict";

  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

  const state = {
    config: null,
    tools: [],
    sessions: [],
    sessionId: localStorage.getItem("friday_session") || null,
    running: false,
    runId: null,
    abort: null,
    approvals: new Map(),
    approvalMode: localStorage.getItem("friday_approval") || null,
    token: localStorage.getItem("friday_token") || "",
    speak: localStorage.getItem("friday_speak") === "1",
    recognition: null,
    listening: false,
    assistantEl: null,
    assistantBuf: "",
    renderQueued: false,
    toolsById: new Map(),
    callSeq: 0,
    lastAnswer: "",
  };

  const els = {
    boot: $("#boot"),
    bootStatus: $("#bootStatus"),
    stream: $("#stream"),
    transcript: $("#transcript"),
    input: $("#input"),
    composer: $("#composer"),
    sendBtn: $("#sendBtn"),
    stopBtn: $("#stopBtn"),
    modeBtn: $("#modeBtn"),
    modeLabel: $("#modeLabel"),
    sessionList: $("#sessionList"),
    bridgeStatus: $("#bridgeStatus"),
    latencyChip: $("#latencyChip"),
    count: $("#count"),
    toasts: $("#toasts"),
  };

  /* ------------------------------------------------------------------ utils */
  const escapeHtml = (value) =>
    String(value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  const uuid = () =>
    (crypto.randomUUID && crypto.randomUUID().replace(/-/g, "").slice(0, 12)) ||
    Math.random().toString(36).slice(2, 14);

  function toast(message, kind = "") {
    const el = document.createElement("div");
    el.className = "toast " + kind;
    el.textContent = message;
    els.toasts.appendChild(el);
    setTimeout(() => {
      el.style.opacity = "0";
      el.style.transition = "opacity .3s";
      setTimeout(() => el.remove(), 320);
    }, 3400);
  }

  function relativeTime(iso) {
    if (!iso) return "—";
    const date = new Date(iso);
    if (Number.isNaN(date.getTime())) return "—";
    const diff = (Date.now() - date.getTime()) / 1000;
    if (diff < 60) return "just now";
    if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
    if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
    if (diff < 86400 * 7) return `${Math.floor(diff / 86400)}d ago`;
    return date.toLocaleDateString();
  }

  function clockTime(iso) {
    const date = iso ? new Date(iso) : new Date();
    if (Number.isNaN(date.getTime())) return "";
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function fmtDuration(ms) {
    if (ms == null) return "—";
    if (ms < 1000) return `${Math.round(ms)}ms`;
    if (ms < 60000) return `${(ms / 1000).toFixed(1)}s`;
    return `${Math.floor(ms / 60000)}m ${Math.round((ms % 60000) / 1000)}s`;
  }

  function fmtUptime(seconds) {
    if (!seconds && seconds !== 0) return "—";
    const s = Math.floor(seconds);
    const d = Math.floor(s / 86400);
    const h = Math.floor((s % 86400) / 3600);
    const m = Math.floor((s % 3600) / 60);
    if (d) return `${d}d ${h}h`;
    if (h) return `${h}h ${m}m`;
    return `${m}m`;
  }

  /* -------------------------------------------------------------------- api */
  async function api(path, options = {}) {
    const headers = { "Content-Type": "application/json", ...(options.headers || {}) };
    if (state.token) headers["X-FRIDAY-Token"] = state.token;
    const response = await fetch(path, { ...options, headers });
    if (response.status === 401) {
      openDrawer();
      throw new Error("API token required — set it in settings.");
    }
    if (!response.ok) {
      let detail = response.statusText;
      try {
        const body = await response.json();
        detail = body.detail || detail;
      } catch (_) {
        /* non-JSON error */
      }
      throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    }
    if (response.status === 204) return null;
    return response.json();
  }

  /* --------------------------------------------------------------- markdown */
  function inlineFormat(text) {
    let out = text;
    const codes = [];
    out = out.replace(/`([^`\n]+)`/g, (_, code) => {
      codes.push(code);
      return `\u0001C${codes.length - 1}\u0001`;
    });
    out = out.replace(
      /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );
    // Auto-link bare URLs, but never inside a tag we just generated.
    out = out
      .split(/(<[^>]+>)/g)
      .map((segment) => {
        if (!segment || segment.startsWith("<")) return segment;
        return segment.replace(/https?:\/\/[^\s<)]+/g, (url) => {
          const clean = url.replace(/[.,;:]+$/, "");
          return `<a href="${clean}" target="_blank" rel="noopener noreferrer">${clean}</a>`;
        });
      })
      .join("");
    out = out.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    out = out.replace(/__([^_]+)__/g, "<strong>$1</strong>");
    out = out.replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>");
    out = out.replace(/~~([^~]+)~~/g, "<del>$1</del>");
    out = out.replace(/\u0001C(\d+)\u0001/g, (_, index) => `<code>${codes[Number(index)]}</code>`);
    return out;
  }

  function renderMarkdown(source) {
    if (!source) return "";
    const fences = [];
    let text = String(source).replace(/```([\w+-]*)\n?([\s\S]*?)```/g, (_, lang, code) => {
      fences.push({ lang: lang || "", code });
      return `\u0000FENCE${fences.length - 1}\u0000`;
    });
    text = escapeHtml(text);

    const lines = text.split("\n");
    const html = [];
    let list = null;
    let paragraph = [];
    let quote = [];
    let table = null;

    const flushParagraph = () => {
      if (paragraph.length) {
        html.push(`<p>${inlineFormat(paragraph.join("<br>"))}</p>`);
        paragraph = [];
      }
    };
    const flushList = () => {
      if (list) {
        html.push(`<${list.tag}>${list.items.map((item) => `<li>${inlineFormat(item)}</li>`).join("")}</${list.tag}>`);
        list = null;
      }
    };
    const flushQuote = () => {
      if (quote.length) {
        html.push(`<blockquote>${inlineFormat(quote.join("<br>"))}</blockquote>`);
        quote = [];
      }
    };

    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const fenceMatch = line.match(/^\u0000FENCE(\d+)\u0000$/);
      if (fenceMatch) {
        flushParagraph();
        flushList();
        flushQuote();
        const fence = fences[Number(fenceMatch[1])] || { lang: "", code: "" };
        html.push(
          `<pre>${fence.lang ? `<span class="lang">${escapeHtml(fence.lang)}</span>` : ""}<code>${escapeHtml(
            fence.code.replace(/\n$/, "")
          )}</code></pre>`
        );
        continue;
      }

      if (!line.trim()) {
        flushParagraph();
        flushList();
        flushQuote();
        table = null;
        continue;
      }

      const heading = line.match(/^(#{1,3})\s+(.*)$/);
      if (heading) {
        flushParagraph();
        flushList();
        flushQuote();
        const level = heading[1].length;
        html.push(`<h${level}>${inlineFormat(heading[2])}</h${level}>`);
        continue;
      }

      if (/^(-{3,}|\*{3,}|_{3,})$/.test(line.trim())) {
        flushParagraph();
        flushList();
        flushQuote();
        html.push("<hr>");
        continue;
      }

      const quoteMatch = line.match(/^&gt;\s?(.*)$/);
      if (quoteMatch) {
        flushParagraph();
        flushList();
        quote.push(quoteMatch[1]);
        continue;
      }

      const listMatch = line.match(/^\s*([-*+]|\d+\.)\s+(.*)$/);
      if (listMatch) {
        flushParagraph();
        flushQuote();
        const ordered = /\d+\./.test(listMatch[1]);
        const tag = ordered ? "ol" : "ul";
        if (!list || list.tag !== tag) {
          flushList();
          list = { tag, items: [] };
        }
        list.items.push(listMatch[2]);
        continue;
      }

      if (line.includes("|") && lines[i + 1] && /^\s*\|?[\s:|-]+\|[\s:|-]*$/.test(lines[i + 1])) {
        flushParagraph();
        flushList();
        flushQuote();
        const header = line.split("|").map((c) => c.trim()).filter(Boolean);
        const rows = [];
        i += 2;
        while (i < lines.length && lines[i].includes("|") && lines[i].trim()) {
          rows.push(lines[i].split("|").map((c) => c.trim()).filter(Boolean));
          i++;
        }
        i--;
        html.push(
          `<table><thead><tr>${header.map((c) => `<th>${inlineFormat(c)}</th>`).join("")}</tr></thead>` +
            `<tbody>${rows
              .map((row) => `<tr>${row.map((c) => `<td>${inlineFormat(c)}</td>`).join("")}</tr>`)
              .join("")}</tbody></table>`
        );
        continue;
      }

      flushList();
      flushQuote();
      paragraph.push(line);
    }
    flushParagraph();
    flushList();
    flushQuote();
    return html.join("");
  }

  /* ----------------------------------------------------------- transcript UI */
  function nearBottom() {
    const t = els.transcript;
    return t.scrollHeight - t.scrollTop - t.clientHeight < 160;
  }

  function scrollDown(force = false) {
    if (force || nearBottom()) {
      requestAnimationFrame(() => {
        els.transcript.scrollTop = els.transcript.scrollHeight;
      });
    }
  }

  function clearStream() {
    hideThinking();
    els.stream.innerHTML = "";
    state.assistantEl = null;
    state.assistantBuf = "";
    state.toolsById = new Map();
    state.callSeq = 0;
  }

  let thinkingEl = null;

  function showThinking() {
    if (thinkingEl) return;
    thinkingEl = document.createElement("div");
    thinkingEl.className = "msg assistant thinking";
    thinkingEl.innerHTML = `
      <div class="avatar">F</div>
      <div class="body">
        <div class="who"><span>FRIDAY</span><time>${clockTime()}</time></div>
        <div class="dots"><span></span><span></span><span></span></div>
      </div>`;
    els.stream.appendChild(thinkingEl);
    scrollDown(true);
  }

  function hideThinking() {
    if (thinkingEl) {
      thinkingEl.remove();
      thinkingEl = null;
    }
  }

  function renderHero() {
    clearStream();
    const cards = [
      ["System status", "live telemetry, CPU, memory, disk", "system status"],
      ["Audit the workspace", "list every file I can reach", "list the files in the workspace"],
      ["Run the numbers", "17 × 42 + 9, no calculator", "what is 17 * 42 + 9"],
      ["Teach me something", "store a durable fact in memory", "remember that the project codename is FRIDAY"],
    ];
    const hero = document.createElement("div");
    hero.className = "hero";
    hero.innerHTML = `
      <div class="kicker">Field interface · ${
        state.config && state.config.simulation ? "simulation core" : "neural link online"
      }</div>
      <h2>What are we building today, <em>boss</em>?</h2>
      <p>I've got files, shell, Python, web, telemetry and a memory bank wired in. Give me a target —
         I'll show you every move I make.</p>
      <div class="prompt-grid">
        ${cards
          .map(
            ([title, desc, prompt]) => `
          <button class="prompt-card" data-prompt="${escapeHtml(prompt)}">
            <b>${escapeHtml(title)}</b>
            <small>${escapeHtml(desc)}</small>
          </button>`
          )
          .join("")}
      </div>`;
    els.stream.appendChild(hero);
    $$(".prompt-card", hero).forEach((card) =>
      card.addEventListener("click", () => {
        els.input.value = card.dataset.prompt || "";
        sendMessage();
      })
    );
  }

  function addUserMessage(text) {
    const wrap = document.createElement("div");
    wrap.className = "msg user";
    const body = document.createElement("div");
    body.className = "body";
    body.textContent = text;
    wrap.appendChild(body);
    els.stream.appendChild(wrap);
    scrollDown(true);
  }

  function ensureAssistant() {
    if (state.assistantEl) return state.assistantEl;
    const wrap = document.createElement("div");
    wrap.className = "msg assistant";
    wrap.innerHTML = `
      <div class="avatar">F</div>
      <div class="body">
        <div class="who"><span>FRIDAY</span><time></time><span class="tag" hidden>sim</span></div>
        <div class="md"></div>
      </div>`;
    const time = $("time", wrap);
    time.textContent = clockTime();
    if (state.config && state.config.simulation) $(".tag", wrap).hidden = false;
    els.stream.appendChild(wrap);
    state.assistantEl = wrap;
    return wrap;
  }

  function renderAssistant(streaming) {
    if (!state.assistantBuf) return;
    const wrap = ensureAssistant();
    const md = $(".md", wrap);
    md.innerHTML = renderMarkdown(state.assistantBuf) + (streaming ? '<span class="caret"></span>' : "");
    scrollDown();
  }

  function queueAssistantRender() {
    if (state.renderQueued) return;
    state.renderQueued = true;
    requestAnimationFrame(() => {
      state.renderQueued = false;
      renderAssistant(true);
    });
  }

  function assistantAppend(text) {
    hideThinking();
    state.assistantBuf += text;
    state.lastAnswer += text;
    queueAssistantRender();
  }

  function assistantFinish() {
    if (!state.assistantEl) {
      state.assistantBuf = "";
      return null;
    }
    if (!state.assistantBuf.trim()) {
      state.assistantEl.remove();
      state.assistantEl = null;
      state.assistantBuf = "";
      return null;
    }
    renderAssistant(false);
    const wrap = state.assistantEl;
    state.assistantEl = null;
    state.assistantBuf = "";
    return wrap;
  }

  function addNote(text, kind = "") {
    const note = document.createElement("div");
    note.className = "msg assistant " + (kind === "error" ? "error" : "");
    note.innerHTML = `
      <div class="avatar" style="${kind === "error" ? "border-color:rgba(255,84,112,.45);color:var(--red)" : ""}">!</div>
      <div class="body"><div class="md" style="color:var(--muted);font-family:var(--mono);font-size:11.5px">${escapeHtml(
        text
      )}</div></div>`;
    els.stream.appendChild(note);
    scrollDown();
  }

  function normaliseArgs(args) {
    if (!args) return {};
    if (typeof args === "string") {
      try {
        return JSON.parse(args);
      } catch (_) {
        return { input: args };
      }
    }
    return args;
  }

  function toolSummary(args) {
    args = normaliseArgs(args);
    try {
      const parts = Object.entries(args || {}).map(([key, value]) => {
        const text = typeof value === "string" ? value : JSON.stringify(value);
        return `${key}=${text.length > 34 ? text.slice(0, 34) + "…" : text}`;
      });
      return parts.join("  ");
    } catch (_) {
      return "";
    }
  }

  function addToolCard(call, status = "running", output = "", duration = null, ok = null) {
    call = { ...call, arguments: normaliseArgs(call.arguments) };
    state.callSeq += 1;
    const card = document.createElement("div");
    card.className = "tool-card " + status;
    card.dataset.callId = call.id || String(state.callSeq);
    const summary = toolSummary(call.arguments);
    const risk = call.risk ? `<span class="risk ${call.risk === "high" ? "high" : ""}">${escapeHtml(call.risk)}</span>` : "";
    card.innerHTML = `
      <button class="tool-head" type="button" aria-expanded="false">
        <svg><use href="#i-chevron"/></svg>
        <span class="tool-name">${escapeHtml(call.name || "tool")}</span>
        ${risk}
        <span class="tool-args">${escapeHtml(summary)}</span>
        <span class="tool-ms">${duration != null ? fmtDuration(duration) : "running"}</span>
        <span class="tool-status"></span>
      </button>
      <div class="tool-body">
        <div class="tool-args-full">${escapeHtml(JSON.stringify(call.arguments || {}, null, 2))}</div>
        <pre class="tool-output">${escapeHtml(output || "")}</pre>
      </div>`;
    const head = $(".tool-head", card);
    head.addEventListener("click", () => {
      card.classList.toggle("open");
      head.setAttribute("aria-expanded", card.classList.contains("open") ? "true" : "false");
    });
    if (output && (ok === false || output.length < 600) && status !== "running") {
      card.classList.add("open");
      head.setAttribute("aria-expanded", "true");
    }
    els.stream.appendChild(card);
    state.toolsById.set(card.dataset.callId, card);
    scrollDown();
    return card;
  }

  function updateToolCard(callId, result) {
    const card = state.toolsById.get(callId);
    if (!card) return;
    card.classList.remove("running");
    card.classList.add(result.ok ? "ok" : "failed");
    $(".tool-ms", card).textContent = fmtDuration(result.duration_ms);
    $(".tool-output", card).textContent = result.output || "";
    if (!result.ok && result.output) {
      card.classList.add("open");
      $(".tool-head", card).setAttribute("aria-expanded", "true");
    }
  }

  function addApprovalCard(ev) {
    const card = document.createElement("div");
    card.className = "tool-card approval";
    card.innerHTML = `
      <div class="approval-body">
        <div class="approval-title"><svg style="width:12px;height:12px"><use href="#i-shield"/></svg>
          Approval required <span class="risk ${ev.risk === "high" ? "high" : ""}">${escapeHtml(ev.risk || "medium")}</span>
        </div>
        <div class="approval-desc">FRIDAY wants to run <b style="font-family:var(--mono);color:var(--cyan-soft)">${escapeHtml(
          ev.name
        )}</b>.</div>
        <pre style="margin:0 0 10px;padding:8px 10px;border-radius:8px;border:1px solid var(--line);background:#04080d;font-family:var(--mono);font-size:10.5px;color:#b8d8e4;max-height:150px;overflow:auto;white-space:pre-wrap">${escapeHtml(
          JSON.stringify(ev.arguments || {}, null, 2)
        )}</pre>
        <div class="approval-actions">
          <button class="btn primary" data-approve="1">Approve</button>
          <button class="btn danger" data-approve="0">Deny</button>
        </div>
      </div>`;
    $$("[data-approve]", card).forEach((button) =>
      button.addEventListener("click", () => resolveApproval(ev.token, button.dataset.approve === "1", card))
    );
    els.stream.appendChild(card);
    state.approvals.set(ev.token, card);
    scrollDown(true);
    if (state.speak) speak("Approval required, boss.");
  }

  function markApprovalResolved(card, approved, label) {
    // Idempotent: the SSE `approval_result` and the POST response can both land
    // here, and whichever arrives second must not try to replace a node twice.
    const actions = card.querySelector(".approval-actions");
    if (!actions) return;
    const note = document.createElement("div");
    note.className = "approval-resolved " + (approved ? "approved" : "denied");
    note.textContent = label || (approved ? "approved" : "denied");
    actions.replaceWith(note);
  }

  async function resolveApproval(token, approved, card) {
    const buttons = $$("button", card);
    buttons.forEach((button) => (button.disabled = true));
    markApprovalResolved(card, approved, approved ? "approved" : "denied");
    try {
      await api(`/api/approvals/${encodeURIComponent(token)}`, {
        method: "POST",
        body: JSON.stringify({ approved, note: approved ? "approved in HUD" : "denied in HUD" }),
      });
    } catch (error) {
      // Put the decision controls back only if the server rejected the decision.
      const resolved = card.querySelector(".approval-resolved");
      const actions = card.querySelector(".approval-actions");
      if (!actions) {
        const restored = document.createElement("div");
        restored.className = "approval-actions";
        restored.innerHTML =
          '<button class="btn primary" data-approve="1">Approve</button>' +
          '<button class="btn danger" data-approve="0">Deny</button>';
        $$("[data-approve]", restored).forEach((button) =>
          button.addEventListener("click", () =>
            resolveApproval(token, button.dataset.approve === "1", card)
          )
        );
        if (resolved) resolved.replaceWith(restored);
      }
      $$("button", card).forEach((button) => (button.disabled = false));
      toast(error.message, "error");
    }
  }

  /* ------------------------------------------------------------- session list */
  function renderSessions() {
    els.sessionList.innerHTML = "";
    if (!state.sessions.length) {
      const empty = document.createElement("div");
      empty.style.cssText = "padding:14px 12px;color:var(--muted);font-family:var(--mono);font-size:10.5px";
      empty.textContent = "No sessions yet.";
      els.sessionList.appendChild(empty);
      return;
    }
    for (const session of state.sessions) {
      const button = document.createElement("button");
      button.className = "session-item" + (session.id === state.sessionId ? " active" : "");
      button.innerHTML = `<span class="stitle">${escapeHtml(session.title || "Untitled")}</span>
        <span class="smeta">${session.message_count || 0} msgs · ${escapeHtml(relativeTime(session.updated_at))}</span>`;
      button.addEventListener("click", () => {
        loadSession(session.id);
        $("#sidebar").classList.remove("show");
      });
      els.sessionList.appendChild(button);
    }
  }

  async function refreshSessions() {
    try {
      state.sessions = await api("/api/sessions");
      renderSessions();
    } catch (error) {
      console.warn("sessions:", error);
    }
  }

  function setActiveSession(id) {
    state.sessionId = id;
    if (id) localStorage.setItem("friday_session", id);
    else localStorage.removeItem("friday_session");
    renderSessions();
  }

  async function loadSession(id) {
    if (state.running) await stopRun();
    try {
      const data = await api(`/api/sessions/${encodeURIComponent(id)}`);
      setActiveSession(id);
      renderHistory(data.messages || []);
      els.bridgeStatus.textContent = data.title || "";
    } catch (error) {
      toast(error.message, "error");
      setActiveSession(null);
      renderHero();
    }
  }

  function renderHistory(messages) {
    clearStream();
    state.lastAnswer = "";
    if (!messages.length) {
      renderHero();
      return;
    }
    const byCall = new Map();
    for (const message of messages) {
      if (message.role === "user") {
        addUserMessage(message.content);
      } else if (message.role === "assistant") {
        if (message.content) {
          const wrap = document.createElement("div");
          wrap.className = "msg assistant";
          const simTag = state.config && state.config.simulation ? '<span class="tag">sim</span>' : "";
          const cancelled = message.meta && message.meta.cancelled ? '<span class="tag">stopped</span>' : "";
          wrap.innerHTML = `
            <div class="avatar">F</div>
            <div class="body">
              <div class="who"><span>FRIDAY</span><time>${escapeHtml(clockTime(message.created_at))}</time>${simTag}${cancelled}</div>
              <div class="md">${renderMarkdown(message.content)}</div>
            </div>`;
          els.stream.appendChild(wrap);
          state.lastAnswer += message.content + "\n";
        }
        for (const call of message.tool_calls || []) {
          const card = addToolCard(call, "ok", "", null, true);
          byCall.set(call.id, card);
          state.toolsById.set(call.id, card);
        }
      } else if (message.role === "tool") {
        const card = byCall.get(message.tool_call_id) || state.toolsById.get(message.tool_call_id);
        if (card) {
          card.classList.remove("running");
          card.classList.add(message.meta && message.meta.ok === false ? "failed" : "ok");
          $(".tool-ms", card).textContent = "";
          $(".tool-output", card).textContent = message.content || "";
        }
      }
    }
    scrollDown(true);
  }

  /* ------------------------------------------------------------------ events */
  function handleEvent(ev) {
    switch (ev.type) {
      case "open":
      case "start":
        if (ev.session_id) setActiveSession(ev.session_id);
        if (ev.run_id) state.runId = ev.run_id;
        break;
      case "step":
        els.bridgeStatus.textContent = `step ${ev.step}/${ev.max_steps} · reasoning`;
        break;
      case "token":
        if (ev.text) assistantAppend(ev.text);
        break;
      case "message":
        // Assistant text arrives as `token` deltas; this is a fallback for
        // non-streaming providers that send the finished message in one piece.
        hideThinking();
        if (ev.role === "assistant" && ev.content && !state.assistantEl && !state.assistantBuf) {
          state.assistantBuf = ev.content;
          state.lastAnswer += ev.content;
          renderAssistant(false);
          assistantFinish();
        }
        break;
      case "tool_call": {
        hideThinking();
        els.bridgeStatus.textContent = `${ev.name} · running`;
        assistantFinish();
        addToolCard({ id: ev.id, name: ev.name, arguments: ev.arguments }, "running");
        break;
      }
      case "tool_result":
        els.bridgeStatus.textContent = `${ev.name} · ${ev.ok ? "ok" : "failed"}`;
        updateToolCard(ev.id, ev);
        break;
      case "approval_required":
        els.bridgeStatus.textContent = "awaiting approval";
        addApprovalCard(ev);
        break;
      case "approval_result": {
        const card = state.approvals.get(ev.token);
        if (card && ev.approved !== undefined) {
          markApprovalResolved(card, ev.approved, ev.approved ? "approved" : ev.note || "denied");
        }
        break;
      }
      case "notice":
        addNote(ev.message || "");
        break;
      case "error":
        addNote(ev.message || "Something went wrong.", "error");
        toast(ev.message || "Agent error", "error");
        break;
      case "cancelled":
        addNote("Run stopped by operator.");
        break;
      case "done": {
        const usage = ev.usage || {};
        els.bridgeStatus.textContent = `${ev.steps || 0} step(s) · ${usage.total_tokens || 0} tokens`;
        if (ev.duration_ms != null) {
          els.latencyChip.hidden = false;
          els.latencyChip.textContent = fmtDuration(ev.duration_ms);
        }
        refreshSessions();
        pollSystem();
        break;
      }
      case "close":
        break;
      default:
        break;
    }
  }

  async function consumeSSE(response, onEvent) {
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let boundary;
      while ((boundary = buffer.indexOf("\n\n")) >= 0) {
        const chunk = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const payload = chunk
          .split("\n")
          .filter((line) => line.startsWith("data:"))
          .map((line) => line.slice(5).trim())
          .join("\n");
        if (!payload) continue;
        try {
          onEvent(JSON.parse(payload));
        } catch (error) {
          console.warn("bad SSE frame", error, payload);
        }
      }
    }
  }

  /* -------------------------------------------------------------------- chat */
  function setRunning(running) {
    state.running = running;
    els.composer.classList.toggle("running", running);
    els.input.disabled = false;
    els.sendBtn.disabled = running;
    $("#statusText").textContent = running ? "working" : "online";
    $("#statusChip").classList.toggle("warn", running);
    if (!running) {
      els.stopBtn.style.display = "";
      state.runId = null;
      state.abort = null;
    }
  }

  async function stopRun() {
    if (!state.running) return;
    try {
      if (state.runId) await api(`/api/chat/${encodeURIComponent(state.runId)}/cancel`, { method: "POST" });
    } catch (error) {
      console.warn("cancel:", error);
    }
    if (state.abort) {
      try {
        state.abort.abort();
      } catch (_) {
        /* ignore */
      }
    }
    setRunning(false);
  }

  async function sendMessage() {
    const text = els.input.value.trim();
    if (!text || state.running) return;
    if (els.stream.querySelector(".hero")) els.stream.innerHTML = "";
    addUserMessage(text);
    els.input.value = "";
    autosize();
    updateCount();
    setRunning(true);
    showThinking();
    els.bridgeStatus.textContent = "establishing link";
    state.lastAnswer = "";
    const controller = new AbortController();
    state.abort = controller;
    state.runId = uuid();
    const runId = state.runId;

    const headers = { "Content-Type": "application/json" };
    if (state.token) headers["X-FRIDAY-Token"] = state.token;

    try {
      const response = await fetch("/api/chat", {
        method: "POST",
        headers,
        signal: controller.signal,
        body: JSON.stringify({
          message: text,
          session_id: state.sessionId || undefined,
          run_id: runId,
          approval: state.approvalMode || undefined,
        }),
      });
      if (response.status === 401) {
        openDrawer();
        throw new Error("API token required — set it in settings.");
      }
      if (!response.ok) {
        let detail = `HTTP ${response.status}`;
        try {
          const body = await response.json();
          detail = body.detail || detail;
        } catch (_) {
          /* ignore */
        }
        throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      }
      await consumeSSE(response, handleEvent);
    } catch (error) {
      if (error.name === "AbortError") {
        addNote("Run cancelled.");
      } else {
        addNote(`Connection error: ${error.message}`, "error");
        toast(error.message, "error");
      }
    } finally {
      hideThinking();
      const finished = assistantFinish();
      setRunning(false);
      els.bridgeStatus.textContent = "";
      if (state.speak && state.lastAnswer.trim() && !controller.signal.aborted) {
        speak(state.lastAnswer);
      }
      refreshSessions();
      updateFacts();
      els.input.focus();
      if (!finished) scrollDown();
    }
  }

  /* ------------------------------------------------------------------- voice */
  function initSpeech() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      $("#voiceBtn").title = "Voice input is not supported in this browser";
      return;
    }
    const recognition = new SpeechRecognition();
    recognition.lang = "en-US";
    recognition.interimResults = true;
    recognition.continuous = false;
    let base = "";
    recognition.onstart = () => {
      state.listening = true;
      base = els.input.value;
      $("#voiceBtn").style.color = "var(--red)";
      $("#voiceBtn").style.borderColor = "rgba(255,84,112,.5)";
      toast("Listening…");
    };
    recognition.onresult = (event) => {
      let interim = "";
      let final = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const transcript = event.results[i][0].transcript;
        if (event.results[i].isFinal) final += transcript;
        else interim += transcript;
      }
      els.input.value = (base ? base + " " : "") + (final || interim);
      autosize();
    };
    recognition.onerror = (event) => {
      if (event.error !== "aborted") toast(`Voice input: ${event.error}`, "error");
    };
    recognition.onend = () => {
      state.listening = false;
      $("#voiceBtn").style.color = "";
      $("#voiceBtn").style.borderColor = "";
    };
    state.recognition = recognition;
  }

  function extractPlainText(markdown) {
    return String(markdown)
      .replace(/```[\s\S]*?```/g, " code block ")
      .replace(/`([^`]+)`/g, "$1")
      .replace(/!\[[^\]]*\]\([^)]*\)/g, "")
      .replace(/\[([^\]]+)\]\([^)]*\)/g, "$1")
      .replace(/[*_#>|~]/g, " ")
      .replace(/\s+/g, " ")
      .trim();
  }

  function pickVoice() {
    if (!window.speechSynthesis) return null;
    const voices = window.speechSynthesis.getVoices();
    if (!voices.length) return null;
    const preferred = [
      /Samantha/i,
      /Google UK English Female/i,
      /Microsoft Zira/i,
      /Google US English/i,
      /female/i,
      /en-US/i,
      /^en/i,
    ];
    for (const pattern of preferred) {
      const match = voices.find((voice) => pattern.test(voice.name) || pattern.test(voice.lang));
      if (match) return match;
    }
    return voices.find((voice) => voice.lang && voice.lang.startsWith("en")) || voices[0];
  }

  function speak(text) {
    if (!window.speechSynthesis) {
      toast("Speech synthesis is not available in this browser.", "error");
      state.speak = false;
      syncSpeakUI();
      return;
    }
    const clean = extractPlainText(text).slice(0, 1800);
    if (!clean) return;
    window.speechSynthesis.cancel();
    const utterance = new SpeechSynthesisUtterance(clean);
    utterance.rate = 1.02;
    utterance.pitch = 1.05;
    utterance.volume = 0.95;
    const voice = pickVoice();
    if (voice) utterance.voice = voice;
    window.speechSynthesis.speak(utterance);
  }

  function syncSpeakUI() {
    $("#speakBtn").style.color = state.speak ? "var(--cyan)" : "";
    $("#speakBtn").style.borderColor = state.speak ? "rgba(56,225,255,.4)" : "";
    $("#speakToggle").checked = state.speak;
  }

  /* ----------------------------------------------------------------- system */
  function setGauge(id, percent) {
    const gauge = $(id);
    if (!gauge) return;
    const value = Math.max(0, Math.min(100, Number(percent) || 0));
    const circle = $(".fill", gauge);
    const circumference = 2 * Math.PI * 26;
    circle.style.strokeDasharray = String(circumference);
    circle.style.strokeDashoffset = String(circumference * (1 - value / 100));
    $(".gauge-value", gauge).textContent = `${Math.round(value)}%`;
    gauge.classList.toggle("warn", value >= 70 && value < 88);
    gauge.classList.toggle("danger", value >= 88);
    gauge.title = id === "#gaugeCpu" ? "1-minute load average, normalised by core count" : "";
  }

  async function pollSystem() {
    try {
      const data = await api("/api/system");
      const cores = data.cpu_count || 1;
      const cpu = data.load_1m != null ? Math.min(100, (data.load_1m / cores) * 100) : 0;
      setGauge("#gaugeCpu", cpu);
      setGauge("#gaugeMem", data.memory ? data.memory.used_pct : 0);
      setGauge("#gaugeDisk", data.disk ? data.disk.used_pct : 0);
      $("#hostLabel").textContent = (data.hostname || "").toUpperCase();
      const friday = data.friday || {};
      $("#metaProvider").textContent = friday.provider || "—";
      $("#metaModel").textContent = friday.model || "—";
      $("#metaUptime").textContent = fmtUptime(friday.uptime_seconds);
      $("#metaSessions").textContent = friday.sessions ?? "—";
      $("#factCount").textContent = friday.memories ?? 0;
      $("#metaVersion").textContent = friday.version || "—";
      $("#statusChip").classList.remove("danger");
      if (!state.running) $("#statusText").textContent = "online";
    } catch (error) {
      $("#statusChip").classList.remove("warn");
      $("#statusText").textContent = "offline";
      $("#statusChip").classList.add("danger");
    }
  }

  function renderTools() {
    const list = $("#toolList");
    list.innerHTML = "";
    $("#toolCount").textContent = state.tools.filter((tool) => tool.enabled).length;
    $("#toolkitCount").textContent = String(state.tools.filter((tool) => tool.enabled).length).padStart(2, "0");
    for (const tool of state.tools) {
      const row = document.createElement("div");
      row.className = "tool-row";
      row.title = tool.description;
      row.innerHTML = `<span class="tdot ${tool.risk}"></span><span class="tname">${escapeHtml(
        tool.name
      )}</span><span class="trisk">${tool.risk === "low" ? "" : tool.risk}</span>`;
      list.appendChild(row);
    }
  }

  /* ------------------------------------------------------------------- facts */
  async function updateFacts() {
    try {
      const data = await api("/api/memory");
      $("#factCount").textContent = data.stats ? data.stats.memories : 0;
    } catch (_) {
      /* non-fatal */
    }
  }

  async function openMemory() {
    $("#memoryModal").classList.add("show");
    $("#memoryModal").setAttribute("aria-hidden", "false");
    await renderFacts();
  }

  function closeMemory() {
    $("#memoryModal").classList.remove("show");
    $("#memoryModal").setAttribute("aria-hidden", "true");
  }

  async function renderFacts() {
    const container = $("#factList");
    container.innerHTML = '<div class="fact-empty">loading…</div>';
    try {
      const data = await api("/api/memory");
      container.innerHTML = "";
      if (!data.items.length) {
        container.innerHTML = '<div class="fact-empty">Nothing stored yet. Teach me something durable.</div>';
        return;
      }
      for (const item of data.items) {
        const row = document.createElement("div");
        row.className = "fact";
        row.innerHTML = `<span class="fid">#${item.id}</span><span class="ftext">${escapeHtml(item.text)}</span>
          <button class="del" title="Forget" aria-label="Forget memory ${item.id}">×</button>`;
        $(".del", row).addEventListener("click", async () => {
          try {
            await api(`/api/memory/${item.id}`, { method: "DELETE" });
            renderFacts();
            updateFacts();
          } catch (error) {
            toast(error.message, "error");
          }
        });
        container.appendChild(row);
      }
    } catch (error) {
      container.innerHTML = `<div class="fact-empty">${escapeHtml(error.message)}</div>`;
    }
  }

  async function addFact() {
    const input = $("#factInput");
    const text = input.value.trim();
    if (text.length < 3) return;
    try {
      await api("/api/memory", { method: "POST", body: JSON.stringify({ text }) });
      input.value = "";
      renderFacts();
      updateFacts();
      toast("Memory stored", "ok");
    } catch (error) {
      toast(error.message, "error");
    }
  }

  /* ----------------------------------------------------------------- drawer */
  function openDrawer() {
    $("#drawer").classList.add("show");
    $("#drawer").setAttribute("aria-hidden", "false");
    $("#scrim").classList.add("show");
  }

  function closeDrawer() {
    $("#drawer").classList.remove("show");
    $("#drawer").setAttribute("aria-hidden", "true");
    $("#scrim").classList.remove("show");
  }

  /* -------------------------------------------------------------------- init */
  function autosize() {
    els.input.style.height = "auto";
    els.input.style.height = Math.min(els.input.scrollHeight, 220) + "px";
  }

  function updateCount() {
    const length = els.input.value.length;
    els.count.textContent = length ? `${length} chars` : "";
  }

  function cycleApproval() {
    const order = ["dangerous", "always", "never"];
    const current = state.approvalMode || (state.config && state.config.approval) || "dangerous";
    const next = order[(order.indexOf(current) + 1) % order.length];
    state.approvalMode = next;
    localStorage.setItem("friday_approval", next);
    syncApprovalUI();
  }

  const MODE_LABELS = { dangerous: "guardrails", always: "strict", never: "autonomous" };
  function syncApprovalUI() {
    const mode = state.approvalMode || (state.config && state.config.approval) || "dangerous";
    els.modeBtn.dataset.mode = mode;
    els.modeLabel.textContent = MODE_LABELS[mode] || mode;
    els.modeBtn.title =
      mode === "dangerous"
        ? "High-risk tools need approval"
        : mode === "always"
        ? "Every tool call needs approval"
        : "No approvals — trusted sandbox only";
    $("#metaApproval").textContent = mode;
  }

  function syncClock() {
    const now = new Date();
    $("#clockTime").textContent = now.toLocaleTimeString([], { hour12: false });
    const offset = -now.getTimezoneOffset() / 60;
    const sign = offset >= 0 ? "+" : "";
    $("#clockZone").textContent = `LOCAL ${sign}${offset} · UTC`;
  }

  function applyConfig() {
    const cfg = state.config || {};
    if (!state.approvalMode) state.approvalMode = cfg.approval || "dangerous";
    $("#modelChip").textContent = cfg.simulation ? "simulation core" : `${cfg.provider} · ${cfg.model}`;
    $("#modelChip").title = `${cfg.provider} / ${cfg.model}`;
    $("#modelChip").classList.toggle("warn", !!cfg.simulation);
    $("#cfgProvider").textContent = cfg.provider || "—";
    $("#cfgModel").textContent = cfg.model || "—";
    $("#cfgBase").textContent = cfg.base_url || "local";
    $("#cfgWorkspace").textContent = cfg.workspace || "—";
    $("#simCallout").hidden = !cfg.simulation;
    $("#metaWorkspace").textContent = cfg.workspace || "—";
    $("#approvalSelect").value = state.approvalMode || cfg.approval || "dangerous";
    $("#tokenInput").value = state.token;
    syncApprovalUI();
    syncSpeakUI();
    if (cfg.auth_required && !state.token) {
      toast("This server requires an API token — set it in settings.", "error");
    }
  }

  async function boot() {
    syncClock();
    setInterval(syncClock, 1000);
    initSpeech();

    els.speakBtn && els.speakBtn.addEventListener("click", () => {
      state.speak = !state.speak;
      localStorage.setItem("friday_speak", state.speak ? "1" : "0");
      syncSpeakUI();
      if (!state.speak && window.speechSynthesis) window.speechSynthesis.cancel();
    });

    try {
      const [health, config, tools] = await Promise.all([
        api("/api/health"),
        api("/api/config"),
        api("/api/tools"),
      ]);
      state.config = config;
      state.tools = tools;
      applyConfig();
      renderTools();
      els.bootStatus.textContent = health.simulation ? "simulation core engaged" : "neural link established";
    } catch (error) {
      els.bootStatus.textContent = "uplink failed — " + error.message;
      toast("Could not reach the FRIDAY server: " + error.message, "error");
    }

    await refreshSessions();
    pollSystem();
    setInterval(pollSystem, 5000);
    setInterval(refreshSessions, 30000);

    if (state.sessionId) {
      await loadSession(state.sessionId);
    } else {
      renderHero();
    }

    setTimeout(() => els.boot.classList.add("done"), 700);
    els.input.focus();
  }

  /* ---------------------------------------------------------------- wiring */
  els.composer.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage();
  });
  els.input.addEventListener("input", () => {
    autosize();
    updateCount();
  });
  els.input.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendMessage();
    }
    if (event.key === "Escape" && state.running) {
      event.preventDefault();
      stopRun();
    }
  });
  els.stopBtn.addEventListener("click", stopRun);
  els.modeBtn.addEventListener("click", cycleApproval);
  $("#newBtn").addEventListener("click", async () => {
    if (state.running) await stopRun();
    setActiveSession(null);
    clearStream();
    renderHero();
    els.bridgeStatus.textContent = "";
    els.input.focus();
  });
  $("#menuBtn").addEventListener("click", () => $("#sidebar").classList.toggle("show"));
  $("#settingsBtn").addEventListener("click", openDrawer);
  $("#drawerClose").addEventListener("click", closeDrawer);
  $("#scrim").addEventListener("click", closeDrawer);
  $("#memoryBtn").addEventListener("click", openMemory);
  $("#memoryLink").addEventListener("click", openMemory);
  $("#memoryClose").addEventListener("click", closeMemory);
  $("#factAdd").addEventListener("click", addFact);
  $("#factInput").addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      event.preventDefault();
      addFact();
    }
  });
  $("#voiceBtn").addEventListener("click", () => {
    if (!state.recognition) {
      toast("Voice input is not supported in this browser.", "error");
      return;
    }
    if (state.listening) state.recognition.stop();
    else {
      try {
        state.recognition.start();
      } catch (error) {
        toast("Could not start the microphone: " + error.message, "error");
      }
    }
  });
  $("#tokenInput").addEventListener("change", (event) => {
    state.token = event.target.value.trim();
    localStorage.setItem("friday_token", state.token);
    toast(state.token ? "Token saved — reloading" : "Token cleared — reloading", "ok");
    setTimeout(() => location.reload(), 500);
  });
  $("#approvalSelect").addEventListener("change", (event) => {
    state.approvalMode = event.target.value;
    localStorage.setItem("friday_approval", state.approvalMode);
    syncApprovalUI();
  });
  $("#speakToggle").addEventListener("change", (event) => {
    state.speak = event.target.checked;
    localStorage.setItem("friday_speak", state.speak ? "1" : "0");
    syncSpeakUI();
  });
  $("#clearLocal").addEventListener("click", () => {
    localStorage.removeItem("friday_session");
    state.sessionId = null;
    closeDrawer();
    clearStream();
    renderHero();
    refreshSessions();
    toast("Local transcript cache cleared");
  });
  $("#toolsLink").addEventListener("click", () => {
    const rail = document.querySelector(".rail");
    if (rail && window.getComputedStyle(rail).display !== "none") {
      rail.scrollIntoView({ behavior: "smooth", block: "start" });
    } else {
      openDrawer();
    }
  });
  document.addEventListener("keydown", (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "k") {
      event.preventDefault();
      $("#newBtn").click();
    }
    if (event.key === "Escape") {
      if ($("#memoryModal").classList.contains("show")) closeMemory();
      if ($("#drawer").classList.contains("show")) closeDrawer();
      if (state.running) stopRun();
    }
  });
  window.addEventListener("beforeunload", () => {
    if (state.running && state.runId) {
      navigator.sendBeacon?.(`/api/chat/${state.runId}/cancel`);
    }
  });

  boot();
})();
