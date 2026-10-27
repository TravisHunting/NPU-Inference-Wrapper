"use strict";

// ---------- helpers ----------------------------------------------------------------------

const $ = (sel) => document.querySelector(sel);

/** Build an element. Strings become text nodes (never innerHTML) since model names, process
 *  names and error text all come from outside. */
function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) if (c != null && c !== false) el.append(c instanceof Node ? c : String(c));
  return el;
}

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const d = body?.detail ?? body?.error ?? {};
    const err = new Error(d.message || `${res.status} ${res.statusText}`);
    Object.assign(err, { hint: d.hint, detail: d.detail, code: d.code });
    throw err;
  }
  return body;
}

function fmtBytes(n) {
  if (n == null) return "–";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(n >= 100 || i === 0 ? 0 : 1)} ${units[i]}`;
}
const fmtMb = (mb) => (mb == null ? "–" : mb >= 1024 ? `${(mb / 1024).toFixed(1)} GB` : `${mb} MB`);
const fmtPct = (v) => (v == null ? "–" : `${v.toFixed(v < 10 ? 1 : 0)}%`);
const fmtMs = (ms) => (ms == null ? "–" : ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`);
const fmtTime = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
const runtimeLabel = (ep) => (ep || "").replace(/ExecutionProvider$/, "") || "–";
const isDirect = (m) => m.extra?.engine === "openvino-genai";
const modelLabel = (m) => `${m.alias} · ${m.device.toUpperCase()}${isDirect(m) ? " · OpenVINO GenAI" : ""}`;

function setRows(tbody, rows, emptyText, cols) {
  tbody.replaceChildren(...(rows.length ? rows : [h("tr", {}, h("td", { colspan: cols, class: "empty" }, emptyText))]));
}

// ---------- tabs -------------------------------------------------------------------------

const tabButtons = document.querySelectorAll(".tabs button");
function showTab(name) {
  tabButtons.forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
  document.querySelectorAll(".tab-panel").forEach((p) => (p.hidden = p.id !== `tab-${name}`));
  if (location.hash !== `#${name}`) history.replaceState(null, "", `#${name}`);
  try { localStorage.setItem("tab", name); } catch {}
  if (name === "models") refreshModels();
  if (name === "playground") refreshModels().then(fillModelSelect);
  if (name === "api") renderApiTab();
  if (name === "dashboard") redrawCharts();
}
tabButtons.forEach((b) => b.addEventListener("click", () => showTab(b.dataset.tab)));

// ---------- charts -----------------------------------------------------------------------

const tooltip = $("#tooltip");

/** Single-series time chart with a crosshair tooltip. points: [{t, v}] */
function lineChart(container, points, { max, format, windowSec = 300 }) {
  const W = container.clientWidth || 400, H = container.clientHeight || 180;
  const pad = { l: 44, r: 8, t: 8, b: 20 };
  const now = Date.now() / 1000, t0 = now - windowSec;
  const yMax = max ?? niceMax(Math.max(1, ...points.map((p) => p.v)));
  const x = (t) => pad.l + ((t - t0) / windowSec) * (W - pad.l - pad.r);
  const y = (v) => H - pad.b - (v / yMax) * (H - pad.t - pad.b);
  const NS = "http://www.w3.org/2000/svg";
  const s = (tag, attrs) => { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; };

  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
  for (const frac of [0, 0.5, 1]) {
    const v = yMax * frac;
    svg.append(s("line", { class: "gridline", x1: pad.l, x2: W - pad.r, y1: y(v), y2: y(v) }));
    const label = s("text", { class: "axis-label", x: pad.l - 6, y: y(v) + 4, "text-anchor": "end" });
    label.textContent = format(v);
    svg.append(label);
  }
  for (const [t, text, anchor] of [[t0, `-${windowSec / 60}m`, "start"], [now, "now", "end"]]) {
    const label = s("text", { class: "axis-label", x: x(t), y: H - 4, "text-anchor": anchor });
    label.textContent = text;
    svg.append(label);
  }
  const pts = points.filter((p) => p.t >= t0);
  if (pts.length > 1) {
    const d = pts.map((p, i) => `${i ? "L" : "M"}${x(p.t).toFixed(1)},${y(p.v).toFixed(1)}`).join("");
    svg.append(s("path", { class: "area", d: `${d}L${x(pts.at(-1).t)},${y(0)}L${x(pts[0].t)},${y(0)}Z` }));
    svg.append(s("path", { class: "line", d }));
  }
  const cross = s("line", { class: "crosshair", y1: pad.t, y2: H - pad.b, visibility: "hidden" });
  const marker = s("circle", { class: "marker", r: 4, visibility: "hidden" });
  const hit = s("rect", { x: pad.l, y: 0, width: W - pad.l - pad.r, height: H, fill: "transparent" });
  svg.append(cross, marker, hit);
  container.replaceChildren(svg);
  container.setAttribute("aria-label", pts.length ? `Latest ${format(pts.at(-1).v)}` : "No data yet");

  hit.addEventListener("pointermove", (ev) => {
    if (!pts.length) return;
    const rect = svg.getBoundingClientRect();
    const px = ((ev.clientX - rect.left) / rect.width) * W;
    const t = t0 + ((px - pad.l) / (W - pad.l - pad.r)) * windowSec;
    const p = pts.reduce((a, b) => (Math.abs(b.t - t) < Math.abs(a.t - t) ? b : a));
    for (const el of [cross, marker]) el.setAttribute("visibility", "visible");
    cross.setAttribute("x1", x(p.t)); cross.setAttribute("x2", x(p.t));
    marker.setAttribute("cx", x(p.t)); marker.setAttribute("cy", y(p.v));
    tooltip.replaceChildren(h("strong", {}, format(p.v)), h("span", { class: "muted" }, fmtTime(p.t)));
    tooltip.hidden = false;
    tooltip.style.left = `${Math.min(ev.clientX + 14, innerWidth - 140)}px`;
    tooltip.style.top = `${ev.clientY - 44}px`;
  });
  hit.addEventListener("pointerleave", () => {
    for (const el of [cross, marker]) el.setAttribute("visibility", "hidden");
    tooltip.hidden = true;
  });
}

function niceMax(v) {
  const p = 10 ** Math.floor(Math.log10(v));
  return [1, 2, 2.5, 5, 10].map((m) => m * p).find((m) => m >= v * 1.1) ?? v;
}

// ---------- NPU telemetry ----------------------------------------------------------------

const history = []; // {ts, utilization, shared_bytes}
let latestSample = null;

function redrawCharts() {
  if ($("#tab-dashboard").hidden) return;
  lineChart($("#chart-util"), history.map((s) => ({ t: s.ts, v: s.utilization })), { max: 100, format: (v) => `${Math.round(v)}%` });
  // Round the memory axis in MB so gridlines land on readable binary values.
  const MB = 1024 * 1024;
  const memMax = niceMax(Math.max(1, ...history.map((s) => s.shared_bytes / MB))) * MB;
  lineChart($("#chart-mem"), history.map((s) => ({ t: s.ts, v: s.shared_bytes })), { max: memMax, format: fmtBytes });
}

function renderSample(s) {
  latestSample = s;
  const peak = Math.max(0, ...history.map((x) => x.utilization));
  $("#t-util").textContent = fmtPct(s.utilization);
  $("#t-util-sub").textContent = `5-min peak ${fmtPct(peak)}`;
  $("#t-mem").textContent = fmtBytes(s.shared_bytes);
  $("#t-procs").textContent = s.processes.length;
  $("#t-ram").textContent = fmtBytes(s.system_memory_available);
  $("#t-ram-sub").textContent = `of ${fmtBytes(s.system_memory_total)} · NPU models live in system RAM`;
  $("#live-npu").textContent = fmtPct(s.utilization);
  setRows($("#proc-rows"), s.processes.map((p) => h("tr", {},
    h("td", {}, p.name), h("td", { class: "num" }, p.pid),
    h("td", { class: "num" }, fmtPct(p.utilization)), h("td", { class: "num" }, fmtBytes(p.shared_bytes)),
  )), "No processes are using the NPU.", 4);
  redrawCharts();
}

function renderDevice(device, error) {
  const banner = $("#npu-error");
  banner.hidden = !error;
  if (error) banner.replaceChildren(h("strong", {}, "NPU telemetry unavailable"), error);
  const rows = device ? [
    ["Name", device.name], ["Vendor", device.vendor ?? "…"], ["Driver", device.driver_version ?? "…"],
    ["Driver date", device.driver_date ?? "…"], ["WDDM", device.wddm_version ? (device.wddm_version / 1000).toFixed(1) : "–"],
    ["Adapter LUID", device.luid],
  ] : [["Status", "No NPU detected"]];
  $("#device-kv").replaceChildren(...rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)]));
}

async function startTelemetry() {
  const [info, hist] = await Promise.all([api("/api/npu"), api("/api/npu/history")]);
  renderDevice(info.device, info.error);
  history.push(...hist);
  if (info.latest) renderSample(info.latest);
  // Driver details are filled in a moment after startup.
  if (info.device && !info.device.driver_version) setTimeout(async () => renderDevice(...Object.values(await api("/api/npu")).slice(0, 2)), 4000);

  const es = new EventSource("/api/npu/stream");
  es.onmessage = (ev) => {
    const s = JSON.parse(ev.data);
    history.push(s);
    while (history.length && history[0].ts < s.ts - 300) history.shift();
    renderSample(s);
  };
}
addEventListener("resize", redrawCharts);

// ---------- backend status ---------------------------------------------------------------

let backend = null;

async function refreshBackend() {
  const pill = $("#service-pill");
  try {
    backend = await api("/api/backend");
  } catch (e) {
    backend = null;
    pill.className = "service bad";
    pill.replaceChildren(h("span", { class: "dot" }), h("span", {}, `Foundry Local error: ${e.message}`));
    return;
  }
  if (!backend.installed) {
    pill.className = "service bad";
    pill.replaceChildren(h("span", { class: "dot" }), h("span", {}, "Foundry Local not installed (winget install Microsoft.FoundryLocal)"));
  } else if (!backend.running) {
    pill.className = "service bad";
    pill.replaceChildren(h("span", { class: "dot" }), h("span", {}, "Foundry Local stopped"),
      h("button", { class: "btn", onclick: startBackend }, "Start"));
  } else {
    pill.className = "service ok";
    pill.replaceChildren(h("span", { class: "dot" }), h("span", {}, `Foundry Local ${backend.system?.service?.cliVersion ?? ""} · running`));
  }
  const sys = backend.system ?? {}, svc = sys.service ?? {};
  const rows = [
    ["Service", backend.running ? `running · ${backend.endpoint}` : "stopped"],
    ["Version", svc.cliVersion ?? "–"],
    ["ONNX Runtime", [svc.ortVersion, svc.ortGenAiVersion && `GenAI ${svc.ortGenAiVersion}`].filter(Boolean).join(" · ") || "–"],
    ["NPUs seen", (sys.system?.npus ?? []).map((n) => n.name).join(", ") || "none"],
    ["GPUs seen", (sys.system?.gpus ?? []).map((g) => g.name).join(", ") || "none"],
  ];
  const ovi = backend.openvino ?? {};
  rows.push(["OpenVINO GenAI", ovi.available
    ? `${ovi.version} · ${(ovi.devices ?? []).join(", ")} · fallback for OpenVINO models`
    : "not installed (no direct-load fallback)"]);
  for (const [runtime, why] of Object.entries(ovi.foundry_fallbacks ?? {})) {
    rows.push(["Bypassing", `${runtimeLabel(runtime)} in Foundry: ${why}`]);
  }
  $("#foundry-kv").replaceChildren(...rows.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)]));
  $("#foundry-warnings").replaceChildren(...(sys.warnings ?? []).map((w) =>
    h("div", { class: "banner banner-warning" }, typeof w === "string" ? w : JSON.stringify(w))));
}

async function startBackend() {
  $("#service-pill").replaceChildren(h("span", { class: "dot" }), h("span", {}, "Starting Foundry Local…"));
  try { await api("/api/backend/start", { method: "POST" }); } catch (e) { alert(e.message); }
  refreshBackend();
}

// ---------- models -----------------------------------------------------------------------

let models = [];
let deviceFilter = "npu";
try { deviceFilter = localStorage.getItem("deviceFilter") ?? "npu"; } catch {}

async function refreshModels(refresh = false) {
  try {
    models = await api(`/api/models${refresh ? "?refresh=true" : ""}`);
  } catch (e) {
    setRows($("#model-rows"), [], `Could not load catalog: ${e.message}`, 7);
    return models;
  }
  const loaded = models.filter((m) => m.loaded);
  $("#t-loaded").textContent = loaded.length;
  $("#t-loaded-sub").textContent = loaded.length ? loaded.map(modelLabel).join(", ") : "none yet";
  renderModels();
  return models;
}

function renderModels() {
  const q = $("#model-search").value.trim().toLowerCase();
  const cachedOnly = $("#cached-only").checked;
  const rows = models
    .filter((m) => !deviceFilter || m.device === deviceFilter)
    .filter((m) => !cachedOnly || m.cached)
    .filter((m) => !q || `${m.alias} ${m.name}`.toLowerCase().includes(q))
    .sort((a, b) => b.loaded - a.loaded || b.cached - a.cached || a.alias.localeCompare(b.alias));
  setRows($("#model-rows"), rows.map(modelRow), deviceFilter === "npu"
    ? "No NPU models match. Foundry Local's catalog may not have NPU builds for this chip yet."
    : "No models match.", 7);
}

function modelRow(m) {
  const busy = jobs.some((j) => j.state === "running" && j.model_id === m.id);
  const act = (action, label, cls = "btn") => h("button", { class: cls, disabled: busy, onclick: () => runAction(action, m) }, label);
  const actions = busy ? [h("span", { class: "spinner", title: "Working…" })]
    : m.loaded ? [act("unload", "Unload")]
    : m.cached ? [act("load", "Load", "btn btn-primary"), act("remove", "Delete", "btn btn-danger")]
    : [act("download", `Download`)];
  return h("tr", {},
    h("td", {}, m.alias, h("div", { class: "sub" }, m.task ?? "")),
    h("td", {}, h("span", { class: "sub" }, m.id)),
    h("td", {}, h("span", { class: `badge device-${m.device}` }, m.device.toUpperCase())),
    h("td", {}, runtimeLabel(m.runtime)),
    h("td", { class: "num" }, fmtMb(m.size_mb)),
    h("td", {}, m.loaded
      ? [h("span", { class: "badge loaded" }, "Loaded"), isDirect(m) && h("div", { class: "sub" }, "via OpenVINO GenAI")]
      : m.cached ? h("span", { class: "badge" }, "Downloaded") : ""),
    h("td", { class: "actions" }, actions),
  );
}

async function runAction(action, m) {
  if (action === "remove" && !confirm(`Delete ${m.id} from the local cache? You can download it again later.`)) return;
  try {
    await api(`/api/models/${action}/${encodeURIComponent(m.id)}`, { method: "POST" });
  } catch (e) {
    alert(e.message);
  }
  pollJobs();
}

document.querySelectorAll("#device-filter button").forEach((b) => {
  b.setAttribute("aria-pressed", String(b.dataset.device === deviceFilter));
  b.addEventListener("click", () => {
    deviceFilter = b.dataset.device;
    try { localStorage.setItem("deviceFilter", deviceFilter); } catch {}
    document.querySelectorAll("#device-filter button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
    renderModels();
  });
});
$("#model-search").addEventListener("input", renderModels);
$("#cached-only").addEventListener("change", renderModels);
$("#refresh-models").addEventListener("click", () => refreshModels(true));

// ---------- jobs -------------------------------------------------------------------------

let jobs = [];
const dismissed = new Set();
let jobTimer = null;
const verbs = { download: "Downloading", load: "Loading", unload: "Unloading", remove: "Deleting" };

async function pollJobs() {
  clearTimeout(jobTimer);
  const before = new Set(jobs.filter((j) => j.state === "running").map((j) => j.id));
  try { jobs = await api("/api/jobs"); } catch { return; }
  renderJobs();
  const finished = jobs.some((j) => before.has(j.id) && j.state !== "running");
  if (finished) { await refreshModels(); fillModelSelect(); refreshBackend(); }
  else renderModels();
  if (jobs.some((j) => j.state === "running")) jobTimer = setTimeout(pollJobs, 1000);
}

function renderJobs() {
  const recent = (j) => j.finished && Date.now() / 1000 - j.finished < 600;
  const visible = (j) => j.state === "running" || (recent(j) && (j.state === "failed" || j.notice));
  // jobs arrive newest first; show only the latest job per model so repeat loads don't stack notices
  const seen = new Set();
  const latest = (j) => !seen.has(j.model_id) && seen.add(j.model_id);
  const els = jobs.filter((j) => latest(j) && !dismissed.has(j.id) && visible(j)).map((j) => {
    if (j.state === "running") {
      const secs = Math.round(Date.now() / 1000 - j.started);
      const note = j.action === "load" ? " · first load on an NPU compiles the model and can take a few minutes" : "";
      return h("div", { class: "job" }, h("span", { class: "spinner" }), `${verbs[j.action]} ${j.model_id}… ${secs}s`, h("span", { class: "muted" }, note));
    }
    const dismiss = h("button", { class: "btn", onclick: () => { dismissed.add(j.id); renderJobs(); } }, "Dismiss");
    if (j.state === "succeeded") {
      return h("div", { class: "banner banner-warning" },
        h("div", { class: "row" }, h("div", {}, h("strong", {}, j.message), j.notice), dismiss));
    }
    const e = j.error ?? {};
    return h("div", { class: "banner banner-critical" },
      h("div", { class: "row" },
        h("div", {}, h("strong", {}, `${verbs[j.action]} ${j.model_id} failed`), e.message ?? "Unknown error"),
        dismiss),
      e.detail && h("div", { class: "detail" }, e.detail),
      e.hint && h("div", { class: "hint" }, e.hint));
  });
  $("#jobs").replaceChildren(...els);
}

// ---------- recent inference -------------------------------------------------------------

async function refreshRequests() {
  let reqs;
  try { reqs = await api("/api/requests"); } catch { return; }
  setRows($("#req-rows"), reqs.slice(0, 8).map((r) => h("tr", { title: `${r.path} · ${fmtTime(r.ts)} · HTTP ${r.status}` },
    h("td", {}, h("span", { class: "sub" }, r.model ?? "–")),
    h("td", { class: "num" }, fmtMs(r.ttft_ms)),
    h("td", { class: "num" }, r.completion_tokens ?? "–"),
    h("td", { class: "num" }, r.tokens_per_sec ?? "–"),
  )), "No requests yet. Try the Playground.", 4);
}

// ---------- playground -------------------------------------------------------------------

const chat = []; // {role, content}
let chatAbort = null;

function fillModelSelect() {
  const sel = $("#chat-model");
  const prev = sel.value;
  const loaded = models.filter((m) => m.loaded && (m.task ?? "chat") === "chat");
  sel.replaceChildren(...(loaded.length
    ? loaded.map((m) => h("option", { value: m.id }, modelLabel(m)))
    : [h("option", { value: "" }, "No chat model loaded")]));
  if (loaded.some((m) => m.id === prev)) sel.value = prev;
  else { const npu = loaded.find((m) => m.device === "npu"); if (npu) sel.value = npu.id; }
  $("#chat-send").disabled = !loaded.length;
}

function addMessage(role, text) {
  $("#chat-empty")?.remove();
  const el = h("div", { class: `msg ${role}` }, text);
  $("#chat-log").append(el);
  el.scrollIntoView({ block: "end" });
  return el;
}

function setChatStats(ttft, tokens, tps) {
  $("#chat-stats").replaceChildren(
    h("dt", {}, "Time to first token"), h("dd", {}, fmtMs(ttft)),
    h("dt", {}, "Tokens"), h("dd", {}, tokens ?? "–"),
    h("dt", {}, "Tokens / sec"), h("dd", {}, tps ?? "–"));
}

async function sendChat(ev) {
  ev.preventDefault();
  const text = $("#chat-input").value.trim();
  const model = $("#chat-model").value;
  if (!text || !model || chatAbort) return;
  $("#chat-input").value = "";
  chat.push({ role: "user", content: text });
  addMessage("user", text);
  const out = addMessage("assistant", "…");
  $("#chat-send").textContent = "Stop";
  $("#chat-send").disabled = false;
  chatAbort = new AbortController();

  const sys = $("#sys-prompt").value.trim();
  const body = {
    model, stream: true, stream_options: { include_usage: true },
    temperature: Number($("#temp").value), max_tokens: Number($("#max-tokens").value),
    messages: [...(sys ? [{ role: "system", content: sys }] : []), ...chat],
  };
  const t0 = performance.now();
  let ttft = null, tokens = 0, exactTokens = null, reply = "";
  try {
    const res = await fetch("/v1/chat/completions", {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: chatAbort.signal,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err?.error?.message ?? `HTTP ${res.status}`);
    }
    const reader = res.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        const data = line.slice(5).trim();
        if (data === "[DONE]") continue;
        const msg = JSON.parse(data);
        if (msg.usage?.completion_tokens) exactTokens = msg.usage.completion_tokens;
        const delta = msg.choices?.[0]?.delta?.content;
        if (delta) {
          if (ttft == null) ttft = performance.now() - t0;
          tokens++;
          reply += delta;
          out.textContent = reply;
          out.scrollIntoView({ block: "end" });
        }
      }
    }
  } catch (e) {
    if (e.name !== "AbortError") { out.remove(); addMessage("error", `Request failed: ${e.message}`); }
  } finally {
    chatAbort = null;
    $("#chat-send").textContent = "Send";
    if (reply) chat.push({ role: "assistant", content: reply });
    else if (out.isConnected && out.textContent === "…") out.textContent = "(no output)";
    // Prefer the backend's exact count: streamed chunks can hold more than one token.
    const n = exactTokens ?? tokens;
    const genSecs = (performance.now() - t0 - (ttft ?? 0)) / 1000;
    setChatStats(ttft, n, n && genSecs > 0 ? (n / genSecs).toFixed(1) : null);
    refreshRequests();
  }
}

$("#chat-form").addEventListener("submit", (ev) => {
  if (chatAbort) { ev.preventDefault(); chatAbort.abort(); return; }
  sendChat(ev);
});
$("#chat-input").addEventListener("keydown", (ev) => {
  if (ev.key === "Enter" && !ev.shiftKey) { ev.preventDefault(); $("#chat-form").requestSubmit(); }
});
$("#chat-clear").addEventListener("click", () => {
  chat.length = 0;
  $("#chat-log").replaceChildren(h("div", { class: "empty", id: "chat-empty" }, "Conversation cleared."));
});

// ---------- API tab ----------------------------------------------------------------------

function renderApiTab() {
  const base = `${location.origin}/v1`;
  const model = models.find((m) => m.loaded)?.id ?? "<model-id>";
  $("#api-base").textContent = base;
  $("#snip-curl").textContent =
`curl ${base}/chat/completions \\
  -H "Content-Type: application/json" \\
  -d '{"model": "${model}",
       "messages": [{"role": "user", "content": "Hello!"}]}'`;
  $("#snip-py").textContent =
`from openai import OpenAI

client = OpenAI(base_url="${base}", api_key="not-needed")
resp = client.chat.completions.create(
    model="${model}",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)`;
}

document.querySelectorAll("[data-copy]").forEach((b) => b.addEventListener("click", async () => {
  await navigator.clipboard.writeText($(`#${b.dataset.copy}`).textContent);
  const old = b.textContent; b.textContent = "Copied"; setTimeout(() => (b.textContent = old), 1200);
}));

// ---------- boot -------------------------------------------------------------------------

startTelemetry().catch((e) => renderDevice(null, e.message));
refreshBackend().then(() => refreshModels()).then(pollJobs);
refreshRequests();
setInterval(refreshBackend, 15000);
setInterval(refreshRequests, 4000);
const TABS = [...tabButtons].map((b) => b.dataset.tab);
const tabFromHash = () => (TABS.includes(location.hash.slice(1)) ? location.hash.slice(1) : null);
let initialTab = tabFromHash();
try { initialTab ??= localStorage.getItem("tab"); } catch {}
showTab(TABS.includes(initialTab) ? initialTab : "dashboard");
addEventListener("hashchange", () => { const t = tabFromHash(); if (t) showTab(t); });
