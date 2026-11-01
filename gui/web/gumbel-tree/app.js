// Gumbel tree viewer: a fight as a line of moves; any decision on it, explained by /api/gumbel/decision.
// The URL holds the whole state, so every view is a link.
import { happened } from "./derive.js";
import { board, inspector, rootTable, timeline, tree } from "./render.js";

const $ = (id) => document.getElementById(id);
const SETTINGS = { sims: 64, particles: 8, gumbel_scale: 0, rollout_mix: 0.5, rescale_values: false };

const state = { run: "", iteration: null, fight: "", moves: [], at: 0, search: { ...SETTINGS }, inspect: null };
const responses = new Map();  // request → response; a decision is a pure function of its request
let seq = 0;  // drops stale renders

async function api(path, body) {
  const res = await fetch("../api/gumbel/" + path, body && { method: "POST", body: JSON.stringify(body) });
  const out = await res.json();
  if (!res.ok || out.error) throw new Error(out.error || `HTTP ${res.status}`);
  return out;
}

function request(k) {
  const { run, iteration, fight, search } = state;
  return { run, iteration, fight, moves: state.moves.slice(0, k), search };
}
async function decision(k) {
  const key = JSON.stringify(request(k));
  if (!responses.has(key)) responses.set(key, api("decision", request(k)));
  try { return await responses.get(key); } catch (e) { responses.delete(key); throw e; }
}
const known = (k) => responses.get(JSON.stringify(request(k)));

// ---------- state changes ----------

function play(bits) {
  state.moves = [...state.moves.slice(0, state.at), bits];
  state.at = state.moves.length;
  show();
}
function go(k) {
  state.at = Math.max(0, Math.min(state.moves.length, k));
  show();
}
async function next() {
  if (state.at < state.moves.length) return go(state.at + 1);
  const current = await decision(state.at);
  if (current.decision) play(current.decision.action);
}

// ---------- rendering ----------

async function show() {
  const my = ++seq;
  saveUrl();
  $("status").textContent = "explaining…";
  const t = performance.now();
  let current, after;
  try {
    current = await decision(state.at);
    after = state.at < state.moves.length ? await decision(state.at + 1) : null;
  } catch (e) {
    if (my === seq) $("status").textContent = "error: " + e.message;
    return;
  }
  if (my !== seq) return;
  $("status").textContent = `decision ${state.at} · ${((performance.now() - t) / 1000).toFixed(2)} s`;
  const happenedNode = after && current.decision ? happened(current, state.moves[state.at], after.observation) : null;
  state.inspect = happenedNode;
  board($("board"), current.view, current.over ? null : play);
  $("decision").hidden = $("treeSection").hidden = current.over;
  if (!current.over) {
    rootTable($("root"), current, play);
    tree($("tree"), current, { happenedNode, onInspect: (id) => { state.inspect = id; inspect(current); } });
  }
  inspect(current);
  renderTimeline(my);
}

function inspect(current) {
  $("inspector").hidden = state.inspect === null;
  if (state.inspect !== null) inspector($("inspector"), current, state.inspect, state.search.rollout_mix);
}

// Every decision on the line, filled in as they arrive (each is one request).
async function renderTimeline(my) {
  const line = [];
  for (let k = 0; k <= state.moves.length && my === seq; k++) {
    const response = known(k) ? await decision(k).catch(() => null) : null;
    if (!response) break;
    line.push({ response });
  }
  if (my !== seq) return;
  timeline($("timeline"), line, state.moves, state.at, go);
  const missing = line.length <= state.moves.length;
  if (missing && (await decision(line.length).catch(() => null)) && my === seq) renderTimeline(my);
}

// ---------- controls ----------

async function loadRuns() {
  const { runs } = await api("runs");
  $("run").innerHTML = runs.map((r) => `<option value="${r.run}">${r.date} · ${r.id}</option>`).join("");
  if (!runs.some((r) => r.run === state.run)) state.run = runs.length ? [...runs].sort((a, b) => a.date.localeCompare(b.date)).at(-1).run : "";
  $("run").value = state.run;
}
async function loadRun(keep) {
  const info = await api("run?" + new URLSearchParams({ run: state.run }));
  $("iteration").innerHTML = info.iterations.map((i) => `<option value="${i}">iter ${i}</option>`).join("");
  $("fight").innerHTML = info.fights.map((f) => `<option>${f}</option>`).join("");
  if (!keep || !info.iterations.includes(state.iteration)) state.iteration = info.iterations.at(-1);
  if (!keep || !info.fights.includes(state.fight)) { state.fight = info.fights[0]; state.moves = []; state.at = 0; }
  $("iteration").value = state.iteration;
  $("fight").value = state.fight;
}
function readSettings() {
  state.search = { sims: +$("sims").value, particles: +$("particles").value,
                   gumbel_scale: +$("noise").value, rollout_mix: +$("mix").value,
                   rescale_values: $("rescale").checked };
}
function writeSettings() {
  $("sims").value = state.search.sims; $("particles").value = state.search.particles;
  $("noise").value = state.search.gumbel_scale; $("mix").value = state.search.rollout_mix;
  $("rescale").checked = state.search.rescale_values;
}

function saveUrl() {
  const { run, iteration, fight, moves, at, search } = state;
  const q = new URLSearchParams({ run, iteration, fight, moves: moves.join(","), at, ...search });
  history.replaceState(null, "", "?" + q);
}
function readUrl() {
  const q = new URLSearchParams(location.search);
  state.run = q.get("run") || "";
  state.iteration = q.has("iteration") ? +q.get("iteration") : null;
  state.fight = q.get("fight") || "";
  state.moves = (q.get("moves") || "").split(",").filter(Boolean).map(Number);
  state.at = Math.min(+(q.get("at") ?? state.moves.length), state.moves.length);
  for (const [k, v] of Object.entries(SETTINGS))
    if (q.has(k)) state.search[k] = typeof v === "boolean" ? q.get(k) === "true" || q.get(k) === "1" : +q.get(k);
}

async function main() {
  readUrl();
  writeSettings();
  try {
    await loadRuns();
    if (!state.run) { $("status").textContent = "no runs found (gui/server.py --runs DIR)"; return; }
    await loadRun(true);
  } catch (e) { $("status").textContent = "error: " + e.message; return; }
  $("run").onchange = async () => { state.run = $("run").value; await loadRun(false); show(); };
  $("iteration").onchange = () => { state.iteration = +$("iteration").value; show(); };
  $("fight").onchange = () => { state.fight = $("fight").value; state.moves = []; state.at = 0; show(); };
  $("explain").onclick = () => { readSettings(); show(); };
  $("prev").onclick = () => go(state.at - 1);
  $("next").onclick = next;
  document.addEventListener("keydown", (e) => {
    if (e.target.matches("input, select")) return;
    if (e.key === "ArrowLeft") go(state.at - 1);
    if (e.key === "ArrowRight") next();
  });
  show();
}
main();
