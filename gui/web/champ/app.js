// Champ viewer: a recorded Champ fight, rebuilt server-side from {base, patch, ops} on every change.
// The fight is on its recording while ops are a prefix of its actions (and nothing is patched): the timeline
// position is then ops.length. Playing any other move, or editing, leaves the recording.
const $ = (id) => document.getElementById(id);
const ART = window.CARDS || {};
const ROOT = "../";
const END_TURN = 4;  // sts::search::ActionType in the top 3 bits of Action::bits
const actionType = (bits) => bits >>> 29;

let meta, rec = null, patch = {}, ops = [], view, pv, pre;
let seq = 0;  // drops stale responses
let playing = false, timer = null;
const pct = (p, d = 1) => (100 * p).toFixed(d) + "%";

async function api(path, options) {
  try {
    const res = await fetch(ROOT + path, options);
    let out;
    try { out = await res.json(); }
    catch { throw new Error(`${path}: HTTP ${res.status}, expected a JSON API response`); }
    if (!res.ok || out.error) throw new Error(`${path}: ${out.error || "HTTP " + res.status}`);
    return out;
  } catch (e) {
    const error = `API error: ${e.message}`;
    $("status").textContent = error;
    return { error };
  }
}
async function post(path, body) {
  return api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
}
async function get(path, params = {}) {
  return api(path + "?" + new URLSearchParams(params));
}

// ---------- cards ----------

function card(c, extraClass = "") {
  const info = meta.cardById[c.id];
  const art = ART[info.name] || {};
  const el = document.createElement("div");
  el.className = "card " + extraClass;
  el.dataset.rarity = info.rarity;
  el.dataset.type = info.type;
  const a = document.createElement("div");
  a.className = "art";
  if (art.art) a.innerHTML = `<img src="${ROOT}vis/${art.art}" alt="" loading="lazy">`;
  let cost = c.upgraded ? art.costUp : art.cost;
  let changed = false;
  if (c.cost_for_turn !== undefined && cost !== null && cost !== undefined && c.cost_for_turn >= 0 && c.cost_for_turn !== cost) {
    cost = c.cost_for_turn; changed = true;
  }
  if (cost !== null && cost !== undefined) a.insertAdjacentHTML("beforeend", `<div class="cost${changed ? " changed" : ""}">${cost < 0 ? "X" : cost}</div>`);
  if (c.count > 1) a.insertAdjacentHTML("beforeend", `<div class="count">×${c.count}</div>`);
  el.append(a);
  const t = document.createElement("div");
  t.className = "name";
  t.textContent = info.name;
  if (c.upgraded) t.insertAdjacentHTML("beforeend", `<span class="up">${c.upgraded > 1 ? "+" + c.upgraded : "+"}</span>`);
  el.append(t);
  return el;
}
const cardName = (c) => meta.cardById[c.id].name + (c.upgraded ? "+" : "");
const nice = (s) => s ? s.replace(/^THE_CHAMP_/, "").toLowerCase().replaceAll("_", " ") : "–";

// ---------- recording / timeline ----------

// Length of the recorded prefix that ops follow; equals ops.length while on the recording.
function recPrefix() {
  if (!rec || Object.keys(patch).length) return 0;
  let k = 0;
  while (k < ops.length && k < rec.actions.length && ops[k].act === rec.actions[k]) k++;
  return k;
}
const onRec = () => rec && Object.keys(patch).length === 0 && recPrefix() === ops.length;
const nextRecorded = () => onRec() && ops.length < rec.actions.length ? rec.actions[ops.length] : null;
function turnStarts() {
  const s = [0];
  rec.actions.forEach((a, i) => { if (actionType(a) === END_TURN) s.push(i + 1); });
  return s;
}
function goTo(k) {
  if (!rec) return;
  k = Math.max(0, Math.min(rec.actions.length, k));
  patch = {}; pre = startPre();
  ops = rec.actions.slice(0, k).map((a) => ({ act: a }));
  refresh();
}
function pos() { return onRec() ? ops.length : recPrefix(); }
function prevTurn() { const p = pos(); goTo(Math.max(0, ...turnStarts().filter((s) => s < p))); }
function nextTurn() { const p = pos(); goTo(turnStarts().find((s) => s > p) ?? rec.actions.length); }

function setPlaying(on) {
  playing = on && !!rec;
  clearTimeout(timer);
  if (playing && !onRec()) goTo(recPrefix());  // resume the recording from where you left it
  else if (playing) schedule();
  renderTimeline();
}
function schedule() {
  clearTimeout(timer);
  if (!playing) return;
  if (!onRec() || ops.length >= rec.actions.length) return setPlaying(false);
  timer = setTimeout(() => goTo(ops.length + 1), 1000 / +$("tSpeed").value);
}

function renderTimeline() {
  const n = rec ? rec.actions.length : 0;
  ["tStart", "tPrevTurn", "tPrev", "tPlay", "tNext", "tNextTurn", "tSlider"].forEach((id) => $(id).disabled = !rec);
  $("tSlider").max = n;
  $("tSlider").value = pos();
  $("tPlay").textContent = playing ? "⏸" : "▶";
  if (!rec) { $("tPos").textContent = "no fight loaded"; return; }
  const nb = nextRecorded();
  $("tPos").textContent = onRec()
    ? `move ${ops.length} / ${n}` + (nb !== null ? ` · next: ${describeBits(nb)}` : ` · end: ${rec.won ? "won, " + rec.final_hp + " HP" : "lost"}`)
    : `your line (left the recording at move ${recPrefix()}${Object.keys(patch).length ? ", setup patched" : ""})`;
}
function describeBits(bits) {
  const l = view && view.legal.find((x) => x.bits === bits);
  return l ? moveLabel(l) : String(bits);
}

// ---------- server round trips ----------

async function refresh() {
  const my = ++seq;
  const out = await post("api/champ/query", { base: rec.base, patch, ops, query: "pv" });
  if (my !== seq) return;
  if (out.error) { $("status").textContent = out.error; ops.pop(); setPlaying(false); return; }  // a rejected op is dropped
  view = out.view; pv = out.pv || null;
  render();
  schedule();
}

// ---------- ops ----------

function act(bits) { ops.push({ act: bits }); if (!onRec()) setPlaying(false); refresh(); }
function edit(op) {
  if (view.kind !== "play") { $("status").textContent = "edits only at a normal play decision"; return; }
  setPlaying(false);
  ops.push(op);
  refresh();
}

// ---------- render ----------

function moveLabel(l) {
  if (l.type === "card") return "play " + cardName(view.hand[l.source]);
  if (l.type === "potion") return "drink " + view.potions[l.source];
  if (l.type === "end_turn") return "end turn";
  return l.desc.replace(/^\{\s*|\s*\}$/g, "");
}

function render() {
  const v = view;
  document.body.classList.toggle("editing", $("editMode").checked);
  // champ
  const c = v.champ;
  const pips = (st) => Object.entries(st).map(([k, n]) =>
    `<span class="pip ${k === "strength" ? "str" : k.startsWith("vuln") ? "vuln" : k === "weak" ? "weak" : ""}">${k.replaceAll("_", " ")} ${n}</span>`).join("");
  $("champ").innerHTML = `<div class="who">The Champ ${c.phase2 ? '<span class="pip str">phase 2</span>' : ""}</div>
    <div class="bar"><i style="width:${(100 * c.hp) / c.max_hp}%"></i></div>
    <div class="nums"><b>${c.hp}</b> / ${c.max_hp} HP ${c.block ? `· <span class="pip block">block ${c.block}</span>` : ""}
      · defensive stances used ${c.stances_used} · turn ${v.turn}</div>
    <div class="intent"><span class="what">${nice(c.intent)}</span>
      ${c.intent_damage !== undefined ? `<span class="dmg">${c.intent_damage}${c.intent_hits > 1 ? "×" + c.intent_hits : ""}</span>` : ""}
      <span class="hint">last: ${nice(c.last_move)}</span></div>
    <div class="pips">${pips(c.statuses)}</div>`;
  // player
  const p = v.player;
  const legalPotion = new Map(v.legal.filter((l) => l.type === "potion").map((l) => [l.source, l]));
  $("player").innerHTML = `<span class="stat">HP <b>${p.hp}</b> / ${p.max_hp}</span>
    <span class="stat">Block <b>${p.block}</b></span>
    <span class="stat energy">Energy <b>${p.energy}</b> / ${p.energy_per_turn}</span>
    <span class="pips">${pips(p.statuses)}</span>
    <div class="relics">${v.relics.map((r) => `<span class="relic">${r}</span>`).join("")}
      ${v.potions.map((x, i) => x ? `<span class="relic potion${legalPotion.has(i) ? " use" : ""}" data-slot="${i}" title="click to drink">${x}</span>` : `<span class="relic potion empty">empty</span>`).join("")}</div>`;
  $("player").querySelectorAll(".potion.use").forEach((el) => el.onclick = () => act(legalPotion.get(+el.dataset.slot).bits));
  // verdict
  const done = v.kind === "done";
  $("pwin").innerHTML = done ? `<span class="done-banner ${v.won ? "won" : "lost"}">${v.won ? "Won" : "Lost"}</span>` :
    pv ? pct(pv.value / 1, 1) : "–";
  // hand
  const next = nextRecorded();
  const prior = (bits) => pv && pv.priors[bits] !== undefined ? pv.priors[bits] : null;
  const legalCard = new Map();
  v.legal.filter((l) => l.type === "card").forEach((l) => { if (!legalCard.has(l.source)) legalCard.set(l.source, l); });
  const nextCard = next !== null ? v.legal.find((l) => l.bits === next && l.type === "card") : null;
  $("hand").replaceChildren(...v.hand.map((h, i) => {
    const l = legalCard.get(i);
    const el = card(h, l ? "" : "illegal");
    if (nextCard && nextCard.source === i) el.classList.add("best");
    const pr = l ? prior(l.bits) : null;
    el.insertAdjacentHTML("beforeend", `<div class="badges"><span class="pv">${pr === null ? "" : pct(pr, 0)}</span></div>`);
    el.onclick = () => {
      if ($("editMode").checked) return pileClick("hand", i);
      if (l) act(l.bits);
    };
    return el;
  }), ...ghosts(v.hand.length));
  // buttons
  const end = v.legal.find((l) => l.type === "end_turn");
  $("endTurn").disabled = !end;
  $("endTurn").onclick = () => end && act(end.bits);
  $("undo").disabled = ops.length === 0;
  // moves table
  const rows = v.legal.map((l) => ({ l, pr: prior(l.bits) }));
  rows.sort((a, b) => (b.pr ?? -1) - (a.pr ?? -1));
  const bar = (x) => x === null ? "" : `<span class="shr" style="width:${Math.round(60 * x)}px"></span>${pct(x, 1)}`;
  $("moves").innerHTML = `<tr><th>Move</th><th>PV prior</th><th>Recorded</th><th></th></tr>` +
    rows.map(({ l, pr }) => `<tr class="${l.bits === next ? "chosen" : ""}">
      <td class="desc" title="${l.desc}">${moveLabel(l)}</td><td>${bar(pr)}</td>
      <td>${l.bits === next ? "◀ next" : ""}</td><td><button data-bits="${l.bits}">play</button></td></tr>`).join("") +
    (done ? `<tr><td colspan="4" class="hint">fight over: ${v.won ? "won with " + p.hp + " HP" : "lost"}</td></tr>` : "");
  $("moves").querySelectorAll("button[data-bits]").forEach((b) => b.onclick = () => act(+b.dataset.bits));
  // piles (draw grouped: order hidden)
  const group = (list) => {
    const m = new Map();
    list.forEach((x, i) => { const k = x.id + "/" + x.upgraded; if (!m.has(k)) m.set(k, { ...x, count: 0, idx: i }); m.get(k).count++; });
    return [...m.values()];
  };
  const pile = (name, list, grouped) => {
    const items = grouped ? group(list) : list.map((x, i) => ({ ...x, idx: i }));
    $(name).replaceChildren(...items.map((x) => {
      const el = card(x);
      el.onclick = () => $("editMode").checked && pileClick(name, x.idx);
      return el;
    }));
    $("n" + name[0].toUpperCase() + name.slice(1)).textContent = `— ${list.length}`;
  };
  pile("draw", v.draw, true); pile("discard", v.discard, false); pile("exhaust", v.exhaust, false);
  renderEdit();
  renderTimeline();
  saveUrl();
}

// Invisible cards filling the hand to 10 slots, so its height never changes.
function ghosts(n) {
  const strike = meta.cards.find((c) => c.name === "Strike") || meta.cards[0];
  return Array.from({ length: Math.max(0, 10 - n) }, () => {
    const el = card({ id: strike.id, upgraded: 0 }, "ghost");
    el.insertAdjacentHTML("beforeend", `<div class="badges"></div>`);
    return el;
  });
}

// The draw pile is shown sorted; the server addresses it in its true (hidden) order, so map back by card.
function pileClick(from, idx) {
  const to = $("moveTo").value;
  if (to === from) return;
  const at = from === "draw" ? { from, match: [view.draw[idx].id, view.draw[idx].upgraded] } : { from, index: idx };
  edit(to === "remove" ? { remove: at } : { move: { ...at, to } });
}

const PLAYER_FIELDS = ["hp", "max_hp", "block", "energy", "strength", "dexterity", "vulnerable", "weak", "frail", "metallicize", "demon_form", "artifact"];
const CHAMP_FIELDS = ["hp", "max_hp", "block", "strength", "metallicize", "vulnerable", "weak", "artifact", "phase2", "stances_used"];
function renderEdit() {
  const on = $("editMode").checked;
  $("editPanel").hidden = !on;
  if (!on) return;
  const p = view.player, c = view.champ;
  const val = (o, f) => o[f] !== undefined ? o[f] : (o.statuses[f] || 0);
  const num = (who, f, v) => `<label>${f.replaceAll("_", " ")}<input type="number" data-key="${who}.${f}" value="${v}"></label>`;
  const moveSel = (k, cur) => `<select data-key="${k}"><option value="INVALID"${cur ? "" : " selected"}>–</option>${meta.moves.map((m) => `<option value="${m}"${m === cur ? " selected" : ""}>${nice(m)}</option>`).join("")}</select>`;
  $("editFields").innerHTML = `<h4>Player</h4>` + PLAYER_FIELDS.map((f) => num("player", f, val(p, f))).join("") +
    `<h4>Champ</h4>` + CHAMP_FIELDS.map((f) => num("champ", f, val(c, f))).join("") +
    `<label>intent ${moveSel("champ.intent", c.intent)}</label><label>last move ${moveSel("champ.last_move", c.last_move)}</label>` +
    `<label>turn<input type="number" data-key="turn" value="${view.turn}"></label>`;
  $("editFields").querySelectorAll("[data-key]").forEach((el) => el.onchange = () =>
    edit({ set: { [el.dataset.key]: el.tagName === "SELECT" ? el.value : +el.value } }));
}

// ---------- URL: ?date=&id=&part=&fight= picks the fight; #<base64 JSON> holds patch / ops ----------

function saveUrl() {
  if (!rec) return;
  const [date, id, part, fight] = rec.base.split("|");
  const q = new URLSearchParams({ date, id, ...(part ? { part } : {}), fight });
  const st = { patch, ops, e: $("editMode").checked ? 1 : 0, m: $("moveTo").value, v: +$("tSpeed").value };
  const enc = btoa(JSON.stringify(st)).replaceAll("+", "-").replaceAll("/", "_").replace(/=+$/, "");
  const url = "?" + q + "#" + enc;
  if (location.search + location.hash !== url) history.replaceState(null, "", url);
}
function readHash() {
  const enc = location.hash.slice(1);
  if (!enc) return {};
  try { return JSON.parse(atob(enc.replaceAll("-", "+").replaceAll("_", "/"))); } catch { return {}; }
}

// ---------- pre-battle setup ----------

function startPre() {
  const s = rec.start;
  return { deck: structuredClone(s.deck), relics: structuredClone(s.relics), potions: [...s.potions], hp: s.hp, max_hp: s.max_hp, seed: s.seed };
}
function renderSetup() {
  $("preHp").value = pre.hp; $("preMax").value = pre.max_hp; $("preSeed").value = pre.seed;
  $("preRelics").replaceChildren(
    ...pre.relics.map((r, i) => chip(meta.relicById[r.id].name, "", () => { pre.relics.splice(i, 1); renderSetup(); })),
    ...pre.potions.map((x, i) => x > 1 ? chip(meta.potById[x].name, "potion", () => { pre.potions[i] = 1; renderSetup(); }) : null).filter(Boolean));
  const groups = new Map();
  pre.deck.forEach((c, i) => { const k = `${c.id}/${c.upgraded}`; if (!groups.has(k)) groups.set(k, []); groups.get(k).push(i); });
  $("preDeck").replaceChildren(...[...groups.values()].map((idx) => {
    const c = pre.deck[idx[0]];
    const el = card({ ...c, count: idx.length });
    el.onclick = (e) => {
      if (e.shiftKey) idx.forEach((i) => pre.deck[i].upgraded = pre.deck[i].upgraded ? 0 : 1);
      else pre.deck.splice(idx[idx.length - 1], 1);
      renderSetup();
    };
    return el;
  }));
  $("preSize").textContent = `— ${pre.deck.length} cards`;
}
function chip(text, klass, onclick) {
  const el = document.createElement("span");
  el.className = "relic " + klass; el.textContent = text; el.title = "click to remove"; el.onclick = onclick;
  return el;
}
function applySetup() {
  pre.hp = +$("preHp").value; pre.max_hp = +$("preMax").value; pre.seed = +$("preSeed").value;
  const s = rec.start, same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  patch = {};
  if (!same(pre.deck, s.deck)) patch.deck = pre.deck;
  if (!same(pre.relics, s.relics)) patch.relics = pre.relics;
  if (!same(pre.potions, s.potions)) patch.potions = pre.potions;
  if (pre.hp !== s.hp) patch.hp = pre.hp;
  if (pre.max_hp !== s.max_hp) patch.max_hp = pre.max_hp;
  if (pre.seed !== s.seed) patch.seed = pre.seed;
  ops = [];
  setPlaying(false);
  refresh();
}

// ---------- fight picker ----------

const dataset = () => meta.datasets[+$("runSel").value];
function fillDates() {
  $("dateSel").innerHTML = [...new Set(meta.datasets.map((d) => d.date))].reverse().map((d) => `<option>${d}</option>`).join("");
}
function fillRuns() {
  $("runSel").innerHTML = meta.datasets.map((d, i) => d.date === $("dateSel").value ?
    `<option value="${i}">${d.id}${d.part ? " / " + d.part : ""}</option>` : "").join("");
}
async function fillFights() {
  const d = dataset();
  if (!d) { $("fightHint").textContent = "No recorded fight datasets found"; return; }
  $("fightSel").innerHTML = "";
  $("fightHint").textContent = "loading fights…";
  const out = await get("api/champ/fights", d);
  if (d !== dataset()) return;
  if (out.error) { $("fightHint").textContent = out.error; return; }
  const w = out.fights.filter((f) => f.won).length;
  $("fightHint").textContent = out.fights.length ? `${out.fights.length} Champ fights, ${w} won` : "no Champ fights in this run";
  $("fightSel").innerHTML = out.fights.map((f) =>
    `<option value="${f.fight_id}">${f.won ? "won " + f.final_hp + " HP" : "lost"} · ${f.n} moves · ${f.fight_id}</option>`).join("");
}
async function selectFromQuery() {
  const q = new URLSearchParams(location.search);
  const i = meta.datasets.findIndex((d) => d.date === q.get("date") && d.id === q.get("id") && d.part === (q.get("part") || ""));
  if (i < 0) { fillRuns(); await fillFights(); return false; }
  $("dateSel").value = meta.datasets[i].date; fillRuns(); $("runSel").value = i;
  await fillFights();
  if (q.get("fight")) $("fightSel").value = q.get("fight");
  return !!q.get("fight");
}
async function load(restore = {}) {
  const fid = $("fightSel").value;
  if (!fid) return;
  setPlaying(false);
  const out = await get("api/champ/fight", { ...dataset(), fight: fid });
  if (out.error) { $("status").textContent = out.error; return; }
  rec = out;
  $("status").textContent = `agent: ${rec.agent || "?"}`;
  patch = restore.patch || {}; ops = restore.ops || [];
  pre = Object.assign(startPre(), patch);
  renderSetup();
  await refresh();
}

// ---------- startup ----------

(async () => {
  meta = await get("api/champ/meta");
  if (meta.error) return;
  const by = (list, k = "id") => Object.fromEntries(list.map((x) => [x[k], x]));
  meta.cardById = by(meta.cards); meta.relicById = by(meta.relics); meta.potById = by(meta.potions);
  const byName = (list) => Object.fromEntries(list.map((x) => [x.name.toLowerCase(), x]));
  const cardByName = byName(meta.cards), relicByName = byName(meta.relics), potByName = byName(meta.potions);
  const addable = meta.cards.filter((c) => ["red", "colorless", "curse"].includes(c.color) || c.type === "status");
  $("cardList").innerHTML = addable.map((c) => `<option value="${c.name}">`).join("");
  $("relicList").innerHTML = meta.relics.map((c) => `<option value="${c.name}">`).join("");
  $("potList").innerHTML = meta.potions.map((c) => `<option value="${c.name}">`).join("");

  $("dateSel").onchange = () => { fillRuns(); fillFights(); };
  $("runSel").onchange = fillFights;
  $("load").onclick = () => load();
  $("fightSel").onchange = () => load();
  $("undo").onclick = () => { if (ops.length) { ops.pop(); setPlaying(false); refresh(); } };
  $("restart").onclick = () => { ops = []; setPlaying(false); refresh(); };
  $("editMode").onchange = render;
  $("share").onclick = async () => {
    await navigator.clipboard.writeText(location.href).catch(() => {});
    $("status").textContent = "link copied (also in the address bar)";
  };
  $("cardAdd").onclick = () => {
    const c = cardByName[$("cardIn").value.trim().toLowerCase()];
    const to = $("moveTo").value;
    if (!c || to === "remove") { $("status").textContent = "pick a listed card and a pile"; return; }
    edit({ add: { id: c.id, upgraded: $("cardUp").checked, to } });
  };
  $("preCardAdd").onclick = () => {
    const c = cardByName[$("preCardIn").value.trim().toLowerCase()];
    if (c) { pre.deck.push({ id: c.id, upgraded: $("preCardUp").checked ? 1 : 0, misc: 0 }); $("preCardIn").value = ""; renderSetup(); }
  };
  $("relicAdd").onclick = () => {
    const r = relicByName[$("relicIn").value.trim().toLowerCase()];
    if (r) { pre.relics.push({ id: r.id, data: 0 }); $("relicIn").value = ""; renderSetup(); }
  };
  $("potAdd").onclick = () => {
    const x = potByName[$("potIn").value.trim().toLowerCase()];
    if (!x) return;
    const empty = pre.potions.indexOf(1);
    if (empty >= 0) pre.potions[empty] = x.id; else if (pre.potions.length < 5) pre.potions.push(x.id);
    $("potIn").value = ""; renderSetup();
  };
  $("applySetup").onclick = applySetup;
  // timeline
  $("tStart").onclick = () => { setPlaying(false); goTo(0); };
  $("tPrev").onclick = () => { setPlaying(false); goTo(pos() - 1); };
  $("tNext").onclick = () => { setPlaying(false); goTo(onRec() ? ops.length + 1 : recPrefix()); };
  $("tPrevTurn").onclick = () => { setPlaying(false); prevTurn(); };
  $("tNextTurn").onclick = () => { setPlaying(false); nextTurn(); };
  $("tPlay").onclick = () => setPlaying(!playing);
  $("tSlider").oninput = () => { setPlaying(false); goTo(+$("tSlider").value); };
  $("tSpeed").onchange = () => { saveUrl(); schedule(); };
  document.addEventListener("keydown", (e) => {
    if (!rec || e.target.closest("input, select, textarea")) return;
    if (e.key === " ") { e.preventDefault(); setPlaying(!playing); }
    else if (e.key === "ArrowRight") $("tNext").click();
    else if (e.key === "ArrowLeft") $("tPrev").click();
  });

  const st = readHash();
  if (st.e) $("editMode").checked = true;
  if (st.m) $("moveTo").value = st.m;
  if (st.v) $("tSpeed").value = st.v;
  fillDates();
  renderTimeline();
  if (await selectFromQuery()) await load(st);
})().catch((e) => { console.error(e); $("status").textContent = "failed to load: " + e; });
