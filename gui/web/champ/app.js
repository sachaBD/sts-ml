// Champ viewer: a sandbox Champ battle rebuilt server-side from {base, patch, ops} on every change.
const $ = (id) => document.getElementById(id);
const ART = window.CARDS || {};
const ROOT = "../";
const END_TURN = 4;  // sts::search::ActionType in the top 3 bits of Action::bits
const actionType = (bits) => bits >>> 29;

let meta, deck, base, patch = {}, ops = [], undos = 0, view, pv, live = {}, teacherRec = null, replayK = null, pre;
let seq = 0;  // drops stale responses
const recorded = new Set();
const pct = (p, d = 1) => (100 * p).toFixed(d) + "%";
const key = () => JSON.stringify([base, patch, ops]);

async function post(path, body) {
  const res = await fetch(ROOT + path, { method: "POST", body: JSON.stringify(body) });
  return res.json();
}
let busy = 0;
function spin(delta, text = "") {
  busy += delta;
  $("spin").hidden = busy === 0;
  if (text || busy === 0) $("status").textContent = text;
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

// ---------- server round trips ----------

async function refresh() {
  const my = ++seq;
  const out = await post("api/champ/query", { base, patch, ops, query: "pv" });
  if (my !== seq) return;
  if (out.error) { $("status").textContent = out.error; ops.pop(); return; }  // a rejected op is dropped
  view = out.view; pv = out.pv || null;
  render();
  if (view.kind === "done") recordIfBench();
  else if ($("autoTeacher").checked && !live[key()]) askTeacher();
}

async function askTeacher() {
  const k = key();
  if (view.kind === "done" || live[k]) return render();
  spin(1, "teacher searching…");
  const out = await post("api/champ/query", { base, patch, ops, query: "search", sims: 20000 });
  spin(-1);
  if (out.error) { $("status").textContent = out.error; return; }
  live[k] = out.search;
  if (k === key()) render();
}

async function playouts() {
  const n = +$("nPlay").value, sims = +$("simsPlay").value;
  const k = key();
  $("playoutRes").textContent = "";
  spin(1, `${n} teacher playouts at ${sims} sims (2 cores)…`);
  const out = await post("api/champ/query", { base, patch, ops, query: "playout", n, sims });
  spin(-1);
  if (out.error) { $("status").textContent = out.error; return; }
  if (k !== key()) return;
  $("playoutRes").innerHTML = `teacher playouts: <b>${out.wins}/${out.n}</b> = ${pct(out.p_win, 0)}
    <br>95% CI ${pct(out.ci95[0], 0)}–${pct(out.ci95[1], 0)}` +
    (out.mean_hp_if_won !== null ? `<br>HP if won ${out.mean_hp_if_won.toFixed(1)}` : "") +
    `<br><span class="hint">${out.sims} sims · ${out.wall_seconds.toFixed(0)} s</span>`;
}

async function recordIfBench() {
  const k = key();
  if (recorded.has(k) || replayK !== null) return;
  recorded.add(k);
  const out = await post("api/champ/record", { base, patch, ops, undos, won: view.won, final_hp: view.player.hp });
  $("status").textContent = out.error ? out.error : `fight recorded${out.edited ? " (edited: not in the tally)" : ""}`;
  loadTally();
}

async function loadTally() {
  const t = await (await fetch(ROOT + "api/champ/tally")).json();
  $("tally").textContent = t.n ? `your unedited bench fights: you ${t.user_wins}/${t.n} · teacher ${t.teacher_wins}/${t.n}` +
    ` (only you ${t.user_only}, only teacher ${t.teacher_only}; ${t.with_undo} used undo)` : "";
}

// ---------- ops ----------

function act(bits) {
  if (replayK !== null) { replayK = null; renderReplay(); }
  ops.push({ act: bits });
  refresh();
}
function edit(op) {
  if (view.kind !== "play") { $("status").textContent = "edits only at a normal play decision"; return; }
  if (replayK !== null) { replayK = null; renderReplay(); }
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

function teacherStats() {
  const s = live[key()];
  if (s) return { moves: s.moves, chosen: s.chosen, src: `live search: ${s.simulations} sims, ${s.seconds.toFixed(1)} s` };
  if (replayK !== null && teacherRec) {
    const r = teacherRec.search[replayK];
    if (r) return { moves: r.moves, chosen: teacherRec.actions[replayK], src: `recorded search (step ${replayK})` };
  }
  return null;
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
      ${v.potions.map((x, i) => x ? `<span class="relic potion${legalPotion.has(i) ? " use" : ""}" data-slot="${i}" title="click to drink">${x}</span>` : "").join("")}</div>`;
  $("player").querySelectorAll(".potion.use").forEach((el) => el.onclick = () => act(legalPotion.get(+el.dataset.slot).bits));
  // verdict
  const done = v.kind === "done";
  $("pwin").innerHTML = done ? `<span class="done-banner ${v.won ? "won" : "lost"}">${v.won ? "Won" : "Lost"}</span>` :
    pv ? pct(pv.value / 1, 1) : "–";
  // hand
  const ts = teacherStats();
  const total = ts ? Object.values(ts.moves).reduce((a, m) => a + m.visits, 0) : 0;
  const share = (bits) => ts && ts.moves[bits] ? ts.moves[bits].visits / total : (ts ? 0 : null);
  const prior = (bits) => pv && pv.priors[bits] !== undefined ? pv.priors[bits] : null;
  const legalCard = new Map();
  v.legal.filter((l) => l.type === "card").forEach((l) => { if (!legalCard.has(l.source)) legalCard.set(l.source, l); });
  $("hand").replaceChildren(...v.hand.map((h, i) => {
    const l = legalCard.get(i);
    const el = card(h, l ? "" : "illegal");
    if (l) {
      if (ts && ts.chosen === l.bits) el.classList.add("best");
      const pr = prior(l.bits), sh = share(l.bits);
      el.insertAdjacentHTML("beforeend", `<div class="badges"><span class="pv">${pr === null ? "" : pct(pr, 0)}</span>
        <span class="tv">${sh === null ? "" : pct(sh, 0)}</span></div>`);
    }
    el.onclick = () => {
      if ($("editMode").checked) return pileClick("hand", i);
      if (l) act(l.bits);
    };
    return el;
  }));
  // buttons
  const end = v.legal.find((l) => l.type === "end_turn");
  $("endTurn").disabled = !end;
  $("endTurn").onclick = () => end && act(end.bits);
  $("undo").disabled = ops.length === 0;
  $("askTeacher").disabled = done; $("playout").disabled = done;
  // moves table
  $("teacherHint").textContent = (ts ? ts.src : "no teacher search yet") +
    (pv ? " · PV prior from " + meta.pv_model.split("/").slice(-3, -2)[0] : (done ? "" : " · PV unavailable"));
  const rows = v.legal.map((l) => ({ l, pr: prior(l.bits), sh: share(l.bits), q: ts && ts.moves[l.bits] ? ts.moves[l.bits].value : null }));
  rows.sort((a, b) => (b.sh ?? -1) - (a.sh ?? -1) || (b.pr ?? -1) - (a.pr ?? -1));
  const bar = (x, cls) => x === null ? "" : `<span class="shr ${cls}" style="width:${Math.round(60 * x)}px"></span>${pct(x, 1)}`;
  $("moves").innerHTML = `<tr><th>Move</th><th>PV prior</th><th>Teacher visits</th><th>Teacher Q</th><th></th></tr>` +
    rows.map(({ l, pr, sh, q }) => `<tr class="${ts && ts.chosen === l.bits ? "chosen" : ""}">
      <td class="desc" title="${l.desc}">${moveLabel(l)}</td><td>${bar(pr, "pv")}</td><td>${bar(sh, "")}</td>
      <td class="band">${q === null ? "" : q.toFixed(3)}</td><td><button data-bits="${l.bits}">play</button></td></tr>`).join("") +
    (done ? `<tr><td colspan="5" class="hint">fight over: ${v.won ? "won with " + p.hp + " HP" : "lost"}</td></tr>` : "");
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
  renderReplay();
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

// ---------- teacher replay ----------

function renderReplay() {
  const show = teacherRec && Object.keys(patch).length === 0;
  $("replay").hidden = !show;
  if (!show) return;
  const n = teacherRec.actions.length;
  $("replayHint").textContent = `${teacherRec.won ? "won" : "lost"} · final HP ${teacherRec.final_hp} · ${n} decisions`;
  $("repPos").textContent = replayK === null ? "(your line)" : `decision ${replayK} / ${n}` +
    (replayK < n ? ` · teacher plays: ${describeBits(teacherRec.actions[replayK])}` : " · end");
}
function describeBits(bits) {
  const l = view && view.legal.find((x) => x.bits === bits);
  return l ? moveLabel(l) : String(bits);
}
function replayTo(k) {
  const n = teacherRec.actions.length;
  replayK = Math.max(0, Math.min(n, k));
  ops = teacherRec.actions.slice(0, replayK).map((a) => ({ act: a }));
  refresh();
}
function replayNextTurn() {
  let k = replayK ?? 0;
  const a = teacherRec.actions;
  while (k < a.length && actionType(a[k]) !== END_TURN) k++;
  replayTo(k + 1);
}

// ---------- pre-battle setup ----------

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
  const orig = meta.fightById[base];
  patch = {};
  const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
  if (!same(pre.deck, deck.deck)) patch.deck = pre.deck;
  if (!same(pre.relics, deck.relics)) patch.relics = pre.relics;
  if (!same(pre.potions, deck.potions)) patch.potions = pre.potions;
  if (pre.hp !== deck.hp) patch.hp = pre.hp;
  if (pre.max_hp !== deck.max_hp) patch.max_hp = pre.max_hp;
  if (pre.seed !== orig.seed) patch.seed = pre.seed;
  ops = []; undos = 0; replayK = null;
  refresh();
}

// ---------- start picker ----------

function inFilter(d) {
  const f = $("filter").value;
  if (f === "all") return true;
  const [lo, hi] = f.split("-").map(Number);
  return d.wins >= lo && d.wins <= (hi ?? lo);
}
function deckLabel(d) {
  const names = d.deck.map((c) => meta.cardById[c.id]).filter((c) => !["basic", "curse", "special"].includes(c.rarity))
    .map((c) => c.name);
  return `${d.wins}/5 · ${d.hp}/${d.max_hp} HP · ${d.deck.length} cards · ${[...new Set(names)].slice(0, 6).join(", ")}`;
}
function fillDecks() {
  const list = meta.decks.filter(inFilter);
  $("deckSel").innerHTML = list.map((d) => `<option value="${d.source}">${deckLabel(d)}</option>`).join("");
  fillSeeds();
}
function fillSeeds() {
  const d = meta.deckBySource[$("deckSel").value];
  $("seedSel").innerHTML = d ? d.fights.map((f, i) => `<option value="${f.fight_id}">#${i + 1} teacher ${f.won ? "won (" + f.final_hp + " HP)" : "lost"}</option>`).join("") : "";
}
async function load() {
  deck = meta.deckBySource[$("deckSel").value];
  base = $("seedSel").value;
  if (!base) return;
  patch = {}; ops = []; undos = 0; replayK = null; teacherRec = null;
  pre = { deck: structuredClone(deck.deck), relics: structuredClone(deck.relics), potions: [...deck.potions],
          hp: deck.hp, max_hp: deck.max_hp, seed: meta.fightById[base].seed };
  renderSetup();
  history.replaceState(null, "", "#" + base);
  await refresh();
  teacherRec = await (await fetch(ROOT + "api/champ/teacher/" + base)).json();
  renderReplay();
}

// ---------- startup ----------

(async () => {
  meta = await (await fetch(ROOT + "api/champ/meta")).json();
  const by = (list, k = "id") => Object.fromEntries(list.map((x) => [x[k], x]));
  meta.cardById = by(meta.cards); meta.relicById = by(meta.relics); meta.potById = by(meta.potions);
  meta.deckBySource = by(meta.decks, "source");
  meta.fightById = {};
  meta.decks.forEach((d) => d.fights.forEach((f) => meta.fightById[f.fight_id] = { ...f, source: d.source }));
  const byName = (list) => Object.fromEntries(list.map((x) => [x.name.toLowerCase(), x]));
  const cardByName = byName(meta.cards), relicByName = byName(meta.relics), potByName = byName(meta.potions);
  const addable = meta.cards.filter((c) => ["red", "colorless", "curse"].includes(c.color) || c.type === "status");
  $("cardList").innerHTML = addable.map((c) => `<option value="${c.name}">`).join("");
  $("relicList").innerHTML = meta.relics.map((c) => `<option value="${c.name}">`).join("");
  $("potList").innerHTML = meta.potions.map((c) => `<option value="${c.name}">`).join("");

  $("filter").onchange = fillDecks;
  $("deckSel").onchange = fillSeeds;
  $("load").onclick = load;
  $("undo").onclick = () => {
    if (!ops.length) return;
    ops.pop(); undos++;
    if (replayK !== null) replayK = ops.length;
    refresh();
  };
  $("restart").onclick = () => { ops = []; replayK = null; refresh(); };
  $("editMode").onchange = render;
  $("askTeacher").onclick = askTeacher;
  $("autoTeacher").onchange = () => $("autoTeacher").checked && askTeacher();
  $("playout").onclick = playouts;
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
  $("repStart").onclick = () => replayTo(0);
  $("repPrev").onclick = () => replayTo((replayK ?? ops.length) - 1);
  $("repNext").onclick = () => replayTo((replayK ?? 0) + 1);
  $("repTurn").onclick = replayNextTurn;

  if (new URLSearchParams(location.search).has("edit")) $("editMode").checked = true;
  fillDecks();
  const hash = location.hash.slice(1);
  if (hash && meta.fightById[hash]) {
    const src = meta.fightById[hash].source;
    $("filter").value = "all"; fillDecks();
    $("deckSel").value = src; fillSeeds(); $("seedSel").value = hash;
  }
  loadTally();
  load();
})().catch((e) => { console.error(e); $("status").textContent = "failed to load: " + e; });
