// Fight-outcome page: edit a state, POST it to /api/predict, render the results.
const $ = (id) => document.getElementById(id);
const ART = window.CARDS || {};
const ROOT = window.GUI_ROOT || "./";
let meta, state, cands = [], timer;

const pct = (p) => (100 * p).toFixed(1) + "%";
const sgn = (x, d = 1) => (x >= 0 ? "+" : "") + x.toFixed(d);
const cls = (x) => (x > 0 ? "pos" : x < 0 ? "neg" : "");
const byId = (list) => Object.fromEntries(list.map((x) => [x.id, x]));
const byName = (list) => Object.fromEntries(list.map((x) => [x.name.toLowerCase(), x]));

/** One card, as in sts_visualiser's app.js. */
function card(c, count) {
  const info = meta.cardById[c.card_id];
  const art = ART[info.name] || {};
  const el = document.createElement("div");
  el.className = "card";
  el.dataset.rarity = info.rarity;
  el.dataset.type = info.type;
  const a = document.createElement("div");
  a.className = "art";
  if (art.art) a.innerHTML = `<img src="${ROOT}vis/${art.art}" alt="" loading="lazy">`;
  const cost = c.upgraded ? art.costUp : art.cost;
  if (cost !== null && cost !== undefined) a.insertAdjacentHTML("beforeend", `<div class="cost">${cost}</div>`);
  if (count > 1) a.insertAdjacentHTML("beforeend", `<div class="count">×${count}</div>`);
  el.append(a);
  const t = document.createElement("div");
  t.className = "name";
  t.textContent = info.name;
  if (c.upgraded) t.insertAdjacentHTML("beforeend", `<span class="up">${c.upgraded > 1 ? "+" + c.upgraded : "+"}</span>`);
  el.append(t);
  return el;
}

const cardName = (c) => meta.cardById[c.card_id].name + (c.upgraded ? "+" : "");

// Deck grouped into identical cards; click removes one, shift-click toggles upgrade.
function renderDeck() {
  const groups = new Map();
  state.pre.deck.forEach((c, i) => {
    const k = `${c.card_id}/${c.upgraded}/${c.misc}`;
    if (!groups.has(k)) groups.set(k, []);
    groups.get(k).push(i);
  });
  const els = [...groups.values()].map((idx) => {
    const c = state.pre.deck[idx[0]];
    const el = card(c, idx.length);
    el.title = "click: remove one · shift-click: toggle upgrade";
    el.onclick = (e) => {
      if (e.shiftKey) c.upgraded = c.upgraded ? 0 : 1;
      else state.pre.deck.splice(idx[idx.length - 1], 1);
      changed();
    };
    return el;
  });
  $("deck").replaceChildren(...els);
  $("size").textContent = `— ${state.pre.deck.length} cards`;
}

function chip(text, klass, onclick, extra = "") {
  const el = document.createElement("span");
  el.className = "relic " + klass;
  el.innerHTML = text + (extra ? ` <small>${extra}</small>` : "");
  el.title = "click to remove";
  el.onclick = onclick;
  return el;
}

function renderRelics() {
  const p = state.pre;
  $("relics").replaceChildren(
    ...p.relics.map((r, i) => chip(meta.relicById[r.relic_id].name, r.relic_id === 86 ? "starter" : "",
      () => { p.relics.splice(i, 1); changed(); }, r.data ? `(${r.data})` : "")),
    ...p.potions.map((x, i) => chip(meta.potById[x.potion_id].name, "potion", () => { p.potions.splice(i, 1); changed(); })),
  );
}

function renderInputs() {
  $("enc").value = state.encounter;
  $("hp").value = state.pre.hp;
  $("maxhp").value = state.pre.max_hp;
  $("cap").value = state.pre.potion_capacity;
}

function render() {
  renderInputs();
  renderDeck();
  renderRelics();
}

// Called after every edit: re-render now, re-predict shortly after.
function changed() {
  delete state.real;  // any edit: the state no longer matches a real fight
  refresh();
}

function refresh() {
  render();
  clearTimeout(timer);
  timer = setTimeout(predict, 120);
}

// Ask the server for the current state, candidates and every fight, then fill the page.
async function predict() {
  const q = { encounter: state.encounter, pre: state.pre, candidates: cands };
  let out;
  if (window.API) out = await window.API.predict(q);  // static site: model runs in the browser (model.js)
  else {
    const res = await fetch(ROOT + "api/predict", { method: "POST", body: JSON.stringify(q) });
    if (!res.ok) { $("pwin").textContent = "error"; return; }
    out = await res.json();
  }
  const b = out.base;
  $("pwin").textContent = pct(b.p_win);
  $("band").innerHTML = `seeds ${b.p_seeds.map(pct).join(" · ")}`;
  $("real").innerHTML = state.real ? `real outcome of this fight: <b>${state.real}</b> (one sample)` : "";
  $("ev").innerHTML = `HP if won <b>${b.hp_if_win.toFixed(1)}</b> · expected HP <b>${b.expected_hp.toFixed(1)}</b>`;

  const hi = Math.max(...b.hp_bins);
  const cur = Math.min(Math.floor((state.pre.hp - 1) / 5), 20);
  $("hist").replaceChildren(...b.hp_bins.map((p, i) => {
    const d = document.createElement("div");
    d.style.height = `${(100 * p) / hi}%`;
    d.title = `${i === 20 ? ">100" : `${5 * i + 1}–${5 * i + 5}`} HP: ${pct(p)}`;
    if (i === cur) d.className = "cur";
    if (i % 4 === 0 || i === 20) d.innerHTML = `<span>${i === 20 ? ">100" : 5 * i + 1}</span>`;
    return d;
  }));

  const rows = cands.map((c, i) => ({ c, i, s: out.candidates[i] }))
    .sort((x, y) => y.s.expected_hp - x.s.expected_hp);
  const head = `<tr><th>Card</th><th>P(win)</th><th>Δ P(win)</th><th>Δ band</th><th>Exp. HP</th><th>Δ Exp. HP</th><th>Δ band</th><th></th></tr>`;
  const band = (a, base) => {
    const d = a.map((x, k) => x - base[k]);
    return `${sgn(Math.min(...d), 2)} … ${sgn(Math.max(...d), 2)}`;
  };
  $("cands").innerHTML = head +
    `<tr class="sel"><td><i>skip (current deck)</i></td><td>${pct(b.p_win)}</td><td></td><td></td><td>${b.expected_hp.toFixed(1)}</td><td></td><td></td><td></td></tr>` +
    rows.map(({ c, i, s }) => {
      const dp = 100 * (s.p_win - b.p_win), dh = s.expected_hp - b.expected_hp;
      return `<tr><td>${cardName(c)}</td><td>${pct(s.p_win)}</td><td class="${cls(dp)}">${sgn(dp)} pp</td>` +
        `<td class="band">${band(s.p_seeds.map((x) => 100 * x), b.p_seeds.map((x) => 100 * x))}</td>` +
        `<td>${s.expected_hp.toFixed(1)}</td><td class="${cls(dh)}">${sgn(dh)}</td>` +
        `<td class="band">${band(s.ev_seeds, b.ev_seeds)}</td>` +
        `<td><button data-take="${i}">take</button> <button data-drop="${i}">×</button></td></tr>`;
    }).join("");

  $("fights").innerHTML = `<tr><th>Fight</th><th>Group</th><th>P(win)</th><th>Seed range</th><th>HP if won</th><th>Exp. HP</th></tr>` +
    Object.entries(meta.groups).flatMap(([g, es]) => es.map((e) => {
      const s = out.fights[e];
      return `<tr class="${e === state.encounter ? "sel" : ""}"><td>${e}</td><td class="grp">${g}</td><td>${pct(s.p_win)}</td>` +
        `<td class="band">${pct(Math.min(...s.p_seeds))} – ${pct(Math.max(...s.p_seeds))}</td>` +
        `<td>${s.hp_if_win.toFixed(1)}</td><td>${s.expected_hp.toFixed(1)}</td></tr>`;
    })).join("");
}

// Resolve a typed name to a table entry (cards, relics, potions).
function lookup(input, table) {
  const x = table[input.value.trim().toLowerCase()];
  if (!x) { input.setCustomValidity("unknown"); input.reportValidity(); return null; }
  input.setCustomValidity("");
  input.value = "";
  return x;
}

function loadPreset(i) {
  const p = meta.presets[i];
  state = JSON.parse(JSON.stringify({ encounter: p.encounter, pre: p.pre }));
  state.real = p.won === null ? null : p.won ? `won with ${p.final_hp} HP` : "lost";
  refresh();
}

// Startup: load tables, fights, presets and calibration, wire up the controls.
(window.API ? window.API.meta() : fetch(ROOT + "api/meta").then((r) => r.json())).then((m) => {
  meta = m;
  meta.cardById = byId(m.cards); meta.relicById = byId(m.relics); meta.potById = byId(m.potions);
  // Ironclad cards first in the picker; duplicates like "Strike" resolve to the red one.
  const cards = [...m.cards].sort((a, b) => (a.color !== "red") - (b.color !== "red") || a.name.localeCompare(b.name));
  const cardByName = {};
  cards.forEach((c) => { cardByName[c.name.toLowerCase()] ??= c; });
  const relicByName = byName(m.relics), potByName = byName(m.potions);
  const opts = (list) => [...new Set(list.map((x) => x.name))].map((n) => `<option value="${n}">`).join("");
  $("cardList").innerHTML = opts(cards);
  $("relicList").innerHTML = opts(m.relics);
  $("potList").innerHTML = opts(m.potions);

  $("enc").innerHTML = Object.entries(m.groups).map(([g, es]) =>
    `<optgroup label="${g}">${es.map((e) => `<option>${e}</option>`).join("")}</optgroup>`).join("");
  $("preset").innerHTML = m.presets.map((p, i) => `<option value="${i}">${p.category === "starter" ? p.label : p.category + " · " + p.label}</option>`).join("");

  $("preset").onchange = (e) => loadPreset(Number(e.target.value));
  $("enc").onchange = (e) => { state.encounter = e.target.value; changed(); };
  $("hp").onchange = (e) => { state.pre.hp = Math.max(1, Number(e.target.value)); changed(); };
  $("maxhp").onchange = (e) => { state.pre.max_hp = Math.max(1, Number(e.target.value)); changed(); };
  $("cap").onchange = (e) => { state.pre.potion_capacity = Number(e.target.value); changed(); };

  $("cardAdd").onclick = () => {
    const c = lookup($("cardIn"), cardByName);
    if (c) { state.pre.deck.push({ card_id: c.id, upgraded: $("cardUp").checked ? 1 : 0, misc: 0, name: c.key }); changed(); }
  };
  $("relicAdd").onclick = () => {
    const r = lookup($("relicIn"), relicByName);
    if (r) { state.pre.relics.push({ relic_id: r.id, data: 0, name: r.key }); changed(); }
  };
  $("potAdd").onclick = () => {
    const p = lookup($("potIn"), potByName);
    if (p) { state.pre.potions.push({ potion_id: p.id, name: p.key }); changed(); }
  };
  $("candAdd").onclick = () => {
    const c = lookup($("candIn"), cardByName);
    if (c) { cands.push({ card_id: c.id, upgraded: $("candUp").checked ? 1 : 0, name: c.key }); changed(); }
  };
  for (const [inp, btn] of [["cardIn", "cardAdd"], ["relicIn", "relicAdd"], ["potIn", "potAdd"], ["candIn", "candAdd"]])
    $(inp).onkeydown = (e) => { if (e.key === "Enter") $(btn).click(); };
  $("cands").onclick = (e) => {
    const take = e.target.dataset.take, drop = e.target.dataset.drop;
    if (take !== undefined) { const [c] = cands.splice(Number(take), 1); state.pre.deck.push({ ...c, misc: 0 }); changed(); }
    if (drop !== undefined) { cands.splice(Number(drop), 1); changed(); }
  };

  // A typical Act 1 reward spread as default candidates.
  cands = ["Shrug It Off", "Pommel Strike", "Inflame", "Carnage", "Feel No Pain", "Uppercut"]
    .map((n) => cardByName[n.toLowerCase()]).filter(Boolean)
    .map((c) => ({ card_id: c.id, upgraded: 0, name: c.key }));
  renderCalibration(m.calibration);
  loadPreset(0);
});

// Real vs predicted win rate on held-out natural fights (computed once at server start).
function renderCalibration(c) {
  $("calN").textContent = `— ${c.n} held-out natural fights`;
  const head = `<tr><th>Fight</th><th>Group</th><th>n</th><th>Real win rate</th><th>Predicted</th><th>Seeds</th><th>Pred − real</th></tr>`;
  const row = (name, g, x, bold) => x.n ? `<tr class="${bold ? "sel" : ""}"><td>${name}</td><td class="grp">${g}</td><td>${x.n}</td>` +
    `<td>${pct(x.real)}</td><td>${pct(x.pred)}</td><td class="band">${x.seeds.map(pct).join(" · ")}</td>` +
    `<td>${sgn(100 * (x.pred - x.real))} pp</td></tr>` : "";
  $("calEnc").innerHTML = head + Object.entries(meta.groups).map(([g, es]) =>
    row(`<b>all ${g}</b>`, g, c.groups[g], true) + es.map((e) => row(e, g, c.encounters[e])).join("")).join("");
  $("calRel").innerHTML = `<tr><th>Predicted P(win)</th>` + Object.keys(c.reliability).map((g) => `<th>${g}: n · real · pred</th>`).join("") + `</tr>` +
    c.reliability.all.map((b, i) => `<tr><td>${pct(b.lo)} – ${pct(b.hi)}</td>` + Object.values(c.reliability).map((r) => {
      const x = r[i];
      return x.n ? `<td>${x.n} · <b>${pct(x.real)}</b> · ${pct(x.pred)}</td>` : `<td class="band">–</td>`;
    }).join("") + `</tr>`).join("");
}
