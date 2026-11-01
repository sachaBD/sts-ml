// DOM for the page's panels. Each function takes data and callbacks and returns or fills elements; no state.
import { cardLabel, intent, moveLabel, outcome, q, rounds, terminal, title } from "./derive.js";

const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const f2 = (x) => (x === null || x === undefined ? "–" : x.toFixed(2));
const statuses = (s) => Object.entries(s).map(([k, n]) => `<span class="status">${esc(title(k))} ${n}</span>`).join("");
const visits = (n, total) => `<span class="visits"><i style="width:${total ? (100 * n) / total : 0}%"></i></span>`;

export function board(el, view, onPlay) {
  const playable = new Map();
  for (const l of view.legal) if (l.type === "card") playable.set(l.source, playable.has(l.source) ? null : l.bits);
  const pile = (name, cards) => `<span title="${esc(cards.map(cardLabel).join(", "))}">${name} ${cards.length}</span>`;
  el.innerHTML = `
    <div class="monsters">${view.monsters.map((m) => `
      <div class="monster${m.alive ? "" : " dead"}"><b>${esc(title(m.id))}</b> ${m.hp}/${m.max_hp}
        ${m.block ? `<span class="block">⛨${m.block}</span>` : ""} <span class="foe-move">${esc(intent(m))}</span> ${statuses(m.statuses)}</div>`).join("")}
    </div>
    <div class="me"><b>Ironclad</b> ${view.player.hp}/${view.player.max_hp}
      ${view.player.block ? `<span class="block">⛨${view.player.block}</span>` : ""}
      · energy ${view.player.energy}/${view.player.energy_per_turn} ${statuses(view.player.statuses)}
      ${view.kind === "done" ? `<b class="verdict">${view.won ? "Won" : "Lost"}</b>` : ""}</div>
    <div class="cards">${view.hand.map((c, i) => `<button class="chip ${esc(c.type)}" data-slot="${i}"
      ${onPlay && playable.get(i) != null ? "" : "disabled"}>${esc(cardLabel(c))} <small>${c.cost_for_turn < 0 ? "X" : c.cost_for_turn}</small></button>`).join("")}</div>
    <div class="piles">${pile("Draw", view.draw)} · ${pile("Discard", view.discard)} · ${pile("Exhaust", view.exhaust)}
      · Potions ${view.potions.map((p) => esc(p || "–")).join(", ") || "–"}</div>`;
  el.querySelectorAll(".cards .chip:not([disabled])").forEach((b) => (b.onclick = () => onPlay(playable.get(+b.dataset.slot))));
}

export function timeline(el, line, moves, at, onGo) {
  const turns = new Map();
  line.forEach((d, k) => {
    const turn = d.response.view.turn;
    if (!turns.has(turn)) turns.set(turn, []);
    turns.get(turn).push(k);
  });
  el.innerHTML = [...turns].map(([turn, ks]) => `<span class="turn"><b>T${turn + 1}</b>${ks.map((k) => {
    const { response } = line[k];
    const now = k === moves.length;
    const label = now ? (response.over ? (response.view.won ? "won" : "lost") : "latest") : moveLabel(response.view, moves[k]);
    const took = !now && response.decision && moves[k] !== response.decision.action;
    const value = response.decision ? f2(response.decision.value) : "";
    return `<button class="dot${k === at ? " at" : ""}${took ? " took" : ""}" data-k="${k}" title="${took ? "your move" : "agent's move"}">
      ${now ? "◉" : took ? "✎" : "●"} ${esc(label)} <small>${value}</small></button>`;
  }).join("")}</span>`).join("");
  el.querySelectorAll(".dot").forEach((b) => (b.onclick = () => onGo(+b.dataset.k)));
}

export function rootTable(el, response, onPlay) {
  const root = response.tree.nodes[0];
  const round = rounds(root.edges);
  const rows = root.edges.map((e, i) => ({ e, i })).sort((a, b) => b.e.visits - a.e.visits);
  el.innerHTML = `<tr><th>Move</th><th>prior</th><th>g</th><th>σ</th><th>π′</th><th>N</th><th>round</th><th>Q</th><th></th></tr>` +
    rows.map(({ e, i }) => `<tr class="${e.bits === response.decision.action ? "chosen" : ""}">
      <td>${e.bits === response.decision.action ? "★ " : ""}${esc(moveLabel(root.view, e.bits))}</td>
      <td>${f2(e.prior)}</td><td>${f2(response.tree.gumbel[i])}</td><td>${f2(e.sigma)}</td><td>${f2(e.improved)}</td>
      <td>${visits(e.visits, root.total)} ${e.visits}</td><td>${round[i] || "–"}</td><td>${f2(q(e))}</td>
      <td><button data-bits="${e.bits}">play</button></td></tr>`).join("");
  el.querySelectorAll("button[data-bits]").forEach((b) => (b.onclick = () => onPlay(+b.dataset.bits)));
}

// The tree as nested <details>: an edge, then its child node, or one entry per chance outcome.
// Open by default: the chosen move and anything with at least 10% of its parent's visits.
export function tree(el, response, { happenedNode, onInspect }) {
  const nodes = response.tree.nodes;
  const stats = (id) => {
    const n = nodes[id];
    return `<a class="node" data-node="${id}">node ${id}</a> N ${n.total} · V ${f2(n.value)} · net ${f2(n.network)}
      · roll ${n.rollout ?? "–"}${id === happenedNode ? ` <b class="happened">✓ happened</b>` : ""}`;
  };
  const edges = (id) => nodes[id].edges.filter((e) => e.visits).sort((a, b) => b.visits - a.visits)
    .map((e) => edge(id, e)).join("");
  const edge = (id, e) => {
    const parent = nodes[id];
    const open = e.visits >= 0.1 * parent.total || (id === 0 && e.bits === response.decision.action);
    const end = terminal(e, nodes);
    const summary = `${id === 0 && e.bits === response.decision.action ? "★ " : ""}<b>${esc(moveLabel(parent.view, e.bits))}</b>
      <span class="num">N ${e.visits} · Q ${f2(q(e))} · π′ ${f2(e.improved)}</span>
      ${end.visits > 0 ? `<span class="end">ends fight: won ${end.wins}/${end.visits}</span>` : ""}`;
    if (e.children.length === 1)
      return `<details${open ? " open" : ""}><summary>${summary} <span class="num">→ ${stats(e.children[0].node)}</span></summary>${edges(e.children[0].node)}</details>`;
    const body = [...e.children].sort((a, b) => nodes[b.node].total - nodes[a.node].total).map((c) => `
      <details${nodes[c.node].total + 1 >= 0.1 * e.visits || c.node === happenedNode ? " open" : ""}>
        <summary>◇ ${esc(outcome(parent.view, nodes[c.node].view))} <span class="num">${stats(c.node)}</span></summary>
        ${edges(c.node)}</details>`).join("");
    return `<details${open ? " open" : ""}><summary>${summary}</summary>${body}</details>`;
  };
  const hidden = nodes[0].edges.some((e) => !e.visits) ? ` <span class="hint">(unvisited moves hidden)</span>` : "";
  el.innerHTML = `<div class="nodeline">root ${stats(0)}${hidden}</div>${edges(0)}`;
  el.querySelectorAll("a.node").forEach((a) => (a.onclick = (ev) => { ev.preventDefault(); onInspect(+a.dataset.node); }));
}

export function inspector(el, response, id, mix) {
  const n = response.tree.nodes[id];
  const parts = n.rollout === null ? `V ${f2(n.value)} = net ${f2(n.network)}`
    : `V ${f2(n.value)} = (1 − ${mix})·net ${f2(n.network)} + ${mix}·roll ${n.rollout}`;
  el.innerHTML = `<p><b>node ${id}</b> · turn ${n.view.turn + 1} · N ${n.total} · ${parts}</p><div class="board"></div>
    <table class="tbl"><tr><th>Move</th><th>prior</th><th>σ</th><th>π′</th><th>N</th><th>Q</th><th>ends fight</th></tr>
    ${n.edges.map((e) => {
      const end = terminal(e, response.tree.nodes);
      return `<tr><td>${esc(moveLabel(n.view, e.bits))}</td><td>${f2(e.prior)}</td><td>${f2(e.sigma)}</td><td>${f2(e.improved)}</td>
        <td>${e.visits}</td><td>${f2(q(e))}</td><td>${end.visits > 0 ? `won ${end.wins}/${end.visits}` : ""}</td></tr>`;
    }).join("")}</table>`;
  board(el.querySelector(".board"), n.view, null);
}
