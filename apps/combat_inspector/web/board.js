export const title = (s) => s.toLowerCase().replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
export const cardLabel = (c) => c.name + (c.upgraded > 1 ? "+" + c.upgraded : c.upgraded ? "+" : "");

export function intent(monster) {
  if (!monster.intent) return "–";
  const prefix = monster.id.toUpperCase() + "_";
  const move = title(monster.intent.startsWith(prefix) ? monster.intent.slice(prefix.length) : monster.intent);
  if (monster.intent_damage === undefined) return move;
  return `${move} ${monster.intent_damage}${monster.intent_hits > 1 ? "×" + monster.intent_hits : ""}`;
}


const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const statuses = (s) => Object.entries(s).map(([k, n]) => `<span class="status">${esc(title(k))} ${n}</span>`).join("");

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

