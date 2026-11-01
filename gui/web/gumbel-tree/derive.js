// What the page derives from an /api/gumbel/decision response: pure functions of its JSON.

export const title = (s) => s.toLowerCase().replaceAll("_", " ").replace(/\b\w/g, (c) => c.toUpperCase());
export const cardLabel = (c) => c.name + (c.upgraded > 1 ? "+" + c.upgraded : c.upgraded ? "+" : "");
export const q = (edge) => (edge.visits ? edge.value_sum / edge.visits : null);

export function intent(monster) {
  if (!monster.intent) return "–";
  const prefix = monster.id.toUpperCase() + "_";
  const move = title(monster.intent.startsWith(prefix) ? monster.intent.slice(prefix.length) : monster.intent);
  if (monster.intent_damage === undefined) return move;
  return `${move} ${monster.intent_damage}${monster.intent_hits > 1 ? "×" + monster.intent_hits : ""}`;
}

export function moveLabel(view, bits) {
  const move = view.legal.find((l) => l.bits === bits);
  if (!move) return String(bits);
  const alive = view.monsters.filter((m) => m.alive);
  const target = alive.length > 1 && view.monsters[move.target] ? " → " + title(view.monsters[move.target].id) : "";
  if (move.type === "card") return cardLabel(view.hand[move.source]) + target;
  if (move.type === "potion") return "Drink " + view.potions[move.source] + target;
  if (move.type === "end_turn") return "End turn";
  const card = move.type === "select" && view.hand[move.source];
  return title(view.kind) + ": " + (card ? cardLabel(card) : move.desc.replace(/^\{\s*|\s*\}$/g, ""));
}

// Sequential halving keeps the most-visited moves longest, so distinct visit counts, in order, are its rounds.
export function rounds(edges) {
  const levels = [...new Set(edges.map((e) => e.visits).filter((n) => n > 0))].sort((a, b) => a - b);
  return edges.map((e) => (e.visits ? levels.indexOf(e.visits) + 1 : 0));
}

// Simulations that ended the fight on this edge, and how many of them were won.
export function terminal(edge, nodes) {
  const children = edge.children.map((c) => nodes[c.node]);
  const subtree = (n) => n.value + n.edges.reduce((s, e) => s + e.value_sum, 0);
  const visits = edge.visits - children.reduce((s, n) => s + n.total + 1, 0);
  const wins = edge.value_sum - children.reduce((s, n) => s + subtree(n), 0);
  return { visits, wins: Math.round(wins) };
}

// What a chance outcome showed the player: the new hand at a new turn, cards drawn otherwise, and new intents.
export function outcome(parent, child) {
  const parts = [];
  if (child.turn > parent.turn) parts.push(`turn ${child.turn + 1}: ` + child.hand.map(cardLabel).join(", "));
  else {
    const before = parent.hand.map(cardLabel), drawn = [];
    for (const card of child.hand.map(cardLabel)) {
      const i = before.indexOf(card);
      if (i < 0) drawn.push(card);
      else before.splice(i, 1);
    }
    if (drawn.length) parts.push("draw " + drawn.join(", "));
  }
  child.monsters.forEach((m, i) => {
    if (m.alive && m.intent !== parent.monsters[i]?.intent) parts.push(title(m.id) + ": " + intent(m));
  });
  return parts.join(" · ") || "no visible change";
}

// The node the fight actually reached after `bits` at the root: the child showing the observation that followed,
// looking through forced moves (single-edge nodes), which the fight plays without a decision.
export function happened(response, bits, observation) {
  const nodes = response.tree.nodes;
  let frontier = nodes[0].edges.filter((e) => e.bits === bits).flatMap((e) => e.children);
  while (frontier.length) {
    const hit = frontier.find((c) => c.observation === observation);
    if (hit) return hit.node;
    frontier = frontier.flatMap((c) => (nodes[c.node].edges.length === 1 ? nodes[c.node].edges[0].children : []));
  }
  return null;
}

