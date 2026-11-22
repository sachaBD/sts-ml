export const esc = (value) => String(value).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
const number = (value) => value == null ? '–' : value.toFixed(3);

// One node at a time: no recursive expansion, so shared nodes and cycles need no special traversal rules.
export function edgeRows(graph, node, chosen) {
  const root = node.observation === graph.root;
  return node.edges.map((edge, i) => ({ edge, noise: root ? graph.root_noise[i] : null }))
    .sort((a, b) => b.edge.visits - a.edge.visits)
    .map(({ edge: e, noise }) => `<tr class="${root && e.bits === chosen ? 'chosen' : ''}">
      <td>${root && e.bits === chosen ? '★ ' : ''}${esc(e.description ?? `action ${e.action}`)}</td>
      <td>${number(e.prior)}</td><td>${number(noise)}</td><td>${number(e.sigma)}</td><td>${number(e.improved_policy)}</td>
      <td>${e.visits}</td><td>${number(e.q)}</td><td>${esc(e.transition)} ${e.children.map((c) =>
        `<button data-node="${esc(c.observation)}">${esc(c.observation)} (${c.visits})${c.bootstrap ? ' ↩ bootstrap' : ''}</button>`).join(' ')}</td>
      <td>${root ? `<button data-play="${e.bits}">Play</button>` : ''}</td></tr>`).join('');
}
