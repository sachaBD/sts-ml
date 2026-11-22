import { board } from './board.js';
import { edgeRows } from './graph.js';
const $ = (id) => document.getElementById(id);
const worker = new Worker(new URL('worker.js', import.meta.url));
const pending = new Map();
let requestId = 0;
worker.onmessage = ({ data: { id, result, error } }) => {
  const call = pending.get(id);
  if (!call) return;
  pending.delete(id);
  if (error) call.reject(new Error(error)); else call.resolve(result);
};
worker.onerror = (e) => {
  for (const call of pending.values()) call.reject(new Error(e.message || 'worker failed; reload to retry'));
  pending.clear();
};
function api(path, body) {
  return new Promise((resolve, reject) => {
    const id = requestId++;
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, path, body });
  });
}
const cache = new Map();
let manifest, current, selected, nodes, moves = [], at = 0, sequence = 0, busy = false, autoplay = false;
let history = [];
const settings = () => Object.fromEntries(['sims', 'particles', 'gumbel_scale', 'rollout_mix'].map((k) => [k, +$(k).value]));
function save() {
  const q = new URLSearchParams({ checkpoint: manifest.checkpoint_id, fight: $('fight').value, at, moves: moves.join(','), ...settings() });
  window.history.replaceState(null, '', '?' + q);
}
function controls() {
  document.querySelectorAll('.controls input, .controls select, .controls button, #timeline button, [data-play]')
    .forEach((el) => { el.disabled = busy || autoplay; });
  $('board').querySelectorAll('button').forEach((el) => { el.disabled = busy || autoplay || !el.onclick; });
  $('auto').disabled = busy && !autoplay;
  $('auto').textContent = autoplay ? 'Stop after this decision' : 'Play fight';
  $('back').disabled = busy || autoplay || !history.length;
  $('prev').disabled = busy || autoplay || at === 0;
  $('next').disabled = busy || autoplay || (current?.over && at === moves.length);
}
function timeline() {
  $('timeline').replaceChildren(...Array.from({ length: moves.length + 1 }, (_, k) => {
    const b = document.createElement('button');
    b.textContent = String(k); b.className = k === at ? 'at' : '';
    b.onclick = () => { at = k; show(); };
    return b;
  }));
  $('position').textContent = `Decision ${at} / ${moves.length}`;
}
function inspect(id, push = true) {
  if (push && selected) history.push(selected);
  selected = id;
  const node = nodes.get(id);
  if (!node) return;
  $('node').textContent = `${id} · evaluation ${node.evaluation_value.toFixed(3)} · backed ${node.backed_value.toFixed(3)}${node.terminal ? ' · terminal' : ''}`;
  $('nodeBoard').hidden = id === current.graph.root || !node.view;
  if (!$('nodeBoard').hidden) board($('nodeBoard'), node.view, null);
  $('edges').innerHTML = edgeRows(current.graph, node, current.decision.action);
  $('edges').querySelectorAll('[data-node]').forEach((b) => { b.onclick = () => inspect(b.dataset.node); });
  $('edges').querySelectorAll('[data-play]').forEach((b) => { b.onclick = () => play(+b.dataset.play); });
  $('back').disabled = !history.length;
  controls();
}
async function show() {
  const seq = ++sequence;
  busy = true; controls(); save();
  $('status').textContent = 'Searching… First use downloads the model and WASM runtime.';
  $('graph').hidden = true;
  const start = performance.now();
  const body = { fight: $('fight').value, moves: moves.slice(0, at), search: settings() };
  const key = JSON.stringify(body);
  if (!cache.has(key)) {
    cache.set(key, api('decision', body));
    if (cache.size > 32) cache.delete(cache.keys().next().value);
  }
  try {
    const result = await cache.get(key);
    if (seq !== sequence) return false;
    current = result;
    board($('board'), result.view, result.over ? null : play);
    timeline();
    $('status').textContent = result.over ? `Fight ended: ${result.view.won ? 'won' : 'lost (including turn/step limit)'}`
      : `Decision ${at} · ${(performance.now() - start).toFixed(0)} ms · agent value ${result.decision.value.toFixed(3)}`;
    $('graph').hidden = result.over;
    if (!result.over) {
      nodes = new Map(result.graph.nodes.map((n) => [n.observation, n]));
      $('graphSize').textContent = `${nodes.size} public states`;
      history = []; selected = null; inspect(result.graph.root, false);
    }
    return true;
  } catch (error) {
    cache.delete(key); current = null;
    $('status').textContent = 'Error: ' + error.message;
    return false;
  } finally {
    if (seq === sequence) { busy = false; controls(); }
  }
}
function play(bits) {
  if (busy || autoplay) return;
  moves = [...moves.slice(0, at), bits]; at = moves.length;
  show();
}
async function auto() {
  if (autoplay) { autoplay = false; controls(); return; }
  autoplay = true; controls();
  try {
    while (autoplay) {
      if (!await show() || !autoplay || current.over) break;
      if (at < moves.length) at++;
      else { moves.push(current.decision.action); at = moves.length; }
      await new Promise((resolve) => setTimeout(resolve, 16));
    }
  } finally { autoplay = false; controls(); }
}
async function main() {
  try {
    manifest = await api('manifest');
    $('checkpoint').textContent = `${manifest.checkpoint} · ${manifest.encoding} · source ${manifest.source_commit.slice(0, 10)}${manifest.source_dirty ? ' (uncommitted changes)' : ''} · simulator ${manifest.simulator_commit.slice(0, 10)}${manifest.simulator_dirty ? ' (uncommitted changes)' : ''}`;
    for (const f of manifest.fights) $('fight').add(new Option(f.id, f.id));
    const q = new URLSearchParams(location.search);
    if (q.get('checkpoint') === manifest.checkpoint_id && manifest.fights.some((f) => f.id === q.get('fight'))) {
      $('fight').value = q.get('fight');
      moves = (q.get('moves') || '').split(',').filter(Boolean).map(Number);
      const k = Number(q.get('at') ?? moves.length);
      at = Number.isInteger(k) ? Math.max(0, Math.min(k, moves.length)) : 0;
      for (const key of Object.keys(settings())) if (q.has(key)) $(key).value = q.get(key);
    }
    $('fight').onchange = () => { moves = []; at = 0; cache.clear(); show(); };
    $('search').onclick = show;
    $('auto').onclick = auto;
    $('restart').onclick = () => { moves = []; at = 0; show(); };
    $('prev').onclick = () => { at--; show(); };
    $('next').onclick = () => { if (at < moves.length) { at++; show(); } else if (current?.decision) play(current.decision.action); };
    $('root').onclick = () => inspect(current.graph.root);
    $('back').onclick = () => { if (history.length) inspect(history.pop(), false); };
    await show();
  } catch (error) { $('status').textContent = 'Error: ' + error.message; }
}
main();
