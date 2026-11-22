import assert from 'node:assert/strict';
import { edgeRows } from './web/graph.js';

const shared = '18446744073709551615';
const edge = (bits, visits) => ({ bits, visits, action: String(bits), description: '<move>', prior: .5,
  sigma: 0, improved_policy: .5, q: null, transition: 'chance',
  children: [{ observation: shared, visits, bootstrap: true }] });
const root = { observation: '1', edges: [edge(7, 1), edge(8, 3)] };
const html = edgeRows({ root: '1', root_noise: [.25, .75] }, root, 8);
assert(html.indexOf('data-play="8"') < html.indexOf('data-play="7"'));
assert(html.indexOf('0.750') < html.indexOf('0.250')); // sorting must not detach noise from its move
assert.equal((html.match(new RegExp(`data-node="${shared}"`, 'g')) || []).length, 2);
assert(html.includes('bootstrap'));
assert(html.includes('&lt;move&gt;'));
assert(!edgeRows({ root: 'other', root_noise: [] }, root, 8).includes('data-play'));
console.log('Graph rendering: shared/cycle references, root noise and escaping OK');
