// Smoke check for the WASM session_json bridge (init, 64-bit start_json parsing, async inference).
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');
const createGumbel = createRequire(import.meta.url)(process.argv[2] ?? path.join(root, 'build/gumbel-wasm/gumbel.js'));

globalThis.ort = { Tensor: class { constructor(type, data, dims) { Object.assign(this, { type, data, dims }); } } };
globalThis.gumbelSession = {
  async run({ tokens: { dims: [, count] } }) {
    return { value: { data: new Float32Array([0.5]) }, logits: { data: new Float32Array(count) } };
  },
};

const mod = await createGumbel();
const call = async (cmd) => {
  const out = JSON.parse(await mod.ccall('session_json', 'string', ['string'], [JSON.stringify(cmd)], { async: true }));
  if (out.error) throw new Error(out.error);
  return out;
};

const startJson = JSON.stringify({
  seed: 0, ascension: 0, act: 1, floor: 1, encounter: 2, cur_room: 4, last_room: 8, burning_elite_buff: -1,
  hp: 80, max_hp: 80, gold: 99,
  misc_rng: { counter: 0, seed0: 1, seed1: 2 }, potion_rng: { counter: 0, seed0: 3, seed1: 4 },
  potion_capacity: 2, potions: [1, 1], relics: [], bottled: [-1, -1, -1],
  deck: [...Array(5).fill({ id: 321, upgraded: false, misc: 0 }),
         ...Array(4).fill({ id: 104, upgraded: false, misc: 0 }),
         { id: 25, upgraded: false, misc: 0 }],
}).replace('"seed":0', '"seed":9007199254740999');

for (const encoding of ['entity', 'inventory_v1']) {
  assert.equal((await call({ op: 'init', encoding })).ok, true);
  const { trees: [tree] } = await call({ op: 'open', trees: [{ start_json: startJson, moves: [], draw: 0 }] });
  assert.equal(tree.over, false);
  const sim = await call({ op: 'simulate', picks: { [String(tree.id)]: [[[0], 0, false]] } });
  assert.equal(sim.roots[String(tree.id)].moves[0].visits, 1);
  await call({ op: 'close', trees: [tree.id] });
}
console.log('WASM bridge OK');
