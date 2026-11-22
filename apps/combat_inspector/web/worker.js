const assets = new URL('../assets/gumbel/', location.href);
let engine;
async function load() {
  const response = await fetch(new URL('manifest.json', assets));
  if (!response.ok) throw new Error(`checkpoint manifest: HTTP ${response.status}`);
  const manifest = await response.json();
  return { manifest };
}
const info = load();
async function start(manifest) {
  importScripts(new URL('ort.wasm.min.js', assets).href, new URL('gumbel.js', assets).href);
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = assets.href;
  globalThis.gumbelSession = await ort.InferenceSession.create(new URL(manifest.model, assets).href, {
    executionProviders: ['wasm'],
    externalData: manifest.external.map((path) => ({ path, data: new URL(path, assets).href })),
  });
  return createGumbel({ locateFile: (path) => new URL(path, assets).href });
}
async function answer(path, body) {
  const { manifest } = await info;
  if (path === 'manifest') return manifest;
  if (path !== 'decision') throw new Error('unknown request');
  const fight = manifest.fights.find((f) => f.id === body.fight);
  if (!fight) throw new Error('fight is not in this checkpoint');
  const s = body.search;
  if (!Number.isInteger(s.sims) || s.sims < 0 || s.sims > 512 ||
      !Number.isInteger(s.particles) || s.particles < 1 || s.particles > 64 ||
      !Number.isFinite(s.gumbel_scale) || s.gumbel_scale < 0 || s.gumbel_scale > 10 ||
      !Number.isFinite(s.rollout_mix) || s.rollout_mix < 0 || s.rollout_mix > 1 ||
      !Array.isArray(body.moves) || body.moves.length > 512 ||
      !body.moves.every((n) => Number.isInteger(n) && n >= 0 && n <= 0xffffffff))
    throw new Error('limits: sims 0–512, particles 1–64, noise 0–10, mix 0–1, up to 512 unsigned action bits');
  if (!engine) engine = start(manifest).catch((error) => { engine = null; throw error; });
  const module = await engine;
  const request = JSON.stringify({ start_json: fight.start_json, encoding: manifest.encoding, moves: body.moves, search: s });
  const result = JSON.parse(await module.ccall('explain_json', 'string', ['string'], [request], { async: true }));
  if (result.error) throw new Error(result.error);
  return result;
}
// Asyncify cannot re-enter a suspended C++ call. All requests share this one serial queue.
let queue = Promise.resolve();
onmessage = ({ data: { id, path, body } }) => {
  queue = queue.then(async () => {
    try { postMessage({ id, result: await answer(path, body) }); }
    catch (error) { postMessage({ id, error: error.message || String(error) }); }
  });
};
