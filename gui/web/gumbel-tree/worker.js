// One single-threaded simulator and ORT session: no SharedArrayBuffer or isolation headers required.
const assets = new URL("../assets/gumbel/", location.href);
const checkedJSON = async (url) => {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`could not load ${url}: HTTP ${response.status}`);
  return response.json();
};
const manifest = checkedJSON(new URL("manifest.json", assets));
let engine;
async function loadEngine(info) {
  importScripts(new URL("ort.wasm.min.js", assets).href, new URL("gumbel.js", assets).href);
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.proxy = false;
  ort.env.wasm.wasmPaths = assets.href;
  globalThis.gumbelSession = await ort.InferenceSession.create(new URL(info.model, assets).href, {
    executionProviders: ["wasm"],
    externalData: info.external.map((path) => ({path, data: new URL(path, assets).href})),
  });
  return createGumbel({ locateFile: (path) => new URL(path, assets).href });
}

async function answer(path, body) {
  const info = await manifest;
  if (path === "runs") return { runs: [{run: info.run, id: info.id, date: info.date}] };
  if (path.startsWith("run?")) return {iterations: [info.iteration], fights: info.fights.map((f) => f.id)};
  if (path !== "decision") throw new Error(`unknown browser request: ${path}`);
  if (body.run !== info.run || body.iteration !== info.iteration) throw new Error("checkpoint is not in this frozen site");
  const fight = info.fights.find((f) => f.id === body.fight);
  if (!fight) throw new Error("fight is not in this frozen site");
  const s = body.search;
  if (!Number.isInteger(s.sims) || s.sims < 0 || s.sims > 512 ||
      !Number.isInteger(s.particles) || s.particles < 1 || s.particles > 64 ||
      !Number.isFinite(s.gumbel_scale) || s.gumbel_scale < 0 || s.gumbel_scale > 10 ||
      !Number.isFinite(s.rollout_mix) || s.rollout_mix < 0 || s.rollout_mix > 1 ||
      !Array.isArray(body.moves) || body.moves.length > 512)
    throw new Error("browser limits: sims 0–512, particles 1–64, noise 0–10, mix 0–1, up to 512 moves");
  if (!engine) engine = loadEngine(info).catch((error) => { engine = null; throw error; });
  const module = await engine;
  const request = JSON.stringify({start_json: fight.start_json, moves: body.moves, search: s});
  const result = JSON.parse(await module.ccall("explain_json", "string", ["string"], [request], {async: true}));
  if (result.error) throw new Error(result.error);
  return result;
}

// Asyncify must not re-enter a suspended C++ explanation when another request arrives.
let queue = Promise.resolve();
onmessage = ({data: {id, path, body}}) => {
  queue = queue.then(async () => {
    try { postMessage({id, result: await answer(path, body)}); }
    catch (error) { postMessage({id, error: error.message || String(error)}); }
  });
};
