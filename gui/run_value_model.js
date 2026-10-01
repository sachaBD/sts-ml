// Browser inference for agents/overworld/value/run_policy_v1.py (evaluation mode).
(() => {
  const BrowserModel = typeof window !== "undefined" ? window.BrowserModel : require("./browser_model.js");
  const { b64, json } = BrowserModel;
  const relu = (x) => x.map((v) => Math.max(0, v));
  const cat = (...xs) => Float64Array.from(xs.flatMap((x) => Array.from(x)));
  const add = (a, b) => { for (let i = 0; i < a.length; i++) a[i] += b[i]; return a; };
  const scale = (a, n) => a.map((x) => x * n);
  const tensor = (x) => ({ shape: x.shape, w: b64(x.w) });
  const row = (e, i) => e.w.subarray(i * e.shape[1], (i + 1) * e.shape[1]);
  const linear = (l, x) => {
    const [out, input] = l.shape, y = Float64Array.from(l.b);
    for (let o = 0; o < out; o++) for (let i = 0; i < input; i++) y[o] += l.w[o * input + i] * x[i];
    return y;
  };
  // run_policy_v1's token/path MLPs are Linear → ReLU → Linear (no final activation).
  const mlp = (a, b, x) => linear(b, relu(linear(a, x)));
  const layerNorm = (p, x) => {
    const mean = x.reduce((s, v) => s + v, 0) / x.length;
    const variance = x.reduce((s, v) => s + (v - mean) ** 2, 0) / x.length;
    return x.map((v, i) => (v - mean) / Math.sqrt(variance + 1e-5) * p.w[i] + p.b[i]);
  };
  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  const ROOM = {"$": 0, R: 1, "?": 2, E: 3, M: 4, T: 5};
  let M;

  function load(raw) {
    const t = (name) => tensor(raw[name]);
    const linearT = (name) => ({ ...tensor(raw[name + ".weight"]), b: b64(raw[name + ".bias"].w) });
    return { args: raw.args, card_id: t("card_id.weight"), relic_id: t("relic_id.weight"), potion_id: t("potion_id.weight"),
      boss: t("boss.weight"), room: t("room.weight"), floor: t("floor.weight"), card_mlp0: linearT("card_mlp.0"),
      card_mlp2: linearT("card_mlp.2"), relic_mlp0: linearT("relic_mlp.0"), relic_mlp2: linearT("relic_mlp.2"),
      path_mlp0: linearT("path_mlp.0"), path_mlp2: linearT("path_mlp.2"), ctx: linearT("ctx"),
      score0: linearT("score_mlp.0"), score2: linearT("score_mlp.2"), inp0: linearT("inp.1"),
      win: linearT("win_out"), blocks: Array.from({ length: raw.args.depth }, (_, i) => ({
        norm: { w: b64(raw[`blocks.${i}.0.weight`].w), b: b64(raw[`blocks.${i}.0.bias`].w) },
        l1: linearT(`blocks.${i}.2`), l2: linearT(`blocks.${i}.4`) })) };
  }
  function card(c, ctx) { return mlp(M.card_mlp0, M.card_mlp2, cat(row(M.card_id, c.card_id), [c.upgraded || 0, (c.misc || 0) / 10], ctx)); }
  function score(state, options, bossName) {
    const W = M.args.width, bossIndex = { slime_boss: 0, the_guardian: 1, hexaghost: 2 }[bossName] ?? 0;
    const hp = state.hp, max = Math.max(state.max_hp, 1), floor = state.floor;
    const scalars = [hp / 100, max / 100, hp / max, state.gold / 100, floor / 17, state.potion_capacity / 5];
    const boss = row(M.boss, bossIndex), ctx0 = cat(boss, [scalars[2], scalars[4]]);
    const deck = new Float64Array(W); for (const c of state.deck) add(deck, card(c, ctx0));
    const relic = new Float64Array(W); for (const r of state.relics) add(relic, mlp(M.relic_mlp0, M.relic_mlp2, cat(row(M.relic_id, r.relic_id), [Math.sign(r.data || 0) * Math.log1p(Math.abs(r.data || 0))])));
    const potion = new Float64Array(W); for (const p of state.potions) add(potion, row(M.potion_id, p.potion_id));
    const paths = state.map.paths.map((p) => {
      const x = new Float64Array(W); for (let f = 0; f < 15; f++) { const room = ROOM[p.rooms[f]] ?? 8; if (room !== 8) { add(x, row(M.room, room)); add(x, row(M.floor, f)); } }
      return mlp(M.path_mlp0, M.path_mlp2, x);
    });
    const meanPath = new Float64Array(W); for (const p of paths) add(meanPath, scale(p, 1 / paths.length));
    return [...options, null].map((option) => {
      const dsum = Float64Array.from(deck); if (option) add(dsum, card(option, ctx0));
      const size = state.deck.length + (option ? 1 : 0), g = cat(dsum, scale(dsum, 1 / Math.max(size, 1)), relic, potion, boss, scalars, [Math.log1p(size)]);
      const context = linear(M.ctx, g); let bestScore = -Infinity, bestPath = paths[0];
      for (const p of paths) { const q = linear(M.score2, relu(linear(M.score0, cat(p, context))))[0]; if (q > bestScore) { bestScore = q; bestPath = p; } }
      let h = relu(linear(M.inp0, cat(g, bestPath, [bestScore], meanPath)));
      for (const b of M.blocks) add(h, linear(b.l2, relu(linear(b.l1, layerNorm(b.norm, h)))));
      return sigmoid(linear(M.win, relu(h))[0]);
    });
  }
  const API = { load(raw) { M = load(raw); }, async ready() { if (!M) M = load(await json("run_value_weights.json")); }, score };
  if (typeof window !== "undefined") window.RunValueModel = API;
  if (typeof module !== "undefined") module.exports = API;
})();
