// In-browser port of combat_outcome_v2 (python/sts_combat_rl/topology/combat_outcome_v2.py), eval mode.
// Exposes window.API = { meta(), predict(q) } with the same shapes as server.py's /api/meta and /api/predict.
(() => {
  const b64 = (s) => new Float32Array(Uint8Array.from(atob(s), (c) => c.charCodeAt(0)).buffer);

  // y = W x + b, W stored row-major (out x in) as in torch.
  function linear(p, x) {
    const [out, inp] = p.shape, W = p.w, y = new Float64Array(out);
    for (let o = 0; o < out; o++) {
      let s = p.b[o];
      for (let i = 0, k = o * inp; i < inp; i++, k++) s += W[k] * x[i];
      y[o] = s;
    }
    return y;
  }
  const relu = (x) => x.map((v) => (v > 0 ? v : 0));
  const row = (emb, i) => emb.w.subarray(i * emb.shape[1], (i + 1) * emb.shape[1]);
  const cat = (...xs) => Float64Array.from(xs.flatMap((x) => Array.from(x)));
  const addTo = (a, b) => { for (let i = 0; i < a.length; i++) a[i] += b[i]; };

  function layerNorm(p, x) {
    const n = x.length, mean = x.reduce((s, v) => s + v, 0) / n;
    const v = x.reduce((s, y) => s + (y - mean) ** 2, 0) / n, d = Math.sqrt(v + 1e-5);
    return x.map((y, i) => ((y - mean) / d) * p.w[i] + p.b[i]);
  }

  // One network, one state -> [win_logit, hp_logits]. Mirrors train.batch + CombatOutcomeV2.forward.
  function forward(net, encounters, encounter, pre) {
    const e = encounters[encounter] ?? 0;
    const s = [pre.hp / 100, pre.max_hp / 100, pre.hp / pre.max_hp, pre.potion_capacity / 5, Math.log1p(pre.deck.length)];
    const enc = row(net.encounter, e);
    const width = enc.length;
    const sum = new Float64Array(width);
    for (const c of pre.deck) {
      const x = cat(row(net.card_id, c.card_id + 1), [c.upgraded, c.misc / 10], enc, [s[2]]);
      addTo(sum, linear(net.card_mlp2, relu(linear(net.card_mlp0, x))));
    }
    const mean = sum.map((v) => v / Math.max(pre.deck.length, 1));
    const relic = new Float64Array(width);
    for (const r of pre.relics) {
      const d = Math.sign(r.data) * Math.log1p(Math.abs(r.data));
      addTo(relic, linear(net.relic_mlp2, relu(linear(net.relic_mlp0, cat(row(net.relic_id, r.relic_id + 1), [d])))));
    }
    const potion = new Float64Array(width);
    for (const p of pre.potions) addTo(potion, row(net.potion_id, p.potion_id + 1));

    let h = relu(linear(net.inp, cat(sum, mean, relic, potion, enc, s)));
    for (const blk of net.blocks) addTo(h, linear(blk.l2, relu(linear(blk.l1, layerNorm(blk.ln, h)))));
    h = relu(h);
    // Per-encounter linear term: intercept + slope * hp / 100, for (win, bins).
    const k = net.enc_linear.shape[2], base = e * 2 * k, EL = net.enc_linear.w;
    const lin = (j) => EL[base + j] + EL[base + k + j] * s[0];
    const win = linear(net.win_out, h)[0] + lin(0);
    const hp = linear(net.hp_out, h).map((v, j) => v + lin(j + 1));
    return [win, hp];
  }

  const sigmoid = (x) => 1 / (1 + Math.exp(-x));
  function softmax(x) {
    const m = Math.max(...x), e = x.map((v) => Math.exp(v - m)), z = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / z);
  }
  const avg = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;

  // Ensemble score for one state: same fields as gui/model.py Ensemble.score.
  function score(W, encounter, pre) {
    const per = W.nets.map((net) => forward(net, W.encounters, encounter, pre));
    const p = per.map(([w]) => sigmoid(w));
    const bins = per.map(([, h]) => softmax(h));
    const hpm = bins.map((b) => b.reduce((s, v, i) => s + v * W.centers[i], 0));
    const ev = p.map((v, i) => v * hpm[i]);
    return { p_win: avg(p), p_seeds: p, hp_if_win: avg(hpm), expected_hp: avg(ev), ev_seeds: ev,
             hp_bins: W.centers.map((_, i) => avg(bins.map((b) => b[i]))) };
  }

  function load(raw) {
    const t = (x) => ({ shape: x.shape, w: b64(x.w), b: x.b ? b64(x.b) : null });
    return { encounters: raw.encounters, centers: raw.centers, nets: raw.nets.map((n) => ({
      card_id: t(n.card_id), card_mlp0: t(n.card_mlp0), card_mlp2: t(n.card_mlp2),
      relic_id: t(n.relic_id), relic_mlp0: t(n.relic_mlp0), relic_mlp2: t(n.relic_mlp2),
      potion_id: t(n.potion_id), encounter: t(n.encounter), inp: t(n.inp), enc_linear: t(n.enc_linear),
      win_out: t(n.win_out), hp_out: t(n.hp_out),
      blocks: n.blocks.map((b) => ({ ln: t(b.ln), l1: t(b.l1), l2: t(b.l2) })) })) };
  }

  let W, M;
  const API = {
    load, score,
    async meta() {
      const [m, w] = await Promise.all(["data/meta.json", "data/weights.json"].map((u) => fetch(u).then((r) => r.json())));
      W = load(w); M = m;
      return m;
    },
    // Same response as server.py predict(): the state, the state plus each candidate, and every fight.
    async predict(q) {
      const base = score(W, q.encounter, q.pre);
      const candidates = (q.candidates || []).map((c) =>
        score(W, q.encounter, { ...q.pre, deck: [...q.pre.deck, { ...c, misc: 0 }] }));
      const fights = {};
      for (const es of Object.values(M.groups)) for (const e of es) fights[e] = score(W, e, q.pre);
      return { base, candidates, fights };
    },
  };
  if (typeof window !== "undefined") window.API = API;
  if (typeof module !== "undefined") module.exports = API;
})();
