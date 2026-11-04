const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const vm = require('node:vm');

function dashboard(saved = {}) {
  const elements = new Map(), plots = new Map(), storage = new Map(Object.entries(saved));
  const context = vm.createContext({
    document: { body: { dataset: { mode: 'mock' } }, getElementById(id) {
      if (!elements.has(id)) elements.set(id, {});
      return elements.get(id);
    } },
    localStorage: { getItem: key => storage.get(key), setItem: (key, value) => storage.set(key, String(value)) },
    performance: { now: () => 0 }, requestAnimationFrame() {},
    Plotly: { react: (id, traces) => plots.set(id, traces) },
  });
  const html = readFileSync(`${__dirname}/web.html`, 'utf8');
  vm.runInContext(html.match(/<script>([\s\S]*?)<\/script>/)[1], context);
  vm.runInContext(`update({train: [2, 4, null, 10].map((value, it) => ({
    it, loss: {value, policy: value}, play: value, fit: value, gps: value,
    gpu: {util: value, vram: value}, nn: value,
    rates: {all: {w: 1, n: 2}}
  })), test: [], system: {}})`, context);
  return { context, elements, plots, storage };
}

test('slider averages each iteration chart, skips missing values and leaves quality smoothing alone', () => {
  const { context, elements, plots, storage } = dashboard();
  assert.deepEqual(Array.from(plots.get('loss')[0].y), [2, 4, null, 10]);
  elements.get('iterationSmooth').oninput({ target: { value: '3' } });
  assert.equal(elements.get('iterationSmoothVal').textContent, 3);
  assert.equal(storage.get('iterationSmooth'), '3');
  for (const id of ['loss', 'gpuPlot', 'nnPlot']) {
    assert.deepEqual(Array.from(plots.get(id)[0].y), [2, 3, 3, 7]);
  }
  assert.deepEqual(Array.from(plots.get('time')[0].y), [2 / 60, 3 / 60, 3 / 60, 7 / 60]);
  assert.equal(vm.runInContext('smooth', context), 1);
});

test('saved iteration window is restored independently of quality', () => {
  const { elements, plots } = dashboard({ iterationSmooth: '2', smooth: '5' });
  assert.equal(elements.get('iterationSmooth').value, 2);
  assert.equal(elements.get('smooth').value, 5);
  assert.deepEqual(Array.from(plots.get('loss')[0].y), [2, 3, 4, 10]);
});

test('heldout losses share head colours, axes and smoothing but use dotted lines', () => {
  const { context, elements, plots } = dashboard();
  vm.runInContext(`data.train.forEach((b, i) => {
    b.loss.heldout_value = [4, 6, null, 12][i];
    b.loss.heldout_policy = [8, 10, null, 16][i];
  }); charts()`, context);
  elements.get('iterationSmooth').oninput({ target: { value: '2' } });
  const [value, policy, heldoutValue, heldoutPolicy] = plots.get('loss');
  assert.equal(heldoutValue.line.color, value.line.color);
  assert.equal(heldoutPolicy.line.color, policy.line.color);
  assert.equal(heldoutValue.yaxis, value.yaxis);
  assert.equal(heldoutPolicy.yaxis, policy.yaxis);
  assert.equal(heldoutValue.line.dash, 'dot');
  assert.equal(heldoutPolicy.line.dash, 'dot');
  assert.deepEqual(Array.from(heldoutValue.y), [4, 5, 6, 12]);
  assert.deepEqual(Array.from(heldoutPolicy.y), [8, 9, 10, 16]);
});
