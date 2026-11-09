const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const vm = require('node:vm');

function fixture() {
  const metrics = Object.fromEntries(Object.entries({ cpu: 40, gpu: 90, vram: 6, ram: 12, disk_free: 30 })
    .map(([key, current]) => [key, { current, stale: false }]));
  return { ready: true, time: Date.now() / 1000, interval: 2, state: 'running',
    config: { run: { id: '<unsafe>' } }, metrics,
    machine: { cores: [10, 80], gpu_name: 'GPU', vram_total: 8, ram_total: 32,
      ram_available: 20, disk_total: 100, disk_used: 65 } };
}

async function dashboard(input, failure = false, mode = 'mock') {
  const elements = new Map(), requests = [], livePanels = [];
  const element = () => ({ hidden: false, children: [], style: { setProperty(key, value) { this[key] = value; } }, attrs: {},
    classList: { toggle() {}, add() {}, remove() {} },
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
    setAttribute(key, value) { this.attrs[key] = value; },
  });
  const get = id => { if (!elements.has(id)) elements.set(id, element()); return elements.get(id); };
  const context = vm.createContext({
    document: { body: { dataset: { mode } }, getElementById: get, createElement: element }, activeView: 'health', data: null,
    trainingPanel: () => () => {},
    telemetryPanel: (...args) => { livePanels.push(args); return () => {}; },
    Plotly: { react() {} }, performance: { now: () => 0 }, requestAnimationFrame() {},
    update() {}, charts() {}, Event: class {}, window: { dispatchEvent() {} }, setTimeout() {},
    fetch: async url => {
      requests.push(url);
      if (failure) throw new Error('offline');
      return { ok: true, json: async () => input };
    },
  });
  vm.runInContext(readFileSync(`${__dirname}/motion.js`, 'utf8'), context);
  vm.runInContext(readFileSync(`${__dirname}/health.js`, 'utf8'), context);
  await new Promise(resolve => setImmediate(resolve));
  return { elements, requests, context, livePanels };
}
const reading = (elements, id) => elements.get(id).children[0];
const valueText = (elements, id) => reading(elements, id).children[0].children[1].textContent;
const fill = (elements, id) => reading(elements, id).children[1].children[0];

test('only resource usage is rendered, without a chart dependency', async () => {
  const { elements, requests } = await dashboard(fixture());
  assert.deepEqual(requests, ['api/mock']);
  assert.equal(elements.get('run').textContent, '<unsafe>');
  assert.equal(valueText(elements, 'resourceCpu'), '40%');
  assert.equal(valueText(elements, 'resourceVram'), '6 / 8 GiB');
  assert.equal(fill(elements, 'resourceRam').style.width, '37.5%');
  assert.equal(fill(elements, 'resourceDisk').style.width, '65%');
  assert.equal(elements.get('resourceDiskNote').textContent, '30 GiB available');
  assert.equal(elements.get('resourceCores').children.length, 2);
  assert.equal(elements.get('resourceCores').children[1].children[0].style.height, '80%');
  assert.equal(elements.get('resourceCores').children[1].children[2].textContent, '80%');
  assert.equal(reading(elements, 'resourceCpu').children[0].children[0].textContent, 'Utilisation');
  assert.equal(elements.get('resourceStatus').hidden, true);
  assert.equal(elements.has('healthFlow'), false);
});

test('missing capacity and stale values are unavailable, not empty headroom', async () => {
  const input = fixture();
  input.metrics.gpu.stale = true;
  delete input.machine.ram_total;
  const { elements } = await dashboard(input);
  assert.equal(valueText(elements, 'resourceGpu'), 'Unavailable');
  assert.equal(valueText(elements, 'resourceRam'), 'Unavailable');
  assert.equal(reading(elements, 'resourceGpu').children[1].children.length, 0);
  input.time -= 60;
  const stale = await dashboard(input);
  assert.equal(valueText(stale.elements, 'resourceDisk'), 'Unavailable');
  assert.match(stale.elements.get('resourceStatus').textContent, /stale/);
});

test('offline measurements are explicitly marked as not current', async () => {
  const { elements } = await dashboard(fixture(), true);
  assert.match(elements.get('resourceStatus').textContent, /not current/);
});

test('view switching preserves the separate quality view', async () => {
  const { elements, context } = await dashboard(fixture());
  elements.get('qualityTab').onclick();
  assert.equal(context.activeView, 'quality');
  assert.equal(elements.get('healthPanel').hidden, true);
  assert.equal(elements.get('qualityPanel').hidden, false);
  elements.get('healthTab').onclick();
  assert.equal(context.activeView, 'health');
});

const { execFileSync } = require('node:child_process');
const { resolve } = require('node:path');
const mockScenarios = JSON.parse(execFileSync(resolve(__dirname, '../../../../.venv/bin/python'), ['-c',
  'import json; from apps.gumbel_nn_combat_expert_iteration.web.health_mock import MockHealth, SCENARIOS; print(json.dumps({s: MockHealth(s).view() for s in SCENARIOS}))'],
  { cwd: resolve(__dirname, '../../../..'), encoding: 'utf8' }));
for (const scenario of ['normal', 'fallback']) {
  test(`execution renders individual components for ${scenario}`, async () => {
    const input = structuredClone(mockScenarios[scenario]);
    input.time = input.topology.time = Date.now() / 1000;
    const { elements } = await dashboard(input);
    assert.equal(elements.get('executionActors').children.length, 9);
    assert.equal(elements.get('executionDispatchers').children.length, 2);
    assert.equal(elements.get('executionLanes').children.length, 2);
    assert.equal(elements.get('executionLanes').children[0].children[1].children.length, 5);
    assert.match(elements.get('executionStage').textContent, /MOCK DATA/);
    assert.equal(elements.get('qualityTab').disabled, true);
  });
}
for (const scenario of ['stale', 'missing']) {
  test(`execution does not present ${scenario} component snapshots as live`, async () => {
    const { elements } = await dashboard(mockScenarios[scenario]);
    assert.equal(elements.get('executionDispatchers').children.length, 0);
    assert.match(elements.get('executionFreshness').textContent, scenario === 'stale' ? /stale/ : /unavailable/);
  });
}


test('live mode uses telemetry renderer, hides mock-only views and never polls mock API', async () => {
  const { elements, requests, livePanels } = await dashboard({ phase: 'play', stage: 'iteration 4', running: true, done: 12, total: 100 }, false, 'live');
  assert.deepEqual(requests, ['api/progress']);
  assert.match(elements.get('executionStage').textContent, /iteration 4/);
  assert.equal(elements.get('fightCount').textContent, '12 / 100 · 12%');
  assert.equal(elements.get('fightFill').style.width, '12%');
  assert.equal(livePanels.length, 1);
  assert.equal(livePanels[0][0], elements.get('executionTopology'));
  for (const id of ['resourceFrame', 'executionLegend'])
    assert.equal(elements.get(id).hidden, true);
  assert.notEqual(elements.get('qualityTab').disabled, true);
});

for (const [phase, stage, running, collecting, fitting] of [
  ['play', 'iteration 4', true, true, false],
  ['play', 'test 4', true, true, false],
  ['train', 'iteration 4', true, false, true],
  ['load', 'iteration 4', true, false, true],
  ['score', 'iteration 4', true, false, true],
  ['export', 'iteration 4', true, false, false],
  ['train', 'iteration 4', false, false, false],
]) {
  test(`phase panels match ${stage}/${phase}, running=${running}`, async () => {
    const { elements } = await dashboard({ phase, stage, running }, false, 'live');
    assert.equal(elements.get('collectionPanel').hidden, !collecting);
    assert.equal(elements.get('fightThroughputPanel').hidden, !collecting);
    assert.equal(elements.get('trainingPanel').hidden, !fitting);
  });
}

test('unavailable phase does not leave phase panels visible', async () => {
  const { elements } = await dashboard({}, true, 'live');
  for (const id of ['collectionPanel', 'fightThroughputPanel', 'trainingPanel'])
    assert.equal(elements.get(id).hidden, true);
});
