const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const vm = require('node:vm');

const sample = () => ({
  actors: [{ id: 0, requests_s: 1200, blocked_pct: 35 }],
  queues: [{ id: 0, requests_s: 1200, wait_ms: 1.25, depth: 3, peak_depth: 7 }],
  lanes: [{ id: 0, queue: 0, requests_s: 1200, busy_pct: 80, fill_pct: 75 }],
});
const element = () => ({ children: [], style: {}, classes: new Set(),
  append(...children) { this.children.push(...children); },
  replaceChildren(...children) { this.children = children; },
  get classList() { return { add: name => this.classes.add(name), remove: name => this.classes.delete(name) }; },
});
const text = node => [node.textContent || '', ...node.children.map(text)].join(' ');

async function panel(data = sample()) {
  const root = element(), status = element(), requests = [], timers = [];
  let response = { data, ok: true };
  const context = vm.createContext({ document: { createElement: element },
    fetch: async url => { requests.push(url); return { ok: response.ok, status: 503,
      json: async () => response.data[url.split('/').pop()] }; },
    setTimeout: callback => timers.push(callback),
  });
  vm.runInContext(readFileSync(`${__dirname}/telemetry.js`, 'utf8'), context);
  context.telemetryPanel(root, status);
  const settle = () => new Promise(resolve => setImmediate(resolve));
  await settle();
  return { root, status, requests, async refresh(data, ok = true) {
    response = { data, ok }; timers.shift()(); await settle();
  } };
}

test('live diagram renders only API metrics and retains nodes across refreshes', async () => {
  const view = await panel();
  assert.deepEqual(view.requests, ['api/actors', 'api/queues', 'api/lanes']);
  const output = text(view.root);
  for (const expected of ['A0', 'blocked time', '35%', 'Inference server', 'Q0 · Narrow', 'Depth 3', 'Peak 7', 'Display scale 0–8 requests', '1.25 ms', 'L0 · Q0', 'Busy', '80%', '75% batch fill', '1,200 req/s'])
    assert.ok(output.includes(expected), expected);
  assert.doesNotMatch(output, /Slots|p95|Dispatchers|H2D|CPU fallback/);
  const actor = view.root.children[0].children[1].children[0];
  const data = sample(); data.actors[0].blocked_pct = 50;
  await view.refresh(data);
  assert.equal(view.root.children[0].children[1].children[0], actor);
  assert.match(text(actor), /50%/);
});

test('unknown wait and fill remain unknown, not zero', async () => {
  const data = sample(); data.queues[0].wait_ms = null; data.lanes[0].fill_pct = null;
  data.queues[0].depth = data.queues[0].peak_depth = null;
  const view = await panel(data);
  assert.match(text(view.root.children[2]), /Mean wait · — ms/);
  assert.match(text(view.root.children[2]), /Depth —.*Peak —/);
  const queue = view.root.children[2].children[1].children[0];
  assert.equal(queue.children[2].hidden, true);
  assert.match(text(view.root.children[4]), /— batch fill/);
});

test('empty telemetry keeps the last layout without presenting it as live', async () => {
  const view = await panel();
  await view.refresh({ actors: [], queues: [], lanes: [] });
  assert.match(view.status.textContent, /showing last sample/);
  assert.ok(view.root.classes.has('unavailable'));
  assert.match(text(view.root), /A0/);
  assert.match(text(view.root), /L0 · Q0/);
});

test('HTTP and API errors retain the layout but mark readings as previous', async () => {
  const view = await panel();
  await view.refresh(sample(), false);
  assert.equal(view.status.textContent, 'Telemetry unavailable · showing last sample');
  assert.ok(view.root.classes.has('unavailable'));
  assert.match(text(view.root), /A0/);
  await view.refresh(sample());
  assert.ok(!view.root.classes.has('unavailable'));
  await view.refresh({ ...sample(), actors: { error: 'file unavailable' } });
  assert.equal(view.status.textContent, 'Telemetry unavailable · showing last sample');
  assert.match(text(view.root), /A0/);
});
