const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const vm = require('node:vm');

const sample = () => ({
  actors: [{ id: 0, requests_s: 1200, blocked_pct: 35 }],
  queues: [{ id: 0, requests_s: 1200, wait_ms: 1.25, depth: 3, peak_depth: 7 }],
  lanes: [{ id: 0, queue: 0, requests_s: 1200, busy_pct: 80, fill_pct: 75 }],
});
const element = () => ({ isConnected: true, children: [], style: {}, classes: new Set(),
  append(...children) { this.children.push(...children); },
  replaceChildren(...children) { this.children = children; this.replacements = (this.replacements || 0) + 1; },
  get classList() { return { add: name => this.classes.add(name), remove: name => this.classes.delete(name) }; },
});
const text = node => [node.textContent || '', ...node.children.map(text)].join(' ');

async function panel(data = sample()) {
  const root = element(), status = element(), requests = [], timers = [];
  let response = { data, ok: true }, now = 0, nextFrame;
  const context = vm.createContext({ window: {}, performance: { now: () => now },
    requestAnimationFrame: callback => { nextFrame = callback; }, document: { createElement: element },
    fetch: async url => { requests.push(url); return { ok: response.ok, status: 503,
      json: async () => response.data[url.split('/').pop()] }; },
    setTimeout: callback => timers.push(callback),
  });
  vm.runInContext(readFileSync(`${__dirname}/motion.js`, 'utf8'), context);
  vm.runInContext(readFileSync(`${__dirname}/telemetry.js`, 'utf8'), context);
  const progress = context.telemetryPanel(root, status);
  const settle = () => new Promise(resolve => setImmediate(resolve));
  await settle();
  return { root, status, requests, progress, advance(ms) { now += ms; nextFrame(now); }, async refresh(data, ok = true, during = () => {}) {
    response = { data, ok }; timers.shift()(); during(); await settle();
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
  view.advance(1000);
  await view.refresh(data);
  assert.equal(view.root.children[0].children[1].children[0], actor);
  assert.equal(view.root.children[0].children[1].replacements, 1, 'unchanged rows are never remounted');
  assert.match(text(actor), /35%/);
  view.advance(1000);
  const fill = actor.children[1].children[0];
  assert.ok(Math.abs(parseFloat(fill.style.width) - 42.5) < 1e-9, 'playback is halfway between the two samples');
  view.advance(500);
  assert.equal(fill.style.width, '50%', 'reaches the sample exactly; no exponential tail');
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
  assert.match(view.status.textContent, /holding display/);
  assert.ok(view.root.classes.has('unavailable'));
  assert.match(text(view.root), /A0/);
  assert.match(text(view.root), /L0 · Q0/);
});

test('HTTP and API errors retain the layout but mark readings as previous', async () => {
  const view = await panel();
  await view.refresh(sample(), false);
  assert.equal(view.status.textContent, 'Telemetry unavailable · holding display');
  assert.ok(view.root.classes.has('unavailable'));
  assert.match(text(view.root), /A0/);
  await view.refresh(sample());
  assert.ok(!view.root.classes.has('unavailable'));
  await view.refresh({ ...sample(), actors: { error: 'file unavailable' } });
  assert.equal(view.status.textContent, 'Telemetry unavailable · holding display');
  assert.match(text(view.root), /A0/);
});

test('worker reset snaps every section and failures stop buffered animation', async () => {
  const view = await panel();
  view.advance(1000);
  const data = sample(); data.actors[0].blocked_pct = 60;
  await view.refresh(data);
  view.advance(1000);
  const actor = view.root.children[0].children[1].children[0];
  assert.match(text(actor), /48%/);
  await view.refresh(sample(), false);
  view.advance(5000);
  assert.match(text(actor), /48%/);
  data.actors[0].blocked_pct = 90;
  await view.refresh(data);
  assert.match(text(actor), /90%/);
  view.advance(1000);
  data.actors[0].blocked_pct = 10; data.queues[0].peak_depth = 1;
  await view.refresh(data);
  assert.match(text(actor), /10%/);
});


test('phase identity resets playback and rejects responses from the previous phase', async () => {
  const view = await panel();
  view.progress({ stage: 'iteration 1', phase: 'play', running: true });
  const actor = view.root.children[0].children[1].children[0];
  const data = sample(); data.actors[0].blocked_pct = 90;
  view.advance(1000);
  await view.refresh(data, true, () => {
    view.progress({ stage: 'iteration 2', phase: 'play', running: true });
  });
  view.advance(3000);
  assert.match(text(actor), /35%/, 'late previous-stage response never reaches the display');
  assert.ok(view.root.classes.has('unavailable'));
  await view.refresh(data);
  assert.match(text(actor), /90%/, 'first new-stage sample snaps rather than blending');
  assert.ok(!view.root.classes.has('unavailable'));
  view.progress({ stage: 'iteration 2', phase: 'train', running: true });
  const count = view.requests.length;
  await view.refresh(data);
  assert.equal(view.requests.length, count, 'inactive inference does not poll');
});
