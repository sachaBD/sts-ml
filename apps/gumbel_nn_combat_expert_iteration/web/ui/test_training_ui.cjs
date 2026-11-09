const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const vm = require('node:vm');

async function panel() {
  const elements = new Map(), timers = new Map();
  const el = id => {
    if (!elements.has(id)) elements.set(id, { isConnected: true, style: {},
      classList: { toggle() {} }, setAttribute() {} });
    return elements.get(id);
  };
  let now = 0, frame, sample = null;
  const context = vm.createContext({ window: {}, document: { getElementById: el },
    performance: { now: () => now }, requestAnimationFrame: callback => { frame = callback; },
    setTimeout: callback => timers.set(callback.name, callback),
    fetch: async url => ({ ok: true, json: async () => url.endsWith('/training') ? sample : {} }),
  });
  for (const file of ['motion.js', 'training.js'])
    vm.runInContext(readFileSync(`${__dirname}/${file}`, 'utf8'), context);
  const progress = context.trainingPanel();
  const settle = () => new Promise(resolve => setImmediate(resolve));
  await settle();
  return { el, progress, at(time) { now = time; frame(time); },
    async poll(value) { sample = value; timers.get('pollTraining')(); await settle(); } };
}
const sample = (step, value) => ({ step, total: 100, steps_s: value,
  value_loss: value, policy_loss: value, window_seconds: 5, seconds: step });

test('training rates replay smoothly, counts remain exact and stages never blend', async () => {
  const p = await panel();
  p.progress({ running: true, phase: 'train', stage: 'iteration 1' });
  await p.poll(sample(10, 2));
  p.at(1000); await p.poll(sample(30, 4));
  assert.equal(p.el('trainingSteps').textContent, '30 / 100');
  assert.equal(p.el('trainingValueLoss').textContent, '2');
  p.at(2000);
  assert.equal(p.el('trainingValueLoss').textContent, '3');
  assert.equal(p.el('trainingFill').style.width, '20%');
  p.progress({ running: true, phase: 'export', stage: 'iteration 1' });
  p.at(5000);
  assert.equal(p.el('trainingValueLoss').textContent, '3');
  p.progress({ running: true, phase: 'train', stage: 'iteration 2' });
  await p.poll(sample(1, 9));
  assert.equal(p.el('trainingValueLoss').textContent, '9');
});
