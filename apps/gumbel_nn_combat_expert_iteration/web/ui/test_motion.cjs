const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const vm = require('node:vm');

function playback(reduced = false) {
  let now = 0, frame, displayed, exact;
  const context = vm.createContext({
    window: { matchMedia: () => ({ matches: reduced }) },
    performance: { now: () => now }, requestAnimationFrame: callback => { frame = callback; },
    element: { isConnected: true }, render: (values, latest) => { displayed = values; exact = latest; },
  });
  vm.runInContext(readFileSync(`${__dirname}/motion.js`, 'utf8'), context);
  const update = vm.runInContext('smoothView(element, render)', context);
  return { update, context, at(time) { now = time; frame(now); },
    get value() { return displayed; }, get exact() { return exact; } };
}

test('timestamped samples replay linearly with bounded delay and no extrapolation', () => {
  const p = playback();
  p.update({ rate: 0 }); p.at(1000); p.update({ rate: 100 });
  p.at(1500); assert.equal(p.value.rate, 0);
  p.at(2000); assert.equal(p.value.rate, 50);
  p.at(2500); assert.equal(p.value.rate, 100);
  p.at(10000); assert.equal(p.value.rate, 100);
});

test('poll jitter does not restart playback or change the prior segment', () => {
  const p = playback();
  p.update({ rate: 0 }); p.at(1000); p.update({ rate: 100 });
  p.at(2100); assert.equal(p.value.rate, 60);
  p.update({ rate: 210 }); assert.equal(p.value.rate, 60);
  p.at(3000); assert.equal(p.value.rate, 150);
  p.at(3600); assert.equal(p.value.rate, 210);
});

test('exact sample is separate from the interpolated presentation', () => {
  const p = playback();
  p.update({ count: 1 }); p.at(1000); p.update({ count: 9 });
  p.at(2000);
  assert.equal(p.value.count, 5); assert.equal(p.exact.count, 9);
});

test('missing values, resets, failures and long gaps never bridge old timelines', () => {
  const p = playback();
  p.update({ rate: 100 }); p.at(1000); p.update({ rate: null });
  assert.equal(p.value.rate, null);
  p.update({ rate: 10 }, true); assert.equal(p.value.rate, 10);
  p.at(2000); p.update({ rate: 20 }); p.at(4000);
  p.update.pause(); p.at(8000); assert.equal(p.value.rate, 20);
  p.update({ rate: 90 }); assert.equal(p.value.rate, 90);
  p.at(14000); p.update({ rate: 1 }); assert.equal(p.value.rate, 1);
});

test('reduced motion presents samples directly and disconnected views stop rendering', () => {
  const reduced = playback(true);
  reduced.update({ rate: 0 }); reduced.at(1000); reduced.update({ rate: 100 });
  assert.equal(reduced.value.rate, 100);
  const p = playback(); p.update({ rate: 0 }); p.at(1000); p.update({ rate: 100 });
  p.context.element.isConnected = false; p.at(2000);
  assert.equal(p.value.rate, 0);
});


test('new detached rows render immediately, animate after mounting and stop after removal', () => {
  const p = playback();
  p.context.element.isConnected = false;
  p.update({ rate: 12 });
  assert.equal(p.value.rate, 12, 'initial content does not depend on a future poll');
  p.context.element.isConnected = true;
  p.at(1000); p.update({ rate: 24 }); p.at(2000);
  assert.equal(p.value.rate, 18);
  p.context.element.isConnected = false; p.at(2500);
  assert.equal(p.value.rate, 18);
});
