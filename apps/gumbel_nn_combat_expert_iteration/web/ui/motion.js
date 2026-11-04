const UI_HALF_LIFE_MS = 3000; // Quality headline only.
const animatedViews = new Set();
const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches;

// Play receipt-timestamped samples 1.5 s behind polling, without extrapolation.
// The second render argument is the latest exact sample (counts, IDs, peaks).
function smoothView(element, render, delay = 1500) {
  let samples = [], latest;
  const frame = now => {
    if (element.isConnected === false) { animatedViews.delete(frame); return; }
    const time = now - delay;
    while (samples.length > 1 && samples[1].time <= time) samples.shift();
    const a = samples[0], b = samples[1] || a;
    const fraction = b.time > a.time ? Math.max(0, Math.min(1, (time - a.time) / (b.time - a.time))) : 1;
    const values = { ...latest };
    for (const key of Object.keys(latest)) {
      // Missing readings are immediate; never blend them through zero.
      if (Number.isFinite(latest[key]) && Number.isFinite(a.values[key]) && Number.isFinite(b.values[key]))
        values[key] = a.values[key] + (b.values[key] - a.values[key]) * fraction;
    }
    render(values, latest);
    if (samples.length === 1) animatedViews.delete(frame);
  };
  const update = (values, snap = false) => {
    const now = performance.now(), previous = samples[samples.length - 1];
    latest = { ...values };
    if (snap || reducedMotion || !previous || now - previous.time > 3 * delay) samples = [];
    if (samples.length && previous.time === now) samples.pop();
    samples.push({ time: now, values: latest });
    frame(now);
    if (!reducedMotion && samples.length > 1) animatedViews.add(frame);
  };
  // Stop playback on failure/phase changes; the next sample starts a new timeline.
  update.pause = () => { samples = []; animatedViews.delete(frame); };
  return update;
}

function animateViews(now) {
  for (const frame of animatedViews) frame(now);
  requestAnimationFrame(animateViews);
}
requestAnimationFrame(animateViews);
