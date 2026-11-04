const UI_HALF_LIFE_MS = 3000;
const animatedViews = new Set();
const reducedMotion = window.matchMedia?.('(prefers-reduced-motion: reduce)')?.matches;

// Polls only replace targets; the animation clock never restarts between polls.
function smoothView(element, render) {
  let current = null, target = null;
  const frame = k => {
    if (!element.isConnected) { animatedViews.delete(frame); return; }
    for (const key of Object.keys(target)) {
      current[key] = Number.isFinite(current[key]) && Number.isFinite(target[key]) ?
        current[key] + (target[key] - current[key]) * k : target[key];
    }
    render(current);
  };
  return (values, snap = false) => {
    target = values;
    if (current === null || snap || reducedMotion) current = { ...target };
    // Unknown values and categorical changes are immediate, never eased through zero.
    for (const key of Object.keys(target))
      if (!Number.isFinite(target[key]) || !Number.isFinite(current[key])) current[key] = target[key];
    render(current);
    if (!reducedMotion) animatedViews.add(frame);
  };
}

let motionTime = performance.now();
function animateViews(now) {
  const k = -Math.expm1(-Math.LN2 * (now - motionTime) / UI_HALF_LIFE_MS);
  motionTime = now;
  for (const frame of animatedViews) frame(k);
  requestAnimationFrame(animateViews);
}
requestAnimationFrame(animateViews);
