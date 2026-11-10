// The live execution view consumes the infra.workers.telemetry states directly.
function telemetryPanel(root, status) {
  const node = (tag, text, className) => {
    const element = document.createElement(tag);
    if (text != null) element.textContent = text;
    if (className) element.className = className;
    return element;
  };
  const number = (value, digits = 0) => Number.isFinite(value) ?
    value.toLocaleString(undefined, { maximumFractionDigits: digits }) : '—';
  const percent = value => Number.isFinite(value) ? `${number(value)}%` : '—';
  const width = (element, value) => {
    element.style.width = Number.isFinite(value) ? `${Math.max(0, Math.min(100, value))}%` : '0';
  };
  const sections = [
    ['Actors · blocked time · req/s', 'topology-actors'], ['Inference server', 'topology-service'], ['Inference lanes', 'topology-lanes'],
  ].map(([title, className]) => {
    const section = node('section', null, className), rows = node('div');
    section.append(node('h3', title), rows);
    return { section, rows, nodes: new Map() };
  });
  const arrow = () => node('div', '→', 'topology-wire');
  root.replaceChildren(sections[0].section, arrow(), sections[1].section, arrow(), sections[2].section);

  function actorView() {
    const root = node('div', null, 'topology-actor'), name = node('span', null, 'topology-actor-id');
    const bar = node('div', null, 'topology-slots'), fill = node('span', null, 'slot-waiting');
    const rate = node('span', null, 'topology-rate');
    bar.append(fill); root.append(name, bar, rate);
    const render = smoothView(root, (actor, latest) => {
      name.textContent = `A${latest.id}`;
      width(fill, actor.blocked_pct);
      fill.textContent = percent(actor.blocked_pct);
      bar.title = `Blocked wall time: ${percent(actor.blocked_pct)}; no episode ready to advance`;
      rate.textContent = number(actor.requests_s);
      rate.title = 'Inference requests submitted per second';
    });
    return { root, update: render, pause: render.pause };
  }

  function queueView() {
    const root = node('div', null, 'topology-queue'), name = node('div', null, 'topology-dispatch-title');
    const heading = node('div', null, 'topology-queue-heading'), depth = node('strong'), peakText = node('span');
    heading.append(depth, peakText);
    const meter = node('div', null, 'topology-queue-meter'), fill = node('span', null, 'queue-fill');
    const peak = node('span', null, 'queue-peak'); meter.append(fill, peak);
    const scaleText = node('div', null, 'topology-dispatch-title');
    scaleText.title = 'Visual scale only, not queue capacity';
    const wait = node('div', null, 'topology-queue-wait'), rate = node('div', null, 'topology-queue-wait');
    const lanes = node('div', null, 'topology-service-arrow');
    root.append(name, heading, meter, scaleText, rate, wait, lanes);
    let laneStates = [], lastPeak = null, lastScale = null;
    const render = smoothView(root, (queue, latest) => {
      name.textContent = `Q${queue.id}`;
      depth.textContent = `Depth ${number(latest.depth)}`;
      depth.title = 'Pending requests at the latest sample; excludes batches already gathered';
      peakText.textContent = `Peak ${number(latest.peak_depth)}`;
      peakText.title = 'Highest pending count since worker start';
      const known = Number.isFinite(queue.depth) && Number.isFinite(queue.peak_depth);
      meter.hidden = scaleText.hidden = !known;
      if (known) {
        const scale = queue.scale;
        width(fill, 100 * queue.depth / scale);
        peak.style.left = `${100 * queue.peak_depth / scale}%`;
        scaleText.textContent = `Display scale 0–${number(scale)} requests`;
      }
      rate.textContent = `${number(queue.requests_s)} req/s`;
      wait.textContent = `Mean wait · ${number(queue.wait_ms, 2)} ms`;
      wait.title = 'Enqueue to batch gathered; excludes evaluation and replies';
      lanes.textContent = '→ ' + laneStates.filter(lane => lane.queue === queue.id).map(lane => `L${lane.id}`).join(' · ');
    });
    return { root, pause: render.pause, update(queue, snap, lanes) {
      laneStates = lanes;
      const scale = Number.isFinite(queue.depth) && Number.isFinite(queue.peak_depth) ?
        2 ** Math.ceil(Math.log2(Math.max(1, queue.depth, queue.peak_depth))) : null;
      const reset = Number.isFinite(lastPeak) && Number.isFinite(queue.peak_depth) && queue.peak_depth < lastPeak;
      lastPeak = queue.peak_depth;
      render({ ...queue, scale }, snap || reset || scale !== lastScale);
      lastScale = scale;
    } };
  }

  function laneView() {
    const root = node('div', null, 'topology-lane'), heading = node('div', null, 'topology-lane-heading');
    const name = node('strong'), busyText = node('span', null, 'topology-lane-phase');
    heading.append(name, busyText);
    const timeline = node('div', null, 'topology-lane-time');
    const busy = node('span', 'Busy', 'lane-time-inference'), idle = node('span', 'Idle', 'lane-time-idle');
    timeline.append(busy, idle);
    const rate = node('div', null, 'topology-lane-throughput'), fillText = node('div', null, 'topology-lane-footer');
    const bar = node('div', null, 'topology-dispatch-batch'), fill = node('span', null, 'dispatch-fill');
    bar.append(fill); root.append(heading, timeline, rate, fillText, bar);
    const render = smoothView(root, (lane, latest) => {
      name.textContent = `L${latest.id} · Q${latest.queue}`;
      busyText.textContent = `${percent(lane.busy_pct)} busy`;
      busyText.title = 'Evaluation and reply wall time; not GPU utilisation';
      busy.textContent = lane.busy_pct >= 6.5 ? 'Busy' : '';
      idle.textContent = Number.isFinite(lane.busy_pct) && lane.busy_pct <= 93.5 ? 'Idle' : '';
      width(busy, lane.busy_pct);
      width(idle, Number.isFinite(lane.busy_pct) ? 100 - lane.busy_pct : null);
      rate.textContent = `${number(lane.requests_s)} req/s`;
      fillText.textContent = `${percent(lane.fill_pct)} batch fill`;
      width(fill, lane.fill_pct);
    });
    return { root, update: render, pause: render.pause };
  }

  function update(section, items, create, snap, lanes) {
    const ids = new Set(items.map(item => item.id));
    for (const id of section.nodes.keys()) if (!ids.has(id)) {
      section.nodes.get(id).pause(); section.nodes.delete(id);
    }
    for (const item of items) {
      if (!section.nodes.has(item.id)) section.nodes.set(item.id, create());
      section.nodes.get(item.id).update(item, snap, lanes);
    }
    const order = items.map(item => item.id).join(',');
    if (order !== section.order) {
      section.rows.replaceChildren(...items.map(item => section.nodes.get(item.id).root));
      section.order = order;
    }
  }

  let resume = false, active = true, generation = 0, phaseKey;
  let previousQueues = [];
  function pause() {
    resume = true;
    for (const section of sections) for (const view of section.nodes.values()) view.pause();
  }

  async function refresh() {
    const requestedGeneration = generation;
    try {
      if (!active) return;
      const [actors, queues, lanes] = await Promise.all(['actors', 'queues', 'lanes'].map(async name => {
        const response = await fetch('api/' + name, { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (!Array.isArray(data)) throw new Error('Invalid telemetry');
        return data;
      }));
      if (requestedGeneration !== generation || !active) return;
      if (actors.length && queues.length && lanes.length) {
        // A falling lifetime peak indicates a restarted self-play worker.
        resume ||= queues.some(q => previousQueues.some(old => old.id === q.id &&
          Number.isFinite(q.peak_depth) && Number.isFinite(old.peak_depth) && q.peak_depth < old.peak_depth));
        previousQueues = queues.map(({ id, peak_depth }) => ({ id, peak_depth }));
        update(sections[0], actors, actorView, resume);
        update(sections[1], queues, queueView, resume, lanes);
        update(sections[2], lanes, laneView, resume);
        resume = false;
        status.textContent = 'Rates over last 5 s · 1 s refresh · 1.5 s playback delay';
        root.classList.remove('unavailable');
      } else {
        pause();
        root.classList.add('unavailable');
        status.textContent = 'No active inference · holding display';
      }
    } catch (error) {
      if (requestedGeneration === generation) {
        pause();
        root.classList.add('unavailable');
        status.textContent = 'Telemetry unavailable · holding display';
      }
    } finally {
      setTimeout(refresh, 1000);
    }
  }
  refresh();
  return progress => {
    const nextActive = Boolean(progress.running && progress.phase === 'play');
    const key = [progress.stage, progress.phase, nextActive].join('|');
    if ((phaseKey !== undefined && phaseKey !== key) || active !== nextActive) {
      generation++;
      pause();
      previousQueues = [];
      root.classList.add('unavailable');
      status.textContent = nextActive ? 'Waiting for current stage telemetry' : 'No active inference';
    }
    phaseKey = key;
    active = nextActive;
  };
}
