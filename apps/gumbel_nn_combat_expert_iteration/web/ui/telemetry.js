// The live execution view consumes the existing monitoring.telemetry states directly.
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
    return { root, update(actor) {
      name.textContent = `A${actor.id}`;
      width(fill, actor.blocked_pct);
      fill.textContent = percent(actor.blocked_pct);
      bar.title = `Blocked wall time: ${percent(actor.blocked_pct)}; no episode ready to advance`;
      rate.textContent = number(actor.requests_s);
      rate.title = 'Inference requests submitted per second';
    } };
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
    return { root, update(queue, laneStates) {
      name.textContent = queue.id === 0 ? 'Q0 · Narrow' : queue.id === 1 ? 'Q1 · Wide / oversized' : `Q${queue.id}`;
      depth.textContent = `Depth ${number(queue.depth)}`;
      depth.title = 'Pending requests at the latest sample; excludes batches already gathered';
      peakText.textContent = `Peak ${number(queue.peak_depth)}`;
      peakText.title = 'Highest pending count since worker start';
      const known = Number.isFinite(queue.depth) && Number.isFinite(queue.peak_depth);
      meter.hidden = scaleText.hidden = !known;
      if (known) {
        const scale = 2 ** Math.ceil(Math.log2(Math.max(1, queue.depth, queue.peak_depth)));
        width(fill, 100 * queue.depth / scale);
        peak.style.left = `${100 * queue.peak_depth / scale}%`;
        scaleText.textContent = `Display scale 0–${number(scale)} requests`;
      }
      rate.textContent = `${number(queue.requests_s)} req/s`;
      wait.textContent = `Mean wait · ${number(queue.wait_ms, 2)} ms`;
      wait.title = 'Enqueue to batch gathered; excludes evaluation and replies';
      lanes.textContent = '→ ' + laneStates.filter(lane => lane.queue === queue.id).map(lane => `L${lane.id}`).join(' · ');
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
    return { root, update(lane) {
      name.textContent = `L${lane.id} · Q${lane.queue}`;
      busyText.textContent = `${percent(lane.busy_pct)} busy`;
      busyText.title = 'Evaluation and reply wall time, including CPU fallback; not GPU utilisation';
      busy.textContent = lane.busy_pct >= 6.5 ? 'Busy' : '';
      idle.textContent = Number.isFinite(lane.busy_pct) && lane.busy_pct <= 93.5 ? 'Idle' : '';
      width(busy, lane.busy_pct);
      width(idle, Number.isFinite(lane.busy_pct) ? 100 - lane.busy_pct : null);
      rate.textContent = `${number(lane.requests_s)} req/s`;
      fillText.textContent = `${percent(lane.fill_pct)} batch fill`;
      width(fill, lane.fill_pct);
    } };
  }

  function update(section, items, create, lanes) {
    const ids = new Set(items.map(item => item.id));
    for (const id of section.nodes.keys()) if (!ids.has(id)) section.nodes.delete(id);
    for (const item of items) {
      if (!section.nodes.has(item.id)) section.nodes.set(item.id, create());
      section.nodes.get(item.id).update(item, lanes);
    }
    section.rows.replaceChildren(...items.map(item => section.nodes.get(item.id).root));
  }

  async function refresh() {
    try {
      const [actors, queues, lanes] = await Promise.all(['actors', 'queues', 'lanes'].map(async name => {
        const response = await fetch('api/' + name, { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (!Array.isArray(data)) throw new Error('Invalid telemetry');
        return data;
      }));
      update(sections[0], actors, actorView);
      update(sections[1], queues, queueView, lanes);
      update(sections[2], lanes, laneView);
      status.textContent = actors.length || queues.length || lanes.length ?
        'Rates over last 5 s · 1 s refresh' : 'Not playing / no telemetry';
      root.classList.remove('unavailable');
    } catch (error) {
      sections.forEach(section => { section.rows.replaceChildren(); section.nodes.clear(); });
      root.classList.add('unavailable');
      status.textContent = 'Telemetry unavailable';
    }
    setTimeout(refresh, 1000);
  }
  refresh();
}
