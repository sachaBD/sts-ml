/* Current resource usage only; run-quality plots remain in their own view. */
(() => {
  const el = id => document.getElementById(id);
  const valid = value => typeof value === 'number' && Number.isFinite(value);
  const fmt = value => valid(value) ? value.toLocaleString(undefined, { maximumFractionDigits: 1 }) : '—';
  const node = (tag, text, className) => {
    const element = document.createElement(tag);
    if (text != null) element.textContent = text;
    if (className) element.className = className;
    return element;
  };

  function usage(id, label, value, total, unit) {
    const known = valid(value) && valid(total) && total > 0;
    const row = node('div', null, 'resource-reading');
    const heading = node('div', null, 'resource-heading');
    heading.append(node('span', label), node('strong', known ?
      (unit === '%' ? `${fmt(value)}%` : `${fmt(value)} / ${fmt(total)} ${unit}`) : 'Unavailable'));
    const meter = node('div', null, 'resource-meter');
    meter.setAttribute('role', 'meter');
    meter.setAttribute('aria-label', label);
    if (known) {
      meter.setAttribute('aria-valuemin', '0'); meter.setAttribute('aria-valuemax', String(total));
      meter.setAttribute('aria-valuenow', String(value));
      const fill = node('div');
      fill.style.width = `${Math.max(0, Math.min(100, value / total * 100))}%`;
      meter.append(fill);
    } else meter.setAttribute('aria-valuetext', 'Unavailable');
    row.append(heading, meter);
    el(id).replaceChildren(row);
  }

  let mock = false;
  const componentNodes = new Map();
  let queueNodes = null;

  function components(id, items, create, update) {
    let group = componentNodes.get(id);
    if (!group) { group = { nodes: new Map(), order: '' }; componentNodes.set(id, group); }
    const order = items.map(item => item.id).join(',');
    for (const item of items) if (!group.nodes.has(item.id)) group.nodes.set(item.id, create(item.id));
    if (group.order !== order || !group.mounted) {
      el(id).replaceChildren(...items.map(item => group.nodes.get(item.id).root));
      group.order = order; group.mounted = true;
      for (const key of group.nodes.keys()) if (!items.some(item => item.id === key)) group.nodes.delete(key);
    }
    for (const item of items) update(group.nodes.get(item.id), item);
  }

  function execution(d, samplerStale) {
    const config = d.config.workers || {}, progress = d.progress || {}, snapshot = d.topology;
    const running = ['running', 'stopping'].includes(d.state), playing = running && progress.phase === 'play';
    const age = snapshot ? Math.max(0, Date.now() / 1000 - snapshot.time) : null;
    const fresh = playing && !samplerStale && age != null && age < 5;
    const state = fresh ? snapshot : null;
    const percent = value => valid(value) ? `${Math.round(value * 100)}%` : '—';
    const width = (element, value) => { element.style.width = valid(value) ? `${100 * Math.max(0, Math.min(1, value))}%` : '0'; };
    const phase = progress.stage?.startsWith('test') ? 'Evaluate' : progress.phase === 'play' ? 'Collect' :
      progress.phase === 'export' ? 'Export' : ['score', 'load', 'train'].includes(progress.phase) ? 'Fit' : null;
    el('executionDriver').replaceChildren(...['Collect', 'Fit', 'Export', 'Evaluate'].map((name, i) => {
      const step = node('span', name, 'execution-phase');
      step.classList.toggle('active', running && !samplerStale && phase === name);
      if (i < 3) step.append(node('span', ' →', 'execution-phase-arrow'));
      return step;
    }));
    el('executionStage').textContent = [d.mock ? 'MOCK DATA' : null, progress.stage,
      valid(progress.done) && valid(progress.total) ? `${fmt(progress.done)} / ${fmt(progress.total)}` : null].filter(Boolean).join(' · ');
    el('executionFreshness').textContent = samplerStale ? 'Telemetry stale' : !running ? d.state :
      !playing ? 'Idle' : !snapshot ? 'Component telemetry unavailable' : !fresh ? 'Component telemetry stale' :
      `${snapshot.window_seconds}s window · ${d.interval}s refresh`;
    el('executionMode').textContent = d.mock && state ? `DEMO · ${state.mode}` : '';
    el('executionTopology').classList.toggle('unavailable', !fresh);
    const actors = state?.actors || Array.from({ length: config.actors || 0 }, (_, id) => ({ id }));
    components('executionActors', actors, id => {
      const root = node('div', null, 'topology-actor');
      const name = node('span', `A${id}`, 'topology-actor-id');
      const slots = node('div', null, 'topology-slots');
      const ready = node('span', null, 'slot-ready'), waiting = node('span', null, 'slot-waiting'), empty = node('span', null, 'slot-empty');
      slots.append(ready, waiting, empty);
      const rate = node('span', null, 'topology-rate');
      root.append(name, slots, rate);
      return { root, name, slots, ready, waiting, empty, rate };
    }, (n, actor) => {
      const total = actor.ready + actor.waiting + actor.empty;
      n.slots.classList.toggle('unknown', !valid(total));
      for (const key of ['ready', 'waiting', 'empty']) width(n[key], actor[key] / total);
      n.slots.title = valid(total) ? `${fmt(actor.ready)} ready · ${fmt(actor.waiting)} awaiting inference · ${fmt(actor.empty)} empty (window averages)` : 'Slot telemetry unavailable';
      n.name.title = `${percent(actor.working_fraction)} working wall time; not CPU utilisation`;
      n.name.style.setProperty('--activity', valid(actor.working_fraction) ? actor.working_fraction : 0);
      n.rate.textContent = fmt(actor.requests_s);
      n.rate.title = 'Inference requests submitted per second';
    });
    if (!queueNodes) {
      const root = node('div', null, 'topology-queue');
      const heading = node('div', null, 'topology-queue-heading');
      const average = node('strong'), peakText = node('span'); heading.append(average, peakText);
      const meter = node('div', null, 'topology-queue-meter'), fill = node('span', null, 'queue-fill'), peak = node('span', null, 'queue-peak');
      meter.append(fill, peak);
      const wait = node('div', null, 'topology-queue-wait');
      root.append(node('div', 'Request queue', 'topology-dispatch-title'), heading, meter, wait);
      el('executionQueue').replaceChildren(root); queueNodes = { average, peakText, fill, peak, wait };
    }
    const queue = state?.queue;
    queueNodes.average.textContent = `Avg ${fmt(queue?.depth)}`;
    queueNodes.peakText.textContent = `Peak ${fmt(queue?.peak)}`;
    width(queueNodes.fill, queue ? queue.depth / queue.capacity : null);
    queueNodes.peak.style.left = queue ? `${Math.min(100, 100 * queue.peak / queue.capacity)}%` : '0';
    queueNodes.peak.hidden = !queue;
    queueNodes.wait.textContent = `Queue wait p95 · ${fmt(queue?.wait_p95_ms)} ms`;
    queueNodes.wait.title = queue ? `p50 ${fmt(queue.wait_p50_ms)} ms · n=${fmt(queue.samples)} requests` : 'Queue telemetry unavailable';
    components('executionDispatchers', state?.dispatchers || [], id => {
      const root = node('div', null, 'topology-dispatcher');
      const rate = node('span'), bar = node('div', null, 'topology-dispatch-batch'), fill = node('span', null, 'dispatch-fill');
      bar.append(fill); root.append(node('strong', `D${id}`), rate, bar);
      return { root, rate, fill, bar };
    }, (n, dispatcher) => {
      n.rate.textContent = `${fmt(dispatcher.batches_s)} batches/s`;
      width(n.fill, dispatcher.batch_fill);
      n.bar.title = `${percent(dispatcher.batch_fill)} batch fill`;
    });
    const laneTypes = [['packing', 'Pack'], ['h2d', 'H2D'], ['inference', 'Infer'], ['d2h', 'D2H'], ['idle', 'Idle']];
    const lanes = state?.lanes || Array.from({ length: d.native?.gpu_stats?.lanes || 0 }, (_, id) => ({ id }));
    components('executionLanes', lanes, id => {
      const root = node('div', null, 'topology-lane');
      const heading = node('div', null, 'topology-lane-heading'), occupied = node('span', null, 'topology-lane-phase');
      heading.append(node('strong', `Lane ${id}`), occupied);
      const timeline = node('div', null, 'topology-lane-time'), parts = {};
      for (const [key, label] of laneTypes) {
        parts[key] = node('span', label, `lane-time-${key}`); timeline.append(parts[key]);
      }
      const rate = node('div', null, 'topology-lane-throughput'), batches = node('span'), fill = node('span');
      const footer = node('div', null, 'topology-lane-footer'); footer.append(batches, fill);
      root.append(heading, timeline, rate, footer);
      return { root, occupied, parts, rate, batches, fill };
    }, (n, lane) => {
      n.occupied.textContent = `${percent(lane.occupied_fraction)} occupied`;
      n.occupied.title = 'Lane occupied wall time, not GPU compute utilisation';
      for (const [key, label] of laneTypes) {
        const fraction = lane.time_fractions?.[key];
        width(n.parts[key], fraction);
        n.parts[key].textContent = fraction >= .065 ? label : '';
        n.parts[key].title = `${label}: ${percent(fraction)} of the ${snapshot?.window_seconds || '—'}s window`;
      }
      n.rate.textContent = `${fmt(lane.evals_s)} evals/s`;
      n.batches.textContent = `${fmt(lane.batches_s)} batches/s`;
      n.fill.textContent = `${percent(lane.batch_fill)} batch fill`;
    });
    const fallback = d.metrics.fallback;
    el('executionFallback').textContent = fallback && valid(fallback.current) && !fallback.stale && playing && !samplerStale ?
      `CPU fallback ${fallback.current.toLocaleString(undefined, { maximumFractionDigits: 3 })}%` : '';
  }

  function render(d) {
    if (!d.ready) { el('resourceStatus').textContent = 'Waiting for measurements…'; return; }
    const stale = Date.now() / 1000 - d.time > 10;
    const current = key => {
      const m = d.metrics[key];
      return !stale && m && !m.stale ? m.current : null;
    };
    const machine = stale ? {} : d.machine;
    el('run').textContent = d.config.run?.id || 'Training';
    el('stateText').textContent = d.state;
    el('state').className = 'muted ' + d.state;
    el('stage').textContent = '';
    el('healthFreshness').textContent = stale ? 'Measurements stale' : `${d.interval}s refresh`;
    usage('resourceCpu', 'Utilisation', current('cpu'), 100, '%');
    usage('resourceGpu', 'Utilisation', current('gpu'), 100, '%');
    usage('resourceVram', 'VRAM used', current('vram'), machine.vram_total, 'GiB');
    usage('resourceRam', 'Used', current('ram'), machine.ram_total, 'GiB');
    usage('resourceDisk', 'Used', machine.disk_used, machine.disk_total, 'GiB');
    const cores = machine.cores || [];
    el('resourceCores').replaceChildren(...cores.map((value, i) => {
      const column = node('div', null, 'resource-core');
      column.title = `CPU ${i}: ${fmt(value)}% utilisation`;
      const fill = node('div');
      fill.style.height = valid(value) ? `${Math.max(0, Math.min(100, value))}%` : '0';
      column.append(fill, node('span', String(i), 'resource-core-index'),
        node('span', valid(value) ? `${Math.round(value)}%` : '—', 'resource-core-value'));
      return column;
    }));
    el('resourceGpuNote').textContent = machine.gpu_name || '';
    el('resourceRamNote').textContent = valid(machine.ram_available) ? `${fmt(machine.ram_available)} GiB available` : 'Available: —';
    el('resourceDiskNote').textContent = valid(current('disk_free')) ? `${fmt(current('disk_free'))} GiB available` : 'Available: —';
    el('resourceStatus').textContent = stale ? 'Measurements stale' : '';
    el('resourceStatus').hidden = !stale;
    el('healthPanel').classList.remove('offline');
    mock = !!d.mock;
    el('qualityTab').disabled = mock;
    execution(d, stale);
  }

  async function pollHealth() {
    if (activeView === 'health') {
      try {
        const response = await fetch('api/health', { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const d = await response.json();
        if (d.error) throw new Error(d.error);
        if (activeView === 'health') render(d);
      } catch (error) {
        el('healthFreshness').textContent = 'Telemetry offline';
        el('resourceStatus').textContent = 'Offline · measurements not current';
        el('resourceStatus').hidden = false;
        el('healthPanel').classList.add('offline');
        el('executionFreshness').textContent = 'Telemetry offline';
      }
    }
    setTimeout(pollHealth, mock ? 1000 : 2000);
  }
  function switchView(view) {
    activeView = view;
    el('healthPanel').hidden = view !== 'health'; el('qualityPanel').hidden = view !== 'quality';
    el('qualitySmooth').hidden = view !== 'quality'; el('healthFreshness').hidden = view !== 'health';
    for (const name of ['health', 'quality']) {
      el(name + 'Tab').classList.toggle('active', view === name);
      el(name + 'Tab').setAttribute('aria-selected', String(view === name));
    }
    if (view === 'quality' && data) { update(data); charts(); }
    window.dispatchEvent(new Event('resize'));
  }
  el('healthTab').onclick = () => switchView('health');
  el('qualityTab').onclick = () => switchView('quality');
  pollHealth();
})();
