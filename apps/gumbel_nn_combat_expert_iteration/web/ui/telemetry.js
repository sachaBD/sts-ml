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
  const sections = ['Actors', 'Queues', 'Inference lanes'].map(title => {
    const section = node('section'), rows = node('div');
    section.append(node('h3', title), rows);
    return { section, rows, nodes: new Map() };
  });
  const arrow = () => node('div', '→', 'topology-wire');
  root.replaceChildren(sections[0].section, arrow(), sections[1].section, arrow(), sections[2].section);

  function metric(label) {
    const row = node('div', null, 'live-metric'), text = node('span');
    const meter = node('div', null, 'live-meter'), fill = node('span');
    meter.append(fill); row.append(node('span', label), meter, text);
    return { row, text, fill };
  }
  function percent(metric, value) {
    metric.text.textContent = `${number(value)}${Number.isFinite(value) ? '%' : ''}`;
    metric.fill.style.width = Number.isFinite(value) ? `${Math.max(0, Math.min(100, value))}%` : '0';
  }
  function update(section, items, kind) {
    const ids = new Set(items.map(item => item.id));
    for (const id of section.nodes.keys()) if (!ids.has(id)) section.nodes.delete(id);
    for (const item of items) {
      let view = section.nodes.get(item.id);
      if (!view) {
        const root = node('div', null, 'topology-lane'), heading = node('div', null, 'topology-lane-heading');
        const name = node('strong'), rate = node('span', null, 'topology-rate');
        heading.append(name, rate); root.append(heading);
        const primary = kind === 'queue' ? null : metric(kind === 'actor' ? 'Blocked' : 'Busy');
        const fill = kind === 'lane' ? metric('Fill') : null;
        const wait = kind === 'queue' ? node('div', null, 'topology-lane-footer') : null;
        if (primary) root.append(primary.row);
        if (fill) root.append(fill.row);
        if (wait) root.append(wait);
        view = { root, name, rate, primary, fill, wait };
        section.nodes.set(item.id, view);
      }
      view.name.textContent = kind === 'actor' ? `A${item.id}` : kind === 'queue' ? `Q${item.id}` : `L${item.id} · Q${item.queue}`;
      view.rate.textContent = `${number(item.requests_s)} req/s`;
      if (kind === 'queue') view.wait.textContent = `Mean wait · ${number(item.wait_ms, 2)} ms`;
      else percent(view.primary, kind === 'actor' ? item.blocked_pct : item.busy_pct);
      if (view.fill) percent(view.fill, item.fill_pct);
    }
    section.rows.replaceChildren(...items.map(item => section.nodes.get(item.id).root));
  }

  async function refresh() {
    try {
      const states = await Promise.all(['actors', 'queues', 'lanes'].map(async name => {
        const response = await fetch('api/' + name, { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const data = await response.json();
        if (!Array.isArray(data)) throw new Error('Invalid telemetry');
        return data;
      }));
      states.forEach((items, i) => update(sections[i], items, ['actor', 'queue', 'lane'][i]));
      status.textContent = states.some(items => items.length) ? 'Last 5 s · 1 s refresh' : 'Not playing / no telemetry';
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
