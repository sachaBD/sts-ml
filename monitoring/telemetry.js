// Execution panel for run_actors telemetry: Actors -> queues -> lanes, from the serving page's
// api/actors, api/queues and api/lanes (monitoring/telemetry.py states), refreshed every second.
const TELEMETRY_STYLE = `
  .telemetry .status { color: var(--dim, #7d8590); }
  .telemetry .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 28px; align-items: start; }
  .telemetry .meter { display: inline-block; width: 90px; height: 6px; margin-right: 8px; vertical-align: middle;
                      border-radius: 3px; background: var(--line, #232830); }
  .telemetry .meter span { display: block; height: 100%; border-radius: 3px; background: var(--train, #6ea8fe); }`;

function telemetryPanel(root) {
  const style = document.createElement('style');
  style.textContent = TELEMETRY_STYLE;
  document.head.append(style);
  root.classList.add('telemetry');
  root.innerHTML = '<div class="status"></div><div class="grid"><table></table><table></table><table></table></div>';
  const status = root.querySelector('.status'), [actors, queues, lanes] = root.querySelectorAll('table');

  const meter = pct => `<span class="meter"><span style="width:${Math.min(100, pct)}%"></span></span>${pct.toFixed(0)}%`;
  const rate = x => Math.round(x).toLocaleString();
  const optional = (x, format) => x == null ? '—' : format(x);
  const table = (element, head, rows) => {
    element.innerHTML = `<tr>${head.map(h => `<th>${h}</th>`).join('')}</tr>` +
      rows.map(cells => `<tr>${cells.map(c => `<td>${c}</td>`).join('')}</tr>`).join('');
  };

  async function refresh() {
    try {
      const [a, q, l] = await Promise.all(['actors', 'queues', 'lanes'].map(async name =>
        (await fetch('api/' + name, { cache: 'no-store' })).json()));
      if (![a, q, l].every(Array.isArray)) throw new Error('telemetry unavailable');
      status.textContent = a.length ? '' : 'not playing';
      table(actors, ['actor', 'blocked', 'requests / s'], a.map(x => [`A${x.id}`, meter(x.blocked_pct), rate(x.requests_s)]));
      table(queues, ['queue', 'requests / s', 'wait', 'depth'], q.map(x => [`Q${x.id}`, rate(x.requests_s),
        optional(x.wait_ms, w => `${w.toFixed(2)} ms`), optional(x.wait_ms, w => (x.requests_s * w / 1000).toFixed(1))]));
      table(lanes, ['lane', 'busy', 'fill', 'requests / s'], l.map(x => [`L${x.id} · Q${x.queue}`, meter(x.busy_pct),
        optional(x.fill_pct, f => `${f.toFixed(0)}%`), rate(x.requests_s)]));
    } catch (e) {
      status.textContent = 'unavailable';
    }
    setTimeout(refresh, 1000);
  }
  refresh();
}
