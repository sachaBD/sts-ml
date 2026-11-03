// Run with: node --test apps/gumbel_nn_combat_expert_iteration/test_web_ui.cjs
const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const vm = require('node:vm');

function dashboard(count, saved = null) {
  const elements = new Map();
  const element = () => ({ value: '', style: {}, dataset: {}, children: [],
    replaceChildren(...children) { this.children = children; } });
  const context = vm.createContext({
    document: {
      getElementById(id) {
        if (!elements.has(id)) elements.set(id, element());
        return elements.get(id);
      },
      createElement: element,
    },
    localStorage: { getItem: key => key === 'decks' ? JSON.stringify(saved) : null, setItem() {} },
    Plotly: { react() {} }, performance: { now: () => 0 }, requestAnimationFrame() {},
  });
  const source = readFileSync(`${__dirname}/web.html`, 'utf8').split('<script>')[1].split('</script>')[0];
  vm.runInContext(source.replace('poll();\nrequestAnimationFrame(frame);', ''), context);
  const decks = Array.from({ length: count }, (_, i) => `deck-${String(i).padStart(5, '0')}`);
  const rate = { w: 1, n: 2, p: .5, ci: [.1, .9] };
  const rows = ['all', ...decks].map(deck => ({ deck, train: rate }));
  context.input = {
    decks, run: 'test', state: 'running', system: {}, progress: {}, speed: {},
    train: [], test: [], trend: rows, windows: { train: 5, eval: [] },
    compare: { n: count, columns: ['a', 'b'], diffs: { a: { b: {} }, b: { a: {} } },
      rows: rows.map(r => ({ deck: r.deck, rates: { a: rate, b: rate } })) },
  };
  vm.runInContext('update(input)', context);
  return { elements, run: code => vm.runInContext(code, context) };
}

for (const count of [0, 3, 10000]) {
  test(`bounded tables and picker for ${count} decks`, () => {
    const { elements } = dashboard(count);
    for (const id of ['trend', 'compare']) {
      // Header plus pinned overall row plus one page.
      assert.equal((elements.get(id).innerHTML.match(/<tr/g) || []).length, 2 + Math.min(count, 25));
    }
    assert.ok(elements.get('chips').children.length <= 25);
    assert.ok(elements.get('selectedChips').children.length <= 6);
  });
}

test('search and pagination stay in sync without hiding overall rows', () => {
  const { elements } = dashboard(10000);
  elements.get('nextDecks').onclick();
  assert.match(elements.get('trend').innerHTML, /deck-00025/);
  assert.doesNotMatch(elements.get('trend').innerHTML, /deck-00000/);
  elements.get('deckSearch').value = 'deck-09999';
  elements.get('deckSearch').oninput();
  for (const id of ['trend', 'compare']) {
    assert.match(elements.get(id).innerHTML, /deck-09999/);
    assert.equal((elements.get(id).innerHTML.match(/<tr/g) || []).length, 3);
  }
  assert.equal(elements.get('deckPage').textContent, '1 / 1');
  assert.equal(elements.get('nextDecks').disabled, true);
  elements.get('deckSearch').value = 'missing';
  elements.get('deckSearch').oninput();
  assert.equal((elements.get('trend').innerHTML.match(/<tr/g) || []).length, 2);
});

test('old all-deck selections are capped and selections survive filtering', () => {
  const saved = Array.from({ length: 10000 }, (_, i) => `deck-${String(i).padStart(5, '0')}`);
  const { elements, run } = dashboard(10000, saved);
  assert.equal(run('selected.length'), 6);
  const extra = elements.get('chips').children[0];
  elements.get('chips').onclick({ target: extra });
  assert.equal(run('selected.length'), 6);
  elements.get('selectedChips').onclick({ target: elements.get('selectedChips').children[0] });
  assert.equal(run('selected.length'), 5);
  elements.get('deckSearch').value = 'deck-09999';
  elements.get('deckSearch').oninput();
  elements.get('chips').onclick({ target: elements.get('chips').children[0] });
  assert.equal(run('selected.length'), 6);
  assert.equal(run("selected.includes('deck-09999')"), true);
});
