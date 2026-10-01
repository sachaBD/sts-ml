const results = [
  [59.9, "SimpleAgent", "all overworld decisions"],
  [74.6, "V: card rewards", "+14.7 ± 1.6 pp"],
  [88.9, "V: + rest + path", "about +14 pp"],
  [91.1, "V: + shop", "+2.2 ± 1.1 pp"],
];
document.querySelector("#results").replaceChildren(...results.map(([rate, label, note]) => {
  const el = document.createElement("div"); el.className = "result";
  el.innerHTML = `<span class="rate">${rate.toFixed(1)}%</span><span class="label">${label}<br>${note}</span><div class="bar"><i style="width:${rate}%"></i></div>`;
  return el;
}));

const choices = (target, rows) => document.querySelector(target).replaceChildren(...rows.map(([name, value, selected]) => {
  const el = document.createElement("div"); el.className = "choice" + (selected ? " selected" : "");
  el.innerHTML = `<span>${name}</span><strong>V ${value.toFixed(3)}</strong>`;
  return el;
}));
// Same browser-only model pattern as Fight Outcome: weights are exported data, never a Pages backend.
(async () => {
  const [example, meta] = await Promise.all([
    fetch("../data/run_value_example.json").then((r) => r.json()),
    fetch("../data/meta.json").then((r) => r.json()),
    RunValueModel.ready(),
  ]);
  const cards = meta.cards.filter((c) => c.color === "red");
  const byName = Object.fromEntries(cards.map((c) => [c.name.toLowerCase(), c]));
  document.querySelector("#card-list").innerHTML = cards.map((c) => `<option value="${c.name}">`).join("");
  let options = example.options.map((c) => ({ card_id: c.card_id, upgraded: c.upgraded, misc: c.misc, name: c.name }));
  const render = () => {
    const values = RunValueModel.score(example.state, options, example.boss);
    const rows = [...options.map((c, i) => [c.name.replaceAll("_", " "), values[i], i]), ["Skip", values.at(-1), null]];
    const best = Math.max(...values);
    document.querySelector("#cards").replaceChildren(...rows.map(([name, value, i]) => {
      const el = document.createElement("div");
      el.className = "choice" + (value === best ? " selected" : "");
      el.innerHTML = `<span>${name}${i !== null ? ` <button class="drop" data-drop="${i}" title="remove">×</button>` : ""}</span><strong>V ${value.toFixed(3)}</strong>`;
      return el;
    }));
  };
  document.querySelector("#candidate-add").onclick = () => {
    const input = document.querySelector("#candidate"), card = byName[input.value.trim().toLowerCase()];
    if (!card) { input.setCustomValidity("Choose a listed Ironclad card"); input.reportValidity(); return; }
    input.setCustomValidity(""); input.value = "";
    options = [{ card_id: card.id, upgraded: 0, misc: 0, name: card.key }]; render();
  };
  document.querySelector("#candidate").onkeydown = (e) => { if (e.key === "Enter") document.querySelector("#candidate-add").click(); };
  document.querySelector("#cards").onclick = (e) => { const i = e.target.dataset.drop; if (i !== undefined) { options.splice(Number(i), 1); render(); } };
  render();
})().catch((error) => { console.error(error); document.querySelector("#cards").textContent = "model failed to load"; });

