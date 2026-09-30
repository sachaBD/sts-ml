// Check the in-browser model (model.js) against the Python ensemble on the fixture written by export.py.
const fs = require("fs");
const path = require("path");
const API = require("./model.js");

const W = API.load(JSON.parse(fs.readFileSync(path.join(__dirname, "site/data/weights.json"))));
const rows = JSON.parse(fs.readFileSync(path.join(__dirname, "build/fixture.json")));
let dp = 0, dh = 0;
for (const r of rows) {
  const s = API.score(W, r.encounter, r.pre);
  dp = Math.max(dp, Math.abs(s.p_win - r.p_win));
  dh = Math.max(dh, Math.abs(s.expected_hp - r.expected_hp));
}
console.log(`${rows.length} rows: max |dp_win| = ${dp.toExponential(2)}, max |d expected_hp| = ${dh.toExponential(2)}`);
if (dp > 1e-5) process.exit(1);
