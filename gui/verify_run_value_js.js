// Browser run-value model parity: the selected v3-shop checkpoint vs logged Python evaluation.
const fs = require("fs");
const path = require("path");
const API = require("./run_value_model.js");
const read = (name) => JSON.parse(fs.readFileSync(path.join(__dirname, "site/data", name)));
API.load(read("run_value_weights.json"));
const x = read("run_value_fixture.json");
const got = API.score(x.state, x.options, x.boss);
const delta = Math.max(...got.map((v, i) => Math.abs(v - x.expected[i])));
console.log(`${got.length} run-value outputs: max |delta| = ${delta.toExponential(2)}`);
if (delta > 1e-5) process.exit(1);
