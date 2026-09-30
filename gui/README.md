# Fight outcome GUI

Local web UI for the pre-combat outcome model (brief: `rundecks/outcome-ui/BRIEF.md`).

```
.venv/bin/python gui/server.py            # http://127.0.0.1:8732/  (~10 s startup, CPU only)
cd gui && ../.venv/bin/python verify.py   # reproduce saved seed-0 p_win through the GUI path
```

- `model.py`: 3-seed ensemble (`tpair1-co2-w32-h64-l1-d30-lr.001-s{0,1,2}`), encoding via `train.batch`; card/relic/potion
  id tables parsed from `../sts_lightspeed/include/constants/*.h`; presets = real natural pre-fight states
  (non-reserved buckets of `act1-eval-mcts-a20-checked`), up to 6 per encounter, bosses first.
- `server.py`: stdlib HTTP server. `GET /api/meta`, `POST /api/predict`; `/vis/*` serves card art from `../sts_visualiser`.
- `static/`: page; `style.css` is copied from `../sts_visualiser`.

The band is min–max over the 3 training seeds: rough, not calibrated. Added cards/relics get `misc`/`data` = 0.

## Static site (GitHub Pages)

```
.venv/bin/python gui/export.py   # -> gui/site/ (~6 MB: page, model.js, data/meta.json, data/weights.json, card art)
node gui/verify_js.js            # model.js vs the Python ensemble (max |dp_win| ~1e-7)
```

`model.js` is a plain-JS port of `combat_outcome_v2` (no WASM, no server); `gui/site/` is fully static and works from a
subpath. `site/` and `build/` are git-ignored build output.

Live: https://sachabd.github.io/sts-ml/ (Pages serves the `gh-pages` branch of sachaBD/sts-ml). To republish:
`rm -rf /tmp/pages && cp -r gui/site /tmp/pages && cd /tmp/pages && git init -b gh-pages && git add -A &&
git commit -m site && git push -f git@github.com:sachaBD/sts-ml.git gh-pages`
