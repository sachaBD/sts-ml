# Fight outcome GUI

Web UI for the pre-combat outcome model (Ironclad A20, Acts 1–2, including the Act 2 bosses). Pick a deck, relics,
potions, HP and a fight to see P(win), the final-HP distribution if won, take-vs-skip for candidate cards, the deck
against every fight, and real vs predicted win rates on held-out fights.

**Live:** https://sachabd.github.io/sts-ml/

## Run locally

```
.venv/bin/python gui/server.py            # http://127.0.0.1:8732/  (~40 s startup, CPU: scores ~61k held-out fights)
```

## Publish (GitHub Pages)

```
.venv/bin/python gui/export.py            # build gui/site/ (static, ~6 MB)
node gui/verify_js.js                     # Fight Outcome JS model vs Python ensemble, expect max |dp_win| ~1e-7
node gui/verify_run_value_js.js           # Run Value JS model vs logged Python evaluation
rm -rf /tmp/pages && cp -r gui/site /tmp/pages && cd /tmp/pages && git init -b gh-pages && git add -A \
  && git commit -m site && git push -f git@github.com:sachaBD/sts-ml.git gh-pages
```

Pages serves the `gh-pages` branch of sachaBD/sts-ml. The live site only changes when you republish.

## Files

| File | Role |
|---|---|
| `model.py` | 3-seed ensemble (`act2-co2-w32-h64-l1-d30-lr.001-s{0,1,2}/natural.pt`), encoded via `learn.batch`; id tables from `../sts_lightspeed` headers; fight groups (act × easy/hard/elite/boss), presets and calibration from held-out natural fights (run_seed % 10 ∈ {0,1,8}) of `combat_transition_v1/2026-10-02/act2-overworld-fights` |
| `server.py` | Local server: `GET /api/meta`, `POST /api/predict`, card art from `../sts_visualiser` at `/vis/` |
| `model.js` | Plain-JS port of `combat_outcome_v2`; on the published fight-outcome page it replaces the server |
| `export.py` | Writes `site/` from `web/`, adds the browser model/data/art, and writes the parity fixture in `build/` |
| `verify.py` / `verify_js.js` | Parity checks: Python path vs saved predictions; JS vs Python |
| `web/` | Source tree for the static site, mirroring published URLs: shared assets, the landing page, and one directory per report |

## Adding a report

Add `web/<report-name>/index.html` (and a page-specific script if needed), then link it from `web/index.html`.
Use paths relative to that page, such as `../assets/style.css` and `../data/...`; this works both locally and on GitHub Pages under `/sts-ml/`.
If a report needs model inference, load `../assets/model.js` in the exported page and set `window.GUI_ROOT = "../"` before it. `export.py` currently does this for `fight-outcome`; extend its small injection step for another inference page.

## Model and data

The ensemble is trained on the natural fights of every `experiments/act2` overworld run (~206k fights with a verified
pre-fight state, ~6k of them Act 2 bosses), extracted by `apps/combat_transition/from_overworld.py`, plus the Act 1
card experiments (`card_marginals_v1`). Retrain with `experiments/act2/train_fight_outcome.sh`. Each run trains a
`natural` arm (natural fights only) and an `augmented` arm. The GUI uses `natural`: it is better calibrated on the
current agent's fights. The card experiments were played by an older, weaker combat agent. Switch with `ARM` in `model.py`.

## Caveats

- Development model. The seed band is the min–max over 3 training seeds: rough, not calibrated.
- Elite/boss card effects explain only ~20–35% of real effect variance; boss P(win) near 0.5 is least reliable.
- Act 2 boss P(win) on held-out fights (n = 1766): Brier 0.149 vs 0.231 for a constant base rate (skill ≈ 0.35). Part of
  that is just HP. Act 2 card effects are learnt from the agent's own decks only, so they are rougher than Act 1's.
- Pre-fight states are rebuilt from overworld records; about 10% of fights are dropped as unverifiable. Final HP is
  inverted from the post-fight heal (approximate in the ~9% of wins where the heal hit max HP).
- Cards and relics you add get `misc` / `data` = 0.
