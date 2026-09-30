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
