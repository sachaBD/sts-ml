"""Build the static site (GitHub Pages) into gui/site/: page + in-browser model (model.js) + JSON data + card art.

  .venv/bin/python gui/export.py && node gui/verify_js.js
"""
import base64
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch

from model import CHECKPOINTS, natural_dev
from server import HERE, META, VIS, ens

SITE = HERE / "site"


def b64(t):
    return base64.b64encode(t.detach().float().numpy().astype("<f4").tobytes()).decode()


def weights():
    """State dicts -> {name: {shape, w, b}} with float32 base64 arrays, in model.js's layout."""
    nets = []
    for path in CHECKPOINTS:
        sd = torch.load(path, map_location="cpu")["state_dict"]
        lin = lambda k: dict(shape=list(sd[k + ".weight"].shape), w=b64(sd[k + ".weight"]), b=b64(sd[k + ".bias"]))
        emb = lambda k: dict(shape=list(sd[k].shape), w=b64(sd[k]))
        depth = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
        nets.append(dict(card_id=emb("card_id.weight"), card_mlp0=lin("card_mlp.0"), card_mlp2=lin("card_mlp.2"),
                         relic_id=emb("relic_id.weight"), relic_mlp0=lin("relic_mlp.0"), relic_mlp2=lin("relic_mlp.2"),
                         potion_id=emb("potion_id.weight"), encounter=emb("encounter.weight"), inp=lin("inp.1"),
                         enc_linear=emb("enc_linear"), win_out=lin("win_out"), hp_out=lin("hp_out"),
                         blocks=[dict(ln=lin(f"blocks.{i}.0"), l1=lin(f"blocks.{i}.2"), l2=lin(f"blocks.{i}.4"))
                                 for i in range(depth)]))
    return dict(encounters=ens.encounters, centers=ens.centers, nets=nets)


def main():
    shutil.rmtree(SITE, ignore_errors=True)
    (SITE / "data").mkdir(parents=True)
    for f in ["app.js", "gui.css", "style.css"]:
        shutil.copy(HERE / "static" / f, SITE / f)
    shutil.copy(HERE / "model.js", SITE / "model.js")
    # Same page; model.js provides window.API so app.js never calls the server.
    html = (HERE / "static/index.html").read_text()
    (SITE / "index.html").write_text(html.replace('<script src="app.js">', '<script src="model.js"></script>\n<script src="app.js">'))
    shutil.copytree(VIS / "data/art", SITE / "vis/data/art")
    shutil.copy(VIS / "data/cards.js", SITE / "vis/data/cards.js")
    (SITE / "data/meta.json").write_text(json.dumps(META))
    (SITE / "data/weights.json").write_text(json.dumps(weights()))
    (SITE / ".nojekyll").write_text("")

    # Parity fixture for verify_js.js: held-out natural states and the Python ensemble's scores.
    rows = natural_dev()[::10][:300]
    scored = ens.score([dict(encounter=r["encounter"], pre=r["pre"]) for r in rows])
    fixture = [dict(encounter=r["encounter"], pre=r["pre"], p_win=s["p_win"], expected_hp=s["expected_hp"])
               for r, s in zip(rows, scored)]
    (HERE / "build").mkdir(exist_ok=True)
    (HERE / "build/fixture.json").write_text(json.dumps(fixture))
    size = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file())
    print(f"site: {SITE} ({size / 1e6:.1f} MB), fixture: {len(fixture)} rows")


if __name__ == "__main__":
    main()
