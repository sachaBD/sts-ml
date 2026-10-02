"""Build the GitHub Pages site from gui/web/ into gui/site/.

The fight-outcome page gets the in-browser model at export time; local development
uses server.py's API instead.

  .venv/bin/python gui/export.py && node gui/verify_js.js
"""
import base64
import json
import re
import shutil
from pathlib import Path

import numpy as np
import torch

from model import CHECKPOINTS
from server import HERE, META, VIS, dev, ens

SITE = HERE / "site"
RUN_VALUE_CHECKPOINT = HERE.parents[0] / "runs/schema=run_rl_v1/date=2026-10-01/id=v3-shop/out/iter003/model.pt"
RUN_VALUE_LOG = HERE.parents[0] / "runs/schema=run_rl_v1/date=2026-10-01/id=v3-shop/out/iter003/eval/runs.jsonl"


def b64(t):
    return base64.b64encode(t.detach().float().numpy().astype("<f4").tobytes()).decode()


def run_value_weights():
    """The selected run_policy_v1 checkpoint in browser-friendly float32 base64 tensors."""
    checkpoint = torch.load(RUN_VALUE_CHECKPOINT, map_location="cpu", weights_only=False)
    return dict(args=checkpoint["args"], **{
        name: dict(shape=list(tensor.shape), w=b64(tensor))
        for name, tensor in checkpoint["state_dict"].items()
    })


def run_value_fixture():
    """One logged card reward for Python/JS parity."""
    for line in RUN_VALUE_LOG.open():
        run = json.loads(line)
        if run["seed"] != 600000000489:
            continue
        for step in run["steps"]:
            if step["kind"] == "pick" and [x["name"] for x in step["options"]] == ["spot_weakness", "twin_strike", "burning_pact"]:
                return dict(boss=run["boss"], state=step["state"], options=step["options"], expected=step["values"])
    raise RuntimeError("run-value parity fixture not found")


def run_value_example():
    """A deliberately simple floor-10 base-deck state for the public card evaluator."""
    for line in RUN_VALUE_LOG.open():
        run = json.loads(line)
        for step in run["steps"]:
            state = step.get("state") or step.get("after")
            if isinstance(state, dict) and state.get("floor") == 10 and state.get("map", {}).get("paths"):
                return dict(boss=run["boss"], state=dict(
                    deck=[dict(card_id=15, upgraded=0, misc=0, name="ascenders_bane"),
                          *[dict(card_id=321, upgraded=0, misc=0, name="strike_red") for _ in range(5)],
                          *[dict(card_id=104, upgraded=0, misc=0, name="defend_red") for _ in range(4)],
                          dict(card_id=25, upgraded=0, misc=0, name="bash")],
                    floor=10, hp=80, max_hp=80, gold=99, potion_capacity=3, potions=[],
                    relics=[dict(relic_id=86, data=0, name="burning_blood")], map=state["map"]), options=[])
    raise RuntimeError("floor-10 map example not found")


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
    # web/ mirrors the published URL layout, so adding a report is just adding a page there.
    shutil.copytree(HERE / "web", SITE)
    (SITE / "data").mkdir()
    shutil.copy(HERE / "browser_model.js", SITE / "assets/browser-model.js")
    shutil.copy(HERE / "model.js", SITE / "assets/model.js")
    shutil.copy(HERE / "run_value_model.js", SITE / "assets/run-value-model.js")
    # On Pages model.js provides window.API; the source page otherwise uses server.py locally.
    page = SITE / "fight-outcome/index.html"
    html = page.read_text()
    page.write_text(html.replace('<script src="app.js">', '<script src="../assets/browser-model.js"></script>\n<script src="../assets/model.js"></script>\n<script src="app.js">'))
    shutil.copytree(VIS / "data/art", SITE / "vis/data/art")
    shutil.copy(VIS / "data/cards.js", SITE / "vis/data/cards.js")
    (SITE / "data/meta.json").write_text(json.dumps(META))
    (SITE / "data/weights.json").write_text(json.dumps(weights()))
    (SITE / "data/run_value_weights.json").write_text(json.dumps(run_value_weights()))
    (SITE / "data/run_value_example.json").write_text(json.dumps(run_value_example()))
    (SITE / "data/run_value_fixture.json").write_text(json.dumps(run_value_fixture()))
    (SITE / ".nojekyll").write_text("")

    # Parity fixture for verify_js.js: held-out natural states and the Python ensemble's scores.
    rows = dev[::50][:300]
    scored = ens.score([dict(encounter=r["encounter"], pre=r["pre"]) for r in rows])
    fixture = [dict(encounter=r["encounter"], pre=r["pre"], p_win=s["p_win"], expected_hp=s["expected_hp"])
               for r, s in zip(rows, scored)]
    (HERE / "build").mkdir(exist_ok=True)
    (HERE / "build/fixture.json").write_text(json.dumps(fixture))
    size = sum(f.stat().st_size for f in SITE.rglob("*") if f.is_file())
    print(f"site: {SITE} ({size / 1e6:.1f} MB), fixture: {len(fixture)} rows")


if __name__ == "__main__":
    main()
