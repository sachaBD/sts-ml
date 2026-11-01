"""Freeze one Gumbel checkpoint and selected fights into a standalone Pages site.

Does not load the Python GUI backend, train anything, or publish the result.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import onnx

ROOT = Path(__file__).resolve().parents[1]
ORT_VERSION = "1.23.2"


def export(checkpoint, fights_file, wasm, runtime, site, fight_count):
    checkpoint, site = checkpoint.resolve(), site.resolve()
    model = checkpoint / "model.onnx"
    graph = onnx.load(model, load_external_data=False)
    external = sorted({entry.value for tensor in graph.graph.initializer for entry in tensor.external_data
                       if entry.key == "location"})
    if any(Path(name).name != name or name in {".", ".."} for name in external):
        raise ValueError("external weights must be files beside model.onnx")
    iteration = int(checkpoint.name.removeprefix("iter"))
    run = checkpoint.parents[2]
    with fights_file.open() as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not rows:
        raise ValueError("no fights to publish")
    # Inputs are often grouped by deck: spread the demo across the file, not just its first group.
    count = min(fight_count, len(rows))
    selected = [rows[i * (len(rows) - 1) // max(count - 1, 1)] for i in range(count)]
    fights = [dict(id=row["fight_id"], start_json=json.dumps(row["start"], separators=(",", ":"))) for row in selected]
    package = json.loads((runtime / "package.json").read_text())
    if package["version"] != ORT_VERSION:
        raise ValueError(f"expected onnxruntime-web {ORT_VERSION}")
    files = [model, *(checkpoint / name for name in external), wasm / "gumbel.js", wasm / "gumbel.wasm",
             *(runtime / "dist" / name for name in (
                 "ort.wasm.min.js", "ort-wasm-simd-threaded.mjs", "ort-wasm-simd-threaded.wasm"))]
    for file in files:
        if not file.is_file():
            raise FileNotFoundError(file)
    # Only a dedicated generated output directory may be updated; never overwrite source or a run.
    if site == ROOT or ROOT.is_relative_to(site) or site.is_relative_to(ROOT / "gui/web") or site.is_relative_to(run):
        raise ValueError("site must be a separate generated directory")
    site.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "gui/web/gumbel-tree", site / "gumbel-tree", dirs_exist_ok=True)
    assets = site / "assets"
    assets.mkdir(exist_ok=True)
    for name in ("style.css", "gui.css"):
        shutil.copy(ROOT / "gui/web/assets" / name, assets / name)
    output = assets / "gumbel"
    output.mkdir(exist_ok=True)
    hashes = {}
    for file in files:
        shutil.copy(file, output / file.name)
        hashes[file.name] = hashlib.sha256(file.read_bytes()).hexdigest()
    shutil.copy(ROOT / "gui/wasm/THIRD_PARTY_NOTICES.txt", output / "THIRD_PARTY_NOTICES.txt")
    manifest = dict(run=run.name, id=run.name.removeprefix("id="), date=run.parent.name.removeprefix("date="),
                    iteration=iteration, model="model.onnx", external=external, fights=fights,
                    onnxruntime_web=ORT_VERSION, sha256=hashes,
                    source_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    source_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)))
    (output / "manifest.json").write_text(json.dumps(manifest))
    page = site / "gumbel-tree/index.html"
    html = page.read_text().replace('<script type="module" src="app.js">',
                                   '<script>window.GUMBEL_BROWSER = true;</script>\n<script type="module" src="app.js">')
    html = html.replace('Local only: needs <code>gui/server.py</code>\n      and <code>nn_explain</code>.',
                        'Frozen checkpoint · all inference and search run in your browser. First use downloads the model and WASM runtime.')
    page.write_text(html)
    (site / "index.html").write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" content="0;url=gumbel-tree/">'
                                     '<a href="gumbel-tree/">Gumbel tree viewer</a>')
    (site / ".nojekyll").touch()
    size = sum(p.stat().st_size for p in site.rglob("*") if p.is_file())
    print(f"Frozen iter{iteration:03d}, {len(fights)} fights: {site} ({size / 1e6:.1f} MB uncompressed)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, type=Path, help="directory containing model.onnx and its external weights")
    parser.add_argument("--fights", required=True, type=Path, help="combat_v4 start records, e.g. out/inputs/test.jsonl")
    parser.add_argument("--wasm", type=Path, default=ROOT / "build/gumbel-wasm")
    parser.add_argument("--runtime", type=Path, default=ROOT / "build/web-deps/node_modules/onnxruntime-web")
    parser.add_argument("--site", type=Path, default=ROOT / "build/gumbel-site")
    parser.add_argument("--fight-count", type=int, default=8)
    args = parser.parse_args()
    if args.fight_count < 1:
        parser.error("--fight-count must be positive")
    export(args.checkpoint, args.fights, args.wasm, args.runtime, args.site, args.fight_count)


if __name__ == "__main__":
    main()
