"""Build a browser-only checkpoint demo, or explicitly publish it to gh-pages."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import tomllib

import onnx

ROOT = Path(__file__).resolve().parents[2]
ORT_VERSION = '1.23.2'
OWNED = ('gumbel-tree', 'assets/gumbel')


def git(*args, cwd=ROOT):
    return subprocess.check_output(['git', '-C', str(cwd), *args], text=True).strip()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(checkpoint, fights_file, wasm, runtime, site, simulator, count=8):
    checkpoint, site = checkpoint.resolve(), site.resolve()
    network = tomllib.loads((checkpoint / 'network.toml').read_text())
    encoding = network['encoding']
    if encoding not in ('entity', 'inventory_v1'):
        raise ValueError(f'unsupported browser encoding: {encoding}')
    model_name = network['cpu']['file']
    if Path(model_name).name != model_name:
        raise ValueError('CPU model must be a file beside network.toml')
    model = checkpoint / model_name
    graph = onnx.load(model, load_external_data=False)
    external = sorted({entry.value for tensor in graph.graph.initializer for entry in tensor.external_data
                       if entry.key == 'location'})
    if any(Path(name).name != name or name in ('.', '..') for name in external):
        raise ValueError('external weights must be files beside the model')
    if json.loads((runtime / 'package.json').read_text())['version'] != ORT_VERSION:
        raise ValueError(f'expected onnxruntime-web {ORT_VERSION}')
    with fights_file.open() as source:
        rows = [json.loads(line) for line in source if line.strip()]
    if not rows or count < 1:
        raise ValueError('need fights and a positive fight count')
    n = min(count, len(rows))
    # Starts are often grouped by deck; spread the selection across the input file.
    selected = [rows[i * (len(rows) - 1) // max(n - 1, 1)] for i in range(n)]
    fights = [dict(id=row['fight_id'], start_json=json.dumps(row['start'], separators=(',', ':'))) for row in selected]
    if len({f['id'] for f in fights}) != len(fights):
        raise ValueError('selected fight IDs must be unique')
    files = [model, *(checkpoint / name for name in external), wasm / 'gumbel.js', wasm / 'gumbel.wasm',
             *(runtime / 'dist' / name for name in
               ('ort.wasm.min.js', 'ort-wasm-simd-threaded.mjs', 'ort-wasm-simd-threaded.wasm'))]
    if len({file.name for file in files}) != len(files):
        raise ValueError('model and runtime filenames must not collide')
    if any(file.stat().st_size >= 100 * 1024**2 for file in files):
        raise ValueError('a deployment file exceeds the gh-pages Git file limit (100 MiB)')
    hashes = {file.name: digest(file) for file in files}
    checkpoint_id = hashlib.sha256(json.dumps({name: hashes[name] for name in [model_name, *external]},
                                             sort_keys=True).encode()).hexdigest()
    label = f'{checkpoint.parents[2].name} / {checkpoint.name}' if checkpoint.parent.name == 'training' else checkpoint.name
    manifest = dict(checkpoint=label, checkpoint_id=checkpoint_id, encoding=encoding, model=model_name,
                    external=external, fights=fights, sha256=hashes, onnxruntime_web=ORT_VERSION,
                    source_commit=git('rev-parse', 'HEAD'), source_dirty=bool(git('status', '--porcelain')),
                    simulator_commit=git('rev-parse', 'HEAD', cwd=simulator),
                    simulator_dirty=bool(git('status', '--porcelain', cwd=simulator)))
    site.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=site.parent) as temporary:
        stage = Path(temporary)
        shutil.copytree(ROOT / 'apps/combat_inspector/web', stage / 'gumbel-tree')
        assets = stage / 'assets/gumbel'
        assets.mkdir(parents=True)
        for file in files:
            shutil.copy(file, assets / file.name)
        shutil.copy(ROOT / 'apps/combat_inspector/wasm/THIRD_PARTY_NOTICES.txt', assets)
        (assets / 'manifest.json').write_text(json.dumps(manifest))
        for owned in OWNED:
            destination = site / owned
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                shutil.rmtree(destination)
            shutil.move(stage / owned, destination)
    (site / 'index.html').write_text('<!doctype html><meta charset="utf-8"><meta http-equiv="refresh" '
                                     'content="0;url=gumbel-tree/"><a href="gumbel-tree/">Gumbel tree viewer</a>')
    (site / '.nojekyll').touch()
    print(f'Built {label}, {len(fights)} fights: {site}')
    return manifest


def publish(site):
    """Replace only the viewer and its assets; keep the existing landing page and other reports."""
    for owned in OWNED:
        if not (site / owned).is_dir():
            raise ValueError(f'missing prepared directory: {owned}')
    git('fetch', 'origin', 'gh-pages')
    base = git('rev-parse', 'FETCH_HEAD')
    with tempfile.TemporaryDirectory() as temporary:
        checkout = Path(temporary) / 'publish'
        git('worktree', 'add', '--detach', str(checkout), base)
        try:
            for owned in OWNED:
                destination = checkout / owned
                if destination.exists():
                    shutil.rmtree(destination)
                shutil.copytree(site / owned, destination)
            (checkout / '.nojekyll').touch()
            git('add', '--', *OWNED, '.nojekyll', cwd=checkout)
            if not git('diff', '--cached', '--name-only', cwd=checkout):
                print('Pages already matches the prepared viewer')
                return
            git('commit', '-m', 'Publish current combat inspector browser demo', cwd=checkout)
            # Normal push rejects concurrent changes: no force-push or replacement of other reports.
            git('push', 'origin', 'HEAD:refs/heads/gh-pages', cwd=checkout)
        finally:
            git('worktree', 'remove', '--force', str(checkout))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build')
    build.add_argument('--checkpoint', type=Path, required=True)
    build.add_argument('--fights', type=Path, required=True)
    build.add_argument('--wasm', type=Path, default=ROOT / 'build/gumbel-wasm')
    build.add_argument('--runtime', type=Path, default=ROOT / 'build/web-deps/node_modules/onnxruntime-web')
    build.add_argument('--simulator', type=Path, default=ROOT.parent / 'lightspeed')
    build.add_argument('--fight-count', type=int, default=8)
    for command in (build, commands.add_parser('publish')):
        command.add_argument('--site', type=Path, default=ROOT / 'build/gumbel-site')
    args = parser.parse_args()
    site = args.site.resolve()
    if not site.is_relative_to(ROOT / 'build') or site == ROOT / 'build':
        parser.error('--site must be a dedicated directory under build/')
    if args.command == 'publish':
        publish(site)
    else:
        export(args.checkpoint, args.fights, args.wasm, args.runtime, site, args.simulator, args.fight_count)


if __name__ == '__main__':
    main()
