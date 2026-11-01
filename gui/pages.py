import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

from gui.export_gumbel import ROOT, export


def git(repository, *args):
    return subprocess.check_output(["git", "-C", str(repository), *args], text=True).strip()


def hashes(site):
    result = {}
    for path in sorted(site.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"site contains a symlink: {path}")
        if path.is_file():
            result[path.relative_to(site).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def prepare(site, checkpoint, fights, wasm, runtime, count, remote="origin", repository=ROOT):
    site = site.resolve()
    build = (repository / "build").resolve()
    if site == build or not site.is_relative_to(build):
        raise ValueError("site must be a directory inside build/")
    ref = f"refs/remotes/{remote}/gh-pages"
    git(repository, "fetch", remote, f"refs/heads/gh-pages:{ref}")
    base = git(repository, "rev-parse", ref)
    site.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=site.parent) as temporary:
        stage = Path(temporary) / "site"
        stage.mkdir()
        with tempfile.TemporaryFile() as archive:
            subprocess.run(["git", "-C", str(repository), "archive", base], stdout=archive, check=True)
            archive.seek(0)
            with tarfile.open(fileobj=archive) as tree:
                tree.extractall(stage, filter="data")
        # Replace only our reports/assets; everything else comes from the published tree.
        for owned in ("gumbel-tree", "assets/gumbel"):
            shutil.rmtree(stage / owned, ignore_errors=True)
        export(checkpoint, fights, wasm, runtime, stage, count)
        shutil.copy(ROOT / "gui/pages.html", stage / "index.html")
        receipt = dict(base=base, remote=remote, sha256=hashes(stage))
        if site.exists():
            shutil.rmtree(site)
        shutil.move(stage, site)
        site.with_name(site.name + ".json").write_text(json.dumps(receipt))
    print(f"Prepared {site}; publish separately with make pages-publish")


def publish(site, repository=ROOT):
    site = site.resolve()
    receipt = json.loads(site.with_name(site.name + ".json").read_text())
    if not receipt["sha256"] or hashes(site) != receipt["sha256"]:
        raise ValueError("prepared site changed; run pages-build again")
    remote, base = receipt["remote"], receipt["base"]
    head = git(repository, "ls-remote", remote, "refs/heads/gh-pages").split()[0]
    if head != base:
        raise ValueError("gh-pages changed since preparation; run pages-build again")
    with tempfile.TemporaryDirectory() as temporary:
        checkout = Path(temporary) / "publish"
        git(repository, "worktree", "add", "--detach", str(checkout), base)
        try:
            git(checkout, "rm", "-r", "--ignore-unmatch", ".")
            shutil.copytree(site, checkout, dirs_exist_ok=True)
            git(checkout, "add", "-A")
            if not git(checkout, "diff", "--cached", "--name-only"):
                print("Pages already matches the prepared site")
                return
            git(checkout, "commit", "-m", "Publish prepared model reports")
            git(checkout, "push", remote, "HEAD:refs/heads/gh-pages")
        finally:
            git(repository, "worktree", "remove", "--force", str(checkout))
    print("Pushed gh-pages; GitHub Pages will deploy it asynchronously")


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("build")
    build.add_argument("--checkpoint", type=Path, required=True)
    build.add_argument("--fights", type=Path, required=True)
    build.add_argument("--wasm", type=Path, default=ROOT / "build/gumbel-wasm")
    build.add_argument("--runtime", type=Path, default=ROOT / "build/web-deps/node_modules/onnxruntime-web")
    build.add_argument("--fight-count", type=int, default=8)
    build.add_argument("--remote", default="origin")
    for command in (build, commands.add_parser("publish")):
        command.add_argument("--site", type=Path, default=ROOT / "build/pages/site")
    args = parser.parse_args()
    try:
        if args.command == "publish":
            publish(args.site)
        else:
            if args.fight_count < 1:
                parser.error("--fight-count must be positive")
            prepare(args.site, args.checkpoint, args.fights, args.wasm, args.runtime, args.fight_count, args.remote)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"pages: {error}\n")


if __name__ == "__main__":
    main()
