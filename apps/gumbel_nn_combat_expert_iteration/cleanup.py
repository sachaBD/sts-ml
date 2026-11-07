"""Remove disposable data from a completed Gumbel NN run; preview unless --apply is given."""
import argparse
import fcntl
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--apply', action='store_true', help='delete instead of preview')
    parser.add_argument('--drop-replay', action='store_true', help='discard replay and indexes; forfeits exact continuation')
    parser.add_argument('--keep-iteration', type=int, action='append', default=[],
                        help='leave this iteration entirely untouched; may be repeated')
    args = parser.parse_args(argv)
    if any(k < 0 for k in args.keep_iteration):
        parser.error('kept iterations must be nonnegative')
    kept = {f'iter{k:03d}' for k in args.keep_iteration}
    run = args.run.resolve()
    record = json.loads((run / 'run.json').read_text())
    if record.get('schema') != 'gumbel_nn_combat_v1' or record.get('status') != 'done':
        parser.error('only completed Gumbel NN runs are supported; failed/interrupted runs need manual review')
    training = (run / 'out/training').resolve()
    lock = training / '.lock'
    if not training.is_relative_to(run) or lock.is_symlink() or not lock.is_file():
        parser.error('expected training data and an existing lock inside this run')

    with lock.open('rb') as owner:
        try:
            fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('run is active')
        selected = []
        for path in sorted(training.rglob('*')):
            if path.is_symlink() or not path.is_file():
                continue
            parts = path.relative_to(training).parts
            if kept.intersection(parts[:-1]):
                continue
            cache = 'tensorrt' in parts[:-1] and path.suffix in ('.engine', '.profile')
            replay = (path.name in ('samples.bin', 'samples.index.npz')
                      or path.match('samples-*.bin') or path.match('samples-*.index.npz'))
            if cache or (args.drop_replay and replay):
                selected.append(path)
        print('APPLY' if args.apply else 'DRY RUN — nothing deleted')
        if args.drop_replay:
            print('Discarding replay forfeits exact continuation; checkpoints are retained.')
        for path in selected:
            print(path.relative_to(run))
        size = sum(path.stat().st_size for path in selected)
        print(f'{len(selected)} files, {size / 1024**3:.2f} GiB of file data')
        if args.apply:
            for path in selected:
                path.unlink()


if __name__ == '__main__':
    main()
