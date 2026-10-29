"""Component benchmarks for infra: python -m infra.benchmarks MODEL_DIR REPLAY_FILE... [--seconds 5] [--build build/main]

MODEL_DIR holds model.onnx and model-gpu.onnx (batch 512). The reference is runs/benchmark/panache-128x4 (`make infra-benchmark`).
REPLAY_FILE: replay sample files to read rows from, e.g. that run's samples-*.bin.
Native benchmarks need `make build` (or --build DIR) first.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from . import reader


def native(build, name, *args):
    """Run a native benchmark and return its JSON line. TensorRT's libraries ship in the venv."""
    packages = Path(sys.prefix) / f'lib/python{sys.version_info.major}.{sys.version_info.minor}/site-packages'
    libs = [packages / 'tensorrt_libs', *sorted((packages / 'nvidia').glob('*/lib'))]
    env = dict(os.environ, LD_LIBRARY_PATH=':'.join(map(str, libs)) + ':' + os.environ.get('LD_LIBRARY_PATH', ''))
    result = subprocess.run([str(build / 'infra/benchmarks' / name), *map(str, args)],
                            capture_output=True, text=True, env=env)
    if result.returncode:
        sys.exit(f'{name} failed:\n{result.stderr[-2000:]}')
    return json.loads(result.stdout.strip().splitlines()[-1])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('models', type=Path)
    p.add_argument('replay', type=Path, nargs='+')
    p.add_argument('--seconds', type=float, default=5)
    p.add_argument('--build', type=Path, default=Path('build/main'))
    a = p.parse_args()

    gpu = native(a.build, 'infra_bench_inference', a.models / 'model.onnx', a.models / 'model-gpu.onnx', a.seconds)
    actors = native(a.build, 'infra_bench_actors', a.seconds)
    disk = reader.run(a.replay, a.seconds)

    print(f"{'component':<28}{'throughput':>17}  notes")
    print(f"{'inference (GPU, batch 512)':<28}{gpu['evals_per_second']:>10,.0f} eval/s  "
          f"{a.models.name}")
    print(f"{'actors + inference server':<28}{actors['requests_per_second']:>10,.0f} req/s   "
          f"9 actor + 1 server threads, free episodes and backend; mean batch {actors['mean_batch']:.0f}")
    print(f"{'reader + prefetch':<28}{disk['rows_per_second']:>10,.0f} rows/s  "
          f"{len(a.replay)} files, {sum(f.stat().st_size for f in a.replay) / 2**30:.1f} GiB")


if __name__ == '__main__':
    main()
