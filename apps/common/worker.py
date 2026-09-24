"""Running the C++ workers (apps/common/worker.hpp) in parallel and writing their combat_v3 rows."""
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

import msgpack
import pyarrow as pa
import pyarrow.parquet as pq
from sts_combat_rl.schemas.combat_v3 import COMBAT_V3


def run_worker(binary, request, weights=None, timeout=None):
    """One worker call: `binary REQUEST.json OUTPUT_DIR [WEIGHTS]` in a temp dir; returns its result.

    A failed worker raises RuntimeError with the request and the end of its stderr (subprocess.TimeoutExpired
    on timeout).
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "request.json").write_text(json.dumps(request))
        argv = [str(binary), str(tmp / "request.json"), str(tmp), *([str(weights)] if weights else [])]
        done = subprocess.run(argv, timeout=timeout, stderr=subprocess.PIPE, text=True)
        if done.returncode:
            raise RuntimeError(f"{Path(binary).name} exited {done.returncode} on request {json.dumps(request)[:300]}: "
                               f"{done.stderr[-2000:].strip()}")
        sys.stderr.write(done.stderr)
        with (tmp / "result.msgpack").open("rb") as result:
            return msgpack.unpack(result, raw=False)


_END = object()


def run_parallel(fn, items, workers, on_result=lambda item, result: None):
    """fn(item) on `workers` threads (named w0, w1, ...), at most `workers` at a time, so `items` may be endless.

    on_result(item, result) runs in the calling thread as each call finishes. The first exception stops
    handing out items, waits for the running calls, and is raised.
    """
    items = iter(items)
    with ThreadPoolExecutor(workers, thread_name_prefix="w") as pool:
        running = {}
        while True:
            while len(running) < workers and (item := next(items, _END)) is not _END:
                running[pool.submit(fn, item)] = item
            if not running:
                return
            finished, _ = wait(running, return_when=FIRST_COMPLETED)
            for future in finished:
                on_result(running.pop(future), future.result())


def write_part(out, key, rows, metadata=None):
    """combat_v3 rows -> out/part-<key>.parquet, atomically (readers never see a half-written part); the table."""
    schema = COMBAT_V3.with_metadata({**COMBAT_V3.metadata, **{k.encode(): v.encode() for k, v in (metadata or {}).items()}})
    table = pa.Table.from_pylist(rows, schema=schema)
    tmp = out / f".part-{key:08d}.tmp"
    pq.write_table(table, tmp, compression="zstd")
    os.replace(tmp, out / f"part-{key:08d}.parquet")
    return table
