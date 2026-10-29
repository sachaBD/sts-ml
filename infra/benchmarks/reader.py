"""Reader + prefetch throughput as the Learner sees it: batches of 2048 random ~10 KiB rows (our mean row size)."""
import os
import random
import time

from infra.workers.reader import Reader, prefetch

BATCH, ROW_BYTES = 2048, 10 * 1024


def run(paths, seconds=5):
    sizes = [os.path.getsize(p) for p in paths]
    rng = random.Random(1)

    def batch(_):
        rows = []
        for _ in range(BATCH):
            f = rng.randrange(len(paths))
            rows.append((f, rng.randrange(sizes[f] - ROW_BYTES), ROW_BYTES))
        return reader.read(rows)

    with Reader(paths) as reader:
        rows, start = 0, time.time()
        for rows_read in prefetch(batch, 1 << 30):
            rows += len(rows_read)
            if time.time() - start >= seconds:
                break
        return dict(rows_per_second=rows / (time.time() - start))
