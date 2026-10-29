"""Readers: parallel exact-size reads from many files, and a bounded prefetch queue for the Learner.

pread, not mmap: random mmap page faults trigger read-ahead (8 MiB per fault under WSL), which turns
~10 KB rows into ~1 MB disk reads and thrashes the page cache once the files outgrow RAM.
"""
from concurrent.futures import ThreadPoolExecutor
import os
import queue
import threading


class Reader:
    """Read batches with coarse tasks. os.pread releases the GIL for parallel reads.

    Each batch uses at most eight chunks per thread: enough to balance uneven I/O
    without creating a future for every row. Results retain the request order.
    """

    def __init__(self, paths, threads=4):
        self.files = [os.open(path, os.O_RDONLY) for path in paths]
        for fd in self.files:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        self.pool = ThreadPoolExecutor(threads, thread_name_prefix='reader')
        self._chunks = threads * 8

    def _read_chunk(self, rows):
        return [os.pread(self.files[file], size, offset) for file, offset, size in rows]

    def read(self, rows):
        """rows: (file index, offset, size) triples -> their bytes, in the same order."""
        rows = list(rows)
        if not rows:
            return []
        size = (len(rows) + self._chunks - 1) // self._chunks
        chunks = (rows[start:start + size] for start in range(0, len(rows), size))
        return [raw for chunk in self.pool.map(self._read_chunk, chunks) for raw in chunk]

    def close(self):
        self.pool.shutdown()
        for fd in self.files:
            os.close(fd)
        self.files = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def prefetch(produce, count, depth=3):
    """Yield produce(0), ..., produce(count - 1) in order, built up to `depth` ahead on a background thread.

    A producer error is re-raised in the consumer. Abandoning the iterator stops the producer.
    """
    ready, stop = queue.Queue(depth), threading.Event()

    def put(kind, value):
        while not stop.is_set():  # never block forever on a consumer that has gone
            try:
                return ready.put((kind, value), timeout=0.1)
            except queue.Full:
                pass

    def run():
        try:
            for i in range(count):
                if stop.is_set():
                    return
                put('item', produce(i))
            put('done', None)
        except BaseException as error:  # noqa: BLE001 - re-raised in the consumer
            put('error', error)

    thread = threading.Thread(target=run, name='prefetch', daemon=True)
    thread.start()
    try:
        while True:
            kind, value = ready.get()
            if kind == 'done':
                return
            if kind == 'error':
                raise value
            yield value
    finally:
        stop.set()
        thread.join()
