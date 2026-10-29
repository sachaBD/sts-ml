"""Readers: parallel exact-size reads from many files, and a bounded prefetch queue for the Learner.

pread, not mmap: random mmap page faults trigger read-ahead (8 MiB per fault under WSL), which turns
~10 KB rows into ~1 MB disk reads and thrashes the page cache once the files outgrow RAM.
"""
from concurrent.futures import ThreadPoolExecutor
import os
import queue
import threading


class Reader:
    """`threads` Readers over `paths`. os.pread releases the GIL, so the threads read in parallel."""

    def __init__(self, paths, threads=12):
        self.files = [os.open(path, os.O_RDONLY) for path in paths]
        for fd in self.files:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)
        self.pool = ThreadPoolExecutor(threads, thread_name_prefix='reader')

    def read(self, rows):
        """rows: (file index, offset, size) triples -> their bytes, in the same order."""
        return list(self.pool.map(lambda row: os.pread(self.files[row[0]], row[2], row[1]), rows, chunksize=64))

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
