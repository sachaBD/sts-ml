from pathlib import Path
import tempfile
import threading
import time
import unittest

from .reader import Reader, prefetch


class ReaderTests(unittest.TestCase):
    def test_reads_exact_rows_in_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / 'a', Path(tmp) / 'b'
            a.write_bytes(bytes(range(100)))
            b.write_bytes(bytes(range(100, 200)))
            with Reader([a, b], threads=4) as reader:
                rows = [(1, 5, 3), (0, 0, 2), (0, 98, 2)] * 50
                self.assertEqual(reader.read(rows), [bytes([105, 106, 107]), bytes([0, 1]), bytes([98, 99])] * 50)
            self.assertEqual(reader.files, [])


class PrefetchTests(unittest.TestCase):
    def test_order_and_depth(self):
        built = []
        items = prefetch(lambda i: built.append(i) or i * i, 10, depth=2)
        self.assertEqual(next(items), 0)
        time.sleep(0.2)
        self.assertLessEqual(len(built), 4)  # one consumed, `depth` queued, one blocked in put
        self.assertEqual(list(items), [i * i for i in range(1, 10)])

    def test_error_reaches_consumer(self):
        def produce(i):
            if i == 3:
                raise ValueError('bad batch')
            return i
        with self.assertRaisesRegex(ValueError, 'bad batch'):
            list(prefetch(produce, 10))

    def test_abandoned_iterator_stops_producer(self):
        items = prefetch(lambda i: i, 1000, depth=1)
        next(items)
        items.close()
        self.assertFalse(any(t.name == 'prefetch' for t in threading.enumerate()))


if __name__ == '__main__':
    unittest.main()
