from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

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

    def test_batch_uses_bounded_tasks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows'
            path.write_bytes(bytes(range(256)))
            rows = [(0, i % 250, 1 + i % 7) for i in range(2048)]
            with Reader([path], threads=4) as reader:
                with patch.object(reader.pool, 'submit', wraps=reader.pool.submit) as submit:
                    actual = reader.read(iter(rows))
                self.assertLessEqual(submit.call_count, 4 * 8)
                self.assertEqual(actual, [bytes(range(256))[offset:offset + size]
                                          for _, offset, size in rows])

    def test_empty_and_small_batches(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows'
            path.write_bytes(b'abcdef')
            with Reader([path]) as reader:
                with patch.object(reader.pool, 'submit', wraps=reader.pool.submit) as submit:
                    self.assertEqual(reader.read(iter(())), [])
                    submit.assert_not_called()
                self.assertEqual(reader.read([(0, 2, 3)]), [b'cde'])
                self.assertEqual(reader.read([(0, 4, 2)] * 3), [b'ef'] * 3)

    def test_read_error_reaches_caller(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rows'
            path.write_bytes(b'abcdef')
            with Reader([path]) as reader:
                with patch('infra.workers.reader.os.pread', side_effect=OSError('read failed')):
                    with self.assertRaisesRegex(OSError, 'read failed'):
                        reader.read([(0, 0, 3)] * 100)
                self.assertEqual(reader.read([(0, 0, 3)]), [b'abc'])


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
