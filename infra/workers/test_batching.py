import unittest

from infra.workers.batching import Batch, group_by


class BatchingTests(unittest.TestCase):
    def test_grouping_keeps_first_seen_order_and_compares_settings_by_value(self):
        settings = [{'a': 1, 'b': 2}, {'a': 2}, {'b': 2, 'a': 1}]
        self.assertEqual(group_by(range(len(settings)), key=settings.__getitem__), [[0, 2], [1]])

    def test_round_robin_keeps_overlapping_ids_separate(self):
        groups = {'first': [{'id': 'same', 'payload': 1}, {'id': 'other', 'payload': 3}],
                  'empty': [], 'second': [{'id': 'same', 'payload': 2}]}
        batch = Batch(groups, id_field='id')
        self.assertEqual([j['payload'] for j in batch.jobs], [1, 2, 3])
        replies = [dict(id=j['id'], result=j['payload'] * 10) for j in reversed(batch.jobs)]
        self.assertEqual(batch.split(replies), {
            'first': [{'id': 'other', 'result': 30}, {'id': 'same', 'result': 10}],
            'second': [{'id': 'same', 'result': 20}],
        })
        self.assertEqual(groups['first'][0]['id'], 'same')
        with self.assertRaisesRegex(ValueError, 'exactly one reply'):
            batch.split(replies[:-1] + replies[:1])


if __name__ == '__main__':
    unittest.main()
