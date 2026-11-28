import unittest
from unittest.mock import patch

from .live_combat import play_branch


class LiveCombatTests(unittest.TestCase):
    def test_branch_commits_only_before_requested_boundary(self):
        roots = [dict(step=i, turn=i//2, combat={}, value=.5, requests=1) for i in range(4)]
        frame = dict(state=dict(root_value=.5, q_calls=1, move='A'))
        recorded = [dict(step=0, turn=0, combat={}, frames=[frame])]

        def run(operation, history):
            moves = []

            def exchange(proc, command):
                if 'play' in command:
                    moves.append(command['play'])
                return roots[len(moves)]

            def search(proc, root, config):
                return dict(step=root['step'], turn=root['turn'], frames=[frame]), config['move']

            with patch('apps.voi_gym.live_combat.exchange', side_effect=exchange), \
                    patch('apps.voi_gym.live_combat.search', side_effect=search):
                result = play_branch(None, {}, recorded, 0, history, {'move': 9}, operation)
            return result, moves

        preview, moves = run('rerun', [])
        self.assertEqual(moves, [])
        self.assertEqual(preview['step'], 0)
        advanced, moves = run('action', [])
        self.assertEqual(moves, [9])
        self.assertEqual(advanced['step'], 1)
        advanced, moves = run('turn', [{'move': 7}])
        self.assertEqual(moves, [7, 9])
        self.assertEqual(advanced['turn'], 1)
        self.assertEqual(advanced['history'], [{'move': 7}, {'move': 9}])
