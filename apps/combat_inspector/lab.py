"""Interactive combat inspection session: run scheduled Gumbel searches and steer waves on the same graph.

    with Lab(checkpoint) as lab:
        tree = lab.tree(position['start'], position['moves'])
        tree.simulate([(0, 0), (1, 0), (0, 1)])    # (move_or_path, world[, descend]): any world >= 0
        tree.root['moves'][0]['q']                  # the search's value of move 0
        tree.root['moves'][0]['worlds']             # per world: its visits and its own value of move 0

Pass `search=dict(sims=64, particles=8, gumbel_scale=0, rollout_mix=0)` to `tree()` / `trees()` to start from a
scheduled Gumbel search and continue steering on top of it. World k is the k-th world of the tree's draw; draw 0
is the draw play uses. `lab.simulate({tree: picks, ...})` steps many trees in one network batch.
"""
import json
import os
from pathlib import Path
import subprocess

from agents.combat.player.play import gpu_environment


def _executable():
    build = os.environ.get('STSRL_BUILD_DIR', 'main')
    return Path(__file__).resolve().parents[2] / 'build' / build / 'apps/combat_inspector/combat_lab'


EXECUTABLE = _executable()


class Lab:
    """One combat_lab process holding the checkpoint's network and every open tree."""

    def __init__(self, checkpoint, *, cpu=False, executable=None):
        binary = _executable() if executable is None else Path(executable)
        command = [str(binary.resolve()), str(Path(checkpoint).resolve())] + (['--cpu'] if cpu else [])
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                         env=gpu_environment())

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
        if self._process.stdin and not self._process.stdin.closed:
            self._process.stdin.close()
        self._process.wait()

    def _send(self, command):
        self._process.stdin.write(json.dumps(command, allow_nan=False) + '\n')
        self._process.stdin.flush()
        line = self._process.stdout.readline()
        if not line:
            raise RuntimeError(f'combat_lab exited with {self._process.wait()}')
        response = json.loads(line)
        if 'error' in response:
            raise RuntimeError(response['error'])
        return response

    def trees(self, positions, *, draw=0, search=None, include_graph=False):
        """Open a tree per (start, moves); None for a fight already over. Evaluates every root in one batch."""
        spec = []
        for start, moves in positions:
            entry = dict(start=start, moves=list(moves))
            if search is not None:
                entry['search'] = dict(search)
            else:
                entry['draw'] = draw
            if include_graph:
                entry['include_graph'] = True
            spec.append(entry)
        opened = self._send(dict(op='open', trees=spec))
        return [Tree(self, t) if t['id'] is not None else None for t in opened['trees']]

    def tree(self, start, moves=(), *, draw=0, search=None, include_graph=False):
        return self.trees([(start, moves)], draw=draw, search=search, include_graph=include_graph)[0]

    def simulate(self, picks, *, include_graph=False):
        """One wave per tree, {tree: [(move_or_path, world[, descend]), ...]}; every network call in one batch."""
        command = dict(op='simulate', picks={str(t.id): [list(p) for p in w] for t, w in picks.items()})
        if include_graph:
            command['include_graph'] = True
        response = self._send(command)
        for tree in picks:
            key = str(tree.id)
            tree.root = response['roots'][key]
            if include_graph:
                tree.decision = response['decisions'][key]
        return response


class Tree:
    """One decision's open search.

    moves: each legal move's bits and description.
    view: public board view (`combat::view`) at the root.
    decision: current best action bits and value.
    root: the search's value, network requests sent, and per move its prior, logit, sigma, improved_policy,
    q (None until visited), visits and worlds ([{world, visits, q}]: q is that world's own expectimax value)."""

    def __init__(self, lab, opened):
        self.lab, self.id, self.steps = lab, opened['id'], opened['steps']
        self.observation, self.view = opened['observation'], opened['view']
        self.moves, self.root, self.decision = opened['moves'], opened['root'], opened['decision']

    def simulate(self, picks, *, include_graph=False):
        return self.lab.simulate({self: picks}, include_graph=include_graph)

    @property
    def sims(self):
        return sum(m['visits'] for m in self.root['moves'])

    def graph(self):
        """The full search graph (with public board views on every node)."""
        return self.lab._send(dict(op='graph', tree=self.id))['graph']

    def state(self, path, world, *, worlds=None):
        """The public board view, legal moves, and observation reached along `path` in `world`."""
        command = dict(op='state', tree=self.id, path=list(path), world=world)
        if worlds is not None:
            command['worlds'] = worlds
        return self.lab._send(command)

    def close(self):
        self.lab._send(dict(op='close', trees=[self.id]))
