"""A search you drive one wave at a time: you pick each simulation's root move and world.

    with Lab(checkpoint) as lab:
        tree = lab.tree(position['start'], position['moves'])
        tree.simulate([(0, 0), (1, 0), (0, 1)])    # (move, world): an index into tree.moves, any world >= 0
        tree.root['moves'][0]['q']                  # the search's value of move 0
        tree.root['moves'][0]['worlds']             # per world: its visits and its own value of move 0

World k is the k-th world of the tree's draw; draw 0 is the draw play uses, whose first `particles` worlds play
samples. Below the root, simulations choose as play does. lab.simulate steps many trees in one network batch.
"""
import json
from pathlib import Path
import subprocess

from agents.combat.player.play import gpu_environment


EXECUTABLE = Path(__file__).resolve().parents[2] / 'build/main/apps/combat_inspector/combat_lab'


class Lab:
    """One combat_lab process holding the checkpoint's network and every open tree."""

    def __init__(self, checkpoint, *, cpu=False, executable=EXECUTABLE):
        command = [str(Path(executable).resolve()), str(Path(checkpoint).resolve())] + (['--cpu'] if cpu else [])
        self._process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                         env=gpu_environment())

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def close(self):
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

    def trees(self, positions, *, draw=0):
        """Open a tree per (start, moves); None for a fight already over. Evaluates every root in one batch."""
        opened = self._send(dict(op='open', trees=[dict(start=s, moves=list(m), draw=draw) for s, m in positions]))
        return [Tree(self, t) if t['id'] is not None else None for t in opened['trees']]

    def tree(self, start, moves=(), *, draw=0):
        return self.trees([(start, moves)], draw=draw)[0]

    def simulate(self, picks):
        """One wave per tree, {tree: [(move, world), ...]}; every network call in one batch."""
        roots = self._send(dict(op='simulate', picks={str(t.id): [list(p) for p in w] for t, w in picks.items()}))
        for tree in picks:
            tree.root = roots['roots'][str(tree.id)]


class Tree:
    """One decision's search. moves: each legal move's bits and description.

    root: the search's value, the network requests sent, and per move its prior, logit, sigma, improved_policy,
    q (None until visited), visits and worlds ([{world, visits, q}]: q is that world's own expectimax value)."""

    def __init__(self, lab, opened):
        self.lab, self.id, self.steps = lab, opened['id'], opened['steps']
        self.moves, self.root = opened['moves'], opened['root']

    def simulate(self, picks):
        self.lab.simulate({self: picks})

    @property
    def sims(self):
        return sum(m['visits'] for m in self.root['moves'])

    def graph(self):
        """The full search graph, as combat_explain reports it."""
        return self.lab._send(dict(op='graph', tree=self.id))['graph']

    def close(self):
        self.lab._send(dict(op='close', trees=[self.id]))
