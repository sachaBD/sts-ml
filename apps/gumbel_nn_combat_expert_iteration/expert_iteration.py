"""Expert iteration on fixed combat starts: Gumbel self-play, train on the replay window, export, evaluate.

  python -m apps.gumbel_nn_combat_expert_iteration.expert_iteration OUT [--resume-from N]

OUT is a prepared run (prepare.py): OUT/config.toml and OUT/inputs/. Writes OUT/training/ (run_control.py):
iterNNN/{samples-0.bin, results.jsonl, decks.json, replay.json, model.pt, model*.onnx} and test/iterNNN/SET/results.jsonl.
iterNNN/decks.json counts each deck's self-play fights and wins up to and including iteration NNN.
"""
import argparse
from dataclasses import asdict
import itertools
import json
from pathlib import Path
import time

from agents.combat.nn.training.learner import Learner
from agents.combat.nn.training.train import LOSSES
from agents.combat.nn.pytorch_to_onnx import ModelExporter
from agents.combat.player.play import play
from agents.combat.nn.training.replay import Replay
from apps.common.app import read_jsonl
from agents.combat.nn.training.telemetry import write as write_training

from .config import experiment_changes, load
from .reporting import deck_of
from .run_control import RunControl, delete_samples
from .sampling import sampling_weights

# Rows of an iteration's own samples scored before training on them: a few seconds, and enough to compare
# with the training loss.
HELDOUT_ROWS = 50_000


def collection_stats(results):
    finished = sorted(r['finished_unix'] for r in results)
    return dict(capped=sum(r['capped'] for r in results),
                tail95_seconds=finished[-1] - finished[int(.95 * (len(finished) - 1))] if finished else None)


def tally(counts, results):
    """A copy of counts ({deck: {fights, wins}}) with results added."""
    counts = {deck: dict(c) for deck, c in counts.items()}
    for r in results:
        c = counts[deck_of(r['fight_id'])]
        c['fights'] += 1
        c['wins'] += bool(r['won'])
    return counts


class ExpertIteration:
    def __init__(self, out, config, control):
        self.config, self.control = config, control
        self.inputs = out / 'inputs'
        self.pool = json.loads((self.inputs / 'starts.json').read_text())
        self.tests = {s.name: read_jsonl(self.inputs / 'evaluation' / f'{s.name}.jsonl')
                      for s in config.evaluation.sets}
        m, t = config.model, config.training
        self.learner = Learner(m.architecture(), t.lr, t.device)
        self.exporter = ModelExporter(m.architecture(), m.deployments, m.cpu(), m.mixed_precision)

    def run(self, resume_from=None):
        self.control.status(stage='initialization', phase='initialize')
        first = self.start() if resume_from is None else self.resume(resume_from)
        if self.config.training.device.startswith('cuda'):
            self.control.status(stage='initialization', phase='compile')
            self.learner.compile()
        for it in itertools.count(first):
            if self.control.stop_requested():
                self.control.status(stage='stopped', phase='stopped', last_completed=it - 1)
                return
            if self.config.run.iterations and it > self.config.run.iterations:
                break
            needed = self.control.disk_needed(self.config.run.min_free_gib, self.control.iteration(it - 1))
            if self.config.run.min_free_gib and self.control.disk_free() < needed:
                self.control.status(stage='stopped', phase='stopped', reason='disk reserve', last_completed=it - 1)
                self.control.log(stopped='disk reserve', required_free_bytes=needed)
                return
            begin = time.time()
            results = self.self_play(it)
            played = time.time()
            states, loss = self.train(it)
            self.control.log(iteration=it, games=len(results), wins=sum(r['won'] for r in results), states=states, **loss,
                             rollout_mix=self.config.search.at(it).rollout_mix, lr=self.config.training.lr_at(it),
                             play_seconds=round(played - begin), train_seconds=round(time.time() - played),
                             **collection_stats(results))
            self.evaluate(it)
            if self.config.training.prune_replay:
                self.control.prune(keep_from=it - self.config.training.replay_window + 1)
        self.control.status(stage='done', phase='done')

    def start(self):
        """Iteration 0: the initial (or warm-started) network, without replay."""
        (self.control.directory / 'config.toml').write_text((self.inputs.parent / 'config.toml').read_text())
        if (initial := self.inputs / 'initial-model.pt').exists():
            self.learner.load(initial)
            self.control.log(initial_checkpoint=str(initial), spec=asdict(self.learner.net.spec))
        directory = self.control.iteration(0)
        directory.mkdir()
        begin = time.time()
        self.exporter.export(self.learner.net, directory)
        self.learner.save(directory / 'model.pt')
        (directory / 'decks.json').write_text(json.dumps({d['fight_id']: dict(fights=0, wins=0) for d in self.pool}))
        self.control.log(iteration=0, states=0, train_seconds=round(time.time() - begin))
        self.evaluate(0)
        return 1

    def resume(self, completed):
        self.control.check_resume(completed)
        if self.config.run.iterations and completed >= self.config.run.iterations:
            raise ValueError(f'iteration {completed} already reached run.iterations')
        recorded = self.control.directory / 'config.toml'
        if fixed := experiment_changes(load(recorded.read_text()), self.config):
            raise ValueError(f'resume changed settings that define the experiment: {fixed}')
        recorded.rename(recorded.with_name(f'config-before-resume-{time.time_ns()}.toml'))
        recorded.write_text((self.inputs.parent / 'config.toml').read_text())
        self.learner.load(self.control.iteration(completed) / 'model.pt')
        self.control.log(resume_from=completed, fights_per_iteration=self.config.starts.per_iteration)
        # A stop or crash between the checkpoint and its evaluation must not skip that evaluation.
        self.evaluate(completed)
        return completed + 1

    def self_play(self, it):
        previous, directory = self.control.iteration(it - 1), self.control.iteration(it)
        s = self.config.starts
        weights = sampling_weights(self.control.directory, it, s) if s.sampling_concentration else None
        fights = s.draw(self.pool, f'{self.config.run.seed_namespace}:{it}', s.per_iteration,
                        weights=[weights[d['fight_id']] for d in self.pool] if weights is not None else None)
        results = self.play(f'iteration {it}', previous, fights, directory, self.config.search.at(it))
        counts = json.loads((previous / 'decks.json').read_text())
        (directory / 'decks.json').write_text(json.dumps(tally(counts, results)))
        return results

    def train(self, it):
        t = self.config.training
        stage, window = f'iteration {it}', range(max(0, it - t.replay_window + 1), it + 1)
        samples = [s for k in window for s in sorted(self.control.iteration(k).glob('samples-*.bin'))]
        # The network has never trained on the samples it just played: their loss is a held-out loss.
        self.control.status(stage=stage, phase='score')
        with Replay(sorted(self.control.iteration(it).glob('samples-*.bin')), balanced=t.balanced_replay) as fresh:
            heldout = self.learner.score(fresh, HELDOUT_ROWS, t.batch)
        self.control.status(stage=stage, phase='load', files=len(samples))
        with Replay(samples, max_rows=t.max_train_rows, balanced=t.balanced_replay) as replay:
            if not set(replay.groups) <= {d['fight_id'] for d in self.pool}:
                raise ValueError('replay contains a deck outside the training decks')
            directory = self.control.iteration(it)
            self.control.status(stage=stage, phase='train', done=0, total=t.steps, directory=str(directory))
            with (directory / 'training.jsonl').open('w') as telemetry:
                def report(counters):
                    write_training(telemetry, counters)
                    self.control.status(stage=stage, phase='train', done=counters.step,
                                        total=counters.total, directory=str(directory))
                trained = self.learner.fit(replay, t.steps, t.batch, value_batch=t.value_batch, lr=t.lr_at(it),
                                          log=lambda line: print(f'  {line}', flush=True), on_counters=report)
            (self.control.iteration(it) / 'replay.json').write_text(json.dumps(dict(
                available=replay.available, selected=replay.selected, drawn=replay.drawn), indent=2))
            states = len(replay)
        self.learner.save(self.control.iteration(it) / 'model.pt')
        self.control.status(stage=stage, phase='export')
        self.exporter.export(self.learner.net, self.control.iteration(it))
        loss = {f'{split}_{name}_loss': value for split, losses in (('heldout', heldout), ('train', trained))
                for name, value in zip(LOSSES, losses)}
        return states, {k: round(v, 4) for k, v in loss.items()}

    def evaluate(self, it):
        completed = {r['set'] for r in self.control.history() if r.get('test') == it}
        for evaluation in self.config.evaluation.sets:
            if not evaluation.due(it) or evaluation.name in completed:
                continue
            directory = self.control.directory / 'test' / f'iter{it:03d}' / evaluation.name
            self.control.archive(directory)
            begin = time.time()
            results = self.play(f'evaluation {evaluation.name} iteration {it}', self.control.iteration(it),
                                self.tests[evaluation.name], directory, evaluation.search)
            self.control.log(test=it, set=evaluation.name, games=len(results), wins=sum(r['won'] for r in results),
                             seconds=round(time.time() - begin), **collection_stats(results))
            if self.config.training.prune_replay:
                delete_samples(directory)

    def play(self, stage, models, games, directory, search):
        self.control.status(stage=stage, phase='play', done=0, total=len(games), directory=str(directory))
        return play(self.inputs / 'nn_play', models, games, directory, search, self.config.workers,
                    progress=lambda done, pid: self.control.status(stage=stage, phase='play', done=done,
                                                                   total=len(games), child_pid=pid,
                                                                   directory=str(directory)))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('out', type=Path, help='prepared run output directory')
    p.add_argument('--resume-from', type=int, help='the latest completed iteration; keeps replay and optimizer')
    a = p.parse_args()
    out = a.out.resolve()
    config = load((out / 'config.toml').read_text())
    training = out / 'training'
    training.mkdir(exist_ok=a.resume_from is not None)
    control = RunControl(training)
    with control.lock():
        ExpertIteration(out, config, control).run(a.resume_from)


if __name__ == '__main__':
    main()
