"""Expert iteration on fixed combat starts: Gumbel self-play, train on the replay window, export, evaluate.

  python -m apps.gumbel_nn_combat_expert_iteration.expert_iteration OUT [--resume-from N]

OUT is a prepared run (prepare.py): OUT/config.toml and OUT/inputs/. Writes OUT/training/ (run_control.py):
iterNNN/{samples-0.bin, results.jsonl, replay.json, model.pt, model*.onnx} and test/iterNNN/results.jsonl.
"""
import argparse
from dataclasses import replace
import itertools
import json
from pathlib import Path
import time

from agents.combat.nn.learner import Learner
from agents.combat.nn.play import play
from agents.combat.nn.replay import Replay
from apps.common.app import read_jsonl
from environments.combat.starts import fights

from .config import experiment_changes, load
from .run_control import RunControl, delete_samples


def collection_stats(results):
    finished = sorted(r['finished_unix'] for r in results)
    return dict(capped=sum(r['capped'] for r in results),
                tail95_seconds=finished[-1] - finished[int(.95 * (len(finished) - 1))] if finished else None)


class ExpertIteration:
    def __init__(self, out, config, control):
        self.config, self.control = config, control
        self.inputs = out / 'inputs'
        self.starts = json.loads((self.inputs / 'starts.json').read_text())
        self.tests, self.tests_in = read_jsonl(self.inputs / 'test.jsonl'), read_jsonl(self.inputs / 'test-in.jsonl')
        m, t = config.model, config.training
        spec = dict(dim=m.dim, layers=m.layers, heads=m.heads)
        if m.kind == 'entity':
            spec['kind'] = m.kind
        self.learner = Learner(spec, t.lr, t.device, m.export_reuse, gpu_attended=m.gpu_attended)
        self.games = config.run.games_per_deck * len(self.starts)
        self.evaluation_search = replace(config.search, sims=config.evaluation.sims, gumbel_scale=0.0)

    def run(self, resume_from=None):
        self.control.status(stage='initialization', phase='initialize')
        first = self.start() if resume_from is None else self.resume(resume_from)
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
            states = self.train(it)
            self.control.log(iteration=it, games=len(results), wins=sum(r['won'] for r in results), states=states,
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
            self.control.log(initial_checkpoint=str(initial), spec=self.learner.net.spec)
        directory = self.control.iteration(0)
        directory.mkdir()
        begin = time.time()
        self.export(directory)
        self.learner.save(directory / 'model.pt')
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
        self.control.log(resume_from=completed, games_per_update=self.games)
        # A stop or crash between the checkpoint and its evaluation must not skip that evaluation.
        if not any(r.get('test') == completed for r in self.control.history()):
            self.control.archive(self.control.directory / 'test' / f'iter{completed:03d}')
            self.evaluate(completed)
        return completed + 1

    def self_play(self, it):
        directory = self.control.iteration(it)
        games = fights(self.starts, f'{self.config.run.seed_namespace}:{it}', self.games)
        return self.play(f'iteration {it}', self.control.iteration(it - 1), games, directory, self.config.search)

    def train(self, it):
        t = self.config.training
        stage, window = f'iteration {it}', range(max(0, it - t.replay_window + 1), it + 1)
        samples = [s for k in window for s in sorted(self.control.iteration(k).glob('samples-*.bin'))]
        self.control.status(stage=stage, phase='load', files=len(samples))
        with Replay(samples, max_rows=t.max_train_rows, balanced=t.balanced_replay) as replay:
            if set(replay.groups) != {f['fight_id'] for f in self.starts}:
                raise ValueError('replay does not contain every configured deck')
            self.control.status(stage=stage, phase='train', done=0, total=t.steps)
            self.learner.fit(replay, t.steps, t.batch, log=lambda line: (
                print(f'  {line}', flush=True), self.control.status(stage=stage, phase='train', detail=line)))
            (self.control.iteration(it) / 'replay.json').write_text(json.dumps(dict(
                available=replay.available, selected=replay.selected, drawn=replay.drawn), indent=2))
            states = len(replay)
        self.learner.save(self.control.iteration(it) / 'model.pt')
        self.control.status(stage=stage, phase='export')
        self.export(self.control.iteration(it))
        return states

    def export(self, directory):
        w = self.config.workers
        self.learner.export(directory, w.gpu_batch, self.config.model.gpu_tokens, w.gpu_bank)

    def evaluate(self, it):
        if not self.tests or not self.evaluation_due(it):
            return
        begin, models = time.time(), self.control.iteration(it)
        directory = self.control.directory / 'test' / f'iter{it:03d}'
        results = self.play(f'test iteration {it}', models, self.tests, directory, self.evaluation_search)
        in_sample = self.play(f'test in-sample iteration {it}', models, self.tests_in, directory / 'in-sample',
                              self.evaluation_search) if self.tests_in else []
        self.control.log(test=it, games=len(results), wins=sum(r['won'] for r in results),
                         seconds=round(time.time() - begin), in_sample_games=len(in_sample),
                         in_sample_wins=sum(r['won'] for r in in_sample), **collection_stats(results))
        if self.config.training.prune_replay:
            delete_samples(directory)

    def evaluation_due(self, it):
        e = self.config.evaluation
        if it in e.at:
            return True
        if e.wall_share:
            history = self.control.history()
            validation = sum(r['seconds'] for r in history if 'test' in r)
            learning = sum(r.get('play_seconds', 0) + r.get('train_seconds', 0) for r in history)
            return validation < e.wall_share * (learning + validation)
        return bool(e.every) and it % e.every == 0

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
