"""Short REAL play/train/export benchmark from an existing checkpoint; no fresh bootstrap.

Uses production Run.play/train/export functions. Complete games/updates only, no automatic resume.
All generated artifacts go under --out; source checkpoint/replay are read-only.
"""
import argparse
from contextlib import contextmanager
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import resource
import subprocess
import time

import torch

from agents.combat.nn.data import read
from agents.combat.nn.loop import PLAY, Run, fights
from agents.combat.nn.model import Net, export, export_static
from agents.combat.nn.metrics import METRIC_DEFINITIONS, play_metrics, sampled_devices
from agents.combat.nn.train import train

ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / 'runs/schema=combat_v4/date=2026-10-25/id=nn-gumbel64-panache-128x4-v1'


def cpu_seconds():
    usages = [resource.getrusage(k) for k in (resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN)]
    return sum(r.ru_utime + r.ru_stime for r in usages)


class MeasuredRun(Run):
    def status(self, **fields):
        if fields.get('phase') == 'play' and fields.get('done', 0):
            with (self.out / 'completion-times.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(monotonic=time.monotonic(), **fields)) + '\n')
        super().status(**fields)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--iterations', type=int, default=2)
    parser.add_argument('--games', type=int, default=2000)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--initial-model', type=Path, help='reuse matching initial model/engines; validated against checkpoint weights')
    parser.add_argument('--fight-namespace', help='reuse a prior benchmark name for identical fight starts')
    args = parser.parse_args()
    if min(args.iterations, args.games, args.steps) <= 0:
        parser.error('iterations, games and training steps must be positive')
    args.out.mkdir(parents=True, exist_ok=False)
    run = MeasuredRun(args.out, gpu_batch=512, gpu_tokens=48, gpu_bank=[256, 32], gpu_wait_us=100,
                      parallel=224, scheduler='local', sample_devices=False)
    phases, updates = [], []
    begin, wall_begin, cpu_begin = time.monotonic(), time.time(), cpu_seconds()

    @contextmanager
    def phase(name, iteration=0, synchronize=False):
        if synchronize:
            torch.cuda.synchronize()
        start, wall, cpu = time.monotonic(), time.time(), cpu_seconds()
        run.status(stage=f'iteration {iteration}', phase=name)
        print(f'BEGIN {iteration} {name}', flush=True)
        try:
            yield
        finally:
            if synchronize:
                torch.cuda.synchronize()
            end, cpu_used = time.monotonic(), cpu_seconds() - cpu
            row = dict(phase=name, iteration=iteration, start=start, end=end, wall_start=wall,
                       wall_end=time.time(), seconds=end-start, cpu_seconds=cpu_used,
                       cpu_logical_cores=cpu_used/(end-start))
            phases.append(row)
            with (args.out/'phases.jsonl').open('a') as stream:
                stream.write(json.dumps(row)+'\n')
            print('PHASE', json.dumps(row), flush=True)

    fields = 'timestamp,utilization.gpu,utilization.memory,memory.used,memory.total'
    telemetry = (args.out/'gpu.csv').open('w')
    sampler = subprocess.Popen(['nvidia-smi', '--id=0', f'--query-gpu={fields}', '--format=csv,nounits', '-lms', '250'],
                               stdout=telemetry, stderr=subprocess.DEVNULL)
    try:
        with phase('load_checkpoint'):
            checkpoint = torch.load(SOURCE/'iter010/model.pt', map_location='cpu', weights_only=True)
            torch.manual_seed(0)
            net = Net(**checkpoint['spec']).cuda()
            optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=1e-4)
            net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'rng' in checkpoint:
                torch.set_rng_state(checkpoint['rng'])
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
            initial_step = next(iter(optimizer.state.values()))['step'].item()
            history = [SOURCE/f'iter{k:03d}/samples-0.bin' for k in range(6, 11)]
            base = json.loads((ROOT/'experiments/25-10-26/a-gumbel/nn/panache_start.json').read_text())
            if args.initial_model:
                initial = torch.load(args.initial_model.parent/'model.pt', map_location='cpu', weights_only=True)
                if initial['spec'] != checkpoint['spec'] or any(not torch.equal(v,initial['net'][k]) for k,v in checkpoint['net'].items()):
                    raise ValueError('initial-model weights/spec do not match warm-start checkpoint')
                del initial
            metadata = dict(benchmark_kind='play_train_export', source=str(SOURCE), spec=net.spec,
                            initial_model=str(args.initial_model) if args.initial_model else None,
                            fight_namespace=args.fight_namespace or args.out.name,
                            gpu_name=torch.cuda.get_device_name(0),
                            gpu_memory_mib=torch.cuda.get_device_properties(0).total_memory/1024**2,
                            logical_cpus=os.cpu_count(), iterations=args.iterations, games=args.games,
                            steps=args.steps, train_batch=2048, replay_window=5, lr=.001, sims=64,
                            workers=9, slots_per_worker=224, gpu_batch=512, gpu_bank=[256,32], tokens=48,
                            gpu_wait_us=100, scheduler='local', rng_restored='rng' in checkpoint,
                            initial_replay=list(map(str,history)), source_optimizer_step=initial_step,
                            binary_sha256=hashlib.sha256(PLAY.read_bytes()).hexdigest(),
                            git_revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                            git_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True)))
            (args.out/'args.json').write_text(json.dumps(metadata,indent=2))
            torch.cuda.synchronize()

        def save_export(directory):
            directory.mkdir(exist_ok=True)
            torch.save(dict(net=net.state_dict(), optimizer=optimizer.state_dict(), spec=net.spec,
                            rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all()), directory/'model.pt')
            export(net, directory/'model.onnx')
            export_static(net, directory/'model-gpu.onnx',512,48)
            for batch in (256,32):
                export_static(net,directory/f'model-gpu-b{batch}.onnx',batch,48)
            net.cuda()

        if args.initial_model:
            previous_model = args.initial_model
        else:
            with phase('initial_export', synchronize=True):
                save_export(args.out/'iter000')
            previous_model = args.out/'iter000/model.onnx'
        for iteration in range(1,args.iterations+1):
            directory=args.out/f'iter{iteration:03d}'
            update_begin=time.monotonic()
            with phase('collect_including_engine_setup',iteration):
                results=run.play(previous_model,
                                 fights(base,f'{args.fight_namespace or args.out.name}:{iteration}',args.games),directory,
                                 64,1.,.5,9,stage=f'iteration {iteration}')
            log=(directory/'worker.log').read_text()
            setup=float(re.search(r'timing: evaluator setup ([\d.]+)',log)[1])
            gameplay=float(re.search(r'timing: gameplay ([\d.]+)',log)[1])
            stats=json.loads(re.findall(r'gpu_stats: (\{[^\n]+\})',log)[-1])
            history=(history+[directory/'samples-0.bin'])[-5:]

            def fit():
                with phase('load_replay',iteration):
                    tokens,policy,won=read(history)
                    (directory/'replay.json').write_text(json.dumps(dict(paths=list(map(str,history)),states=len(won),
                         padded_tokens=tokens.shape[1]),indent=2))
                with phase('train',iteration,synchronize=True):
                    train(net,optimizer,tokens,policy,won,args.steps,2048,'cuda',
                          log=lambda text: print('TRAIN',iteration,text,flush=True))
                with phase('save_export',iteration,synchronize=True):
                    save_export(directory)
                return len(won)
            states=fit()
            update_seconds=time.monotonic()-update_begin
            row=dict(iteration=iteration,games=len(results),wins=sum(r['won'] for r in results),states=states,
                     update_seconds=update_seconds,setup_seconds=setup,gameplay_seconds=gameplay,
                     useful_gpu_evals_per_second=stats['leaves']/gameplay,
                     games_per_hour_gameplay=len(results)*3600/gameplay,
                     games_per_hour_full_update=len(results)*3600/update_seconds,
                     batch_fill=stats['leaves']/stats['capacity'],gpu_stats=stats)
            row['standard_metrics'] = play_metrics(directory)
            row['standard_metrics']['games_per_hour_full_update'] = row['games_per_hour_full_update']
            updates.append(row);run.log(**row)
            previous_model = directory/'model.onnx'
        end, cpu_end = time.monotonic(), cpu_seconds()
        final_state=net.cpu().state_dict()
        changed=max((final_state[k]-v).abs().max().item() for k,v in checkpoint['net'].items())
        optimizer_step=next(iter(optimizer.state.values()))['step'].item()
        report=dict(config=metadata,metric_definitions=METRIC_DEFINITIONS,phases=phases,updates=updates,wall_seconds=end-begin,
                    wall_start=wall_begin,wall_end=time.time(),cpu_seconds=cpu_end-cpu_begin,
                    optimizer_step_final=optimizer_step,max_parameter_change=changed,
                    clock_drift_seconds=abs((time.time()-wall_begin)-(time.monotonic()-begin)))
    finally:
        sampler.terminate();sampler.wait();telemetry.close()
    # NVML samples are overlapping active-time windows, not independent measures of peak FLOPs.
    samples=[]
    for raw in csv.DictReader((args.out/'gpu.csv').open()):
        row={k.strip():v.strip() for k,v in raw.items()}
        row['epoch']=datetime.strptime(row['timestamp'],'%Y/%m/%d %H:%M:%S.%f').timestamp()
        samples.append(row)
    metrics=dict(gpu_util='utilization.gpu [%]',memory_util='utilization.memory [%]',
                 vram_mib='memory.used [MiB]')
    for p in phases:
        selected=[r for r in samples if p['wall_start']<=r['epoch']<=p['wall_end']]
        p['telemetry_samples']=len(selected)
        p['telemetry']={}
        for label,field in metrics.items():
            values=[float(r[field]) for r in selected if r[field]!='[N/A]']
            if values:p['telemetry'][label]=dict(mean=sum(values)/len(values),maximum=max(values),minimum=min(values))
    completions=[json.loads(line) for line in (args.out/'completion-times.jsonl').read_text().splitlines()]
    for update in updates:
        events=[e for e in completions if e['stage']==f'iteration {update["iteration"]}']
        n=update['games'];last=events[-1]['monotonic']
        update['tail_last_10pct_seconds']=last-events[(n*90+99)//100-1]['monotonic']
        update['tail_last_1pct_seconds']=last-events[(n*99+99)//100-1]['monotonic']
        standard=update['standard_metrics']
        start=standard['gameplay_wall_start']
        devices=sampled_devices(args.out/'gpu.csv',start,start+standard['gameplay_seconds'] if start is not None else None,
                                metadata['gpu_memory_mib'])
        standard.update(devices['metrics'])
        update['gameplay_device_samples']=devices['samples']
    (args.out/'report.json').write_text(json.dumps(report,indent=2))
    run.status(stage='done',phase='done')
    print('COMPLETE',args.out/'report.json',flush=True)


if __name__=='__main__':
    main()
