"""Read-only hybrid continuation dashboard; safe before setup and during partial writes."""
import argparse
from collections import Counter
import json
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def read_lines(path):
    try:
        lines = path.read_text().splitlines()
    except FileNotFoundError:
        return []
    result = []
    for line in lines:
        try:
            result.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # live writer may be between writes; dashboard never repairs a ledger
    return result


def counts(rows):
    c = Counter(r.get('status', 'unknown') for r in rows)
    c['wins'] = sum(bool(r.get('won', False)) for r in rows if r.get('status') == 'completed')
    c['logged'] = len(rows)
    return c


def render(run):
    run = Path(run); out = run/'out'
    cfg = read_json(out/'config.json')
    if not cfg:
        return f'RUN: {run.name}\nNOT CONFIGURED; no jobs launched by this dashboard.'
    state = read_json(out/'state.json') or {}
    ledger = read_lines(out/'ledger.jsonl')
    results = [r for r in ledger if r.get('kind') == 'result']
    done_keys = {r.get('key') for r in results}
    unresolved = sum(r.get('kind') == 'intent' and r.get('key') not in done_keys for r in ledger)
    lines = [f'RUN: {run.name.removeprefix("id=")}   {state.get("status", cfg["status"])}',
             'Controller: apps.run_rl.expert_champ — one sequential pipeline',
             f'CURRENT: {state.get("phase", "Unknown")}',
             f'Workers: at most {cfg["workers"]}; no second training branch',
             f'Plan: {cfg["updates"]} × {cfg["batch_fights"]} fresh fights; {cfg["optimizer_steps"]} optimizer steps/update',
             f'Replay: {cfg["anchor_states"]} anchor + {cfg["online_states"]} online states/batch; online half-life {cfg["half_life_updates"]:g} update(s)',
             f'Search: {cfg["sims"]} simulations; rollout mix {cfg["rollout_mix"]}; augmentation OFF',
             f'Unresolved dispatch intents: {unresolved} (may be in-flight or interrupted; not automatically retried)', '']
    if state.get('deadline_epoch'):
        from datetime import datetime
        lines += [f'Hard deadline: {datetime.fromtimestamp(state["deadline_epoch"]).astimezone().isoformat()}']
    if state.get('failure'):
        lines += [f'FAILURE: {state["failure"]}']
    ref = counts([r for r in results if r.get('cell')=='monitor-incumbent'])
    lines += [f'Incumbent monitor: {ref["completed"]}/{cfg.get("monitor_fights", 200)} completed, {ref["wins"]} wins, {ref["capped"]} caps, {ref["error"]} errors', '']
    for u in range(1,cfg['updates']+1):
        c = counts([r for r in results if r.get('stage')=='collect' and r.get('update')==u])
        train = read_json(out/f'update{u}/model/train.json')
        logs = read_lines(out/f'update{u}/train.log')
        progress = next((r.get('step') for r in reversed(logs) if 'step' in r),0)
        complete = read_json(out/f'update{u}/model/complete.json')
        label = 'verified DONE' if train and complete else f'{progress}/{cfg["optimizer_steps"]} steps logged'
        mon = counts([r for r in results if r.get('cell')==f'monitor-{u}'])
        lines.append(f'Update {u}: collection {c["completed"]}/{cfg["batch_fights"]}, wins {c["wins"]}, caps {c["capped"]}, errors {c["error"]}; train {label}; monitor {mon["completed"]}/{cfg.get("monitor_fights", 200)}, wins {mon["wins"]}')
    curves = read_json(out/'curve.json') or []
    lines += ['', 'MONITOR (selection-consumed; not independent final evidence):']
    if not curves:
        lines.append('Not evaluated. Frozen incumbent reference will be evaluated once on the same monitor starts.')
    for r in curves:
        lines.append(f'Update {r["update"]}: {r["wins"]}/{r["n"]}; incumbent {r["incumbent_wins"]}/{r["n"]}; paired gap {100*r["gap"]:+.1f} pp')
    final = read_json(out/'final-summary.json')
    lines.append('FINAL: '+(json.dumps(final) if final else 'NOT COMPLETE; selection/confirmation gates still apply'))
    for cell in ('final-incumbent', 'final-challenger'):
        c = counts([r for r in results if r.get('cell')==cell])
        if c['logged']:
            lines.append(f'{cell}: {c["completed"]}/{cfg.get("final_fights", 600)} completed, {c["wins"]} wins; caps {c["capped"]}, errors {c["error"]}')
    if not cfg.get('seed_manifest'):
        lines.append('Seeds NOT YET FROZEN. Setup is not launch-ready.')
    lines += [f'Log: {run / "logs/main.log"}', f'State: {out / "state.json"}',
              'Collection wins are training observations, not a held-out strength estimate.']
    return '\n'.join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    print(render(p.parse_args().run))


if __name__=='__main__':
    main()
