"""Extend a managed single-deck run without deleting any existing outputs."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import os


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--updates',type=int,required=True)
    a=p.parse_args();run=a.run.resolve()
    config=json.loads((run/'out/config.json').read_text())
    if a.updates<=config['updates']:p.error('updates must increase the recorded horizon')
    metadata_path=run/'run.json';metadata=json.loads(metadata_path.read_text())
    if metadata['status']=='running':p.error('run is already marked running')
    command=[sys.executable,'-m','apps.run_rl.single_deck']
    keys=['source','selection','worker','out','namespace','bootstrap','monitor','batch_fights','workers','sims','teacher_sims','bootstrap_epochs','epochs','eval_every','device']
    if 'deck_id' in config:keys.append('deck_id')
    for key in keys:command += ['--'+key.replace('_','-'),str(config[key])]
    command += ['--updates',str(a.updates)]
    metadata.setdefault('continuations',[]).append(dict(previous_updates=config['updates'],updates=a.updates,started=datetime.now(timezone.utc).isoformat(),command=command))
    metadata.update(status='running',finished=None,exit_code=None,pid=os.getpid())
    metadata_path.write_text(json.dumps(metadata,indent=2))
    code=1
    try:
        with (run/'logs/stdout.log').open('a') as stdout,(run/'logs/stderr.log').open('a') as stderr:
            code=subprocess.call(command,stdout=stdout,stderr=stderr)
    finally:
        metadata.update(status='done' if code==0 else 'failed',exit_code=code,finished=datetime.now(timezone.utc).isoformat())
        if code==0:metadata['summary']=json.loads((run/'out/summary.json').read_text())
        metadata_path.write_text(json.dumps(metadata,indent=2))
    raise SystemExit(code)


if __name__=='__main__':main()
