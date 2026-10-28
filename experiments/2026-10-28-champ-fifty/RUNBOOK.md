# Champ fifty-deck expansion — 2026-10-28

## Goal and guardrails
Expand the original ten Champ decks to fifty, warm-starting the working Gumbel agent.
Keep architecture fixed (64/2/2), 5,000 total games/update, five-update replay,
deck-balanced training, minibatch-only GPU loading, approximately 20% validation wall time.
Keep the original ten as a separate retention cohort. Report paired 95% improvements
against the starting checkpoint; historical MCTS20k/PUCT comparisons stay explicitly historical.

The existing ten-deck run must continue throughout preparation. Stop it only when the
replacement is ready. On repeated preparation failure abandon the expansion; on
post-switchover failure restore the known-working run rather than waste the day.

## Working run / rollback anchor
`runs/schema=gumbel_nn_combat_v1/date=2026-10-27/id=gumbel-nn-champ-ten-20261027-205535`
Current session log: `logs/resume.log`; dashboard uses actual loop lock, not stale run.json.
Stop: `./apps/gumbel_nn_combat_expert_iteration/run.sh stop RUN_DIRECTORY`.
Latest completed checkpoint and retained replay must be preserved before switching.

## Progress
- Preparation started. Initial working-tree snapshot will be committed before implementation.
- No replacement training job launched; existing job not stopped.

## Recovery
Detailed verified launch/recovery commands and validation results will be added here
before switchover. Never append partial gameplay. Preserve interrupted stages and
inspect native worker processes before resuming. No silent simulator fixes/reversions,
no filtering failures into losses, no automatic endless restarts.

### Preparation checks
- Initial snapshot committed: `51764b3`.
- First CPU probe stopped safely after28 completed games: candidate pool contained a
  cross-class Ragnarok card outside the NN encoder capability. Existing job unaffected.
  Selection is being restricted to the actual encoder's supported card boundary before retry.
- Eleven existing/new CPU unit checks pass (balanced sampling, cap, format validation,
  deterministic RNG, CPU minibatch training, existing loop/model compatibility).

### Qualified implementation (before switchover)
- Revised fifty-deck selection passed150 CPU games (3seeds/deck,16sims), all50 decks
  produced training rows. This checks capability, not strength or exhaustive simulator coverage.
- Minibatch:2048 rows,40–41/deck; largest sampled sequence59tokens in this probe.
- Replay-scale check:882,150 available rows across50 synthetic deck groups made by
  replicating an existing ten-deck batch;498,810 selected rows. Indexing1.28s;
  CPU3-step check51s total, peakRSS1.97GiB (including mapped pages).
  This is a scaling probe, not independent fifty-deck experience or a GPU memory measurement.
- Warm-start CPU lifecycle check: exact source weights preserved initer000; all50decks
  trained; checkpoints, evaluation commits and pruning verified. Native/export mocked
  only for this small CPU lifecycle check; separate real gameplay probe above.
- Sixteen owner CPU tests pass. Original deleted app tests not restored.
- Prepared inputs contain5,000 unique validation seeds,100/deck. Original ten retain
  their50historical monitoring starts and add50fresh; added40use100fresh each.
  Historical MCTS20k and PUCTu6 comparisons cover only the matching500original starts.
- Validation scheduling uses `validation_share=0.20`: evaluate at update boundaries
  when cumulative recorded validation wall share is below20%, including initialeval.
  This avoids guessing an iteration cadence as runtime changes. Share oscillates
  around target; one full evaluation can overshoot it. There is no hard time cap.
- New replay keeps only compact file/offset indices on CPU; only current minibatch
  is materialized/padded and moved toGPU. Native binary format/topology unchanged.

## Launch and monitoring commands
Run from `/home/sborowsk/project/sts/combat_rl`.
```sh
APP=./apps/gumbel_nn_combat_expert_iteration/run.sh
CONFIG=experiments/2026-10-28-champ-fifty/champ_fifty.toml
OLD=runs/schema=gumbel_nn_combat_v1/date=2026-10-27/id=gumbel-nn-champ-ten-20261027-205535
NEW=runs/schema=gumbel_nn_combat_v1/date=2026-10-28/id=gumbel-nn-champ-fifty-20261028
# Only after qualification, request OLD's clean stop and wait for state=stopped/process exit.
"$APP" stop "$OLD"
# Then launch (never --overwrite):
nohup "$APP" "$CONFIG" > scratch/champ-fifty-launch.log 2>&1 &
watch -n 10 "$APP" status "$NEW"
"$APP" status "$NEW" --json
# Clean stop:
"$APP" stop "$NEW"
```
Baseline validation precedes the first collection; it may take10+minutes. Do not
mistake it for a hang. Inspect current counter age and stage worker.log. Validation
has5,000games, collection has5,000games. Train100steps, batch2048, width64/2/2.
Per-update `replay.json`:50decks, draw counts approximately4096/deck, total204800.
No full action traces. Original-ten cohort and all50 must be monitored separately.
95% intervals are conditional on one training run; repeated inspections/multiple
comparisons are not corrected. Low absolute win rate is not evidence of bad play.

## Recovery / rollback (monitoring-agent instructions)
No automatic restart daemon. Check `status --json` and logs every1–2minutes during
switchover, then every5minutes. A stale counter is not proof of failure (load/export
may pause it). Check RAM/VRAM and disk too. Do not use historical run.json alone to
infer liveness: direct resumed sessions keep their old launcher record.

On deterministic failure, roll back immediately. At most ONE retry of the expansion
is allowed for a plausible transient GPU/driver failure; on repeat failure restore
OLD. Never leave the GPU idle for hours trying speculative fixes. Do not quietly
filter failed games, reduce search quality, or change deck count/topology.

To recover either run after confirming its loop AND native worker have exited,
use the following with RUN set to NEW for its one retry or OLD for rollback.
Redirect output to a NEW session log (e.g. `logs/recovery-$(date -u +%Y%m%d-%H%M%S).log`).
```sh
RUN="$OLD" .venv/bin/python - <<'PY'
import json,os,time
from pathlib import Path
out=Path(os.environ['RUN']).resolve()/'out';training=out/'training'
rows=[json.loads(s) for s in (training/'log.jsonl').read_text().splitlines()]
k=max(r['iteration'] for r in rows if 'iteration' in r)
partial=training/f'iter{k+1:03d}'
if partial.exists(): partial.rename(training/f'.interrupted-{time.time_ns()}-{partial.name}')
(training/'STOP').unlink(missing_ok=True)
cmd=json.loads((out/'command.json').read_text())+['--resume-from',str(k)]
os.execv(cmd[0],cmd)
PY
```
This preserves partial samples; never append them. Latest completed checkpoint plus
its five retained sample batches is the rollback state. Missing checkpoint/replay
is a reason to stop and report, not improvise. Original OLD run config/inputs/worker
remain untouched; new run records its frozen initial checkpoint source and SHA256
in `out/inputs/warm-start.json`.

After first real update, verify finite losses, all50decks in replay.json, bounded
memory, successful export and the next native worker accepting that model. Monitor
first scheduled post-training validation as well. If original-ten performance
shows a sustained loss, report it with paired95%bounds; do not automatically change
sampling weights or architecture. Stop at the user's daytime budget endpoint rather
than assume that run-until-stopped imposes a timer.
