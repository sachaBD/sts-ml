# Champ fifty-deck expansion — 2026-10-28

## CURRENT REVIEW — 22:26 UTC
**NEW RUNNING**,40completed updates/200,000new training games at review; latest
completed validationcheckpoint38. Original run remains stopped atcheckpoint74.
- Live URL:**http://localhost:8742**. Config:`champ_fifty.toml` in this folder.
- NEW:`runs/schema=gumbel_nn_combat_v1/date=2026-10-28/id=gumbel-nn-champ-fifty-20261028`.
- Onall5,000validationstarts:42.52→49.76%, paired+7.24pp[95%:+6.06,+8.42].
- Original10(n=1000):51.1→52.0%,+0.9pp[-2.12,+3.92]. Early retention regression
  is no longer evident at cohort level; this is not a formal non-inferiority guarantee.
  Exhume recovered54→79/100 (initial79); Corruption14→24 (initial25).
- Added40(n=4000):40.38→49.20%,+8.83pp[+7.56,+10.09].
- Last five all50evaluations49.22,49.40,49.24,49.24,49.76%; recent gains small.
- Same500originalstarts: MCTS35.0%,PUCT47.4%,Gumbel10checkpoint74=51.8%,
  Gumbel50checkpoint38=53.4%. G50-G10+1.6pp[-2.73,+5.93]; G50-PUCT+6.0pp
  [+1.63,+10.37]. Historical simulator confounding still applies.
- Recordedphase validation share20.42%. Run has no automatic21:00 cutoff and was
  still active when reviewed. No stop requested by this review.
- Full review JSON/Markdown inNEW/out/analysis/comparison-20261028.*.
- Source/implementation commits and recovery commands below remain unchanged.

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
- At that initial stage no replacement was launched; superseded by the CURRENT STATUS above.

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

## Actual switchover — 08:48 UTC
- Implementation commit:`978f154`; baseline snapshot:`51764b3`.
- OLD stopped cleanly atcheckpoint74. All rollback inputs retained. Exact rollback
  command also saved locally in `scratch/champ-fifty-preflight/rollback-command.json`.
- NEW launched successfully; its frozen initial weights are OLDiter074, SHA256
  `53b973a83d74a89cd86b95410d071c4de7b762acd6e7ee558450e68a4988a04a`.
- Active pointer:`scratch/gumbel-nn-active-run.json` now points toNEW. Old pointer
  preserved at`scratch/gumbel-nn-active-run-before-fifty.json`.
- Launch:`scratch/champ-fifty-launch.log`; transition:`scratch/champ-fifty-transition.log`;
  managed stdout/stderr inNEW/logs/.
- Read-only web dashboard:`http://localhost:8742` (old dashboard8741 left untouched).
  Web log:`scratch/champ-fifty-web.log`. CLI commands above useNEW.
- At08:51UTC, baseline validation active; webAPI responsive; device used~975MiB VRAM.
  First full training update and recurring validation still being watched.

### First production validation
All5,000baseline fights completed successfully:2,126wins (42.52%),992s including
setup. Initial GPU gameplay is therefore qualified across50decks; this is the
warm-start agent, not evidence of learning yet. Broader states incur CPU fallback
above48tokens; no truncation. First collection is active. Actual validation is
slower than the early estimate; measured20% scheduling handles that automatically.

### First full GPU update qualified — 09:25 UTC
- Collection:5,000games,2,060wins;201,985native decision rows;1,072s including setup.
- Training/save/export:12s; final value loss0.3098, policy1.2541 (finite).
- All50decks sampled;204,800training draws total,4,091–4,100/deck.
- Watcher sampled peak loopRSS3,545MiB and device-wide VRAM2,690MiB. One update;
  sampled device peak is not an exact allocator maximum or long-run upper bound.
  /proc loopVmHWM after update~3.50GiB. Native gameplay and Python training are staged,
  so loopRSS alone does not describe total systemRAM. Newworker/model is completing
  iteration2games successfully. No restart or recovery has been needed.
- Proof:`scratch/champ-fifty-preflight/first-update-qualified.json`;
  resource log:`scratch/champ-fifty-first-update-watch.log`.
- Broader collection/validation is materially slower than the old ten-deck run;
  do not assume the previous5–7minute update cadence. CPUfallback above48tokens is
  visible in native gpu_stats. We keep transport/search settings fixed for this run,
  rather than silently tune them mid-experiment. Wall-time validation budget adapts.
- Continuing to watch the first scheduled post-training validation. Its comparison
  against baseline0 is the first learning measurement; baseline42.52% alone is not one.

### First post-training validation — checkpoint5, 10:24 UTC
Full5,000paired validation games completed after five real updates (25,000training
games). No runtime failures/restarts. Model5 is now collecting iteration6.

| Cohort | Initial | Checkpoint5 | Paired improvement, pp [95%] |
|---|---:|---:|---:|
| All50 (n=5000) | 2126/5000=42.52% | 2320/5000=46.40% | +3.88 [+2.71,+5.05] |
| Added40 (n=4000) | 1615/4000=40.38% | 1853/4000=46.33% | +5.95 [+4.72,+7.18] |
| Original10 (n=1000) | 511/1000=51.10% | 467/1000=46.70% | -4.40 [-7.48,-1.32] |

**Retention alert, not a runtime failure:** Exhume/Immolate fell79→54/100
(-25pp[-36.3,-13.7]); Corruption/FNP25→14/100(-11pp[-19.3,-2.7]).
Other original-deck differences have intervals including0. These are conditional,
unadjusted95% intervals, not multiple-comparison-corrected or replicated training
runs. Do not infer deck "mastery" or a capacity diagnosis from absolute win rates.

Keep the run unchanged for the next scheduled evaluation: overall/new-cohort gains
are positive, but original-ten retention is NOT established. If that cohort remains
below its initial checkpoint with paired bounds excluding0, alert the user for
retention/capacity/training-budget review. Do not silently reweight decks or change
architecture. Runtime failures still follow the one-retry/rollback rule above.

Historical500matchedstarts atcheckpoint5:245/500 wins vsMCTS20k175/500 andPUCT237/500;
paired gaps+14.0pp[+9.2,+18.8] and+1.6pp[-2.7,+5.9], respectively. Historical simulator
mismatch remains; these are NOT comparisons on all50decks.

Validation wall share immediately after this large evaluation29.7%; this is the
expected early chunk/startup overshoot, not an every-update29.7% target. The20% rule
will wait for collection/training time to catch up; over a long run overshoot is
bounded by one evaluation chunk plus update-boundary granularity.

Full-window GPU recorder sampled3,774points through this checkpoint at requested
250ms intervals: device-wide peak2,282MiB (not exact allocator peak, not just learner
usage). Earlier first-update watcher saw2,690MiB. LoopVmHWM~8.12GiB includes clean
memory-mapped replay pages; currentRSS~1.53GiB, systemMemAvailable~11GiB afterval.
No OOM occurred. Source:`scratch/champ-fifty-preflight/first-post-validation.json`;
telemetry:`scratch/champ-fifty-preflight/gpu-window.csv`. Future lengths can vary.

To restart only the read-only web dashboard (leave training alone):
```sh
nohup .venv/bin/python -m apps.gumbel_nn_combat_expert_iteration.web \
  "$NEW" --port 8742 >> scratch/champ-fifty-web.log 2>&1 &
```

### Evening review — 22:26 UTC
The original-ten retention alert fromcheckpoint5 has recovered atcheckpoint38;
continue monitoring rather than treating early loss as permanent. Overalllearning
is clear versus the initial model, but fromcheckpoint24to38 gain is only+0.54pp
with paired95%[-0.56,+1.64] on5,000starts: no clear evidence of continued gains in
that recent interval. Comparing checkpoint38 against historical teacher/PUCT must
use their500matchedoriginalstarts, not the full5,000different task mix. Review is
read-only; training was not stopped despite the daytime21:00planning endpoint.
