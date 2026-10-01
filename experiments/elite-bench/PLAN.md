# elite-bench: a trustworthy elite benchmark and a map of the search (round 1)

**Author: elite-bench consultant (session `subagent-chat-01a0e977`). Executor: orchestrator-28.** Owner away; approved to
run for up to ~9 h of compute. Log every launch/finish/decision in `LOG.md`.

## Goals

Overall: the strongest autonomous agent on Act 1 A20 Ironclad elite fights (Gremlin Nob, Lagavulin, Three Sentries).

Round 1 trains nothing. Its purpose is to make comparisons reliable and to understand the problem first:
1. **Headroom:** how many fights are winnable at all (the oracle sees the draws and RNG), and how far fair MCTS and
   our best net (elite-v3-t2 + policy priors) fall short.
2. **Noise floor:** the same agent re-run with a different search seed (`search_salt`, new). How much of a
   single-run comparison is chance?
3. **Real ranking:** v3 vs MCTS on 1,800 fresh fights with each fight scored by its mean over 4 salts (a win
   probability, not one coin flip).
4. **Bottleneck:** MCTS budget curve (1k–50k), v3 vs MCTS at 5k/20k/50k, 32 vs 8 particles on pivotal fights.

Background: `experiments/act-1-combat/2026-09-28-elite-specialist/`, `experiments/elite-v3/REPORT.md`,
`experiments/act-1-easy-combats/README.md`.

## Design

- **Fights:** `bench_fights.csv`, 600 per elite from buckets 0–1 of `act1-all-bosses-a20-scaled-search` (no v3 net
  trained on them), hash-selected (`make_fights.py`). `bench_subset.csv` = the first 200 per elite in the same order.
  About 5% of the replays diverge on the current simulator; all arms skip the same ones (`skip_diverged`).
- **Seeds:** new teacher setting `search_salt` (agents/teacher_leaves.cpp): 0 = historical seeds, bit-identical
  (verified on 11 stored fights); other values reseed particles and rollouts. Run ids end in `-s<salt>`.
- **Arms:** `make_configs.py` writes every config into `configs/` (id = file name). MCTS = guided rollouts;
  v3 = `elite-v3-t2`, leaf policy_net, c_puct 1.0, fpu 0.05, prior floor 0.03; 8 particles; no random move.
- **Pivotal fights:** some fair 20k run lost, but the fight is winnable (the oracle or some fair run won it).
  They are chosen from salts 0–3, so extra salts 4–7 give a held-out estimate free of selection bias.
- **Analysis:** `analyze.py` (sections: headroom, A/A, v3 vs MCTS, pivotal held-out, budget, particles, power).
  It works on partial results.

## Ground rules (executor)

- Run from the repo root `/home/sborowsk/project/sts_combat_rl`. **Do not edit code, commit, stash or reset.** The
  deck depends on uncommitted working-tree changes (the `search_salt` tweak and the owner's own changes).
- **One CPU-heavy app at a time.** `run.sh` runs its ids sequentially; never start two `run.sh` at once.
- Run ids are never reused. If a run fails: log it, and if the cause is transient (e.g. out of memory), copy
  `configs/<id>.toml` to `configs/<id>-r2.toml` with `id = "<id>-r2"` and rerun. Otherwise stop the stage and
  message the author.
- Record each run's `skipped` count (from run.sh output) in LOG.md.
- Clock: T0 = start of Stage 0. Estimates below are from measured rates (MCTS 20k 4.6 s/fight/worker, oracle 5.5,
  v3 8–10) and are ±50%.

## Stages

Launch each stage with `experiments/elite-bench/run.sh <ids...>` (ids exactly as listed). Suggested:
`nohup experiments/elite-bench/run.sh ... >> experiments/elite-bench/logs/stages.log 2>&1 &`

| Stage | Ids | Est. |
|---|---|---:|
| 0 preflight | `make build`, `./build/main/teacher_leaves_test`, record `git rev-parse HEAD` (both repos), `free -g` | 5 min |
| 1 oracle | `bench-oracle-20k bench-oracle-50k` | 25 min |
| 2 MCTS salts | `bench-mcts-20k-s0 bench-mcts-20k-s1 bench-mcts-20k-s2 bench-mcts-20k-s3` | 50 min |
| 3 v3 salts | `bench-v3-20k-s0 bench-v3-20k-s1 bench-v3-20k-s2 bench-v3-20k-s3` | 95 min |
| **checkpoint A** | see below | 5 min |
| 5 pivotal held-out | `bench-mcts-20k-s4 bench-v3-20k-s4 bench-mcts-20k-s5 bench-v3-20k-s5 bench-mcts-20k-s6 bench-v3-20k-s6 bench-mcts-20k-s7 bench-v3-20k-s7` | 25 min |
| 7 particles | `bench-mcts-20k-p32-s0 bench-mcts-20k-p32-s1 bench-mcts-20k-p32-s2 bench-mcts-20k-p32-s3` | 15 min |
| 4 MCTS budget | `bench-mcts-1k-s0 bench-mcts-1k-s1 bench-mcts-2k-s0 bench-mcts-2k-s1 bench-mcts-10k-s0 bench-mcts-10k-s1 bench-mcts-50k-s0 bench-mcts-50k-s1` | 80 min |
| 6 v3 scaling (subset) | `bench-mcts-5k-s0 bench-mcts-5k-s1 bench-v3-5k-s0 bench-v3-5k-s1 bench-v3-50k-s0 bench-v3-50k-s1` | 50 min |
| **checkpoint B** | final analysis, message author | 5 min |

Stages 5 and 7 need `pivotal_fights.csv` from checkpoint A. The order 5, 7, 4, 6 is deliberate: highest value first.

**Checkpoint A** (after stage 3):
1. `PYTHONPATH=python .venv/bin/python experiments/elite-bench/pivotal.py`. It prints `fair 20k runs used: 8` and the
   pivotal count. Gate: 8 runs used and between 20 and 900 pivotal fights; otherwise stop and message the author.
2. `PYTHONPATH=python .venv/bin/python experiments/elite-bench/analyze.py --output experiments/elite-bench/results/analysis-A.md`
3. Message the author (intercom `subagent-chat-01a0e977`, or whoever `LOG.md` names as author): "elite-bench checkpoint A:
   results/analysis-A.md". **Do not wait for a reply**; continue with stage 5.

**Time gates:** skip stage 6 if it would start after T0 + 7h00; skip the `bench-mcts-50k-*` runs of stage 4 if they
would start after T0 + 6h30. Log any skip.

**Checkpoint B:** `analyze.py --output experiments/elite-bench/results/analysis.md`; add a per-run summary line (id,
fights, skipped, wall min) to LOG.md; message the author "elite-bench round 1 done". The author writes REPORT.md
and may append a round-2 plan to this file (below). Follow it the same way.

## Round 2 (approved; appended by the author 2026-09-28 ~23:50Z)

**Why.** Early results: the benchmark works. MCTS against itself (A/A) differs by SD ≈ 10.5 HP-eq per fight and
3.9% of fights flip win/loss; the old v3-vs-MCTS "death differences" were at that floor. With 2+ salts, v3-t2
beats MCTS by about +0.9 HP-eq (CI roughly +0.5 to +1.3). The oracle wins 96.7% against MCTS's 91%. So the next
question is whether **more training data** makes a better net. t2 was trained on only 4,884 teacher fights.
Stored v4-labelled elite data not used by t2 (and never from bench buckets 0–1): 1,831 MCTS-played fights with
new decks (`elite-v3-f-mcts`, `elite-v3-v-mcts`) and 2,335 fights played by t2 + policy search (`elite-v3-selfplay`).

**Candidates** (the t2 recipe from scratch, only the data differs):
- `bench-r2-t4`: t2 data + the 1,831 MCTS-played fights.
- `bench-r2-t5`: t4 data + the 2,335 self-play fights.

**Steps**
1. **Now, on the GPU, in parallel with the running CPU chain** (training is GPU plus light CPU; that's allowed): run
   `./apps/value_train/run.sh experiments/elite-bench/configs/bench-r2-t4.toml`, then when it finishes
   `./apps/value_train/run.sh experiments/elite-bench/configs/bench-r2-t5.toml`, one after the other. Log each
   training's best epoch, `validation_mse` and the final `v3:` line. If a training dies of memory, retry it
   (as `-r2`: copy the config, change the id, and edit `templates/bench-r2t*` accordingly) when no value_play
   query phase is running. If it fails twice, drop that candidate.
2. **After stage 7 and before stage 4**, render and run the evaluations (2 salts each, ~28 min per run):
   ```
   for t in bench-r2t4-20k-s0 bench-r2t5-20k-s0 bench-r2t4-20k-s1 bench-r2t5-20k-s1; do
     .venv/bin/python experiments/elite-bench/render.py templates/$t.toml; done
   experiments/elite-bench/run.sh bench-r2t4-20k-s0 bench-r2t5-20k-s0 bench-r2t4-20k-s1 bench-r2t5-20k-s1
   ```
   Render needs the trained net (`done`). If one candidate's training was dropped, skip its two runs.
3. Then continue with stage 4, then stage 6, as in the table above.
4. After the evaluations: run `analyze.py --output experiments/elite-bench/results/analysis-R2.md` (section G compares
   the candidates with t2 and MCTS) and message the author "elite-bench R2 evals done". Don't wait for a reply.

**Revised time gates** (these replace the ones above): the `bench-mcts-50k-*` runs must start before
**T0 + 8h00 (05:12Z)**; stage 6 must start before **T0 + 8h30 (05:42Z)**; nothing new starts after
**T0 + 9h15 (06:27Z)**. Log any skip.

## Round 2b (approved; appended by the author 2026-09-29 ~02:25Z)

**Why.** Salt-0 results: t5 (t4 data + self-play) looks like the new best, about +1.3 HP-eq/fight against MCTS and
+0.4 against t2. It also seems to close t2's deficit on pivotal Sentries fights. Firm that up in the time left
after stage 6.

**Steps, after stage 6 finishes (or is skipped), in this order, under the same "nothing new after 06:27Z" gate:**
1. Pivotal held-out salts for t5 (210 fights, ~4 min each):
   render `templates/bench-r2t5-20k-s4.toml` … `-s7.toml`, then
   `run.sh bench-r2t5-20k-s4 bench-r2t5-20k-s5 bench-r2t5-20k-s6 bench-r2t5-20k-s7`.
2. More full-benchmark salts for t5 (~28 min each): render and run `bench-r2t5-20k-s2`, then `bench-r2t5-20k-s3`,
   each only if it can start before 06:27Z.
3. Final `analyze.py --output experiments/elite-bench/results/analysis.md` (checkpoint B), with the per-run summary
   lines in LOG.md, then message the author "elite-bench round 1+2 done".

## Round 2c: confirmation on fresh fights (approved; appended by the author ~03:35Z; replaces Round 2b step 2)

**Why.** t5 was chosen from two candidates on the bench fights. A claim needs fights that weren't used for the
choice. `confirm_fights.csv` holds the other 1,262 bucket 0–1 elite fights, not in `bench_fights.csv`
(Nob 412, Lagavulin 476, Sentries 374). No agent here trained on them, and no model or setting was chosen on them.

**Pre-registered success criterion** (written before any confirm run): t5 − MCTS HP-eq/fight overall > 0 with the
95% run-seed cluster-bootstrap CI excluding 0, and no elite with a point estimate below −1.0.

**Steps** (replace Round 2b step 2, the t5 s2/s3 runs; keep 2b step 1):
1. After Round 2b step 1: `run.sh bench-confirm-mcts-20k-s0`, then render `templates/bench-confirm-r2t5-20k-s0.toml`
   and `run.sh bench-confirm-r2t5-20k-s0` (~10 + ~20 min).
2. `PYTHONPATH=python .venv/bin/python experiments/elite-v3/compare.py --baseline bench-confirm-mcts-20k-s0 --candidate bench-confirm-r2t5-20k-s0 --title "elite-bench confirm: t5 vs MCTS (fresh fights, salt 0)" --output experiments/elite-bench/results/confirm.md --json experiments/elite-bench/results/confirm.json`
3. If time remains before 06:27Z: `bench-r2t5-20k-s2` as in 2b.

If the chain can't be reordered safely, run the confirm right after stage 6 instead of after 2b step 1. Either is fine.
