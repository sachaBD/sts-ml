# Human-deck expert corpus — overnight plan (2026-10-05/06)

Status: **stopped 2026-10-06; handed over to user — see HANDOFF.md.**
impl-26-10-5 (well-specified code tasks). Reviewed with single-fight-investigation (points adopted below).

## Core goal and bet

Terminal goal: an autonomous agent that beats the whole game. Combat is the bottleneck; Champ is the testbed.
General expert iteration (combat-pv, champ-oracle-exit, human-combat-r01) plateaued at or below MCTS20k.
Concentrated data on one deck beat MCTS decisively (595/600 vs 514/600); the win-only diagnostic (2/44 rescued)
argues against the HP objective explaining that edge, but does not isolate the learned value net as the cause.

**Hypothesis (user's bet, not established):** scale sideways — above-MCTS expert play on ~50–100 diverse human decks
is a stronger starting point for general expert iteration than MCTS distillation. Tonight: build as much of that
corpus as possible and learn how the recipe scales. Explicit choice: **fresh multi-task joint training (corpus
breadth)**, not the warm-start A→B 75/25 skill-accumulation design (different question; recorded as alternative).

## Questions (priority order)
1. Corpus: how many diverse decks reach provisional expert status (learned2k ≥ MCTS20k on monitor seeds), and how
   many confirm on fresh reserved seeds?
2. Joint vs separate: experience-matched comparison on one deck (directional, n=1 deck).
3. Curriculum: descriptive only (allocation vs progress); no uniform ablation.
4. Zero-shot on newly entering decks vs their MCTS screen, as a function of decks learned (selection-consumed
   signal, not a pristine generalization test).

## Splits (frozen before any outcomes; family-level, fail closed)
- **Held out, never screened or trained:** r01 validation families + original-431 benchmark families, **minus
  consumed controls**: A 7a9ada48, B 00205aa1, 478ffcb8, 82a04d39, 02d4f101, 35a58920, ca29d1e5, 351e9475
  (already trained-on or screened). Manifest written before S0.
- **Training pool:** r01 training-side families (800 decks before exclusions; actual count computed and logged)
  + playable library decks that are not r01-validation. **Training-protected** (previously screened, never trained,
  not pristine): de0524bd, 9308c94d, ebad9892 (r01 `val`). Library rows are filtered by the split, never appended after it.
- Per deck, hash seed namespaces: `screen`, `teacher`, `monitor`(30), `learner-k`, `final`(30, reserved).
- Original reconstructed HP, fixed relics/counters, no potions. Scrawl/unsupported decks are software scope
  exclusions; cost exclusions are compute scope — neither is a "bad deck".

## Stages (hard wall caps; S3 time reserved up front)
**S0 screen (≤75 min).** Probe ~40 new decks × 4 MCTS20k seeds (drops unsupported and slow: mean > 60 s; 478ffcb8
excluded now), then 16 more seeds on survivors → 20/deck. Band: 1–19/20. More waves screened later while S2 runs
only if the active set needs them. Starts while the controller is built.

**S1 pilot (≤2 h incl. control).** Fresh width64 sigmoid net, existing recipe losses. 8 decks (A + 7 band decks
across archetypes/MCTS rates). Teacher: 50 MCTS20k fights/deck (20 screen + 30 `teacher`; A uses 50 of its existing
200). 8 updates × 50 learner2k fights/deck. Monitor 30 seeds/deck vs MCTS reference at bootstrap, update 4, update 8.
**Control:** one non-A pilot deck trained alone, sequentially after, with identical teacher fight IDs, learner seed
namespaces, init seed, sampling rules and epochs (experience-matched, not compute-matched).
Decision: joint ≈/≥ control and most decks rising → S2 joint. If joint lags: first raise teacher coverage
(100/deck) and inspect losses/search/replay mix; width128 only as a separate fresh run.

**S2 scale (until S3 reserve).** Add band decks in waves of 8. Entry: one zero-shot batch, then 50 teacher fights.
Per update: explicit per-deck collection quotas, minimum 10 fights/deck; remaining budget weighted by
p(1−p)/fight-seconds with recent improvement as tie-break (heuristic, not measured learnability).
Statuses: active / **provisional-graduated** (monitor ≥ MCTS ref and ≥ 90%: collection drops to the minimum, replay
mass reserved) / parked (low p after ≥150 fights — "not learned yet", not "unlearnable") / software-failed (errors,
shown separately with attempted/completed counts).

**S3 confirm (45 min reserved).** Freeze checkpoint and graduate list first, then fresh `final` seeds (30/deck)
learned2k vs MCTS20k, paired per seed; deck-clustered aggregate. Predeclared budget: at most 12 graduated decks in
graduation order (ties by deck hash), 30 seeds each; missing pairs reported, deadline never extended; per-deck and macro rates. A forgetting check on
its 100 monitor seeds. Denominators reported separately: screened / active / parked / graduated / unusable.
This is selected-corpus confirmation, not performance across all decks.

## Training/data rules (from review)
- Per-deck, per-batch parquet shards; every fight ID owned by exactly one shard; manifest covers every fight in every
  included shard (fail closed). Explicit per-deck sample quotas with **genuinely mixed-deck minibatches** (bounded
  per-shard iterators + batch scheduler; shard shuffling alone still gives deck-homogeneous batches); realized
  per-deck quotas logged and checked.
- Teacher taper by **per-deck age**, not global update index. Graduated decks' data kept replay-eligible
  independent of global window eviction.
- RSS measured and capped (15 GB box, prior OOM): bounded shards, prune obsolete replay files, no DuckDB fan-out.
- Resume must persist iteration, used seeds, deck lifecycle and sampler state; no ID collisions as decks enter.

## Hard limits
- **≤ 10 gameplay workers at once across all jobs**; one gameplay stage at a time.
- Human pulls: none expected; if needed, bounded `apps/megacrit_dump` pull ≤ ~10k runs.
- No use of the held-out families tonight without explicit user approval.

## Wake-ups (cache-aware)
One resumable controller; orch-26-10-5 polls every ~20–30 min and wakes me only on failure/stall/RSS alarm.
Planned: S0 done, S1 decision, mid-S2, S3/report.
