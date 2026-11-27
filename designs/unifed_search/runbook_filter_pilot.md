# Paired belief-filter pilot — recovery index

From the repository root, open [`runs/benchmark/unified-filter/RUNBOOK.md`](../../runs/benchmark/unified-filter/RUNBOOK.md). It contains the exact arm definitions, input verification, safe resume commands, bootstrap comparison and recovery instructions. The implementation is in `agents/combat/search/filter/`; `filter` is the candidate, `filter-root` and `filter-fixed` are ablations.

The run directory has the base revision and source patch, frozen starts/configs, CPU model, raw fight journals, per-arm logs, analysis script and report. It is gitignored; copying this index alone does not back up the experiment. No final-test data or training run is involved.
See also: [direct-cost 2,400-deck validation and live page](../../runs/benchmark/filter-2400/RUNBOOK.md).
