# Phase 2 item 3: held-out bench2k

Output: `runs/schema=pv_starts_v1/id=champ-bench2k/starts.parquet`.

```sh
taskset -c 10 .venv/bin/python apps/pv/starts.py generate \
  --source runs/schema=combat_v4/date=2026-10-04/id=champ-bench-nodome/out/fights-00000.parquet \
  --encounter 39 --bench-copies 5 --n 2045 --seed0 998000000000 \
  --add-p 0 --remove-p 0 --hp-p 0 \
  --out runs/schema=pv_starts_v1/id=champ-bench2k/starts.parquet
```

`--bench-copies` is independent opt-in; existing `--source` selects the supplied
held-out corpus. It repeats every encounter/Act≥2/no-Dome source deterministically,
adds source_fight_id provenance and changes only seed. Unlike normal generation,
it neither samples without replacement nor excludes the held-out source block,
and does not relocate the room or augment loadouts. Augmentation, exclusion ranges,
nonpositive copies or an n different from source_count×copies are errors.
Normal generate still requires both seed exclusions and is unchanged.

Full output verification:2045 unique fight IDs/seeds,409 source IDs with exactly
5 copies each; seed range[998000000000,998000002045);0Dome,0augmentation;
all start fields (including deck, HP, relics, potions, bottled indices and RNG
counters) exactly equal their source exceptseed. SHA256:
`bb0e9c5bb3243e52e7c0deb384e0d5892a873258754091ce597f9c93a1e1d70e`.

Unit test covers copies, provenance, source preservation, Dome filtering and
invalid-mode arguments. Normal augmented generation also has exact Arrow equality
with the pre-change generator (n10,rng23); logs `bench2k-tests.log`,
`bench2k-generate.log`, `bench2k-verify.log` in this directory.

Evaluation only: these409 source decks remain held out; do not add this output
to training data. No teacher run, full2045 native play or agent comparison was
launched by this checkpoint; those remain for orchestration on released cores.
