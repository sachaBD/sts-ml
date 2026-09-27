#!/usr/bin/env bash
# Endless self-play with one value net, from the repo root:
#   experiments/act-1-combat/2026-09-26-all-bosses/forever.sh value_net_v1/<date>/<id> [start_pass]
# Pass p plays train bucket b = 4 + p % 6 (run_seed % 10 in 4..9; never dev 0-1 / confirm 2-3) in the self-play shape
# of this experiment: every boss fight x2 samples, other encounters <= 330 fights x1. Sample numbers advance each
# time a bucket comes round again (first_sample = 2 * (p / 6)), so every pass plays new versions of its fights.
# One run per pass and part: combat_v3/<date>/ab-forever-<net id>-p<pass>-{boss,rest}. Stop it with kill.
set -uo pipefail
HERE=experiments/act-1-combat/2026-09-26-all-bosses
NET=$1; PASS=${2:-0}; NET_ID=${NET##*/}
DS="'act1-all-bosses-a20-scaled-search'"
mkdir -p "$HERE/configs/forever" "$HERE/results/forever"
while true; do
  b=$((4 + PASS % 6)); first=$((2 * (PASS / 6)))
  for part in boss rest; do
    id="ab-forever-$NET_ID-p$PASS-$part"
    if [ $part = boss ]; then
      where="category = 'boss'"; samples=2
    else
      where="category <> 'boss' and episode_id in (select episode_id from (select episode_id, row_number() over (partition by category, encounter order by hash(episode_id)) k from (select distinct episode_id, category, encounter from combat_v3 where id = $DS and source_episode_id is null and run_seed % 10 = $b and category <> 'boss')) where k <= 330)"
      samples=1
    fi
    cat > "$HERE/configs/forever/$id.toml" <<EOF
# ./apps/fight_resample/run.sh $HERE/configs/forever/$id.toml   (written by forever.sh, pass $PASS)
[run]
id = "$id"
query = """
select * from combat_v3
where id = $DS and source_episode_id is null and run_seed % 10 = $b and $where
"""
samples = $samples
first_sample = $first
hp_sd = 10
random_potions = false
value_run = "$NET"
simulations = 20000
particles = 8
random_move = false
workers = 10
EOF
    echo "$(date -u +%FT%TZ) start $id" >> "$HERE/results/forever/forever.log"
    ./apps/fight_resample/run.sh "$HERE/configs/forever/$id.toml" > "$HERE/results/forever/$id.console.log" 2>&1
    echo "$(date -u +%FT%TZ) end   $id exit=$?" >> "$HERE/results/forever/forever.log"
  done
  PASS=$((PASS + 1))
done
