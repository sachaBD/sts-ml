"""Write configs/*.toml: the value_play runs of this experiment. Run from anywhere: python make_configs.py

Fights: act1-a20-8 dev runs (run_seed % 10 in (0, 1); no value net here trained on them), floor 1, one of the 4
easy-pool encounters. Within each encounter fights are ordered by md5(episode_id); the pilot takes ranks 3-32
(ranks 1-2 were a timing probe), the main study ranks 33-132 (100 per encounter, sized from the pilot).
"""
from pathlib import Path

HERE = Path(__file__).parent
QUERY = """
select * from combat_v3
where date = '2026-09-24' and id = 'act1-a20-8' and episode_id in (
  select episode_id from (
    select episode_id, row_number() over (partition by encounter order by md5(episode_id::varchar)) k
    from (select distinct episode_id, encounter from combat_v3
          where date = '2026-09-24' and id = 'act1-a20-8' and run_seed % 10 in (0, 1) and source_episode_id is null
            and row_kind = 'decision' and decision_index = 0 and floor = 1 and category = 'easy'))
  where k between {lo} and {hi})
"""
GEN0 = "value_net_v1/2026-09-26/act1-gen0"  # guided_rollout uses no weights; the run only supplies the train guard
GEN1 = "value_net_v1/2026-09-26/act1-gen1"
PILOT, MAIN = (3, 32), (33, 132)

# name: (fights, leaf, simulations, oracle, particles, value run)
RUNS = {
    "pilot-fair1k": (PILOT, "guided_rollout", 1_000, False, 8, GEN0),
    "pilot-fair10k": (PILOT, "guided_rollout", 10_000, False, 8, GEN0),
    "pilot-fair100k": (PILOT, "guided_rollout", 100_000, False, 8, GEN0),
    "pilot-fair100k-p32": (PILOT, "guided_rollout", 100_000, False, 32, GEN0),
    "pilot-fair100k-p64": (PILOT, "guided_rollout", 100_000, False, 64, GEN0),
    "pilot-oracle10k": (PILOT, "guided_rollout", 10_000, True, None, GEN0),
    "pilot-oracle100k": (PILOT, "guided_rollout", 100_000, True, None, GEN0),
    "main-fair1k": (MAIN, "guided_rollout", 1_000, False, 8, GEN0),
    "main-fair10k": (MAIN, "guided_rollout", 10_000, False, 8, GEN0),
    "main-fair100k": (MAIN, "guided_rollout", 100_000, False, 8, GEN0),
    "main-fair100k-p64": (MAIN, "guided_rollout", 100_000, False, 64, GEN0),
    "main-gen1net20k": (MAIN, "value_net", 20_000, False, 8, GEN1),
    "main-oracle100k": (MAIN, "guided_rollout", 100_000, True, None, GEN0),
}


def config(name, fights, leaf, sims, oracle, particles, value_run):
    return f"""# ./apps/value_play/run.sh experiments/act-1-easy-combats/configs/{name}.toml   (written by make_configs.py)
[run]
id = "easy-{name}"
input = "{value_run}"
workers = 12
leaf = "{leaf}"
simulations = {sims}
oracle = {str(oracle).lower()}
{"" if oracle else f"particles = {particles}{chr(10)}"}random_move = false
skip_diverged = true
query = \"\"\"{QUERY.format(lo=fights[0], hi=fights[1])}\"\"\"
"""


if __name__ == "__main__":
    for name, spec in RUNS.items():
        (HERE / "configs" / f"{name}.toml").write_text(config(name, *spec))
