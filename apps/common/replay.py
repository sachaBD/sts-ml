"""Replaying stored fights of combat_v3 bootstrap runs; apps/common/fight_replay.hpp rebuilds them in C++.

A fight is rebuilt by replaying its act 1 run from the run seed, with every earlier fight's stored chosen
actions; the worker request is {run_seed, ascension, fight_index, actions: [[chosen_action...] per earlier fight]}.
"""
from collections import defaultdict

from sts_combat_rl import query

# The replayed decision_index 0 row must match the stored one on these columns: it is the same fight.
START = ("encounter", "floor", "starting_hp", "starting_max_hp", "global_numeric", "cards", "monsters")
COLUMNS = ("run_seed", "fight_index", "episode_id", "decision_index", "chosen_action", "ascension", *START)


def decision_rows(run_ids, columns=COLUMNS, run_seeds=None):
    """The decision rows of the runs (only `run_seeds` if given)."""
    where = "row_kind = 'decision'" + (f" and run_seed in {query.sql_list(sorted(run_seeds))}" if run_seeds is not None else "")
    return query.rows(f"select * from combat_v3 where run_id in {query.sql_list(run_ids)}", columns, where)


def replay_requests(rows, episodes):
    """episode_id -> (worker request, stored decision 0 row) for each of `episodes`.

    rows: decision rows (any order) holding each episode's fight and every earlier fight of its run.
    """
    actions, starts = defaultdict(dict), {}  # (run_seed, fight_index) -> {decision_index: chosen_action}
    for row in rows:
        fight = actions[row["run_seed"], row["fight_index"]]
        if row["decision_index"] in fight:
            raise RuntimeError(f"run seed {row['run_seed']} fight {row['fight_index']} decision {row['decision_index']}"
                               " is stored twice (is the run seed in two source runs?)")
        fight[row["decision_index"]] = row["chosen_action"]
        if row["decision_index"] == 0:
            starts[row["episode_id"]] = row
    if missing := sorted(set(episodes) - set(starts)):
        raise RuntimeError(f"{len(missing)} episodes missing from the source runs, e.g. {missing[:5]}")
    requests = {}
    for episode in episodes:
        start = starts[episode]
        seed, index = start["run_seed"], start["fight_index"]
        earlier = []
        for i in range(index):
            fight = actions.get((seed, i), {})
            if not fight or sorted(fight) != list(range(len(fight))):
                raise RuntimeError(f"episode {episode}: decisions of earlier fight {i} are missing")
            earlier.append([fight[d] for d in range(len(fight))])
        requests[episode] = ({"run_seed": seed, "ascension": start["ascension"], "fight_index": index,
                              "actions": earlier}, start)
    return requests


def check_start(episode, replayed, stored, columns=START):
    """Raise unless the replayed decision 0 row matches the stored one on `columns`."""
    if differs := [c for c in columns if replayed[c] != stored[c]]:
        raise RuntimeError(f"episode {episode}: replayed start differs from the stored fight on {differs}")
