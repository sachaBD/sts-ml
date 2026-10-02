# Overworld v3.1 / v3.2: implementation and observation contract

**Goal:** autonomous A20 Ironclad through all three acts and the Heart.
**Status:** implemented, not trained/promoted. Keep the selected Act 2 checkpoint as incumbent.
See [the concise map explainer](overworld-map.md) for backward graph message passing.

## Two controlled variants

| Variant / checkpoint kind | Parameters | Deck–relic interactions | Map / value head |
|---|---:|---|---|
| v3.1: `run_policy_v3_1` | 350,594 | Independent context-conditioned tokens, sum + mean pooling | Shared state-conditioned backward DAG encoder |
| v3.2: `run_policy_v3_2` | 384,066 | Same tokens + one card/relic/global self-attention layer | Identical |

Named specifications live in `agents/overworld/value/architectures/rp3.{1,2}-w64-h128-l2.toml`.
Defaults: token width 64, head width 128, two residual blocks, dropout 0.1; v3.2 has four
attention heads. Offered cards produce separate after-state decks; skip is the final column.
Attention has no deck-position embedding: deck order is not an overworld fact.

Card tokens include ID, upgrade, persistent misc data, type, rarity, printed cost/base damage,
and innate/ethereal/exhaust/self-retain/X-cost flags. These are public static mechanics,
not estimates of effective damage or energy cost in a hypothetical combat.
Owned relic tokens include ID and signed-log counter. A permanent global token prevents
fully masked attention even for an empty deck/relic set. Padding never contributes to pools.

Global context includes boss, act, current room, numerical state, potions, encounter distributions/history,
and pooled event/relic-candidate tokens. v3.1 cards and relics already receive global context;
v3.2 adds direct token-to-token interactions. The final head predicts a scalar after-state value;
existing chooser and TD learner interfaces still use output `[0]`, shape `[batch, choices+skip]`.

## Added public observations

`state_json` keeps all old fields and adds `overworld.version=1`:

- Explicit act, local map floor, character, ascension, ruby/sapphire/emerald keys.
- Rarity modifier and common/uncommon/rare probabilities for the **next individual** hallway,
  elite and shop rarity roll. N'loth's Gift is accounted for outside shops. Boss offers are rare;
  per-offer odds cannot be one fixed number because the modifier changes between cards.
- Potion modifier, base rolled-drop probability, obtainable-drop probability (Sozu distinction).
  White Beast Statue is accounted for. These describe standard combat reward generation, not
  every event reward, pickup decision, or a guarantee of a free potion slot.
- Question-mark combat/shop/treasure counters and effective fight/shop/treasure/event probabilities,
  including an after-shop alternative. Juzu Bracelet, Tiny Chest and consecutive-shop rules are
  applied; integer bucket truncation matches the simulator.
- Shop-removal count and next standard removal price, including relic discounts.
- Remaining normal/shrine/one-time event IDs with current eligibility flags, separately typed.
  These lists change with public encounters; current ineligibility is not permanent depletion.
  The simulator repopulates normal/shrine pools each act; one-time depletion persists. This is
  not one global "every event only once per run" mask.
- Public eligible relic **candidates** by tier, minus owned relics, with ordinary/shop eligibility.
  `relic_candidates_exact=false`: this is deliberately an upper bound, not the actual secret pool.
  Exact depleted membership can reveal invisible spawn-rejection draws. Previously offered,
  unchosen relics are not yet tracked in a dedicated public-history ledger.
- Last elite and possible next elites: the act's three minus the previous elite; Act 4 uses Shield/Spear.
  Encoded as weights over encounter IDs, rather than a redundant act-specific nine-bit interface.
- Currently revealed encounter (only on a battle screen), including the second Act 3 boss once revealed.
- Consumed current-act hallway history/count, and weighted possible next hallway encounters.
  Hallway fights **can recur**. Normal generation excludes the last two; the weak→strong transition
  has separate slime/louse restrictions. Scripted event fights are not ordinary hallway draws.

No seeds, RNG counters/state, shuffled relic order, unconsumed monster schedule or hidden second
Act 3 boss enter the neural observation. The history exporter accesses only already-consumed
encounter prefixes. Encounter rules cover ordinary current-act generated sequences; the simulator's
rare list-exhaustion/regeneration edge cases are not a full persistent history system.

## Collection, training and compatibility

Use the rebuilt worker. New graph observations are retained in ordinary and hypothetical sampled
end states, JSONL traces and canonical `overworld_v1` JSON records. v3 explicitly rejects old
path-only or unversioned records; recollect data instead of pretending missing history is known.
Old v1/v2 model definitions, checkpoint kinds and weight names are unchanged.

Example new collection (real key requirements, existing SimpleAgent baseline):

```sh
.venv/bin/python apps/run_rl/play.py --out runs/overworld-v3-collect \
  --first-seed 980000000000 --seeds 100 --workers 2 --policy simple \
  --max-act 4 --target heart --worker build/overworld-v3/run_rl_worker
```

Example training on **newly collected** observations:

```sh
.venv/bin/python apps/run_rl/train.py --data runs/overworld-v3-collect \
  --out runs/overworld-v3.1/model.pt --target heart \
  --arch-spec agents/overworld/value/architectures/rp3.1-w64-h128-l2.toml
```

Substitute the v3.2 spec for the attention comparison. The learner records the spec hash and appends
its first successful training run to `FROZEN.tsv`; changed frozen specs are rejected. Graph training
uses batch 32 / validation 64 by default (`--batch-size` / `--val-batch-size` override them). These
are conservative starting sizes, not a GPU memory guarantee for every offered-card count.
The `heart` target is binary **actual Heart defeat**, never merely Act 3 clear. Existing
`act1`, `floors`, `floors3` targets remain available. With no positive Heart examples, binary
training will collapse; bootstrap via three-act progress data/targets or improve collection,
then fine-tune toward Heart defeat. Do not interpret untrained outputs as useful decisions.

## Heart plumbing and remaining limits

The worker now accepts `max_act=4`, carries on after the second A20 Act 3 boss, and records
`heart_cleared`. Missing keys at Act 3 completion produce `heart_locked`, not a false Heart clear.
Heart-mode learned campfire options include recall with real key-bearing after-states.
Baseline Heart mode takes sapphire before its competing chest relic; if campfires remain under
SimpleAgent, it recalls at the final Act 3 campfire. Emerald still requires reaching a burning elite.
These are minimal key fallbacks, **not a fully learned key-acquisition policy**.

Shop relic after-states remain approximate: public hypothetical ownership/candidate eligibility
is updated without revealing random pickup outcomes; all on-pickup effects are not simulated.
Other existing SimpleAgent fallbacks and simulator gaps are not solved by this topology.

## Verification and initial cost check

Final isolated build: `build/overworld-v3/`; all six selected CTest entries passed. The new Python
suite contains eight tests, including real native-export observations for Acts 1 and 4, graph
ordering/connectivity, attention padding, gradients, policy dispatch and checkpoint round trips.
Native checks cover public-observation privacy, probabilities, key costs, A20 double bosses and
Act 4 control flow with injected combat wins. One-epoch **CPU synthetic-fixture** learner tests
passed for both kinds (40 fixtures); their checkpoints are only plumbing artifacts, not agents.
Named default-size specifications were also instantiated and evaluated on a real exported state.

A small CPU timing check (one thread, **one** native Act 1 start context, two synthetic candidate
slots populated from deck cards plus skip, ten warmups, **50 timed repetitions per model**) measured
encoding + forward:

| Model | Median ms | Interquartile range, ms |
|---|---:|---:|
| Selected v1 | 0.879 | 0.865–0.910 |
| Untrained v3.1 defaults | 8.350 | 8.059–8.596 |
| Untrained v3.2 defaults | 9.104 | 8.636–9.836 |

Ranges describe timing variability, **not confidence intervals or bounds across game states**.
This narrow check shows the richer graph network is substantially more expensive per decision;
it does not establish whole-run overhead or a strength–cost win. The default architectures have
not been trained on gameplay; GPU training, broad performance/memory tests and live full-game
Heart evaluation remain unverified.

Logs and raw timing data: `runs/schema=topology_report/date=2026-10-02/id=overworld-v3/`
(`build.log`, `learner-smoke.log`, `out/inference-timing.json`). Logs retain initial fixture/stale-binary
failures followed by the final successful passes. No incumbent checkpoint was changed.

## Adoption gate

Tests can establish correct connectivity, masking, history boundaries and checkpoint compatibility,
not gameplay strength. Compare v3.1 vs v3.2 on paired unseen seeds, report Act 1/2/3/Heart clears,
key acquisition, survival/HP at bosses and inference/worker time. First compare with common
observations/targets; then measure whether attention earns its additional cost. Keep fresh seeds
out of model selection. No improvement in win rate is claimed by this implementation.
