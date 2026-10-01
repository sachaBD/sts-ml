# CARDS: Combat-Approximated Rollouts for Deck Selection

**Status: main direction.** Planning only, nothing implemented.

## Idea

Card selection is a sequential decision problem under uncertainty: each pick changes what later picks are worth. Instead of learning "how good is this card" directly, learn the (arguably easier) **combat transition** and let card values fall out of search:

1. **Combat approximator:** (deck, HP, relics, potions, encounter) → *distribution* of combat end states (death, HP left, potions used). It approximates one frozen combat agent at one search budget.
2. **Rollouts:** from a reward state, for each legal choice (each card, skip), sample many public-information futures. Map, rewards, rest sites and events run in the exact simulator; fights are sampled from the approximator. Later card picks inside a rollout are made by a **rollout policy**. Average success per choice, and share sampled futures across choices to reduce noise.
3. **Distil and iterate:** train a picker network on the search's choices. Use it as the next rollout policy, search again, distil again.

This is policy iteration / expert iteration over the run, with a learned sub-model (combat) under the environment model. Full trees are infeasible: branching is about 4 per reward, and there are heavy chance nodes between rewards (encounters, offers, events). Rollouts evaluate root choices only.

## Why the learned policy and value come back

- **Rollout policy (synergy):** the search computes "value of card X given the rollout policy makes later picks". With a dumb rollout policy, synergy cards (e.g. Demon Form needing later block support) are undervalued. Nested search is multiplicatively expensive; a distilled picker network is the cheap fix. The search stays the source of truth; the network caches it.
- **Cutoff value (horizon):** rollouts that stop at the Act 1 boss favour short-term power over scaling. Rolling to the Heart needs an Act 2–3 combat model (planned eventually) and suffers large variance. A learned run value at a cutoff (see [run value](../run-value/README.md)) may be needed.

## Why it suits one machine

Combat labels are dense: roughly 15 labelled fights per run versus one outcome per run for direct value learning. Rough cost estimates (unmeasured):

- **Real fights:** a real fight costs ~2 s, so ~25–30k fights/hour.
- **Rollouts:** with a fast approximator a rollout costs ~ms instead of ~30 s, which affords thousands of rollouts per choice.

## Main risks

1. **Resolution (the crux).** The approximator's error on the *paired difference* a single card makes may exceed that difference. Test it against real paired data first ([Gauntlet](../gauntlet/README.md) produces it). Consider training directly on with/without-card pairs.
2. **Model exploitation (inherent).** Search seeks decks where the approximator is optimistic. Include counterfactual/random-card decks in training data, and audit recommendations with real combat on a schedule.
3. **Coupling to the combat agent.** Retrain or revalidate the approximator whenever combat changes. Freeze combat during a selection cycle.
4. **Simulator support.** Replacing a fight with a sampled outcome and continuing the run, with public-information resampling, is not yet checked in sts_lightspeed.

## Path

1. [Gauntlet](../gauntlet/README.md) with real combat and shared seeds: signal test and reference set.
2. Combat approximator v0. Gate: reproduces real per-card differences, not just low loss.
3. Gauntlet with the approximator: first cheap learned-model picker, tested in real Act 1 runs.
4. CARDS root rollouts, with rollout policy = current baseline. Gate: better real Act 1 outcomes than the baseline on paired held-out seeds.
5. Distil into a picker network, then iterate. Add a cutoff value as the horizon extends.

## Detailed notes (earlier drafts)

- [COMPONENTS.md](COMPONENTS.md): the sub-problems and how they compose.
- [PLAN.md](PLAN.md): staged experiments and gates.
- [explainer.html](explainer.html): combat-outcome model workshop with an interactive toy lab (synthetic data only).
