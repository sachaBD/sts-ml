# Sift: one root decision

| | `ThinkingHead` — enter search? | `SieveGate` — continue search? |
|---|---|---|
| **Model** | One shared-trunk head; one scalar per Gumbel halving round | Separate tiny CPU tree learner (e.g. shallow boosted trees) |
| **Input** | Same state features as the policy and value heads; no budget | Partial-tree statistics, summaries of emitted thinking scalars, proposed extra eval budget; no trunk features |
| **Predicts** | Signed gain at each round: current root choice vs. original prior move | Signed gain in fight win probability: root move after the proposed block vs. root move now |
| **Inference cost** | Extra head arithmetic in the same Q forward pass | CPU prediction; no Q eval |

## Play

1. **Eval root once:** obtain policy, value, thinking.
2. **Enter?** Below entry threshold → play prior. Otherwise buy the first search block, subject to cap.
   - Entry threshold is a policy knob: full-probe gain is **not** first-block gain.
3. **Continue?** At each checkpoint, `SieveGate` scores affordable blocks. Stop if none has positive predicted gain after compute cost; commit current root move.
4. **Spend:** buy the best positive-net block. Local allocator chooses depth, width, or worlds. Return to 3. Never re-eval a state just to run a gate.

## Train — [starting probe](probe_candidates.md) and supervision options

1. **Goal:** `ThinkingHead(state)` estimates signed fight-win gain from a specified probe move over the prior move. Search evidence is supervision, not head input.
2. **Probe:** uniformly sample a self-play decision, snapshot it, run **one search**, retain the explored tree / DAG and chosen move.
   - **Gumbel candidate:** classic tree search, e.g. 256 Q evals across the whole tree. Count actual evals, not simulations.
   - **Horizon candidate:** retain its adaptive search graph, shared-world comparisons and turn-boundary values. Record actual spend.
   - **Target depends on teacher:** same state, different probe → potentially different gain. Do not silently pool labels.

| Data available or purchasable | Possible use | Cost beyond probe / limitation |
|---|---|---|
| Root prior, probe choice, backed-up action estimates | Same move → exact zero move gain. Otherwise the estimated value difference is a proxy | No new Q evals; selecting and grading on the same evidence can inflate gain |
| Interior decision states, priors, explored edges, visits, backups | State-level proxy labels wherever prior and alternative both have evidence | No new Q evals; logging, storage and training cost. Unequal subtree budgets; correlated, search-selected states. No comparison at an unbranched node ≠ zero gain |
| Per-world returns, boundary values, terminal outcomes where available | Compare actions on shared worlds; assess evidence behind each proxy | No new Q evals if retained; search worlds are not independent grading worlds |
| Checkpoint summaries and root choices | Candidate `SieveGate` records: earlier statistics → gain from continuing | Snapshot cost; choice flips alone give no gain magnitude |
| Main fight outcome | Existing sparse value target | No extra Q evals; only the played line is observed |
| Fresh paired continuations on selected disagreements | Independent grading of two moves under a common continuation policy; [candidate recipe](readme.md) | Extra simulation and continuation evals for both branches; finite-world noise |
| Fresh probe at a selected interior state | Measure what a full root budget would choose there | Another probe plus any grading |

- **Open:** probe algorithm and budget; interior-node eligibility and weighting; proxy vs. fresh grading; continuation policy and endpoint; trace retention vs. cost.
