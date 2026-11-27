# ThinkingHead + probe

## Starting design

- **Probe:** Gumbel, 256 simulations over a pool of 16 worlds (not 256 per world).
- **Head:** shared context-token readout; one signed scalar per sequential-halving round. A round is an allocation checkpoint, not a tree depth.
- **Target at round r:** expected fight-win gain of the move we would commit at the end of r vs. the original prior move. Root state only; no future search evidence as input.
- **Trace:** record the current leader at each completed round. Mask rounds not reached. Do not assume later gains are larger.
- **Candidate grading:** compare distinct proposed moves against the prior on fresh matched worlds; reuse the grade when rounds choose the same move. Comparison policy and effort still open.

## Alternatives, not the starting design

| Candidate | One output per… | Possible play decision | Label requirement |
|---|---|---|---|
| **H1: endpoint** | Full reference search | Buy or skip one fixed product | Grade final root move |
| **H2: depth curve** | Action-depth cap | Buy shallow or deep lookahead | Specify leaf evaluation, width and world allocation; grade root move under each capped search |
| **H3: compute curve** | Q-eval checkpoint | Buy a small or large compute block | Record the move an interruptible search would commit at each checkpoint, before later evidence |

- **Depth is not spend:** many worlds at depth one can cost more than a narrow depth-three search.
- **Label cost:** grade each distinct proposed move once per fresh world. Depth-capped searches may need separate runs: trimming a final tree does not undo earlier allocations informed by deeper results. A compute checkpoint describes a prefix, not an independently tuned shorter search.
- **No monotonicity constraint:** more search can pick a worse move. `SieveGate` still uses actual tree evidence after search begins.

## Probe candidates

| | A: reference Gumbel | B: deployment-style Horizon |
|---|---|---|
| **Run** | Existing multi-particle Gumbel; evaluation-mode root choice, sequential halving. Start from a known strong preset; record its exact settings. | Existing Horizon; screen contenders on shared worlds, refine and stop under its native compute-price rule. Fix the preset. |
| **Question answered** | Where does paying for the deep reference change the move for the better? | Where does paying for adaptive Horizon change the move for the better? |
| **Trade-off** | Clear reference target; costly even at easy roots. | Can generate more roots per eval budget; early exits may miss gains the deep teacher finds. |

1. **Collect:** uniformly sample player decisions, independent of the head. Save state and prior choice. Run one candidate search as a side branch; retain visited graph, root choice, actual Q evals and settings. `256 simulations` is not a `256 Q-eval` cap.
2. **Grade (candidate):** if choices agree, label zero. Otherwise force both moves on fresh matched worlds and finish with the same frozen prior. Average signed win differences. Keep grading evals separate from search evals.
3. **Reuse:** retain tree evidence for later auxiliary targets or checkpoint records; do not assign the root label to every visited state. Initial head comparison uses root labels only.
4. **Compare if useful:** same checkpoint, sampled roots and continuation policy; grade each distinct move once per world. Fit each teacher's labels separately. On held-out fights compare **realised gain captured at a fixed total play Q-eval budget**, counting actual search spend. Also report **training Q evals** (probe + grading). No need to re-establish that deep search beats the prior.

- **Starting teacher:** A. B is an alternative, not a second teacher mixed into these labels.
- **Open:** exact presets, grading horizon / world count, auxiliary use of the tree, loss weight, and entry threshold.
