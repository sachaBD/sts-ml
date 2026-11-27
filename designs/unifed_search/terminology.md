# Terminology

| Term | Meaning |
|---|---|
| **Eval** | One NN evaluation of one state; returns policy and value. A batch of 32 states counts as 32 evals. |
| **Search** | Tree search from a single root decision. |
| **Rung** | The set of evals at the same action depth in a search; not necessarily one batch. |
| **Sieve** | A budget decision in a partial search: stop and commit the current root move, or buy more compute. Not a depth or a batch. |
| **Probe** | A training-time search from a sampled decision. Produces a proposed move and an explored tree or DAG. Algorithm and budget are open. |
| **World** | One sample of hidden state and future randomness consistent with the root observation. |
| **Benefit of thinking** | Expected change in fight win probability from choosing the probe move instead of the prior move, with subsequent play held to the same policy. |
