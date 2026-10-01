# models/combat_outcome/architectures/

Named pre-combat outcome topologies for `models/combat_outcome/learn_marginals.py --topology <file>`.
A spec = `kind` (class in `agents/combat/value/<kind>.py`) + every constructor argument except
`encounters` (set from data). Name: `co<kind version>-w<width>-h<hidden>[-L<depth>]-d<dropout x100>` (run ids are lower-cased; new specs use `l`).

**Frozen once trained.** Training appends `name<TAB>sha256<TAB>run out dir` to `FROZEN.tsv`; a later run refuses a
spec whose bytes changed. Never edit a frozen spec or a used kind's code: add a new spec / new kind (`combat_outcome_v3`).
`co1-w16-h32-d30` is the legacy hard-coded default (no `--topology`), trained throughout experiments/card-outcomes.
