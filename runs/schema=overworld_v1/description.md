# overworld_v1

Records actual run-level decisions and public macro-state transitions, independently of the agent architecture. The contract is [schema.py](schema.py).

Macro states are sufficient for the current run-value encoding, not a promise of lossless GameContext replay. Combat links point to independently replayable combat_v4 records. NN values and exploration provenance are separate agent annotations. Event lookahead branches are hypothetical and are not emitted as actual run steps.
