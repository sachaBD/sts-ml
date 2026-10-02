# combat_v4

Goal: lossless, independently replayable combat traces without complete snapshots at every decision. The contract is [schema.py](schema.py).

Canonical facts are a supported initial simulator boundary plus exact executed action bits. RNG is retained for reconstruction, never for NN inputs. Ordinary player boundaries with empty callback/card queues are supported by the initial writer; unsupported boundaries retain explicitly marked outcome/context only, not invented snapshots or training labels. Snapshot scalar supplements preserve exact status/relic bits and caches beyond the typed projection.

Search estimates are agent annotations. Encoded NN tensors are disposable derived caches, not the definition of simulator state. Realized results are separate from search estimates and distinguish battle-end HP from persistent post-exit state. Opaque fight identity links independently recorded combat branches to overworld traces.
