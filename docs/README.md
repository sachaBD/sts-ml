# Documentation

The project layout and quickstart are in the root README. Component contracts live beside their owners; experiment recipes, analysis, and conclusions live in `experiments/`.

`research/` preserves the former `slop_docs/` notes: proposals, explanations, and handoffs of mixed quality. Treat these as working research notes, not validated findings; their original caveats are retained. Historical references to removed experimental files may not be runnable.

## Ownership

- Environments own what can happen.
- Agents own what to choose and how the chooser is learned, including its model architecture and inference.
- Shared predictive models do not choose actions and live in `models/`.
- Apps execute workflows; `runs/` owns provenance and generated outputs.
- Tests live with the component they exercise; `make test` aggregates them.
