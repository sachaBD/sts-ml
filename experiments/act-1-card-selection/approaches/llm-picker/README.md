# LLM picker

**Status: option.** There is recent literature on LLM agents for Slay the Spire; not yet reviewed here.

## Idea

Serialize the public run state (deck, HP, relics, potions, floor, boss, offered cards) into a prompt and ask an LLM for a choice, optionally with reasoning.

## Trade-offs

- **For:** strong built-in game knowledge of synergies and archetypes, and no training data needed.
- **Against:**
  - Slow and costly per decision.
  - Hard to verify, and non-deterministic.
  - Hard to improve from our own outcome data.
  - Local models on one machine may be weak; API use changes the "single machine" framing.

## Possible roles

- A prior or initial rollout policy for CARDS.
- A sanity check or disagreement probe against search recommendations.
- An evaluation opponent.
