Goal: Slime boss is our biggest run killer currently
 - Build a custom network for it
 - We want to see the agent playing around the split points

We use the existing card picking from the other designers project to generate potential
action 1 decks + hp + state.

The we use policy iteration to try to train a policy with notably better outcomes on the
hardest boss.


This is a great test for our approach as the slime boss has extreme non-linearity in how its splits.
Its often better to not do damage to leave it about 75 hp to get a good split.
