# Combat environment

The rules of a fight, independent of any agent. The simulator itself is `../lightspeed`.

- `public.hpp`: the fight as a player sees it. Legal moves, each named by what it chooses rather than where that sits (identical cards are one move), keys for "the same
  visible situation", and sampled worlds for what is hidden (draw order, future rolls), so a search cannot cheat.
- `fight_start.hpp`: build a fight from its saved start (JSON). `starts.py` makes and varies those starts.
- `scenarios/`: fixed fights for tests.

Turning a fight into network inputs is the network's job: `agents/combat/nn/features.hpp`.
