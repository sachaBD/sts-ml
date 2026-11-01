# Coding guidelines

The smallest honest change, using what is already there, that explains itself.

1. **No magic.** Prefer what a reader already knows (the standard library, an existing helper) over a clever
   local construct. Never add another copy of something that needs explaining.
2. **Code is the documentation.** Names, types and structure carry the meaning. Comments say only what code
   cannot: why, and invariants. Do not describe code in prose; plans live outside the repository.
3. **No abstraction without a user.** A new concept must beat "we managed without it". It arrives with its
   second consumer, not before.
4. **Respect existing boundaries.** Build on the interfaces that exist. Concrete choices stay at the edge, in
   the driver that needs them, so general layers stay general.
5. **Tests state invariants.** A test checks a property that holds for any state, seed or evaluator. Do not
   restate constants, outputs or wiring.
6. **Each PR justifies itself.** Small, in scope, self-evidently correct, with its evidence. Nothing that needs
   a future PR to make sense.
7. **Design before code.** Agree the shape first; it is cheapest to change before code exists.
