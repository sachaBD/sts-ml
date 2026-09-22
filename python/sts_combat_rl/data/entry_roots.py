"""Deck-group splitting for bootstrap combat data (runs/README.md).

Writing combat rows moved to apps/bootstrap/generate.py.  It owns continuous
scheduling, safe interruption, and the runs/ output contract.
"""

from __future__ import annotations

from random import Random


def deck_signature_split(rows, validation_fraction, seed):
    """Split whole decks, never decisions from one deck across train/validation."""
    signatures = sorted({row["deck_signature"] for row in rows})
    if len(signatures) < 2:
        raise ValueError("deck-signature split requires at least two signatures")
    Random(seed).shuffle(signatures)
    cut = min(max(1, round(len(signatures) * (1 - validation_fraction))), len(signatures) - 1)
    train_signatures, valid_signatures = set(signatures[:cut]), set(signatures[cut:])
    return (
        [row for row in rows if row["deck_signature"] in train_signatures],
        [row for row in rows if row["deck_signature"] in valid_signatures],
        sorted(train_signatures),
        sorted(valid_signatures),
    )
