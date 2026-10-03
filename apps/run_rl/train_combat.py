"""Fine-tune the frozen combat architecture on combat_v4 fights.

Not available: combat_v4 no longer stores NN encodings (runs/schema=combat_v4/schema.py). Training needs them
derived from the fights by the combat_v4 decompressor (runs/schema=combat_v4_full), which is not implemented yet.
"""
import sys


def main():
    sys.exit('apps/run_rl/train_combat.py: combat_v4 has no stored NN encodings; derive them from the fights '
             '(combat_v4 decompressor, not implemented yet) before training.')


if __name__ == '__main__':
    main()
