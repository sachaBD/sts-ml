"""Small prepared-run configurations for app boundary tests."""
CONFIG = """
[run]
id = "test"
worker = "nn_play"
seed_namespace = "test"
games_per_deck = 2
iterations = 2

[search]
sims = 4
gumbel_scale = 1.0
rollout_mix = 0.5
particles = 2

[training]
fights = "unused"
steps = 1
batch = 4
lr = 0.001
replay_window = 1
max_train_rows = 0
balanced_replay = false
prune_replay = true
device = "cpu"

[model]
kind = "transformer"
dim = 8
layers = 1
heads = 1
gpu_tokens = 16
export_reuse = false

[workers]
actors = 2
slots = 1
gpu_batch = 512
gpu_bank = [256, 32]
gpu_wait_us = 100
timeout_ms = 321
shutdown_timeout_ms = 123

[evaluation]
sims = 0
at = [0]
every = 0
wall_share = 0
fights = "unused"
"""

POOLED_CONFIG = CONFIG.replace('kind = "transformer"', 'kind = "entity"\ngpu_attended = 8')
