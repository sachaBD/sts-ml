"""The network and its optimizer: trains on replay, saves and restores checkpoints. Deployment is owned by ModelExporter."""
from dataclasses import asdict

import torch

from agents.combat.nn.model import THINKING_ROUNDS, build_model
from agents.combat.nn.training.replay import TARGETS
from agents.combat.nn.training.train import losses, score, train


class Learner:
    def __init__(self, spec, lr, device, seed=0):
        torch.manual_seed(seed)
        self.net = build_model(spec).to(device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.device = device
        self.losses = losses  # compile() replaces it; checkpoints, scoring and export never use it

    def compile(self):
        """Compile the training losses once per process; one synthetic step pays most of the cost up front.
        Losses, not the module: training calls heads(), which a compiled module leaves eager.
        Weights, optimizer state and RNG are untouched."""
        self.losses = torch.compile(losses, dynamic=True)
        tokens = self.net.example_tokens()
        self.net.train()
        policy = torch.zeros(tokens.shape[:2])
        policy[:, 1:] = 1 / (tokens.shape[1] - 1)  # uniform over example_tokens' legal actions
        self.losses(self.net, tokens, policy, torch.ones(len(tokens), len(TARGETS) + THINKING_ROUNDS), self.device).sum().backward()
        self.optimizer.zero_grad(set_to_none=True)

    def fit(self, replay, steps, batch, log=print, *, value_batch=0, on_counters=None, lr=None):
        """lr, when given, replaces the optimizer's, including one restored from a checkpoint."""
        if lr is not None:
            for group in self.optimizer.param_groups:
                group['lr'] = lr
        return train(self.net, self.optimizer, replay, None, None, steps, batch, self.device,
                     log=log, value_batch=value_batch, on_counters=on_counters, loss_fn=self.losses)

    def score(self, replay, rows, batch):
        return score(self.net, replay, rows, batch, self.device)

    def save(self, path):
        cuda = self.device.startswith('cuda')
        torch.save(dict(net=self.net.state_dict(), optimizer=self.optimizer.state_dict(), spec=asdict(self.net.spec),
                        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if cuda else []), path)

    def load(self, path):
        """Restores weights, optimizer and RNG, so training continues exactly where the checkpoint left off."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['spec'] != asdict(self.net.spec):
            raise ValueError(f'{path}: architecture {checkpoint["spec"]} differs from {self.net.spec}')
        weights = checkpoint['net']
        optimizer = checkpoint['optimizer']
        if not {'thinking.weight', 'thinking.bias'} & weights.keys():
            # Thinking parameters are last. Preserve Adam moments for every existing parameter.
            weights = {**weights, **{k: v for k, v in self.net.state_dict().items() if k.startswith('thinking.')}}
            saved, = optimizer['param_groups']
            current, = self.optimizer.state_dict()['param_groups']
            if saved['params'] != current['params'][:-2]:
                raise ValueError(f'{path}: unsupported legacy optimizer parameter layout')
            saved['params'] = current['params']
        self.net.load_state_dict(weights)
        self.optimizer.load_state_dict(optimizer)
        torch.set_rng_state(checkpoint['rng'])
        if checkpoint['cuda_rng'] and self.device.startswith('cuda'):
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
