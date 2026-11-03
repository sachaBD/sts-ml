"""The network and its optimizer: trains on replay, saves and restores checkpoints. Deployment is owned by ModelExporter."""
from dataclasses import asdict

import torch

from .model import ModelSpec, build_model
from .tokens import TokenKind, WIDTH
from .train import losses, score, train


class Learner:
    def __init__(self, spec, lr, device, seed=0):
        torch.manual_seed(seed)
        self.net = build_model(spec).to(device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.device = device
        self.training_net = self.net  # compile() replaces it; checkpoints, scoring and export use self.net

    def compile(self):
        """Compile training once per process, paying the cost now with one synthetic forward and backward.
        Weights, optimizer state and RNG are untouched, so checkpoints and resume are unaffected."""
        self.training_net = torch.compile(self.net, dynamic=True)
        tokens = torch.zeros(2, 3, WIDTH)
        tokens[:, 0, 0], tokens[:, 1:, 0] = TokenKind.CONTEXT, TokenKind.ACTION
        self.net.train()
        policy = torch.tensor([[0., .5, .5]] * 2)
        sum(losses(self.training_net, tokens, policy, torch.ones(2), self.device)).backward()
        self.optimizer.zero_grad(set_to_none=True)

    def fit(self, replay, steps, batch, log=print, *, value_batch=0, on_counters=None, lr=None):
        """lr, when given, replaces the optimizer's, including one restored from a checkpoint."""
        if lr is not None:
            for group in self.optimizer.param_groups:
                group['lr'] = lr
        return train(self.training_net, self.optimizer, replay, None, None, steps, batch, self.device,
                     log=log, value_batch=value_batch, on_counters=on_counters)

    def score(self, replay, rows, batch):
        return score(self.net, replay, rows, batch, self.device)

    def save(self, path):
        cuda = self.device.startswith('cuda')
        torch.save(dict(net=self.net.state_dict(), optimizer=self.optimizer.state_dict(), spec=asdict(self.net.spec),
                        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if cuda else []), path)

    def load(self, path):
        """Restores weights, optimizer and RNG, so training continues exactly where the checkpoint left off."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if ModelSpec(**checkpoint['spec']) != self.net.spec:  # specs saved before a field existed take its default
            raise ValueError(f'{path}: architecture {checkpoint["spec"]} differs from {self.net.spec}')
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['rng'])
        if checkpoint['cuda_rng'] and self.device.startswith('cuda'):
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
