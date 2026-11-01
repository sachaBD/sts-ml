"""The network and its optimizer: trains on replay, saves and restores checkpoints. Deployment is owned by ModelExporter."""
from dataclasses import asdict

import torch

from .model import build_model
from .train import train


class Learner:
    def __init__(self, spec, lr, device, seed=0):
        torch.manual_seed(seed)
        self.net = build_model(spec).to(device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.device = device

    def fit(self, replay, steps, batch, log=print):
        train(self.net, self.optimizer, replay, None, None, steps, batch, self.device, log=log)

    def save(self, path):
        cuda = self.device.startswith('cuda')
        torch.save(dict(net=self.net.state_dict(), optimizer=self.optimizer.state_dict(), spec=asdict(self.net.spec),
                        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if cuda else []), path)

    def load(self, path):
        """Restores weights, optimizer and RNG, so training continues exactly where the checkpoint left off."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['spec'] != asdict(self.net.spec):
            raise ValueError(f'{path}: architecture {checkpoint["spec"]} differs from {self.net.spec}')
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['rng'])
        if checkpoint['cuda_rng'] and self.device.startswith('cuda'):
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
