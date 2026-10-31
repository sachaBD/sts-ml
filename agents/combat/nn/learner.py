"""The network and its optimizer: trains on replay, saves and restores checkpoints, exports what nn_play loads."""
import torch

from .graphs import Exporter
from .model import Net, export, export_static
from .play import model_files
from .train import train


class Learner:
    def __init__(self, spec, lr, device, export_reuse=False, seed=0):
        torch.manual_seed(seed)
        self.net = Net(**spec).to(device)
        self.optimizer = torch.optim.AdamW(self.net.parameters(), lr=lr, weight_decay=1e-4)
        self.device = device
        exporter = Exporter() if export_reuse else None
        self._export, self._export_static = (exporter.export, exporter.export_static) if exporter else (export, export_static)

    def fit(self, replay, steps, batch, log=print):
        train(self.net, self.optimizer, replay, None, None, steps, batch, self.device, log=log)

    def save(self, path):
        cuda = self.device.startswith('cuda')
        torch.save(dict(net=self.net.state_dict(), optimizer=self.optimizer.state_dict(), spec=self.net.spec,
                        rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all() if cuda else []), path)

    def load(self, path):
        """Restores weights, optimizer and RNG, so training continues exactly where the checkpoint left off."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
        if checkpoint['spec'] != self.net.spec:
            raise ValueError(f'{path}: architecture {checkpoint["spec"]} differs from {self.net.spec}')
        self.net.load_state_dict(checkpoint['net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        torch.set_rng_state(checkpoint['rng'])
        if checkpoint['cuda_rng'] and self.device.startswith('cuda'):
            torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])

    def export(self, directory, gpu_batch, gpu_tokens, gpu_bank):
        files = model_files(directory, gpu_bank)
        self._export(self.net, files['dynamic'])
        self._export_static(self.net, files['gpu'], gpu_batch, gpu_tokens)
        for batch, path in zip(gpu_bank, files['bank']):
            self._export_static(self.net, path, batch, gpu_tokens)
        self.net.to(self.device)  # the exporters move it to the CPU
        if self.device.startswith('cuda'):
            torch.cuda.empty_cache()  # leave VRAM for the next native TensorRT worker
