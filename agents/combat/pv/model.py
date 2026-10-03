"""PV model and ONNX export. PyTorch is the only implementation of the network.

Inputs are padded token sets produced by features.cpp (zero-ID tokens are padding).
Cards sum separately by zone; other sets sum once. Legal actions share the token encoders.
"""
import torch
from torch import nn

NAMES = ('context', 'cards', 'monsters', 'potions', 'relics', 'actions')
WIDTHS = (65, 18, 29, 19, 4, 261)
CONTRACT = 'pv_champ_hp_v2'
VALUE_SCALE = 100.0  # fixed training-conditioning scale, never dependent on state/max HP


class TokenEncoder(nn.Module):
    def __init__(self, vocab, numeric, width):
        super().__init__()
        self.id = nn.Embedding(vocab, 16)
        self.mlp = nn.Sequential(nn.Linear(16 + numeric, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())

    def forward(self, x):
        return self.mlp(torch.cat((self.id(x[..., 0].long()), x[..., 1:]), -1)) * (x[..., :1] != 0)


class PolicyValue(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.width = width
        self.card = TokenEncoder(512, 17, width)
        self.monster = TokenEncoder(128, 28, width)
        self.potion = TokenEncoder(64, 18, width)
        self.relic = TokenEncoder(192, 3, width)
        self.trunk = nn.Sequential(nn.Linear(65 + 7 * width, 2 * width), nn.ReLU(),
                                   nn.Linear(2 * width, width), nn.ReLU())
        self.value = nn.Linear(width, 1)
        self.kind = nn.Embedding(8, 4)
        self.task = nn.Embedding(32, 4)
        self.policy = nn.Sequential(nn.Linear(5 * width + 20, width), nn.ReLU(), nn.Linear(width, 1))

    def forward(self, context, cards, monsters, potions, relics, actions):
        c = self.card(cards)
        pools = [(c * (cards[..., 1:2] == z)).sum(1) for z in range(4)]
        pools += [self.monster(monsters).sum(1), self.potion(potions).sum(1), self.relic(relics).sum(1)]
        h = self.trunk(torch.cat((context, *pools), -1))
        a = actions
        # A multi-card move embeds the actual subset, not its hand-slot bit mask.
        subset = self.card(a[..., 80:260].reshape(*a.shape[:2], 10, 18)).sum(2)
        move = torch.cat((h[:, None].expand(-1, a.shape[1], -1),
                          self.kind(a[..., 0].long()), self.task(a[..., 1].long()), a[..., 2:8],
                          self.card(a[..., 8:26]), self.monster(a[..., 26:55]),
                          self.potion(a[..., 55:74]), subset, a[..., 74:80]), -1)
        logits = self.policy(move).squeeze(-1).masked_fill(a[..., 260] == 0, -1e9)
        # Unbounded, nonnegative HP-equivalent score. No root/leaf normalization or clipping.
        return VALUE_SCALE * torch.nn.functional.softplus(self.value(h).squeeze(-1)), logits

    def export(self, example, path):
        """Export dynamic batch/token dimensions; all graph computation stays in PyTorch/ONNX."""
        import onnx

        self.eval()
        dims = {n: {0: torch.export.Dim('batch')} for n in NAMES}
        for n in NAMES[1:]:
            dims[n][1] = torch.export.Dim(n + '_count')
        # Export sample dimensions must exceed one for torch.export to keep them dynamic.
        torch.onnx.export(self, tuple(example[n] for n in NAMES), str(path), dynamo=True,
                          input_names=list(NAMES), output_names=['value', 'policy_logits'], dynamic_shapes=dims)
        graph = onnx.load(path)
        onnx.helper.set_model_props(graph, {'pv_contract': CONTRACT, 'width': str(self.width)})
        onnx.save(graph, path)
