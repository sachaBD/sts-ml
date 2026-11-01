"""Small combat inputs and real training updates shared by contract tests."""
import torch

from .model import ModelSpec
from .tokens import CardZone, TokenKind, TokenRows, WIDTH
from .train import train


def small_spec(kind):
    return ModelSpec(kind, dim=16, layers=1, heads=2)


def combat_tokens():
    kinds = [TokenKind.CONTEXT, TokenKind.MONSTER, TokenKind.RELIC,
             TokenKind.CARD, TokenKind.CARD, TokenKind.CARD, TokenKind.CARD,
             TokenKind.ACTION, TokenKind.ACTION, TokenKind.ACTION]
    tokens = torch.zeros(2, len(kinds), WIDTH)
    tokens[..., 0] = torch.tensor(kinds)
    tokens[..., 1] = torch.arange(len(kinds))
    tokens[:, 4:7, 3] = torch.tensor([CardZone.DRAW, CardZone.DISCARD, CardZone.EXHAUST])
    tokens[1, 3, 3] = CardZone.OFFERED
    tokens[..., 4:] = torch.rand(2, len(kinds), WIDTH - 4)
    return tokens


def train_once(learner, tokens):
    actions = TokenRows(tokens).is_kind(TokenKind.ACTION).float()
    policy = actions / actions.sum(dim=1, keepdim=True)
    outcomes = torch.tensor([1., 0.])
    train(learner.net, learner.optimizer, tokens, policy, outcomes,
          steps=1, batch=2, device='cpu', log=lambda _: None)
