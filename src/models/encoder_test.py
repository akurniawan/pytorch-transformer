import torch

from .encoder import TransformerEncoder


def test_encoder():
    encoder = TransformerEncoder(512, 8, 4)
    X = torch.randn(3, 5, 512)
    enc = encoder(X)

    assert enc.size() == X.size()
