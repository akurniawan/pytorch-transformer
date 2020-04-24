import torch

from .decoder import TransformerDecoder


def test_decoder():
    decoder = TransformerDecoder(512, 8, 6)
    query = torch.randn(3, 10, 512, requires_grad=False)
    key = torch.randn(3, 5, 512, requires_grad=False)

    result = decoder(query, key)
    assert result.size() == query.size()
