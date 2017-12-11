import torch.nn as nn

from modules.attention import MultiHeadAttention
from modules.ffn import PositionWiseFFN


class TransformerDecoder(nn.Module):
    def __init__(self, query_dim, key_dim, num_units):
        super(TransformerDecoder, self).__init__()

        self.decoders = self._build_model(query_dim, key_dim, num_units)

    def _build_model(self, query_dim, key_dim, num_units):
        layers = []

        for unit in num_units:
            layer = nn.ModuleList([
                MultiHeadAttention(query_dim, query_dim, unit, is_masked=True),
                MultiHeadAttention(unit, key_dim, unit),
                PositionWiseFFN(unit)
            ])
            layers.append(layer)

        return nn.ModuleList(layers)

    def forward(self, query, key):
        for dec in self.decoders:
            res1 = dec[0](query, query)
            res2 = dec[1](res1, key)
            res3 = dec[2](res2)
            query = res3

        return res3
