import torch
import torch.nn as nn

from modules.attention import MultiHeadAttention
from modules.ffn import PositionWiseFFN


class TransformerEncoder(nn.Modules):
    def __init__(self, dim, num_units):
        super(TransformerEncoder, self).__init__()

        self.enc = self._build_model(dim, num_units)

    def _build_model(self, dim, num_units):
        layers = []
        # for encoder, we use self-attention, which means we
        # have query_dim and key_dim with same size
        dim = dim
        for unit in num_units:
            layers.append(MultiHeadAttention(dim, dim, unit))
            layers.append(PositionWiseFFN(unit))
            dim = unit

        return nn.Sequential(*layers)

    def forward(self, inputs):
        return self.enc(inputs)