import torch
import torch.nn as nn

from modules.attention import MultiHeadAttention
from modules.ffn import PositionWiseFFN
from modules.nn import ExtendedSequential


class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_units):
        super(TransformerEncoder, self).__init__()

        self.encoders = self._build_model(dim, num_units)

    def _build_model(self, dim, num_units):
        layers = []
        # for encoder, we use self-attention, which means we
        # have query_dim and key_dim with same size
        dim = dim
        for unit in num_units:
            layer = ExtendedSequential(
                MultiHeadAttention(dim, dim, unit),
                PositionWiseFFN(unit))
            layers.append(layer)
            dim = unit

        return nn.ModuleList(layers)

    def forward(self, inputs):
        net_inputs = inputs
        for enc in self.encoders:
            net_inputs = enc(net_inputs, net_inputs)

        return net_inputs
