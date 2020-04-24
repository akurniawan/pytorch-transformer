import torch.nn as nn

from modules.ffn import PositionWiseFFN


class _Layer(nn.Module):
    def __init__(self, dim, num_head):
        super(_Layer, self).__init__()

        self.attn = nn.MultiheadAttention(dim, num_head)
        self.pffn = PositionWiseFFN(dim)

    def forward(self, src):
        out = self.attn(src, src, src)[0]
        out = self.pffn(out)

        return out


class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_head, num_layers):
        super().__init__()

        self.layer = nn.Sequential(
            *[_Layer(dim, num_head) for _ in range(num_layers)])

    def forward(self, src):
        return self.layer(src)
