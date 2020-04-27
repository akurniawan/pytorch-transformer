import torch.nn as nn

from ..modules.ffn import PositionWiseFFN


class _Layer(nn.Module):
    def __init__(self, dim, num_head):
        super(_Layer, self).__init__()

        self.self_attn = nn.MultiheadAttention(dim, num_head)
        self.lookbehind_attn = nn.MultiheadAttention(dim, num_head)
        self.pffn = PositionWiseFFN(dim)

        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)

    def forward(self, src, tgt):
        out = self.self_attn(tgt, tgt, tgt)[0]
        out += tgt
        out = self.ln1(out)
        look_out = self.lookbehind_attn(tgt, src, src)[0]
        out += look_out
        out = self.ln2(out)
        out = self.pffn(out)

        return out


class TransformerDecoder(nn.Module):
    def __init__(self, dim, num_head, num_layers):
        super().__init__()

        self.layers = nn.ModuleList(
            [_Layer(dim, num_head) for _ in range(num_layers)])

    def forward(self, src, tgt):
        out = tgt
        for layer in self.layers:
            out = layer(out, src)

        return out
