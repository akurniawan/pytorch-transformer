import torch.nn as nn


class PositionWiseFFN(nn.Module):
    def __init__(self, feature_size, num_units=2048, dropout=0.1):
        super(PositionWiseFFN, self).__init__()
        self._dropout = dropout
        self.ffn = nn.Sequential(nn.LayerNorm(feature_size),
                                 nn.Linear(feature_size, num_units), nn.ReLU(),
                                 nn.Dropout(dropout),
                                 nn.Linear(num_units, feature_size))
        self.ln = nn.LayerNorm(feature_size)

    def forward(self, X):
        ffn = self.ffn(X)
        # residual network
        ffn += X
        ffn = self.ln(ffn)

        return ffn

    def init_weight(self):
        for idx in range(len(self.ffn)):
            if hasattr(self.ffn[idx], "weight"):
                nn.init.uniform_(self.ffn[idx].weight, -0.1, 0.1)
