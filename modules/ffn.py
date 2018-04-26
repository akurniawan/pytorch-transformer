import torch.nn as nn


class PositionWiseFFN(nn.Module):
    def __init__(self, feature_size, num_units=[2048, 512]):
        super(PositionWiseFFN, self).__init__()
        self.ffn = self._build_ffn(feature_size, num_units)

    def _build_ffn(self, feature_size, num_units):
        layers = []
        features = feature_size
        for unit in num_units:
            layers.append(nn.Linear(features, unit))
            features = unit

        return nn.Sequential(*layers)

    def forward(self, X):
        # assert if the feature size of inputs not the same as
        # the last ffn layer, since we need both of them
        # the same for residual network
        assert X.size(-1) == self.ffn[-1].bias.size(-1)
        ffn = self.ffn(X)
        # residual network
        ffn += X

        return ffn
