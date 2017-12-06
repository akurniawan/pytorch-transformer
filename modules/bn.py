import torch
import torch.nn as nn

from torch.nn.parameter import Parameter


class DynamicBatchNormalization(nn.Module):
    def __init__(self, feature_size, epsilon=1e-8):
        super(DynamicBatchNormalization, self).__init__()
        self.beta = Parameter(torch.randn(feature_size))
        self.gamma = Parameter(torch.randn(feature_size))
        self._epsilon = epsilon

    def forward(self, inputs):
        mean = inputs.mean(0)
        variance = inputs.var(0)
        normalized = (inputs - mean) / ((variance + self._epsilon) ** (.5))
        outputs = normalized * self.gamma + self.beta

        return outputs
