import torch
import torch.nn as nn

from torch.nn.parameter import Parameter


class DynamicBatchNormalization(nn.Module):
    def __init__(self, feature_size, epsilon=1e-8):
        super(DynamicBatchNormalization, self).__init__()
        self.beta = Parameter(torch.zeros(feature_size))
        self.gamma = Parameter(torch.ones(feature_size))
        self._epsilon = epsilon

    def forward(self, inputs):
        mean = inputs.mean(-1).unsqueeze(-1)
        variance = inputs.var(-1).unsqueeze(-1)
        normalized = (inputs - mean) / ((variance + self._epsilon) ** (.5))
        outputs = normalized * self.gamma.data + self.beta.data

        return outputs
