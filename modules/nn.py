import torch
import torch.nn as nn


class ExtendedSequential(nn.Sequential):
    def __init__(self, *args):
        super(ExtendedSequential, self).__init__(*args)

    def forward(self, *inputs):
        for module in self._modules.values():
            if isinstance(inputs, list) or isinstance(inputs, tuple):
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs