import unittest
import math
import torch

from torch.autograd import Variable
from hypothesis import given
from hypothesis.strategies import integers

from modules.embedding import OneHotEmbedding
from modules.smoothing import label_smoothing


class SmoothingTest(unittest.TestCase):
    @given(integers(1, 1000))
    def test_label_smoothing(self, C):
        onehot = OneHotEmbedding(1000)
        x = Variable(torch.LongTensor([C]))
        smoothed_label = label_smoothing(onehot(x))

        test = smoothed_label.sum(1).data.numpy()[0]
        self.assertAlmostEqual(round(test), 1.0)
