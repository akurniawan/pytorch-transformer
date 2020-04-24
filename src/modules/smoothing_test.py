import unittest
import torch

from hypothesis import given
from hypothesis.strategies import integers

from modules.embedding import OneHotEmbedding
from modules.smoothing import label_smoothing


class SmoothingTest(unittest.TestCase):
    @given(integers(1, 1000))
    def test_label_smoothing(self, C):
        onehot = OneHotEmbedding(1000)
        smoothed_label = label_smoothing(onehot(torch.LongTensor([C])))

        test = smoothed_label.sum(1).data.numpy()[0]
        self.assertAlmostEqual(round(test), 1.0)


if __name__ == '__main__':
    unittest.main()
