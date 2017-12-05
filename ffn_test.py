import unittest
import torch

from torch.autograd import Variable
from ffn import PositionWiseFFN

class PositionWiseFFNTest(unittest.TestCase):
    def test_ffn(self):
        inputs = Variable(torch.randn(3, 5, 512))
        num_units = [2048, 512]
        pwffn = PositionWiseFFN(inputs.size(-1), num_units)
        result = pwffn(inputs)

        self.assertEqual(result.size(), inputs.size())
