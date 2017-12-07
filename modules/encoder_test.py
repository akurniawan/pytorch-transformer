import unittest
import torch

from torch.autograd import Variable

from encoder import TransformerEncoder


class TransformerEncoderTest(unittest.TestCase):
    def test_encoder(self):
        encoder = TransformerEncoder(512, [512] * 6)
        X = Variable(torch.randn(3, 5, 512))
        enc = encoder(X)
        self.assertEqual(enc.size(), X.size())
