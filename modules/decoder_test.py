import unittest
import torch

from torch.autograd import Variable

from decoder import TransformerDecoder


class TransformerDecoderTest(unittest.TestCase):
    def test_decoder(self):
        decoder = TransformerDecoder(512, 512, [512] * 6)
        query = Variable(torch.randn(3, 10, 512))
        key = Variable(torch.randn(3, 5, 512))

        result = decoder(query, key)
        self.assertEqual(result.size(), query.size())
