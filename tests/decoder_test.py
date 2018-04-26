import unittest
import torch

from modules.decoder import TransformerDecoder


class TransformerDecoderTest(unittest.TestCase):
    def test_decoder(self):
        decoder = TransformerDecoder(512, 512, [512] * 6)
        query = torch.randn(3, 10, 512, requires_grad=False)
        key = torch.randn(3, 5, 512, requires_grad=False)

        result = decoder(query, key)
        self.assertEqual(result.size(), query.size())


if __name__ == "__main__":
    unittest.main()