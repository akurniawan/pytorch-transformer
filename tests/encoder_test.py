import unittest
import torch

from modules.encoder import TransformerEncoder


class TransformerEncoderTest(unittest.TestCase):
    def test_encoder(self):
        encoder = TransformerEncoder(512, [512] * 6)
        X = torch.randn(3, 5, 512)
        enc = encoder(X)
        self.assertEqual(enc.size(), X.size())


if __name__ == "__main__":
    unittest.main()
