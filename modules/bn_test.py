import unittest

import torch

from bn import DynamicBatchNormalization

class BatchNormalizationTest(unittest.TestCase):
    def test_dynamic_bn(self):
        dbn = DynamicBatchNormalization(20)
        inputs = torch.randn(3, 15, 20)
        result = dbn(inputs)

        self.assertEqual(inputs.size(), result.size())
