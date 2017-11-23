import unittest

import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable

from attention import BahdanauAttention


class AttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        np.random.seed(123)

    def test_bahdanau(self):
        query_size = (3, 20)
        keys_size = (3, 15, 10)
        context_size = (keys_size[0], keys_size[2])
        alignment_size = (keys_size[0], keys_size[1])

        query = Variable(
            torch.from_numpy(np.random.randn(*query_size).astype(np.float32)))
        keys = Variable(
            torch.from_numpy(np.random.randn(*keys_size).astype(np.float32)))

        bahdanau_attention = BahdanauAttention(128, query.size(1),
                                               keys.size(2))
        context, alignment_score = bahdanau_attention(query, keys)

        self.assertEqual(context.size(), context_size)
        self.assertEqual(alignment_score.size(), alignment_size)
