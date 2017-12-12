import unittest
import numpy as np
import torch

from torch.autograd import Variable
from modules.embedding import TransformerEmbedding


class TransformerEmbeddingTest(unittest.TestCase):
    def test_embedding(self):
        vocab_size = 100
        max_length = 150
        embedding_size = 300

        sequence = Variable(
            torch.LongTensor(
                np.random.randint(0, 100, size=(3, 5))), requires_grad=False)
        embedding = TransformerEmbedding(vocab_size, max_length,
                                         embedding_size)

        result = embedding(sequence)
        self.assertEqual(result.size(), (3, 5, embedding_size))
