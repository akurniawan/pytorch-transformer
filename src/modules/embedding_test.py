import numpy as np
import torch

from modules.embedding import TransformerEmbedding


def test_embedding():
    vocab_size = 100
    max_length = 150
    embedding_size = 300

    sequence = torch.tensor(np.random.randint(0, 100, size=(3, 5)), requires_grad=False)
    embedding = TransformerEmbedding(vocab_size, max_length, embedding_size, 1)

    result = embedding(sequence)
    assert result.size() == (3, 5, embedding_size)
