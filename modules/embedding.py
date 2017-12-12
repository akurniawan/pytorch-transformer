import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn.parameter import Parameter


class TransformerEmbedding(nn.Module):
    def __init__(self,
                 vocab_size,
                 max_length,
                 embedding_size,
                 padding_idx,
                 use_positional_embedding=True):
        super(TransformerEmbedding, self).__init__()
        self._use_positional_embedding = use_positional_embedding

        # add + 3 for pad, unk and eos/bos
        self.word_embedding = nn.Embedding(
            vocab_size + 3, embedding_size, padding_idx=padding_idx)

        if use_positional_embedding:
            self.pos_embedding = nn.Embedding(max_length, embedding_size)
            pos_enc_weight = self._sin_cos_enc(max_length, embedding_size)
            self.pos_embedding.weight = pos_enc_weight

    def _sin_cos_enc(self, max_length, embedding_size):
        position_enc = np.array(
            [[
                pos / np.power(10000, 2 * i / embedding_size)
                for i in range(embedding_size)
            ] for pos in range(max_length)],
            dtype=np.float32)

        # put sinusodial on even position
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])
        # put cosine on odd position
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])

        return Parameter(torch.from_numpy(position_enc))

    def forward(self, X):
        word_embedding = self.word_embedding(X)
        if self._use_positional_embedding:
            T = X.size(1)
            pos = Variable(
                torch.arange(T).expand(X.size()).long(), requires_grad=False)
            pos_embedding = self.pos_embedding(pos)
            word_embedding += pos_embedding

        return word_embedding
