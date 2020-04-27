import math

import torch
import torch.nn as nn


class TransformerEmbedding(nn.Module):
    def __init__(self, word_embedding, positional_embedding=None):
        super(TransformerEmbedding, self).__init__()
        # add + 3 for pad, unk and eos/bos
        self.word_embedding = word_embedding
        # self.word_embedding = nn.Embedding(vocab_size + 3,
        #                                    embedding_size,
        #                                    padding_idx=padding_idx)

        self.positional_embedding = positional_embedding

    def forward(self, X):
        out = self.word_embedding(X)
        if self.positional_embedding:
            out = self.positional_embedding(out)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class OneHotEmbedding(nn.Module):
    def __init__(self, num_class):
        super(OneHotEmbedding, self).__init__()
        self.embed = nn.Embedding(num_class, num_class)
        self.embed.weight.data = self._build_onehot(num_class)
        # to prevent the weight getting trained
        self.embed.weight.requires_grad = False

    def _build_onehot(self, num_class):
        onehot = torch.eye(num_class)
        return onehot

    def forward(self, x):
        return self.embed(x)
