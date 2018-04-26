import unittest

import numpy as np
import torch
from hypothesis import given
from hypothesis.extra.numpy import arrays
from hypothesis.strategies import floats

from modules.attention import (BahdanauAttention, LuongAttention,
                               MultiHeadAttention)


class AttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        np.random.seed(123)

        query_size = (3, 512)
        keys_size = (3, 15, 512)
        self.context_size = (keys_size[0], keys_size[2])
        self.alignment_size = (keys_size[0], keys_size[1])

        self.query = torch.from_numpy(
            np.random.randn(*query_size).astype(np.float32))
        self.keys = torch.from_numpy(
            np.random.randn(*keys_size).astype(np.float32))

    def test_bahdanau_attention(self):
        bahdanau_attention = BahdanauAttention(
            num_units=128,
            query_size=self.query.size(1),
            memory_size=self.keys.size(2))
        context, alignment_score = bahdanau_attention(self.query, self.keys)

        self.assertEqual(context.size(), self.context_size)
        self.assertEqual(alignment_score.size(), self.alignment_size)

    def test_local_luong_attention_dot(self):
        luong_attention = LuongAttention(
            attention_window_size=3,
            num_units=128,
            query_size=self.query.size(1),
            memory_size=self.keys.size(2),
            score_fn="dot")
        sentence_lengths = torch.FloatTensor(
            [self.keys.size(1)] * self.keys.size(0))
        context, alignment_score = luong_attention(self.query, self.keys,
                                                   sentence_lengths)

        self.assertEqual(context.size(), self.context_size)
        self.assertEqual(alignment_score.size(), self.alignment_size)

    def test_local_luong_attention_general(self):
        luong_attention = LuongAttention(
            attention_window_size=3,
            num_units=128,
            query_size=self.query.size(1),
            memory_size=self.keys.size(2),
            score_fn="general")
        sentence_lengths = torch.FloatTensor(
            [self.keys.size(1)] * self.keys.size(0))
        context, alignment_score = luong_attention(self.query, self.keys,
                                                   sentence_lengths)

        self.assertEqual(context.size(), self.context_size)
        self.assertEqual(alignment_score.size(), self.alignment_size)

    def test_local_luong_attention_concat(self):
        luong_attention = LuongAttention(
            attention_window_size=3,
            num_units=128,
            query_size=self.query.size(1),
            memory_size=self.keys.size(2),
            score_fn="concat")
        sentence_lengths = torch.FloatTensor(
            [self.keys.size(1)] * self.keys.size(0))
        context, alignment_score = luong_attention(self.query, self.keys,
                                                   sentence_lengths)

        self.assertEqual(context.size(), self.context_size)
        self.assertEqual(alignment_score.size(), self.alignment_size)

    def test_global_luong_attention_dot(self):
        luong_attention = LuongAttention(
            attention_window_size=3,
            num_units=128,
            query_size=self.query.size(1),
            memory_size=self.keys.size(2),
            alignment="global",
            score_fn="dot")
        sentence_lengths = torch.FloatTensor(
            [self.keys.size(1)] * self.keys.size(0))
        context, alignment_score = luong_attention(self.query, self.keys,
                                                   sentence_lengths)

        self.assertEqual(context.size(), self.context_size)
        self.assertEqual(alignment_score.size(), self.alignment_size)

    @given(
        arrays(
            dtype=np.float32, shape=(16, 70, 512), elements=floats(-10, 10)))
    def test_multi_head_attention(self, xs):
        _keys = torch.from_numpy(xs)
        num_units = 512
        mh_attention = MultiHeadAttention(
            query_dim=_keys.size(2),
            key_dim=_keys.size(2),
            num_units=num_units)

        count_vars = 0
        for params in mh_attention.parameters():
            if len(params.size()) == 2:
                self.assertEqual(params.size(0), num_units)
                self.assertEqual(params.size(1), num_units)
                self.assertEqual(params.size(0), params.size(1))
            else:
                self.assertEqual(params.size(0), num_units)
            count_vars += 1
        self.assertEqual(count_vars, 5)

        batch_size = _keys.size(0)
        num_units = _keys.size(2)
        keys = torch.cat([_keys, torch.zeros(batch_size, 1, num_units)], dim=1)
        attention_result = mh_attention(keys, keys)

        self.assertEqual(attention_result.size(), keys.size())
