import unittest

import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable

from attention import (BahdanauAttention, LuongAttention, MultiHeadAttention)
from attention import DynamicBatchNormalization


class AttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)
        np.random.seed(123)

        query_size = (3, 24)
        keys_size = (3, 15, 24)
        self.context_size = (keys_size[0], keys_size[2])
        self.alignment_size = (keys_size[0], keys_size[1])

        self.query = Variable(
            torch.from_numpy(np.random.randn(*query_size).astype(np.float32)))
        self.keys = Variable(
            torch.from_numpy(np.random.randn(*keys_size).astype(np.float32)))

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
        sentence_lengths = Variable(
            torch.FloatTensor([self.keys.size(1)] * self.keys.size(0)))
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
        sentence_lengths = Variable(
            torch.FloatTensor([self.keys.size(1)] * self.keys.size(0)))
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
        sentence_lengths = Variable(
            torch.FloatTensor([self.keys.size(1)] * self.keys.size(0)))
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
        sentence_lengths = Variable(
            torch.FloatTensor([self.keys.size(1)] * self.keys.size(0)))
        context, alignment_score = luong_attention(self.query, self.keys,
                                                   sentence_lengths)

        self.assertEqual(context.size(), self.context_size)
        self.assertEqual(alignment_score.size(), self.alignment_size)

    def test_multi_head_attention(self):
        mh_attention = MultiHeadAttention(
            query_dim=self.keys.size(2),
            key_dim=self.keys.size(2),
            num_units=24)
        attention_result = mh_attention(self.keys, self.keys)
        print("running")

        self.assertEqual(attention_result.size(), self.keys.size())
