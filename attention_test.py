import unittest

import numpy as np
import torch
import torch.nn as nn

from torch.autograd import Variable

from attention import BahdanauAttention


class AttentionTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(123)

    def test_bahdanau(self):
        # [B x embedding]
        q = np.array(
            [[0.8599, 0.0769, 0.3757, 0.2706, 0.3908], [
                0.1807, 0.9998, 0.6734, 0.6757, 0.6111
            ], [0.6556, 0.3764, 0.0744, 0.2015, 0.5144]],
            dtype=np.float32)
        # [B X S x embedding]
        k = np.array(
            [[[0.3487, -0.6849, 0.9872, 0.6897, -0.1069], [
                -0.3527, 0.3667, 0.3183, 0.7418, 1.7904
            ], [-2.2037, -0.2246, -2.6717, -1.1718, -0.3740], [
                -1.0758, 2.4612, 0.0645, -0.7556, -0.7677
            ], [-0.6289, 0.2707, 0.4799, 0.2649, 0.8066]], [[
                0.9987, 0.2274, 2.3021, 1.0512, 0.2179
            ], [-2.1130, -0.3766, -1.0890, 0.1726, 0.6302], [
                -1.4765, 0.2976, 1.3884, 0.4130, -0.1273
            ], [-0.5778, 1.0577, 0.7660, 0.9553, 0.1284], [
                0.7613, 1.0447, 1.2169, -0.1073, -1.1696
            ]], [[-0.0400, -0.5368, -2.1448, 0.3482, 2.1307], [
                0.3554, -0.7236, -0.7085, -0.2511, -1.3995
            ], [-0.5165, -0.3775, -1.6791, -0.7991, 0.4681], [
                -0.2399, 0.2564, -0.5338, -0.1205, -0.2918
            ], [-0.1358, -0.3561, -0.7635, -0.0650, -0.5470]]],
            dtype=np.float32)

        result_context = np.array(
            [[
                -0.5459883213043213, 0.45361143350601196, -0.19813239574432373,
                -0.050519876182079315, 0.2608298063278198
            ], [
                -0.4240841865539551, 0.5524493455886841, 0.9416685104370117,
                0.44516095519065857, -0.062102362513542175
            ], [
                -0.1296997368335724, -0.280361533164978, -1.4591201543807983,
                -0.16699394583702087, 0.06320875883102417
            ]],
            dtype=np.float32)

        result_alignment = np.array(
            [[
                -0.32089388370513916, 0.07440998405218124, 0.22585320472717285,
                0.12836413085460663, 0.005609042942523956
            ], [
                -0.34516963362693787, -0.01351994276046753,
                -0.19160181283950806, -0.3283522129058838, -0.24962684512138367
            ], [
                0.11065413057804108, -0.22125248610973358, 0.2177956998348236,
                -0.06752332299947739, -0.13812166452407837
            ]],
            dtype=np.float32)

        query = Variable(torch.from_numpy(q))
        keys = Variable(torch.from_numpy(k))

        bahdanau_attention = BahdanauAttention(128, query.size(1),
                                               keys.size(2))
        context, alignment_score = bahdanau_attention(query, keys)

        self.assertEqual(context.size(), (3, 5))
        self.assertEqual(alignment_score.size(), (3, 5))
        self.assertTrue(np.all(context.data.numpy() == result_context))
        self.assertTrue(np.all(alignment_score.data.numpy() == result_alignment))
