import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, num_units, query_size, memory_size):
        super(BahdanauAttention, self).__init__()

        self._num_units = num_units
        self._softmax = nn.Softmax()

        self.query_layer = nn.Linear(query_size, num_units, bias=False)
        self.memory_layer = nn.Linear(memory_size, num_units, bias=False)
        self.alignment_layer = nn.Linear(num_units, 1, bias=False)

    def _score(self, query, keys):
        # Put the query and the keys into Dense layer
        processed_query = self.query_layer(query)
        values = self.memory_layer(keys)

        # since the sizes of processed_query i [B x embedding],
        # we can't directly add it with the keys. therefore, we need
        # to add extra dimension, so the dimension of the query
        # now become [B x 1 x alignment unit size]
        extended_query = processed_query.unsqueeze(1)

        # The original formula is v * tanh(extended_query + values).
        # We can just use Dense layer and feed tanh as the input
        alignment = self.alignment_layer(F.tanh(extended_query + values))

        # Now the alignment size is [B x S x 1], We need to squeeze it
        # so that we can use Softmax later on. Converting to [B x S]
        return alignment.squeeze()

    def forward(self, query, keys):
        # Calculate the alignment score
        alignment_score = self._score(query, keys)

        # Put it into softmax to get the weight of every steps
        weight = F.softmax(alignment_score)

        # To get the context, this is the original formula
        # context = sum(weight * keys)
        # In order to multiply those two, we need to reshape the weight
        # from [B x S] into [B x S x 1] for broacasting.
        # The multiplication will result in [B x S x embedding]. Remember,
        # we want the score as the sum over all the steps. Therefore, we will
        # sum it over the 1st index
        context = weight.unsqueeze(2) * keys
        total_context = context.sum(1)

        return total_context, alignment_score


class LuongLocalAttention(nn.Module):
    _SCORE_FN = {
        "dot": "_dot_score",
        "general": "_general_score",
        "concat": "_concat_score"
    }

    def __init__(self,
                 attention_window_size,
                 num_units,
                 query_size,
                 memory_size,
                 score_fn="dot"):
        super(LuongLocalAttention, self).__init__()

        if score_fn not in self._SCORE_FN.keys():
            raise ValueError()

        self._attention_window_size = attention_window_size
        self._softmax = nn.Softmax()
        self._score_fn = score_fn

        self.query_layer = nn.Linear(query_size, num_units, bias=False)
        self.memory_layer = nn.Linear(memory_size, num_units, bias=False)
        self.predictive_alignment_layer = nn.Linear(
            num_units, 1, bias=False)
        self.alignment_layer = nn.Linear(num_units, 1, bias=False)

    def _dot_score(self, query, keys, key_lengths=None):
        depth = query.size(-1)
        key_units = keys.size(-1)
        if depth != key_units:
            raise ValueError(
                "Incompatible inner dimensions between query and keys. "
                "Query has units: %d. Keys have units: %d. "
                "Dot score requires you to have same size between num_units in "
                "query and keys" % (depth, key_units))

        # Expand query to [B x 1 x embedding dim] for broadcasting
        extended_query = query.unsqueeze(1)

        # Transpose the keys so that we can multiply it
        tkeys = keys.transpose(1, 2)

        alignment = torch.matmul(extended_query, tkeys)

        # Result of the multiplication will be in size [B x 1 x embedding dim]
        # we can safely squeeze the dimension
        return alignment.squeeze(1)

    def forward(self, query, keys, key_lengths):
        score_fn = getattr(self, self._SCORE_FN[self._score_fn])
        alignment_score = score_fn(query, keys, key_lengths)

        weight = F.softmax(alignment_score)

        extended_key_lengths = key_lengths.unsqueeze(1)
        preprocessed_query = self.query_layer(query)

        activated_query = F.tanh(preprocessed_query)
        sigmoid_query = F.sigmoid(self.predictive_alignment_layer(activated_query))
        predictive_alignment = extended_key_lengths * sigmoid_query

        ai_start = predictive_alignment - self._attention_window_size
        ai_end = predictive_alignment + self._attention_window_size

        std = torch.FloatTensor([self._attention_window_size / 2.]).pow(2)
        alignment_point_dist = (extended_key_lengths - predictive_alignment).pow(2)

        alignment_point_dist = (-(alignment_point_dist/(2 * std[0]))).exp()
        # TODO: add masking for value whose index is not in
        # range of ai_start:ai_end
        weight = weight * alignment_point_dist

        context = weight.unsqueeze(2) * keys

        total_context = context.sum(1)

        return total_context, alignment_score

    @property
    def attention_window_size(self):
        return self._attention_window_size


class MultiHeadAttention(nn.Module):
    def __init__(self):
        pass