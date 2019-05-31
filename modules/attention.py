import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    def __init__(self,
                 query_dim,
                 key_dim,
                 num_units,
                 dropout_p=0.1,
                 h=8,
                 is_masked=False):
        super(MultiHeadAttention, self).__init__()

        if query_dim != key_dim:
            raise ValueError("query_dim and key_dim must be the same")
        if num_units % h != 0:
            raise ValueError("num_units must be dividable by h")
        if query_dim != num_units:
            raise ValueError("to employ residual connection, the number of "
                             "query_dim and num_units must be the same")

        self._num_units = num_units
        self._h = h
        self._key_dim = float(key_dim)**-0.5
        self._dropout_p = dropout_p
        self._is_masked = is_masked

        self.query_layer = nn.Linear(query_dim, num_units, bias=False)
        self.key_layer = nn.Linear(key_dim, num_units, bias=False)
        self.value_layer = nn.Linear(key_dim, num_units, bias=False)
        self.proj_layer = nn.Linear(num_units, num_units)
        self.ln = nn.LayerNorm(query_dim)

    def forward(self, query, keys):
        """
        Args:
            query (torch.Tensor): [seq_len, batch, embed_dim]
            keys (torch.Tensor): [seq_len, batch, embed_dim]

        Returns:
            torch.Tensor: [seq_len, batch, embed_dim]
        """
        Q = self.query_layer(query)
        K = self.key_layer(keys)
        V = self.value_layer(keys)

        batch_size = query.size(0)
        seq_len = query.size(1)
        # split each Q, K and V into h different values from dim 2
        # and then merge them back together in dim 0
        chunk_size = int(self._num_units / self._h)
        Q = Q.view(batch_size * self._h, seq_len, chunk_size)
        K = K.view(batch_size * self._h, -1, chunk_size)
        V = V.view(batch_size * self._h, -1, chunk_size)

        # calculate QK^T
        attention = torch.bmm(Q, K.transpose(1, 2))
        # normalize with sqrt(dk)
        attention = attention * self._key_dim
        # use masking (usually for decoder) to prevent leftward
        # information flow and retains auto-regressive property
        # as said in the paper
        if self._is_masked:
            diag_vals = attention[0].sign().abs()
            diag_mat = diag_vals.tril(abs(Q.size(1) - K.size(1)))
            diag_mat = diag_mat.unsqueeze(0).expand(attention.size())
            mask = torch.ones(
                diag_mat.size(), device=query.device) * (-2**32 + 1)
            # this is some trick that I use to combine the lower diagonal
            # matrix and its masking. (diag_mat-1).abs() will reverse the value
            # inside diag_mat, from 0 to 1 and 1 to zero. with this
            # we don't need loop operation andn could perform our calculation
            # faster
            attention = (attention * diag_mat) + (mask * (diag_mat - 1).abs())
        # put it to softmax
        attention = F.softmax(attention, dim=-1)
        # apply dropout
        attention = F.dropout(
            attention, p=self._dropout_p, training=self.training)
        # multiplyt it with V
        attention = torch.bmm(attention, V)
        # convert attention back to its input original size
        attention = attention.view(batch_size, seq_len, -1)

        # apply  projection
        attention = self.proj_layer(attention.view(-1, attention.size(-1)))
        attention = attention.view(batch_size, seq_len, -1)
        # residual connection
        attention += query
        # apply layer normalization
        attention = self.ln(attention)

        return attention

    def init_weight(self):
        nn.init.uniform_(self.query_layer.weight, -0.1, 0.1)
        nn.init.uniform_(self.key_layer.weight, -0.1, 0.1)
        nn.init.uniform_(self.value_layer.weight, -0.1, 0.1)
        nn.init.uniform_(self.proj_layer.weight, -0.1, 0.1)