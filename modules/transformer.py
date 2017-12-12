import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.embedding import TransformerEmbedding
from modules.encoder import TransformerEncoder
from modules.decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(self,
                 max_length,
                 enc_vocab,
                 dec_vocab,
                 enc_emb_size,
                 dec_emb_size,
                 enc_units,
                 dec_units,
                 dropout_rate=0.8):
        super(Transformer, self).__init__()
        self._dropout_rate = dropout_rate
        enc_vocab_size = len(enc_vocab.itos)
        dec_vocab_size = len(dec_vocab.itos)

        self.encoder_embedding = TransformerEmbedding(
            vocab_size=enc_vocab_size,
            padding_idx=enc_vocab.stoi["<pad>"],
            max_length=max_length,
            embedding_size=enc_emb_size)
        self.decoder_embedding = TransformerEmbedding(
            vocab_size=dec_vocab_size,
            padding_idx=enc_vocab.stoi["<pad>"],
            max_length=max_length,
            embedding_size=dec_emb_size)

        self.encoder = TransformerEncoder(enc_emb_size, enc_units)
        self.decoder = TransformerDecoder(dec_emb_size, enc_emb_size,
                                          dec_units)

        self.output_layer = nn.Linear(
            in_features=enc_units[-1], out_features=dec_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, enc_input, dec_input):
        enc_embed = self.encoder_embedding(enc_input)
        enc_embed = F.dropout(enc_embed, self._dropout_rate)
        encoder_result = self.encoder(enc_embed)
        encoder_result = F.dropout(encoder_result, self._dropout_rate)

        dec_embed = self.decoder_embedding(dec_input)
        dec_embed = F.dropout(dec_embed, self._dropout_rate)
        decoder_result = self.decoder(dec_embed, encoder_result)
        decoder_result = F.dropout(decoder_result, self._dropout_rate)

        output = self.output_layer(decoder_result)
        softmax = self.softmax(output)

        return softmax, output
