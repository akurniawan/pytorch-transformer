import torch
import torch.nn as nn

from modules.embedding import TransformerEmbedding
from modules.encoder import TransformerEncoder
from modules.decoder import TransformerDecoder


class Transformer(nn.Module):
    def __init__(self, max_length, enc_vocab_size, dec_vocab_size,
                 enc_emb_size, dec_emb_size, enc_units, dec_units):
        super(Transformer, self).__init__()
        self.encoder_embedding = TransformerEmbedding(
            vocab_size=enc_vocab_size,
            max_length=max_length,
            embedding_size=enc_emb_size)
        self.decoder_embedding = TransformerEmbedding(
            vocab_size=dec_vocab_size,
            max_length=max_length,
            embedding_size=dec_emb_size)

        self.encoder = TransformerEncoder(enc_emb_size, enc_units)
        self.decoder = TransformerDecoder(dec_emb_size, enc_emb_size,
                                          dec_units)

        self.output_layer = nn.Linear(in_features=enc_units[-1], out_features=dec_vocab_size)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, enc_input, dec_input):
        enc_embed = self.encoder_embedding(enc_input)
        encoder_result = self.encoder(enc_embed)
        dec_embed = self.decoder_embedding(dec_input)
        decoder_result = self.decoder(dec_embed, encoder_result)
        output = self.output_layer(decoder_result)
        softmax = self.softmax(output)

        return softmax, decoder_result
