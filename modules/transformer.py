import torch.nn as nn

from modules.decoder import TransformerDecoder
from modules.embedding import TransformerEmbedding
from modules.encoder import TransformerEncoder


class Transformer(nn.Module):
    def __init__(self,
                 max_length,
                 enc_vocab,
                 dec_vocab,
                 enc_emb_size,
                 dec_emb_size,
                 enc_units,
                 dec_units,
                 dropout_rate=0.1):
        super(Transformer, self).__init__()
        enc_vocab_size = len(enc_vocab.itos)
        dec_vocab_size = len(dec_vocab.itos)

        self.encoder_embedding = nn.Sequential(
            TransformerEmbedding(
                vocab_size=enc_vocab_size,
                padding_idx=enc_vocab.stoi["<pad>"],
                max_length=max_length,
                embedding_size=enc_emb_size), nn.Dropout(p=dropout_rate))
        self.decoder_embedding = nn.Sequential(
            TransformerEmbedding(
                vocab_size=dec_vocab_size,
                padding_idx=enc_vocab.stoi["<pad>"],
                max_length=max_length,
                embedding_size=dec_emb_size), nn.Dropout(p=dropout_rate))

        self.encoder = nn.Sequential(
            TransformerEncoder(enc_emb_size, enc_units),
            nn.Dropout(p=dropout_rate))
        self.decoder = TransformerDecoder(dec_emb_size, enc_emb_size,
                                          dec_units)
        self.decoder_drop = nn.Dropout(p=dropout_rate)

        self.output_layer = nn.Linear(
            in_features=enc_units[-1], out_features=dec_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, enc_input, dec_input):
        enc_embed = self.encoder_embedding(enc_input)
        encoder_result = self.encoder(enc_embed)

        dec_embed = self.decoder_embedding(dec_input)
        decoder_result = self.decoder(dec_embed, encoder_result)
        decoder_result = self.decoder_drop(decoder_result)

        output = self.output_layer(
            decoder_result.view(-1, decoder_result.size(-1)))
        softmax = self.softmax(output)
        softmax = softmax.view(
            decoder_result.size(0), decoder_result.size(1), -1)

        return softmax, output
