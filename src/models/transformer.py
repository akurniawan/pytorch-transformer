import torch
import torch.nn as nn

from models.encoder import TransformerEncoder
from models.decoder import TransformerDecoder
from modules.embedding import PositionalEncoding, TransformerEmbedding


class Transformer(nn.Module):
    def __init__(self,
                 max_length,
                 enc_vocab,
                 dec_vocab,
                 enc_emb_size,
                 dec_emb_size,
                 enc_dim,
                 enc_num_head,
                 enc_num_layer,
                 dec_dim,
                 dec_num_head,
                 dec_num_layer,
                 dropout_rate=0.1):
        super(Transformer, self).__init__()
        enc_vocab_size = len(enc_vocab.itos)
        dec_vocab_size = len(dec_vocab.itos)

        word_enc_embedding = nn.Embedding(enc_vocab_size + 3,
                                          enc_emb_size,
                                          padding_idx=enc_vocab.stoi["<pad>"])
        pos_encoder = PositionalEncoding(enc_emb_size)
        word_dec_embedding = nn.Embedding(dec_vocab_size + 3,
                                          dec_emb_size,
                                          padding_idx=dec_vocab.stoi["<pad>"])
        pos_decoder = PositionalEncoding(dec_emb_size)
        self.encoder_embedding = TransformerEmbedding(word_enc_embedding,
                                                      pos_encoder)
        self.decoder_embedding = TransformerEmbedding(word_dec_embedding,
                                                      pos_decoder)

        self.encoder = TransformerEncoder(enc_dim, enc_num_head,
                                          enc_num_layer),
        self.decoder = TransformerDecoder(dec_dim, dec_num_head, dec_num_layer)

        self.output_layer = nn.Linear(in_features=dec_dim,
                                      out_features=dec_vocab_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, enc_input, dec_input):
        enc_embed = self.encoder_embedding(enc_input)
        encoder_result = self.encoder(enc_embed)

        dec_embed = self.decoder_embedding(dec_input)
        decoder_result = self.decoder(dec_embed, encoder_result)
        decoder_result = decoder_result.transpose(0, 1)

        output = self.output_layer(
            decoder_result.view(-1, decoder_result.size(-1)))
        softmax = self.softmax(output)
        softmax = softmax.view(decoder_result.size(0), decoder_result.size(1),
                               -1)

        return softmax, output

    def generate_square_subsequent_mask(self, sz):
        """Generate a square mask for the sequence.
        The masked positions are filled with float('-inf').
                Unmasked positions are filled with float(0.0).
            """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(
            mask == 1, float(0.0))
        return mask
