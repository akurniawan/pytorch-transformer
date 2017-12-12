# pytorch-transformer
Implementation of "Attention is All You Need" paper with PyTorch

## Requirements
* [pytorch](http://pytorch.org/)
* [torchtext](https://github.com/pytorch/text/)
* [ignite](https://github.com/pytorch/ignite/)

## How to use?
```
usage: train.py [-h] [--batch_size BATCH_SIZE] [--epochs EPOCHS]
                [--enc_max_vocab ENC_MAX_VOCAB]
                [--dec_max_vocab DEC_MAX_VOCAB]
                [--encoder_units ENCODER_UNITS]
                [--decoder_units DECODER_UNITS]
                [--encoder_emb_size ENCODER_EMB_SIZE]
                [--decoder_emb_size DECODER_EMB_SIZE]
                [--log_interval LOG_INTERVAL] [--save_interval SAVE_INTERVAL]
                [--compare_interval COMPARE_INTERVAL]
                [--decay_step DECAY_STEP] [--decay_percent DECAY_PERCENT]
                [--model_dir MODEL_DIR]

Google's transformer implementation in PyTorch

optional arguments:
  -h, --help            show this help message and exit
  --batch_size BATCH_SIZE
                        Number of batch in single iteration
  --epochs EPOCHS       Number of epochs
  --enc_max_vocab ENC_MAX_VOCAB
                        Maximum vocabs for encoder
  --dec_max_vocab DEC_MAX_VOCAB
                        Maximum vocabs for decoder
  --encoder_units ENCODER_UNITS
                        Number of encoder units for every layers. Separable by
                        commas
  --decoder_units DECODER_UNITS
                        Number of decoder units for every layers. Separable by
                        commas
  --encoder_emb_size ENCODER_EMB_SIZE
                        Size of encoder's embedding
  --decoder_emb_size DECODER_EMB_SIZE
                        Size of decoder's embedding
  --log_interval LOG_INTERVAL
                        Print loss for every N steps
  --save_interval SAVE_INTERVAL
                        Save model for every N steps
  --compare_interval COMPARE_INTERVAL
                        Compare current prediction with its true label for
                        every N steps
  --decay_step DECAY_STEP
                        Learning rate will decay after N step
  --decay_percent DECAY_PERCENT
                        Percent of decreased in learning rate decay
  --model_dir MODEL_DIR
                        Location to save the model
```

Still Work in Progress