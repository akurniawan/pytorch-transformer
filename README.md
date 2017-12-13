# pytorch-transformer
Implementation of "Attention is All You Need" paper with PyTorch

## Requirements
* [pytorch](http://pytorch.org/)
* [torchtext](https://github.com/pytorch/text/)
* [ignite](https://github.com/pytorch/ignite/)

## How to use?
### Arguments
```
  --batch_size BATCH_SIZE
                        Number of batch in single iteration
  --source_train_path SOURCE_TRAIN_PATH
                        Path for source training data. Ex: data/train.en
  --target_train_path TARGET_TRAIN_PATH
                        Path for target training data. Ex: data/train.de
  --source_val_path SOURCE_VAL_PATH
                        Path for source validation data. Ex: data/val.en
  --target_val_path TARGET_VAL_PATH
                        Path for target validation data. Ex: data/val.de
  --epochs EPOCHS       Number of epochs
  --learning_rate LEARNING_RATE
                        Learning rate size
  --max_len MAX_LEN     Maximum allowed sentence length
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