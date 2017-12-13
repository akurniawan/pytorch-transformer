# pytorch-transformer
Implementation of "Attention is All You Need" paper with PyTorch.
Installed components are:
1. Multi-Head Attention
2. Positional Encoding with sinusodial
3. Position Wise FFN
4. Label Smoothing (unfortunately still can't use this because PyTorch has no support for loss calculation with logits yet :( )

## Requirements
* [pytorch](http://pytorch.org/)
* [torchtext](https://github.com/pytorch/text/)
* [ignite](https://github.com/pytorch/ignite/)

## How to use?
### Training
You can run the training with simply `python train.py`. If you want to add other arguments. Please see the example below
```
python train.py --batch_size 3 --source_train_path wmt14/train.en --target_train_path wmt14/train.de --source_val_path wmt14/eval.en --target_val_path wmt14/eval.de
```
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

## Results
I haven't had a chance to fully run this code over the whole WMT data, I only tested it using 100 training data. The result, however, is pretty exciting remembering we are not using neither CNN or RNN! Below I present the result after several epochs of training.

====================================================================================================
arbeitsp@@ lan <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
arbeitsp@@ lan <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>

zum mittwoch : <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
zum mittwoch : <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>

wiederaufnahme der sitzungsperiode <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
wiederaufnahme der sitzungsperiode <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>

das war der beschluß . <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
das war der beschluß . <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>

( beifall der pse-fraktion ) <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
( beifall der pse-fraktion ) <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>

warum finden keine brand@@ schutz@@ bel@@ ehr@@ ungen statt ? <eos> <pad> <pad> <pad> <pad> <pad> <pad>
warum finden keine brand@@ schutz@@ bel@@ ehr@@ ungen statt ? <eos> <pad> <pad> <pad> <pad> <pad> <pad>

wir haben dann abgestimmt . <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
wir haben dann abgestimmt . <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>

wir wissen nicht , was passiert . <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
wir wissen nicht , was passiert . <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>

frau präsidentin , zur geschäftsordnung . <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
frau präsidentin , zur geschäftsordnung . <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>

frau präsidentin , zur geschäftsordnung . <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
frau präsidentin , zur geschäftsordnung . <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>

alle anderen waren anderer meinung . <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
alle anderen waren anderer meinung . <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>

es gab eine abstimmung zu diesem punkt . <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
es gab eine abstimmung zu diesem punkt . <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad> <pad>

der kollege hän@@ sch hat sie dort vertreten . <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
der kollege hän@@ sch hat sie dort vertreten . <eos> <pad> <pad> <pad> <pad> <pad> <pad> <pad>
====================================================================================================
The result below are from training data and not validation data. The loss still embarassing and I will try to improve it little by little since I don't have GPU to train the model. Will keep you guys posted!