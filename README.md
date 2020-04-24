# pytorch-transformer

**WARNING! Still in the middle of refactoring, please revert to bb88f0a for the latest working version of this repo**

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

```bash
python train.py --batch_size 3 --source_train_path wmt14/train.en --target_train_path wmt14/train.de --source_val_path wmt14/eval.en --target_val_path wmt14/eval.de
```

### Arguments

```bash
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

With the following command

```bash
python train.py --batch_size 16 --max_len 500 --decoder_units 512,512,512 --save_interval 500 --enc_max_vocab 37000 --dec_max_vocab 37000 --decay_step 4000 --epochs 5
```

I was able to get result as the following

```bash
Train loss: 2.17e-02
Validation loss: 0

Validation result
=================
Prediction: zwei anlagen so nah bei@@ einander : absicht oder schil@@ d@@ bürger@@ st@@ reich ? <eos>
Target: zwei anlagen so nah bei@@ einander : absicht oder schil@@ d@@ bürger@@ st@@ reich ? <eos>
Difference of length: 0


Prediction: al@@ fred ab@@ el , der das grundstück derzeit verwaltet , hatte die ver@@ schön@@ erungs@@ aktion mit seinem kollegen rein@@ hard dom@@ ke von der b@@ i ab@@ gesprochen . <eos>
Target: al@@ fred ab@@ el , der das grundstück derzeit verwaltet , hatte die ver@@ schön@@ erungs@@ aktion mit seinem kollegen rein@@ hard dom@@ ke von der b@@ i ab@@ gesprochen . <eos>
Difference of length: 0


Prediction: andere komit@@ e@@ em@@ it@@ gli@@ eder sagten , es gebe nur ver@@ einzel@@ te berichte von pi@@ loten , die für eine stör@@ ung von flugzeug@@ systemen durch die geräte sp@@ rä@@ chen , und die meisten davon seien sehr alt . <eos>
Target: andere komit@@ e@@ em@@ it@@ gli@@ eder sagten , es gebe nur ver@@ einzel@@ te berichte von pi@@ loten , die für eine stör@@ ung von flugzeug@@ systemen durch die geräte sp@@ rä@@ chen , und die meisten davon seien sehr alt . <eos>
Difference of length: 0


Prediction: der mann aus lamp@@ recht@@ sh@@ au@@ sen wollte an der außen@@ fass@@ ade eines gas@@ th@@ auses einen def@@ ekten heiz@@ ungs@@ füh@@ ler aus@@ wechseln . <eos> <pad> <pad>
Target: der mann aus lamp@@ recht@@ sh@@ au@@ sen wollte an der außen@@ fass@@ ade eines gas@@ th@@ auses einen def@@ ekten heiz@@ ungs@@ füh@@ ler aus@@ wechseln . <eos> <pad> <pad>
Difference of length: 0


Prediction: falls allerdings in den kommenden monaten keine neuen bestellungen bekannt gegeben werden , dann erwarten wir , dass der markt dem programm gegenüber skep@@ tischer wird . <eos>
Target: falls allerdings in den kommenden monaten keine neuen bestellungen bekannt gegeben werden , dann erwarten wir , dass der markt dem programm gegenüber skep@@ tischer wird . <eos>
Difference of length: 0


Prediction: es recht@@ fertigt nicht die ergebnisse eines berichts , in dem es heißt , die anstrengungen des wei@@ ßen hauses zur ru@@ hi@@ g@@ stellung der medien seien die „ aggres@@ si@@ v@@ sten ... seit der ni@@ x@@ on-@@ regierung “ . <eos>
Target: es recht@@ fertigt nicht die ergebnisse eines berichts , in dem es heißt , die anstrengungen des wei@@ ßen hauses zur ru@@ hi@@ g@@ stellung der medien seien die „ aggres@@ si@@ v@@ sten ... seit der ni@@ x@@ on-@@ regierung “ . <eos>
Difference of length: 0


Prediction: &quot; der schutz möglichst vieler be@@ bau@@ ter grund@@ stücke ist das ziel &quot; , so h@@ äuß@@ ler . <eos> <pad> <pad>
Target: &quot; der schutz möglichst vieler be@@ bau@@ ter grund@@ stücke ist das ziel &quot; , so h@@ äuß@@ ler . <eos> <pad> <pad>
Difference of length: 0


Prediction: das kon@@ sist@@ orium im nächsten jahr sei deshalb bedeut@@ sam , weil es das erste seit der wahl von fran@@ z@@ is@@ kus im märz diesen jahres sei , so val@@ ero . <eos> <pad>
Target: das kon@@ sist@@ orium im nächsten jahr sei deshalb bedeut@@ sam , weil es das erste seit der wahl von fran@@ z@@ is@@ kus im märz diesen jahres sei , so val@@ ero . <eos> <pad>
Difference of length: 0


Prediction: samsung , hu@@ a@@ wei und ht@@ c stellen hand@@ ys her , die mit goo@@ g@@ les betriebssystem andro@@ id arbeiten , das in schar@@ f@@ em wettbewerb zu den mobil@@ produkten von apple und microsoft steht . <eos>
Target: samsung , hu@@ a@@ wei und ht@@ c stellen hand@@ ys her , die mit goo@@ g@@ les betriebssystem andro@@ id arbeiten , das in schar@@ f@@ em wettbewerb zu den mobil@@ produkten von apple und microsoft steht . <eos>
Difference of length: 0


Prediction: etw@@ a gegen 14 : 15 uhr am mittwoch sah ein spaziergän@@ ger , der seinen hund aus@@ führte , die gest@@ rand@@ ete ru@@ by auf dem 15 meter hohen absatz im stein@@ bruch . <eos> <pad> <pad> <pad>
Target: etw@@ a gegen 14 : 15 uhr am mittwoch sah ein spaziergän@@ ger , der seinen hund aus@@ führte , die gest@@ rand@@ ete ru@@ by auf dem 15 meter hohen absatz im stein@@ bruch . <eos> <pad> <pad> <pad>
Difference of length: 0


Prediction: c@@ ook sagte : „ nach den erhöh@@ ungen bei der st@@ emp@@ el@@ steuer auf hoch@@ prei@@ si@@ ge wohnungen und der einführung der damit verbundenen gesetzgebung gegen ein um@@ gehen kann man schwer@@ lich behaupten , hochwertige immobilien seien zu niedrig best@@ euert , ungeachtet der auswirkungen des ver@@ alt@@ eten gemein@@ dest@@ euer@@ systems . “ <eos> <pad> <pad>
Target: c@@ ook sagte : „ nach den erhöh@@ ungen bei der st@@ emp@@ el@@ steuer auf hoch@@ prei@@ si@@ ge wohnungen und der einführung der damit verbundenen gesetzgebung gegen ein um@@ gehen kann man schwer@@ lich behaupten , hochwertige immobilien seien zu niedrig best@@ euert , ungeachtet der auswirkungen des ver@@ alt@@ eten gemein@@ dest@@ euer@@ systems . “ <eos> <pad> <pad>
Difference of length: 0


Prediction: ohne die unterstützung des einzigen anderen herstell@@ ers großer moderner j@@ ets sagen experten , dass der ruf nach einem neuen branchen@@ standard vermutlich ver@@ pu@@ ffen werde , aber von der welle von 7@@ 7@@ 7@@ x-@@ verk@@ äu@@ fen ab@@ lenken könnte . <eos> <pad> <pad>
Target: ohne die unterstützung des einzigen anderen herstell@@ ers großer moderner j@@ ets sagen experten , dass der ruf nach einem neuen branchen@@ standard vermutlich ver@@ pu@@ ffen werde , aber von der welle von 7@@ 7@@ 7@@ x-@@ verk@@ äu@@ fen ab@@ lenken könnte . <eos> <pad> <pad>
Difference of length: 0


Prediction: &quot; bildung ist ein wichtiger standor@@ t@@ faktor &quot; , unter@@ strich clau@@ dia st@@ eh@@ le , direkt@@ or@@ in der hans@@ -@@ th@@ om@@ a-@@ schule , die das vern@@ etz@@ te schul@@ projekt bildungs@@ zentrum hoch@@ schwar@@ zw@@ ald vor@@ stellte . <eos> <pad> <pad>
Target: &quot; bildung ist ein wichtiger standor@@ t@@ faktor &quot; , unter@@ strich clau@@ dia st@@ eh@@ le , direkt@@ or@@ in der hans@@ -@@ th@@ om@@ a-@@ schule , die das vern@@ etz@@ te schul@@ projekt bildungs@@ zentrum hoch@@ schwar@@ zw@@ ald vor@@ stellte . <eos> <pad> <pad>
Difference of length: 0


Prediction: es bestand die möglichkeit , dass sie sehr schwer verletzt war oder sch@@ lim@@ mer@@ es . <eos>
Target: es bestand die möglichkeit , dass sie sehr schwer verletzt war oder sch@@ lim@@ mer@@ es . <eos>
Difference of length: 0


Prediction: zu dem unfall war es nach angaben der polizei gekommen , als ein 26 jahre alter mann am donn@@ erst@@ ag@@ abend , gegen 22 uhr , mit einem dam@@ en@@ fahr@@ rad ordnungs@@ widri@@ g auf dem linken geh@@ weg vom bahn@@ hof@@ platz in richtung markt@@ stätte unterwegs war . <eos> <pad> <pad> <pad>
Target: zu dem unfall war es nach angaben der polizei gekommen , als ein 26 jahre alter mann am donn@@ erst@@ ag@@ abend , gegen 22 uhr , mit einem dam@@ en@@ fahr@@ rad ordnungs@@ widri@@ g auf dem linken geh@@ weg vom bahn@@ hof@@ platz in richtung markt@@ stätte unterwegs war . <eos> <pad> <pad> <pad>
Difference of length: 0


Prediction: &quot; dieser aufwand ist nun weg &quot; , freut sich ma@@ ier . <eos> <pad>
Target: &quot; dieser aufwand ist nun weg &quot; , freut sich ma@@ ier . <eos> <pad>
Difference of length: 0


Prediction: thomas op@@ per@@ mann , der abgeordnete , der den für den geheim@@ dienst zuständigen parlamentarischen ausschuss leitet , erklärte , man solle die gelegenheit ergreifen , snow@@ den als zeu@@ ge anzu@@ hören , wenn dies möglich sei , „ ohne ihn zu gefährden und die beziehungen zu den usa völlig zu ru@@ in@@ ieren “ . <eos> <pad> <pad> <pad>
Target: thomas op@@ per@@ mann , der abgeordnete , der den für den geheim@@ dienst zuständigen parlamentarischen ausschuss leitet , erklärte , man solle die gelegenheit ergreifen , snow@@ den als zeu@@ ge anzu@@ hören , wenn dies möglich sei , „ ohne ihn zu gefährden und die beziehungen zu den usa völlig zu ru@@ in@@ ieren “ . <eos> <pad> <pad> <pad>
Difference of length: 0


Prediction: r@@ eine pflanzen@@ mar@@ gar@@ ine sei eine gute alternative zu but@@ ter , jo@@ gh@@ urt lasse durch so@@ ja@@ jo@@ gh@@ urt ersetzen . <eos> <pad>
Target: r@@ eine pflanzen@@ mar@@ gar@@ ine sei eine gute alternative zu but@@ ter , jo@@ gh@@ urt lasse durch so@@ ja@@ jo@@ gh@@ urt ersetzen . <eos> <pad>
Difference of length: 0


Prediction: der handel am nas@@ da@@ q op@@ tions market wurde am frei@@ tag@@ nach@@ mittag deutscher zeit unterbrochen . <eos>
Target: der handel am nas@@ da@@ q op@@ tions market wurde am frei@@ tag@@ nach@@ mittag deutscher zeit unterbrochen . <eos>
Difference of length: 0


Prediction: ö@@ z@@ dem@@ ir will j@@ azz@@ ausbildung in stuttgart erhalten <eos>
Target: ö@@ z@@ dem@@ ir will j@@ azz@@ ausbildung in stuttgart erhalten <eos>
Difference of length: 0
```

I'm still not really sure why the validation can have such a good accuracy, will need to drill down and debug the model further.