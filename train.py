import argparse
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from hooks import *
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torchtext import data
from torchtext import datasets
from ignite.trainer import Trainer, TrainingEvents
from ignite.handlers.logging import log_training_simple_moving_average
from ignite.handlers.logging import log_validation_simple_moving_average

from modules.transformer import Transformer


def run(model_path,
        encoder_emb_size,
        decoder_emb_size,
        encoder_units,
        decoder_units,
        batch_size,
        epochs,
        decay_step,
        decay_percent,
        log_interval,
        save_interval,
        compare_interval=2):
    de = data.Field(batch_first=True)
    en = data.Field(batch_first=True)

    train, val, _ = datasets.WMT14.splits(
        exts=(".de", ".en"),
        root="./",
        fields=(de, en),
        train="train",
        validation="eval",
        test="test")

    de.build_vocab(train.trg, min_freq=3, max_size=80000)
    en.build_vocab(train.src, min_freq=3, max_size=80000)

    transformer = Transformer(
        max_length=100,
        enc_vocab_size=len(en.vocab.freqs),
        dec_vocab_size=len(de.vocab.freqs),
        enc_emb_size=encoder_emb_size,
        dec_emb_size=decoder_emb_size,
        enc_units=encoder_units,
        dec_units=decoder_units)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(transformer.parameters())
    lr_decay = StepLR(opt, step_size=decay_step, gamma=decay_percent)

    if torch.cuda.is_available():
        device_data = 0
        transformer.cuda()
        loss_fn.cuda()
    else:
        device_data = -1

    train_iter, val_iter = data.BucketIterator.splits(
        (train, val),
        batch_size=batch_size,
        repeat=False,
        shuffle=True,
        device=device_data)

    def training_update_function(batch):
        transformer.train()
        lr_decay.step()
        opt.zero_grad()

        predictions = transformer(batch.src, batch.trg)

        flattened_predictions = predictions.view(-1, decoder_units[-1])
        flattened_target = batch.trg.view(-1)

        loss = loss_fn(flattened_predictions, flattened_target)

        loss.backward()
        opt.step()

        return predictions, loss.data[0]

    def validation_inference_function(batch):
        transformer.eval()
        predictions = transformer(batch.src, batch.trg)

        flattened_predictions = predictions.view(-1, decoder_units[-1])
        flattened_target = batch.trg.view(-1)

        loss = loss_fn(flattened_predictions, flattened_target)

        return loss.data[0]

    trainer = Trainer(train_iter, training_update_function, val_iter,
                      validation_inference_function)
    trainer.add_event_handler(TrainingEvents.TRAINING_STARTED,
                              restore_checkpoint_hook(transformer, model_path))
    trainer.add_event_handler(
        TrainingEvents.TRAINING_ITERATION_COMPLETED,
        log_training_simple_moving_average,
        window_size=10,
        metric_name="CrossEntropy",
        should_log=
        lambda trainer: trainer.current_iteration % log_interval == 0,
        history_transform=lambda history: history[-1])
    trainer.add_event_handler(
        TrainingEvents.TRAINING_ITERATION_COMPLETED,
        save_checkpoint_hook(transformer, model_path),
        should_save=
        lambda trainer: trainer.current_iteration % save_interval == 0)
    trainer.add_event_handler(
        TrainingEvents.TRAINING_ITERATION_COMPLETED,
        print_current_prediction_hook(en.vocab),
        should_print=
        lambda trainer: trainer.current_iteration % compare_interval == 0)
    trainer.add_event_handler(
        TrainingEvents.VALIDATION_COMPLETED,
        log_validation_simple_moving_average,
        window_size=10,
        metric_name="CrossEntropy")
    trainer.add_event_handler(
        TrainingEvents.TRAINING_COMPLETED,
        save_checkpoint_hook(transformer, model_path),
        should_save=lambda trainer: True)
    trainer.run(max_epochs=epochs, validate_every_epoch=True)


if __name__ == '__main__':
    run(model_path="./transformer-cp.pt",
        encoder_emb_size=512,
        decoder_emb_size=512,
        encoder_units=[512] * 6,
        decoder_units=[512] * 6,
        batch_size=2,
        epochs=20,
        decay_step=100,
        decay_percent=0.1,
        log_interval=2,
        save_interval=5)
