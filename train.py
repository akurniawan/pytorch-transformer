import argparse
import multiprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from hooks import *
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from ignite.trainer import Trainer, TrainingEvents
from ignite.handlers.logging import log_training_simple_moving_average
from ignite.handlers.logging import log_validation_simple_moving_average

from modules.transformer import Transformer
from data import create_dataset


def run(model_dir, max_len, source_train_path, target_train_path,
        source_val_path, target_val_path, enc_max_vocab, dec_max_vocab,
        encoder_emb_size, decoder_emb_size, encoder_units, decoder_units,
        batch_size, epochs, learning_rate, decay_step, decay_percent,
        log_interval, save_interval, compare_interval):

    train_iter, val_iter, source_vocab, target_vocab = create_dataset(
        batch_size, enc_max_vocab, dec_max_vocab, source_train_path,
        target_train_path, source_val_path, target_val_path)
    transformer = Transformer(
        max_length=max_len,
        enc_vocab=source_vocab,
        dec_vocab=target_vocab,
        enc_emb_size=encoder_emb_size,
        dec_emb_size=decoder_emb_size,
        enc_units=encoder_units,
        dec_units=decoder_units)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(transformer.parameters(), lr=learning_rate)
    lr_decay = StepLR(opt, step_size=decay_step, gamma=decay_percent)

    if torch.cuda.is_available():
        transformer.cuda()
        loss_fn.cuda()

    def training_update_function(batch):
        transformer.train()
        lr_decay.step()
        opt.zero_grad()

        softmaxed_predictions, predictions = transformer(batch.src, batch.trg)

        flattened_predictions = predictions.view(-1, len(target_vocab.itos))
        flattened_target = batch.trg.view(-1)

        loss = loss_fn(flattened_predictions, flattened_target)

        loss.backward()
        opt.step()

        return softmaxed_predictions, loss.data[0], batch.trg

    def validation_inference_function(batch):
        transformer.eval()
        softmaxed_predictions, predictions = transformer(batch.src, batch.trg)

        flattened_predictions = predictions.view(-1, len(target_vocab.itos))
        flattened_target = batch.trg.view(-1)

        loss = loss_fn(flattened_predictions, flattened_target)

        return loss.data[0]

    trainer = Trainer(train_iter, training_update_function, val_iter,
                      validation_inference_function)
    trainer.add_event_handler(TrainingEvents.TRAINING_STARTED,
                              restore_checkpoint_hook(transformer, model_dir))
    trainer.add_event_handler(
        TrainingEvents.TRAINING_ITERATION_COMPLETED,
        log_training_simple_moving_average,
        window_size=10,
        metric_name="CrossEntropy",
        should_log=
        lambda trainer: trainer.current_iteration % log_interval == 0,
        history_transform=lambda history: history[1])
    trainer.add_event_handler(
        TrainingEvents.TRAINING_ITERATION_COMPLETED,
        save_checkpoint_hook(transformer, model_dir),
        should_save=
        lambda trainer: trainer.current_iteration % save_interval == 0)
    trainer.add_event_handler(
        TrainingEvents.TRAINING_ITERATION_COMPLETED,
        print_current_prediction_hook(target_vocab),
        should_print=
        lambda trainer: trainer.current_iteration % compare_interval == 0)
    trainer.add_event_handler(
        TrainingEvents.VALIDATION_COMPLETED,
        log_validation_simple_moving_average,
        window_size=10,
        metric_name="CrossEntropy")
    trainer.add_event_handler(
        TrainingEvents.TRAINING_COMPLETED,
        save_checkpoint_hook(transformer, model_dir),
        should_save=lambda trainer: True)
    trainer.run(max_epochs=epochs, validate_every_epoch=True)


if __name__ == '__main__':
    PARSER = argparse.ArgumentParser(
        description="Google's transformer implementation in PyTorch")
    PARSER.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Number of batch in single iteration")
    PARSER.add_argument(
        "--source_train_path",
        default=None,
        help="Path for source training data. Ex: data/train.en")
    PARSER.add_argument(
        "--target_train_path",
        default=None,
        help="Path for target training data. Ex: data/train.de")
    PARSER.add_argument(
        "--source_val_path",
        default=None,
        help="Path for source validation data. Ex: data/val.en")
    PARSER.add_argument(
        "--target_val_path",
        default=None,
        help="Path for target validation data. Ex: data/val.de")
    PARSER.add_argument(
        "--epochs", type=int, default=10000, help="Number of epochs")
    PARSER.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate size")
    PARSER.add_argument(
        "--max_len",
        type=int,
        default=100,
        help="Maximum allowed sentence length")
    PARSER.add_argument(
        "--enc_max_vocab",
        type=int,
        default=80000,
        help="Maximum vocabs for encoder")
    PARSER.add_argument(
        "--dec_max_vocab",
        type=int,
        default=80000,
        help="Maximum vocabs for decoder")
    PARSER.add_argument(
        "--encoder_units",
        default="512,512,512,512,512,512",
        help="Number of encoder units for every layers. Separable by commas")
    PARSER.add_argument(
        "--decoder_units",
        default="512,512,512,512,512,512",
        help="Number of decoder units for every layers. Separable by commas")
    PARSER.add_argument(
        "--encoder_emb_size",
        type=int,
        default=512,
        help="Size of encoder's embedding")
    PARSER.add_argument(
        "--decoder_emb_size",
        type=int,
        default=512,
        help="Size of decoder's embedding")
    PARSER.add_argument(
        "--log_interval",
        type=int,
        default=2,
        help="""Print loss for every N steps""")
    PARSER.add_argument(
        "--save_interval",
        type=int,
        default=10,
        help="""Save model for every N steps""")
    PARSER.add_argument(
        "--compare_interval",
        type=int,
        default=10,
        help=
        """Compare current prediction with its true label for every N steps""")
    PARSER.add_argument(
        "--decay_step",
        type=int,
        default=500,
        help="Learning rate will decay after N step")
    PARSER.add_argument(
        "--decay_percent",
        type=float,
        default=0.1,
        help="Percent of decreased in learning rate decay")
    PARSER.add_argument(
        "--model_dir",
        type=str,
        default="./transformer-cp.pt",
        help="Location to save the model")
    ARGS = PARSER.parse_args()

    ENCODER_UNITS = [int(unit) for unit in ARGS.encoder_units.split(",")]
    DECODER_UNITS = [int(unit) for unit in ARGS.decoder_units.split(",")]

    run(model_dir=ARGS.model_dir,
        max_len=ARGS.max_len,
        source_train_path=ARGS.source_train_path,
        target_train_path=ARGS.target_train_path,
        source_val_path=ARGS.source_val_path,
        target_val_path=ARGS.target_val_path,
        enc_max_vocab=ARGS.enc_max_vocab,
        dec_max_vocab=ARGS.dec_max_vocab,
        encoder_emb_size=ARGS.encoder_emb_size,
        decoder_emb_size=ARGS.decoder_emb_size,
        encoder_units=ENCODER_UNITS,
        decoder_units=DECODER_UNITS,
        batch_size=ARGS.batch_size,
        epochs=ARGS.epochs,
        learning_rate=ARGS.learning_rate,
        decay_step=ARGS.decay_step,
        decay_percent=ARGS.decay_percent,
        log_interval=ARGS.log_interval,
        save_interval=ARGS.save_interval,
        compare_interval=ARGS.compare_interval)
