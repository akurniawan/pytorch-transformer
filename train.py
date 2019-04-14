import logging
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from hooks import *
from torch.optim.lr_scheduler import StepLR
from ignite.engine import Events, Engine
from ignite.metrics import RunningAverage
from ignite.handlers import Timer
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar

from modules.transformer import Transformer
from data import create_dataset


def run(model_dir, max_len, source_train_path, target_train_path,
        source_val_path, target_val_path, enc_max_vocab, dec_max_vocab,
        encoder_emb_size, decoder_emb_size, encoder_units, decoder_units,
        batch_size, epochs, learning_rate, decay_step, decay_percent,
        val_interval, save_interval, compare_interval):

    logging.basicConfig(
        filename="validation.log", filemode="w", level=logging.INFO)

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

    def training_step(engine, batch):
        transformer.train()
        lr_decay.step()
        opt.zero_grad()

        _, predictions = transformer(batch.src, batch.trg)

        flattened_predictions = predictions.view(-1, len(target_vocab.itos))
        flattened_target = batch.trg.view(-1)

        loss = loss_fn(flattened_predictions, flattened_target)

        loss.backward()
        opt.step()

        return loss.cpu().item()

    def validation_step(engine, batch):
        transformer.eval()
        with torch.no_grad():
            softmaxed_predictions, predictions = transformer(
                batch.src, batch.trg)

            flattened_predictions = predictions.view(-1,
                                                     len(target_vocab.itos))
            flattened_target = batch.trg.view(-1)

            loss = loss_fn(flattened_predictions, flattened_target)

            if not engine.state.output:
                predictions = softmaxed_predictions.argmax(
                    -1).cpu().numpy().tolist()
                targets = batch.trg.cpu().numpy().tolist()
            else:
                predictions = engine.state.output[
                    "predictions"] + softmaxed_predictions.argmax(
                        -1).cpu().numpy().tolist()
                targets = engine.state.output["targets"] + batch.trg.cpu(
                ).numpy().tolist()

            return {
                "loss": loss.cpu().item(),
                "predictions": predictions,
                "targets": targets
            }

    trainer = Engine(training_step)
    evaluator = Engine(validation_step)
    checkpoint_handler = ModelCheckpoint(
        model_dir,
        "Transformer",
        save_interval=save_interval,
        n_saved=10,
        require_empty=False)

    timer = Timer(average=True)
    timer.attach(
        trainer,
        start=Events.EPOCH_STARTED,
        resume=Events.ITERATION_STARTED,
        pause=Events.ITERATION_COMPLETED,
        step=Events.ITERATION_COMPLETED)

    # Attach training metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "train_loss")
    # Attach validation metrics
    RunningAverage(output_transform=lambda x: x["loss"]).attach(
        evaluator, "val_loss")

    pbar = ProgressBar()
    pbar.attach(trainer, ["train_loss"])

    # trainer.add_event_handler(Events.TRAINING_STARTED,
    #                           restore_checkpoint_hook(transformer, model_dir))
    trainer.add_event_handler(
        Events.ITERATION_COMPLETED,
        handler=validation_result_hook(
            evaluator,
            val_iter,
            target_vocab,
            val_interval,
            logger=logging.info))

    trainer.add_event_handler(
        event_name=Events.ITERATION_COMPLETED,
        handler=checkpoint_handler,
        to_save={
            "nmt": {
                "transformer": transformer,
                "opt": opt,
                "lr_decay": lr_decay
            }
        })

    # Run the prediction
    trainer.run(train_iter, max_epochs=epochs)


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
        "--val_interval",
        type=int,
        default=1000,
        help="""Run evaluation for every N steps""")
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
        default="./checkpoints",
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
        val_interval=ARGS.val_interval,
        save_interval=ARGS.save_interval,
        compare_interval=ARGS.compare_interval)
