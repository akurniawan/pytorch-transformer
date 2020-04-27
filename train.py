import logging
import hydra
import torch
import torch.nn as nn
import torch.optim as optim

from omegaconf import DictConfig
# from src.utils.hooks import (validation_result_hook, restore_checkpoint_hook)
from src.utils import print_current_prediction
from torch.optim.lr_scheduler import StepLR
from ignite.engine import Events, Engine
from ignite.metrics import RunningAverage
from ignite.handlers import Timer
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import ProgressBar

from src.models.transformer import Transformer
from data import create_dataset


@hydra.main(config_path="hyperparams/config.yaml")
def run(cfg: DictConfig):
    logging.basicConfig(filename="validation.log",
                        filemode="w",
                        level=logging.INFO)

    train_iter, val_iter, test_iter, source_vocab, target_vocab = create_dataset(
        cfg.dataset)
    transformer = Transformer(max_length=cfg.training.max_len,
                              enc_vocab=source_vocab,
                              dec_vocab=target_vocab,
                              enc_emb_size=cfg.model.encoder_emb_size,
                              dec_emb_size=cfg.model.decoder_emb_size,
                              enc_dim=cfg.model.enc_dim,
                              enc_num_head=cfg.model.enc_num_head,
                              enc_num_layer=cfg.model.enc_num_layer,
                              dec_dim=cfg.model.dec_dim,
                              dec_num_head=cfg.model.dec_num_head,
                              dec_num_layer=cfg.model.dec_num_layer)
    loss_fn = nn.CrossEntropyLoss()
    opt = optim.Adam(transformer.parameters(), lr=cfg.training.learning_rate)
    lr_decay = StepLR(opt,
                      step_size=cfg.training.decay_step,
                      gamma=cfg.training.decay_percent)

    if torch.cuda.is_available():
        transformer.cuda()
        loss_fn.cuda()

    def training_step(engine, batch):
        transformer.train()
        opt.zero_grad()

        _, logits = transformer(batch.src, batch.trg)

        flattened_target = batch.trg.view(-1)
        loss = loss_fn(logits, flattened_target)

        loss.backward()
        opt.step()
        lr_decay.step()

        return loss.cpu().item()

    def validation_step(engine, batch):
        transformer.eval()
        with torch.no_grad():
            probs, logits = transformer(batch.src, batch.trg)

            flattened_target = batch.trg.view(-1)
            loss = loss_fn(logits, flattened_target)

            preds = probs.argmax(-1).tolist()
            targets = batch.trg.t().tolist()

            if engine.state.output:
                preds = engine.state.output["predictions"] + preds
                targets = engine.state.output["targets"] + targets

            return {
                "loss": loss.item(),
                "predictions": preds,
                "targets": targets
            }

    trainer = Engine(training_step)
    evaluator = Engine(validation_step)
    checkpoint_handler = ModelCheckpoint(cfg.training.checkpoint,
                                         "Transformer",
                                         n_saved=10,
                                         require_empty=False)

    timer = Timer(average=True)
    timer.attach(trainer,
                 start=Events.EPOCH_STARTED,
                 resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED,
                 step=Events.ITERATION_COMPLETED)

    # Attach training metrics
    RunningAverage(output_transform=lambda x: x).attach(trainer, "train_loss")
    # Attach validation metrics
    RunningAverage(output_transform=lambda x: x["loss"]).attach(
        evaluator, "loss")

    pbar = ProgressBar()
    pbar.attach(trainer, ["train_loss"])

    # trainer.add_event_handler(
    #     Events.TRAINING_STARTED,
    #     restore_checkpoint_hook(transformer, cfg.training.checkpoint))

    @trainer.on(Events.ITERATION_COMPLETED(every=cfg.training.val_log))
    def log_validation_result(engine):
        evaluator.run(val_iter)
        metrics = evaluator.state.metrics
        avg_loss = metrics["loss"]
        logging.info("Validation loss: %d" % avg_loss)
        print_current_prediction(evaluator, target_vocab)

    @trainer.on(Events.COMPLETED)
    def log_test_result(engine):
        evaluator.run(test_iter)
        metrics = evaluator.state.metrics
        avg_loss = metrics["loss"]
        logging.info("Test loss: %d" % avg_loss)
        print_current_prediction(evaluator, target_vocab)

    trainer.add_event_handler(event_name=Events.ITERATION_COMPLETED,
                              handler=checkpoint_handler,
                              to_save={
                                  "transformer": transformer,
                                  "opt": opt,
                                  "lr_decay": lr_decay
                              })

    # Run the prediction
    trainer.run(train_iter, max_epochs=cfg.training.epochs)


if __name__ == '__main__':
    run()
