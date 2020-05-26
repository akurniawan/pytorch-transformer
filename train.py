import logging
import hydra
import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl

from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer, seed_everything
# from src.utils.hooks import (validation_result_hook, restore_checkpoint_hook)
from src.utils import print_current_prediction
from torch.optim.lr_scheduler import StepLR

from src.models.transformer import Transformer
from data import create_dataset


class MachineTranslationModel(pl.LightningModule):
    def __init__(self, hparams, source_vocab, target_vocab):
        super().__init__()

        self.hparams = hparams
        self.model = Transformer(
            max_length=hparams["training"]["max_len"],
            enc_vocab=source_vocab,
            dec_vocab=target_vocab,
            enc_emb_size=hparams["model"]["encoder_emb_size"],
            dec_emb_size=hparams["model"]["decoder_emb_size"],
            enc_dim=hparams["model"]["enc_dim"],
            enc_num_head=hparams["model"]["enc_num_head"],
            enc_num_layer=hparams["model"]["enc_num_layer"],
            dec_dim=hparams["model"]["dec_dim"],
            dec_num_head=hparams["model"]["dec_num_head"],
            dec_num_layer=hparams["model"]["dec_num_layer"])
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        return self.model(batch.src, batch.trg)

    def training_step(self, batch, batch_idx):
        _, logits = self(batch)

        flattened_target = batch.trg.view(-1)
        loss = self.criterion(logits, flattened_target)

        tensorboard_logs = {'train_loss': loss.item()}

        return {'loss': loss, 'log': tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            probs, logits = self(batch)

            flattened_target = batch.trg.view(-1)
            loss = self.criterion(logits, flattened_target)

            # preds = probs.transpose(0, 1).argmax(-1).tolist()
            # targets = batch.trg.t().tolist()

            return {
                "loss": loss,
                # "predictions": preds,
                # "targets": targets
            }

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            probs, logits = self.model(batch.src, batch.trg)

            flattened_target = batch.trg.view(-1)
            loss = self.criterion(logits, flattened_target)

            # preds = probs.transpose(0, 1).argmax(-1).tolist()
            # targets = batch.trg.t().tolist()

            return {
                "loss": loss.item(),
                # "predictions": preds,
                # "targets": targets
            }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        # avg_loss = avg_loss.item()

        tensorboard_logs = {"avg_val_loss": avg_loss}
        return {"val_loss": avg_loss, "log": tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams["training"]["learning_rate"])
        scheduler = StepLR(optimizer,
                           step_size=self.hparams["training"]["decay_step"],
                           gamma=self.hparams["training"]["decay_percent"])
        return [optimizer], [scheduler]

    # def train_dataloader(self):
    #     return self.train_iter

    # def val_dataloader(self):
    #     return self.val_iter

    # def test_dataloader(self):
    #     return self.test_iter


@hydra.main(config_path="hyperparams/config.yaml")
def run(cfg: DictConfig):
    # logging.basicConfig(filename="validation.log",
    #                     filemode="w",
    #                     level=logging.INFO)

    train_iter, val_iter, test_iter, source_vocab, target_vocab = create_dataset(
        cfg.dataset)
    model = MachineTranslationModel(OmegaConf.to_container(cfg, resolve=True),
                                    source_vocab, target_vocab)
    trainer = Trainer()
    trainer.fit(model, train_iter, val_iter)
    trainer.test(test_iter)


if __name__ == '__main__':
    run()
