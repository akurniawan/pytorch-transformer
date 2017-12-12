import torch

from pathlib import Path


def save_checkpoint_hook(model, model_path, logger=print):
    def save_checkpoint(trainer, should_save):
        if should_save(trainer):
            try:
                logger("Saving model...")
                torch.save(model.state_dict(), model_path)
                logger("Finish saving model!")
            except Exception as e:
                logger("Something wrong while saving the model: %s" % str(e))

    return save_checkpoint


def restore_checkpoint_hook(model, model_path, logger=print):
    def restore_checkpoint(trainer):
        try:
            model_file = Path(model_path)
            if model_file.exists():
                logger("Start restore model...")
                model.load_state_dict(torch.load(model_path))
                logger("Finish restore model!")
            else:
                logger("Model not found, skip restoring model")
        except Exception as e:
            logger("Something wrong while restoring the model: %s" % str(e))

    return restore_checkpoint


def print_current_prediction_hook(vocab, logger=print):
    def print_current_condition(trainer, should_print):
        if should_print(trainer):
            logger("Current state of the model")
            logger("=" * 100)
            print(type(vocab.itos))
            last_hist = trainer.training_history[-1][0]
            for hist in last_hist:
                _, idx = hist.max(1)
                print(idx.data.numpy())
            logger("=" * 100)

    return print_current_condition
