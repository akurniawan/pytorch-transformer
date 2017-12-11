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
