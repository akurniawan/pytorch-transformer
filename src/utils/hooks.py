import torch
import random

from pathlib import Path


def restore_checkpoint_hook(model, model_path, logger=print):
    def restore_checkpoint(engine):
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
