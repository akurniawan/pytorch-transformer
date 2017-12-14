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
            result_str = ""
            result_str += "Current state of the model\n"
            result_str += ("=" * 100) + "\n"
            pred_last_hist = trainer.training_history[-1][0]
            trg_last_hist = trainer.training_history[-1][-1]
            batch_size = pred_last_hist.size(0)
            for this_idx, (pred, trg) in enumerate(
                    zip(pred_last_hist, trg_last_hist)):
                _, idx = pred.max(1)
                preds = []
                trgs = []
                for pred_idx, trg_idx in zip(idx.numpy(),
                                             trg.numpy()):
                    preds.append(vocab.itos[pred_idx])
                    trgs.append(vocab.itos[trg_idx])
                result_str += (" ".join(preds)) + "\n"
                result_str += (" ".join(trgs)) + "\n"
                if this_idx < batch_size - 1:
                    result_str += "\n"
            result_str += ("=" * 100)
            logger(result_str)

    return print_current_condition
