import torch
import random

from pathlib import Path


def validation_result_hook(evaluator,
                           loader,
                           trg_vocab,
                           val_interval,
                           logger=print):
    def _print_current_prediction(engine):
        n_sample = 100
        result_str = ""
        bold_code = "\033[1m"
        end_bold_code = "\033[0m"
        result_str += "Current state of the model\n"
        result_str += ("=" * 100) + "\n"
        rand_ids = random.sample(
            range(len(engine.state.output["targets"])), n_sample)
        pred_sample = [engine.state.output["predictions"][i] for i in rand_ids]
        trg_sample = [engine.state.output["targets"][i] for i in rand_ids]
        for this_idx, (pred, trg) in enumerate(zip(pred_sample, trg_sample)):
            vocab_mapper = lambda x: trg_vocab.itos[x]
            preds = list(map(vocab_mapper, pred))
            trgs = list(map(vocab_mapper, trg))
            result_str += "{}Prediction{}: {}\n".format(
                bold_code, end_bold_code, " ".join(preds[:len(trgs)]))
            result_str += "{}Target{}: {}\n".format(bold_code, end_bold_code,
                                                    " ".join(trgs))
            result_str += "{}Difference of length{}: {}\n\n".format(
                bold_code, end_bold_code, abs(len(preds) - len(trgs)))
            if this_idx < len(pred_sample) - 1:
                result_str += "\n"
        result_str += ("=" * 100)
        logger(result_str)

    def validation_result(engine):
        if engine.state.iteration % val_interval == 0:
            evaluator.run(loader)
            metrics = evaluator.state.metrics
            avg_loss = metrics["val_loss"]
            logger("Validation loss: %d" % avg_loss)
            _print_current_prediction(evaluator)

    return validation_result


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
