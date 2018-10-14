import torch

from pathlib import Path


def print_logs_hook(print_freq, max_epochs, total_data):
    def print_logs(engine):
        if (engine.state.iteration - 1) % print_freq == 0:
            columns = engine.state.metrics.keys()
            values = [
                str(round(float(value), 5))
                for value in engine.state.metrics.values()
            ]
            message = '[{epoch}/{max_epoch}][{i}/{max_i}]'.format(
                epoch=engine.state.epoch,
                max_epoch=max_epochs,
                i=(engine.state.iteration % total_data),
                max_i=total_data)
            for name, value in zip(columns, values):
                message += ' | {name}: {value}'.format(name=name, value=value)

            print(message)

    return print_logs


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


def print_current_prediction_hook(vocab, interval, logger=print):
    def print_current_condition(engine):
        if engine.state.iteration % interval == 0:
            result_str = ""
            result_str += "Current state of the model\n"
            result_str += ("=" * 100) + "\n"
            pred_last_hist = engine.state.output[0]
            trg_last_hist = engine.state.output[-1]
            batch_size = pred_last_hist.size(0)
            for this_idx, (pred, trg) in enumerate(
                    zip(pred_last_hist, trg_last_hist)):
                _, idx = pred.max(1)
                preds = []
                trgs = []
                for pred_idx, trg_idx in zip(idx.numpy(), trg.numpy()):
                    preds.append(vocab.itos[pred_idx])
                    trgs.append(vocab.itos[trg_idx])
                result_str += (" ".join(preds)) + "\n"
                result_str += (" ".join(trgs)) + "\n"
                if this_idx < batch_size - 1:
                    result_str += "\n"
            result_str += ("=" * 100)
            logger(result_str)

    return print_current_condition
