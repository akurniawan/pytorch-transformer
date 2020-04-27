import random


def print_current_prediction(engine, vocab):
    n_sample = 100
    result_str = ""
    bold_code = "\033[1m"
    end_bold_code = "\033[0m"
    result_str += "Current state of the model\n"
    result_str += ("=" * 100) + "\n"
    rand_ids = random.sample(range(len(engine.state.output["targets"])),
                             n_sample)
    pred_sample = [engine.state.output["predictions"][i] for i in rand_ids]
    trg_sample = [engine.state.output["targets"][i] for i in rand_ids]
    for this_idx, (pred, trg) in enumerate(zip(pred_sample, trg_sample)):
        vocab_mapper = lambda x: vocab.itos[x]
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
    print(result_str)