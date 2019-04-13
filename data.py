import torch

from torchtext import data
from torchtext import datasets


def create_dataset(batch_size,
                   enc_max_vocab,
                   dec_max_vocab,
                   source_train_path=None,
                   target_train_path=None,
                   source_val_path=None,
                   target_val_path=None,
                   logger=print):

    if torch.cuda.is_available():
        device_data = "cuda"
    else:
        device_data = "cpu"

    source = data.Field(batch_first=True, lower=True, init_token="<bos>")
    target = data.Field(batch_first=True, lower=True, eos_token="<eos>")

    fields = (source, target)
    if source_train_path and target_train_path and \
        source_val_path and target_val_path:
        source_train_name, source_train_ext = source_train_path.split(".")
        target_train_name, target_train_ext = target_train_path.split(".")
        source_val_name, source_val_ext = source_val_path.split(".")
        target_val_name, target_val_ext = target_val_path.split(".")

        if source_train_name != target_train_name:
            raise ValueError("the name of source and target for training " +
                             "data must be the same")
        if source_val_name != target_val_name:
            raise ValueError("the name of source and target for validation " +
                             "data must be the same")
        if source_train_ext != source_val_ext:
            raise ValueError("extension of source training and validation " +
                             "data must be the same")
        if target_train_ext != target_val_ext:
            raise ValueError("extension of target training and validation " +
                             "data must be the same")
        source_train_ext = "." + source_train_ext
        target_train_ext = "." + target_train_ext
        source_val_ext = "." + source_val_ext
        target_val_ext = "." + target_val_ext
        exts = (source_train_ext, target_train_ext)
        train_path = source_train_name
        val_path = source_val_name

        train, val, _ = get_mt_datasets(exts, fields, train_path, val_path)
    else:
        logger("neither train_path or val_path were defined. "
               "using WMT14 dataset as a fallback")
        train, val, _ = get_wmt_dataset((".de", ".en"), fields)

    source.build_vocab(train.src, min_freq=1, max_size=enc_max_vocab)
    target.build_vocab(train.trg, min_freq=1, max_size=dec_max_vocab)

    train_iter, val_iter = data.BucketIterator.splits((train, val),
                                                      batch_size=batch_size,
                                                      repeat=False,
                                                      shuffle=True,
                                                      device=device_data)

    return train_iter, val_iter, source.vocab, target.vocab


def get_wmt_dataset(exts, fields):
    train, val, test = datasets.WMT14.splits(exts=exts, fields=fields)
    return train, val, test


def get_mt_datasets(exts, fields, train_path, val_path, test_path=""):
    train = datasets.TranslationDataset(
        path=train_path, exts=exts, fields=fields)
    val = datasets.TranslationDataset(path=val_path, exts=exts, fields=fields)
    return train, val, None
