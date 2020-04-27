import torch

from torchtext import data
from torchtext import datasets


def create_dataset(cfg, logger=print):

    if torch.cuda.is_available():
        device_data = "cuda"
    else:
        device_data = "cpu"

    source = data.Field(batch_first=False, lower=True, init_token="<bos>")
    target = data.Field(batch_first=False, lower=True, eos_token="<eos>")

    fields = (source, target)
    if cfg.train and cfg.eval and cfg.test:
        exts = ("." + cfg.source_ext, "." + cfg.target_ext)
        train_path = cfg.train
        val_path = cfg.eval
        test_path = cfg.test

        train, val, test = get_mt_datasets(exts, fields, train_path, val_path,
                                           test_path)
    else:
        logger("neither train_path or val_path were defined. "
               "using WMT14 dataset as a fallback")
        train, val, test = get_wmt_dataset((".de", ".en"), fields)

    source.build_vocab(train.src,
                       min_freq=cfg.source_min_freq,
                       max_size=cfg.source_max_freq)
    target.build_vocab(train.trg,
                       min_freq=cfg.target_min_freq,
                       max_size=cfg.target_max_freq)

    train_iter, val_iter, test_iter = data.BucketIterator.splits(
        (train, val, test),
        batch_size=cfg.batch_size,
        repeat=False,
        shuffle=True,
        device=device_data)

    return train_iter, val_iter, test_iter, source.vocab, target.vocab


def get_wmt_dataset(exts, fields):
    train, val, test = datasets.WMT14.splits(exts=exts, fields=fields)
    return train, val, test


def get_mt_datasets(exts, fields, train_path, val_path, test_path):
    train = datasets.TranslationDataset(path=train_path,
                                        exts=exts,
                                        fields=fields)
    val = datasets.TranslationDataset(path=val_path, exts=exts, fields=fields)
    test = datasets.TranslationDataset(path=test_path,
                                       exts=exts,
                                       fields=fields)
    return train, val, test
