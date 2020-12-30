from itertools import *

from absl import flags
import numpy as np
from loguru import logger
from pathlib import Path
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import utils

FLAGS = flags.FLAGS


class RaggedArray:
    def __init__(self, data, indptr, seperate=False):
        self.data = data
        self.indptr = indptr
        self.seperate = seperate

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self.data[self.indptr[key] : self.indptr[key + 1]]
        elif type(key) is slice:
            start, stop, stride = key.indices(len(self.indptr))
            assert stride == 1
            if self.seperate:
                indptr = self.indptr[start : stop + 1]
                return [self.data[start : stop] for start, stop in zip(indptr[:-1], indptr[1:])]
            else:
                return self.data[self.indptr[start] : self.indptr[stop]]
        else:
            raise RuntimeError()

    def __len__(self):
        return len(self.indptr) - 1


'''
class CFQDataset(Dataset):
    def __init__(self, idx, dat, tok_vocab, tag_vocab, typ_vocab):
        super().__init__()
        self.idx = idx
        self.dat = {"n" : dat["n"]}
        self.dat["seq"] = RaggedArray(dat["seq"], np.cumsum(np.hstack([[0], dat["len"]])))

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, key):
        idx = self.idx[key]
        dat = {"idx" : idx}
        dat.update({k: v[idx] for k, v in self.dat.items()})

        return dat


class CollateFunction:
    def collate_fn(self, dats):
        bat = {"idx" : torch.tensor([dat["idx"] for dat in dats])}
        bat["n"] = torch.tensor([dat["n"] for dat in dats])
        bat["seq"] = pack_sequence([torch.from_numpy(dat["seq"]) for dat in dats], enforce_sorted=False)

        return bat
'''


class CFQDataset(Dataset):
    def __init__(self, idx, dat, tok_vocab, tag_vocab, typ_vocab):
        super().__init__()
        self.idx = idx
        self.dat = {}
        indptr = lambda len: np.cumsum(np.hstack([[0], len]))
        self.dat["seq_tag"] = RaggedArray(dat["seq_tag"], indptr(dat["len_tag"]))
        self.dat["seq_noun"] = RaggedArray(RaggedArray(dat["seq_noun"], indptr(dat["len_noun"]), seperate=True), indptr(dat["len_np"]))
        self.dat["len_np"] = dat["len_np"]
        self.dat["pos_np"] = RaggedArray(dat["pos_np"], indptr(dat["len_np"]))
        self.dat["n_var"] = dat["n_var"]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, key):
        idx = self.idx[key]
        dat = {"idx" : idx}
        dat.update({k: v[idx] for k, v in self.dat.items()})

        return dat


class CollateFunction:
    def collate_fn(self, dats):
        collate = lambda key: torch.tensor([dat[key] for dat in dats])
        pack = lambda key: pack_sequence([torch.from_numpy(dat[key]) for dat in dats], enforce_sorted=False)
        bat = {"idx" : collate('idx'), "n_var" : collate("n_var"), "seq_tag" : pack("seq_tag")}
        bat["seq_noun"] = pack_sequence(list(chain(*((torch.from_numpy(seq) for seq in dat["seq_noun"]) for dat in dats))), enforce_sorted=False)
        len_np = collate("len_np")
        bat["seq_np"] = pack_sequence(torch.arange(len_np.sum()).split(len_np.tolist()), enforce_sorted=False)
        assert all(len(dat["pos_np"]) == dat["len_np"] for dat in dats)
        bat["idx_np"] = torch.arange(len(dats)).repeat_interleave(len_np)
        bat["pos_np"] = torch.cat([torch.tensor(dat["pos_np"]) for dat in dats])

        return bat


class CFQDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, tok_vocab, tag_vocab, typ_vocab):
        super().__init__()
        self.batch_size = batch_size
        self.tok_vocab, self.tag_vocab, self.typ_vocab = tok_vocab, tag_vocab, typ_vocab
        collate_fn = CollateFunction()
        self.data_kwargs = {"num_workers": FLAGS.num_workers, "collate_fn": collate_fn.collate_fn, "pin_memory": True}

    def setup(self, stage=None):
        logger.info(f"Initializing dataset at stage {stage}")
        data = np.load(FLAGS.cfq_data_path)
        print(np.unique(data["n_var"], return_counts=True))
        split_data_dir_path = Path(FLAGS.cfq_split_data_dir) / f"{FLAGS.cfq_split}.npz"
        logger.info(f"Loading data from {split_data_dir_path}")
        split_data = np.load(split_data_dir_path)
        self.train_dataset = CFQDataset(split_data["trainIdxs"], data, self.tok_vocab, self.tag_vocab, self.typ_vocab)
        self.dev_dataset = CFQDataset(split_data["devIdxs"], data, self.tok_vocab, self.tag_vocab, self.typ_vocab)
        self.test_dataset = CFQDataset(split_data["testIdxs"], data, self.tok_vocab, self.tag_vocab, self.typ_vocab)

    def train_step_count_per_epoch(self):
        return len(self.train_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, **self.data_kwargs)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, **self.data_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, **self.data_kwargs)
