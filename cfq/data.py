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
    def __init__(self, data, indptr):
        self.data = data
        self.indptr = indptr

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self.data[self.indptr[key] : self.indptr[key + 1]]
        elif type(key) is slice:
            start, stop, stride = key.indices(len(self.indptr))
            assert stride == 1
            return self.data[self.indptr[start] : self.indptr[stop]]
        else:
            raise RuntimeError()

    def __len__(self):
        return len(self.indptr) - 1


class CFQDataset(Dataset):
    def __init__(self, idx, dat, tok_vocab, tag_vocab, typ_vocab):
        super().__init__()
        self.tok_vocab, self.tag_vocab, self.typ_vocab = tok_vocab, tag_vocab, typ_vocab

        self.idx = idx
        get = lambda k: dat[k] if type(k) is str else k
        rag = lambda x, y: RaggedArray(get(x), np.cumsum(np.hstack([[0], get(y)])))
        self.dat = dict(dat.items())
        self.dat["seq"] = rag("seq", "n_grp")
        self.dat["mem"] = rag("mem", "n_mem")
        self.dat["src"], self.dat["dst"], self.dat["typ"] = rag("src", "n_rel"), rag("dst", "n_rel"), rag("typ", "n_rel")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, key):
        idx = self.idx[key]
        dat = {k: v[idx] for k, v in self.dat.items()}
        dat["index"] = idx

        n = self.dat["n"][idx]
        n_ = torch.arange(n)
        u, v = torch.meshgrid(n_, n_)
        dat["u"], dat["v"] = u.flatten(), v.flatten()

        return dat


class CollateFunction:
    def __init__(self, tok_vocab, tag_vocab, typ_vocab):
        self.tok_vocab = tok_vocab
        self.tag_vocab = tag_vocab
        self.typ_vocab = typ_vocab
        _, self.tok2idx = self.tok_vocab
        _, self.tag2idx = self.tag_vocab
        _, self.typ2idx = self.typ_vocab

    def collate_fn(self, ds):
        bat = {}

        bat["index"] = torch.tensor([d["index"] for d in ds])
        seq = bat["seq"] = pack_sequence([torch.from_numpy(d["seq"]) for d in ds], enforce_sorted=False)
        bat["mask"], _ = pad_packed_sequence(utils.pack_as(torch.ones_like(seq.data), seq))

        hstack = lambda k: torch.from_numpy(np.hstack([d[k] for d in ds]))

        bat["mem"] = hstack("mem")

        n = hstack("n")
        m = n * n
        m_ = m.sum()
        typ_ = torch.zeros(m_, m_, len(self.typ2idx))
        n_rel, src, dst, typ = hstack("n_rel"), hstack("src"), hstack("dst"), hstack("typ")
        disp = torch.cat([torch.zeros(1).type_as(m), n[:-1]]).cumsum(0)
        disp_ = disp.repeat_interleave(n_rel)
        typ_[src + disp_, dst + disp_, typ] = 1

        disp_ = disp.repeat_interleave(m)
        bat["src"], bat["dst"] = hstack("u") + disp_, hstack("v") + disp_
        bat["idx"] = torch.arange(len(ds)).repeat_interleave(m)

        return bat


class CFQDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, tok_vocab, tag_vocab, typ_vocab):
        super().__init__()
        self.batch_size = batch_size
        self.tok_vocab = tok_vocab
        self.tag_vocab = tag_vocab
        self.typ_vocab = typ_vocab
        collate_fn = CollateFunction(tok_vocab, tag_vocab, typ_vocab)
        self.data_kwargs = {"num_workers": FLAGS.num_workers, "collate_fn": collate_fn.collate_fn, "pin_memory": True}

    def setup(self, stage=None):
        logger.info(f"Initializing dataset at stage {stage}")
        data = np.load(FLAGS.cfq_data_path)
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
