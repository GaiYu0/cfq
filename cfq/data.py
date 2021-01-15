from absl import flags
import numpy as np
from loguru import logger
import os
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
        idx2typ, _ = typ_vocab
        self.n_typ = len(idx2typ)

        self.idx = idx
        get = lambda k: dat[k] if type(k) is str else k
        rag = lambda x, y: RaggedArray(get(x), np.cumsum(np.hstack([[0], get(y)])))
        self.dat = dict(dat.items())
        self.dat["seq"] = rag("seq", "n_grp")
        self.dat["n_mem"] = rag("n_mem", "n_grp")
        self.dat["mem"] = rag(rag("mem", "n_mem"), "n_grp")
        self.dat["src"], self.dat["dst"], self.dat["typ"] = rag("src", "n_rel"), rag("dst", "n_rel"), rag("typ", "n_rel")
        
        self.dat["filter"] = rag("filter", "n_filter")
        self.dat["idx2grp"] = rag("idx2grp", "n")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, key):
        idx = self.idx[key]
        for k, v in self.dat.items():
            try:
                v[idx]
            except:
                assert False, k
        dat = {k: v[idx] for k, v in self.dat.items()}

        dat["index"] = idx

        n = self.dat["n"][idx]
        typ = np.zeros([n, n, self.n_typ], dtype=bool)
        typ[dat["src"], dat["dst"], dat["typ"]] = True
        dat["typ"] = typ.reshape([-1, self.n_typ])

        src, dst = np.meshgrid(dat["idx2grp"], dat["idx2grp"])
        dat["src"], dat["dst"] = src.flatten(), dst.flatten()

        return dat


class CollateFunction:
    def collate_fn(self, ds):
        hstack = lambda k: torch.from_numpy(np.hstack([d[k] for d in ds]))
        vstack = lambda k: torch.from_numpy(np.vstack([d[k] for d in ds]))

        bat = {"index" : torch.tensor([d["index"] for d in ds])}

        seq = bat["seq"] = pack_sequence([torch.from_numpy(d["seq"]) for d in ds], enforce_sorted=False)
        msk, _ = bat["msk"], _ = pad_packed_sequence(utils.pack_as(torch.ones_like(seq.data), seq), batch_first=True)

        bat["mem"] = hstack("mem")
        n_mem = hstack("n_mem")
        bat["grp"] = torch.arange(len(n_mem)).repeat_interleave(n_mem)
        bat["pos2grp"] = msk.flatten().cumsum(0).sub(1).view(*msk.shape)

        n = hstack("n")
        m = n * n
        bat["src"], bat["dst"] = hstack("src"), hstack("dst")
        bat["typ"] = vstack("typ")
        bat["idx"] = torch.arange(len(m)).repeat_interleave(m)
        bat["m"] = m

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
        data = np.load(os.path.join(FLAGS.data_root_path, FLAGS.cfq_data_path))
        split_data_dir_path = os.path.join(FLAGS.data_root_path, FLAGS.cfq_split_data_dir, f"{FLAGS.cfq_split}.npz")
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
        val_loader = DataLoader(self.dev_dataset, batch_size=self.batch_size, **self.data_kwargs)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, **self.data_kwargs)
        return [val_loader, test_loader]

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, **self.data_kwargs)
