from absl import flags
import numpy as np
from loguru import logger
from pathlib import Path
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

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
    def __init__(self, idx, data, tok_vocab, rel_vocab):
        super().__init__()
        self.tok_vocab, self.rel_vocab = tok_vocab, rel_vocab
        _, tok2idx = self.tok_vocab
        ntok = len(tok2idx)

        self.idx = idx
        get = lambda k: data[k] if type(k) is str else k
        rag = lambda x, y: RaggedArray(get(x), np.cumsum(np.hstack([[0], get(y)])))
        self.data = dict(data.items())
        self.data["seq"] = rag("seq", "n_tok")
        self.data["pos2grp"] = rag("pos2grp", "n_tok")
        self.data["isconcept"], self.data["isvariable"] = rag("isconcept", "n_tok"), rag("isvariable", "n_tok")
        self.data["n_idx"], self.data["idx"] = rag("n_idx", "n"), rag(rag("idx", "n_idx"), "n")
        self.data["tok"] = rag("tok", "n")
        self.data["src"], self.data["dst"], self.data["rel"] = rag("src", "m"), rag("dst", "m"), rag("rel", "m")

        '''
        disp = np.hstack([np.zeros(1, dtype=int), data["n"]]).cumsum()
        disp_ = disp[:-1].repeat(data["m"])
        tok_src = data["tok"][data["src"] + disp_]
        tok_dst = data["tok"][data["dst"] + disp_]
        tok_rel = data["rel"] + ntok
        self.data["q"] = RaggedArray(np.vstack([tok_src, tok_rel, tok_dst]).T, disp)
        '''

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, key):
        i = self.idx[key]
        d = {k: v[i] for k, v in self.data.items()}

        d["ispad"] = len(d["seq"]) * [False]

        d["cfq_idx"] = i

        n = self.data["n"][i]
        n_ = np.arange(n)
        u, v = np.meshgrid(n_, n_)
        d["u"], d["v"] = u.flatten(), v.flatten()

        idx2rel, _ = self.rel_vocab
        nrel = len(idx2rel)
        mask = np.zeros([n, n, nrel], dtype=bool)
        mask[d["src"], d["dst"], d["rel"]] = True
        d["mask"] = mask.reshape([-1, nrel])

        return d


class CollateFunction:
    def __init__(self, tok_vocab, rel_vocab):
        self.tok_vocab = tok_vocab
        self.rel_vocab = rel_vocab
        _, self.tok2idx = self.tok_vocab
        _, self.rel2idx = self.rel_vocab
        self.ntok = len(self.tok2idx)
        self.nrel = len(self.rel2idx)

    def collate_fn(self, samples):
        b = {}

        max_len = max(len(s["seq"]) for s in samples)
        pad = lambda k, p: torch.from_numpy(np.vstack([np.hstack([s[k], np.full(max_len - len(s[k]), p)]) for s in samples]))
        if FLAGS.seq_model == "lstm":
            b["seq"] = pack_sequence([torch.from_numpy(s["seq"]) for s in samples], enforce_sorted=False)
            b["pos2grp"] = pack_sequence([torch.from_numpy(s["pos2grp"]) for s in samples], enforce_sorted=False)
        elif FLAGS.seq_model == "transformer":
            b["seq"] = {
                "tok": pad("seq", self.tok2idx["[PAD]"]).t(),
                "ispad": pad("ispad", True),
                "isconcept": pad("isconcept", False).t(),
                "isvariable": pad("isvariable", False).t(),
            }
        else:
            raise Exception()

        cat = lambda k: torch.from_numpy(np.hstack([s[k] for s in samples]))
        b["n"], b["n_idx"], b["idx"], b["n_grp"] = cat("n"), cat("n_idx"), cat("idx"), cat("n_grp")
        b["m"], src, dst, b["rel"] = cat("m"), cat("src"), cat("dst"), cat("rel")

        offset = torch.tensor([0] + [s["n"] for s in samples[:-1]]).cumsum(0)

        offset_ = offset.repeat_interleave(b["m"])
        b["src"] = src + offset_
        b["dst"] = dst + offset_

        offset_ = offset.repeat_interleave(torch.tensor([len(s["u"]) for s in samples]))
        b["u"] = cat("u") + offset_
        b["v"] = cat("v") + offset_
        b["mask"] = torch.from_numpy(np.vstack([s["mask"] for s in samples]))

        b["tok"] = cat("tok")
        b["cfq_idx"] = cat("cfq_idx")

        '''
        if FLAGS.seq_model == "lstm":
            max_len = b["m"].max() + 2
            hd = self.ntok + self.nrel
            tl = hd + 1
            b["q"] = torch.from_numpy(
                np.vstack(
                    [np.hstack([[hd], s["q"].reshape([-1]), 3 * (max_len - len(s["q"])) * [self.tok2idx["[PAD]"]], [tl]]) for s in samples]
                )
            )
        '''

        return b


class CFQDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, tok_vocab, rel_vocab):
        super().__init__()
        self.batch_size = batch_size
        self.tok_vocab = tok_vocab
        self.rel_vocab = rel_vocab
        collate_fn = CollateFunction(tok_vocab, rel_vocab)
        self.data_kwargs = {"num_workers": FLAGS.num_workers, "collate_fn": collate_fn.collate_fn, "pin_memory": True}

    def setup(self, stage=None):
        logger.info(f"Initializing dataset at stage {stage}")
        _, tok2idx = self.tok_vocab
        _, rel2idx = self.rel_vocab
        data = np.load(FLAGS.cfq_data_path)
        split_data_dir_path = Path(FLAGS.cfq_split_data_dir) / f"{FLAGS.cfq_split}.npz"
        logger.info(f"Loading data from {split_data_dir_path}")
        split_data = np.load(split_data_dir_path)
        self.train_dataset = CFQDataset(split_data["trainIdxs"], data, self.tok_vocab, self.rel_vocab)
        self.dev_dataset = CFQDataset(split_data["devIdxs"], data, self.tok_vocab, self.rel_vocab)
        self.test_dataset = CFQDataset(split_data["testIdxs"], data, self.tok_vocab, self.rel_vocab)

    def train_step_count_per_epoch(self):
        return len(self.train_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, **self.data_kwargs)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, **self.data_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, **self.data_kwargs)
