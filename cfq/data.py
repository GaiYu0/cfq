from itertools import *

from absl import flags
import dgl
import numpy as np
from loguru import logger
from pathlib import Path
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import *
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import utils

FLAGS = flags.FLAGS


class RaggedArray:
    def __init__(self, data, indptr, separate=False):
        self.data = data
        self.indptr = indptr
        self.separate = separate

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self.data[self.indptr[key] : self.indptr[key + 1]]
        elif type(key) is slice:
            start, stop, stride = key.indices(len(self.indptr))
            assert stride == 1
            if self.separate:
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
        idx2typ, _ = typ_vocab
        self.n_typ = len(idx2typ)

        self.idx = idx
        get = lambda k: dat[k] if type(k) is str else k
        rag = lambda x, y, **kwargs: RaggedArray(get(x), np.cumsum(np.hstack([[0], get(y)])), **kwargs)
        self.dat = dict(dat.items())
        self.dat["seq"] = rag("seq", "n_grp")
        self.dat["n_mem"] = rag("n_mem", "n_grp")
        self.dat["mem"] = rag(rag("mem", "n_mem"), "n_grp")
        self.dat["src"], self.dat["dst"], self.dat["typ"] = rag("src", "n_rel"), rag("dst", "n_rel"), rag("typ", "n_rel")

        self.dat["filter"] = rag("filter", "n_filter")
        self.dat["idx2grp"] = rag("idx2grp", "n")

        self.dat["seq_tag"] = rag("seq_tag", "len_tag")
        self.dat["seq_noun"] = rag(rag("seq_noun", "len_noun", separate=True), "len_np")
        self.dat["pos_np"] = rag("pos_np", "len_np")

        self.dat["len_noun"] = rag("len_noun", "len_np")
        self.dat["isnoun"] = rag("isnoun", "n_grp")
        self.dat["pos_noun"] = rag("pos_noun", "n_grp")
        self.dat["pos_tag"] = rag("pos_tag", "n_grp")

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
'''


'''
class CFQDataset(Dataset):
    def __init__(self, idx, dat, tok_vocab, tag_vocab, typ_vocab):
        super().__init__()
        idx2typ, _ = typ_vocab
        self.n_typ = len(idx2typ)

        self.idx = idx
        get = lambda k: dat[k] if type(k) is str else k
        rag = lambda x, y, **kwargs: RaggedArray(get(x), np.cumsum(np.hstack([[0], get(y)])), **kwargs)
        self.dat = dict(dat.items())
        self.dat["seq"] = rag("seq", "n_grp")
        self.dat["n_mem"] = rag("n_mem", "n_grp")
        self.dat["mem"] = rag(rag("mem", "n_mem"), "n_grp")
        self.dat["src"], self.dat["dst"], self.dat["typ"] = rag("src", "n_rel"), rag("dst", "n_rel"), rag("typ", "n_rel")

        self.dat["filter"] = rag("filter", "n_filter")
        self.dat["idx2grp"] = rag("idx2grp", "n")

        self.dat["seq_tag"] = rag("seq_tag", "len_tag")
        self.dat["seq_noun"] = rag(rag("seq_noun", "len_noun", separate=True), "len_np")

        self.dat["pos_noun"] = rag("pos_noun", "len_nps")
        self.dat["pos_tag"] = rag("pos_tag", "n_grp")
        self.dat["pos_np"] = rag("pos_np", "len_np")
        self.dat["pos_nps"] = rag("pos_nps", "n_grp")

        self.dat["len_noun"] = rag("len_noun", "len_np")

        self.dat["isnp"] = rag("isnp", "n_grp")
        self.dat["issep"] = rag("issep", "len_nps")

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
'''


class CFQDataset(Dataset):
    def __init__(self, idx, dat, tok_vocab, tag_vocab, typ_vocab):
        super().__init__()
        idx2typ, _ = typ_vocab
        self.n_typ = len(idx2typ)

        self.idx = idx
        get = lambda k: dat[k] if type(k) is str else k
        rag = lambda x, y, **kwargs: RaggedArray(get(x), np.cumsum(np.hstack([[0], get(y)])), **kwargs)
        self.dat = dict(dat.items())
        self.dat["n_mem"] = rag("n_mem", "n_grp")
        self.dat["mem"] = rag(rag("mem", "n_mem"), "n_grp")
        self.dat["idx2grp"] = rag("idx2grp", "n")
        self.dat["src"], self.dat["dst"], self.dat["typ"] = rag("src", "n_rel"), rag("dst", "n_rel"), rag("typ", "n_rel")

        self.dat["seq_tag"] = rag("seq_tag", "len_tag")
        self.dat["len_noun"] = rag("len_noun", "n_np")
        self.dat["seq_noun"] = rag(rag("seq_noun", "len_noun", separate=True), "n_np")
        self.dat["seq_var"] = rag("seq_var", "len_var")

        self.dat["pos_np"] = rag("pos_np", "n_np")
        self.dat["pos_all"] = rag("pos_all", "n_all")
        self.dat["istag"] = rag("istag", "n_all")
        self.dat["isnoun"] = rag("isnoun", "n_all")
        self.dat["isvar"] = rag("isvar", "n_all")

        self.dat["symb"] = rag(rag("symb", "n_tree", separate=True), "n_np")
        self.dat["isleaf"] = rag(rag("isleaf", "n_tree", separate=True), "n_np")
        self.dat["src_tree"] = rag(rag("src_tree", "m_tree", separate=True), "n_np")
        self.dat["dst_tree"] = rag(rag("dst_tree", "m_tree", separate=True), "n_np")

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, key):
        idx = self.idx[key]
        dat = {k : v[idx] for k, v in self.dat.items()}

        dat["index"] = idx

        n = self.dat["n"][idx]
        typ = np.zeros([n, n, self.n_typ], dtype=bool)
        typ[dat["src"], dat["dst"], dat["typ"]] = True
        dat["typ"] = typ.reshape([-1, self.n_typ])

        src, dst = np.meshgrid(dat["idx2grp"], dat["idx2grp"])
        dat["src"], dat["dst"] = src.flatten(), dst.flatten()

        return dat


"""
class CollateFunction:
    def collate_fn(self, ds):
        dats = ds

        hstack = lambda k: torch.from_numpy(np.hstack([d[k] for d in ds]))
        vstack = lambda k: torch.from_numpy(np.vstack([d[k] for d in ds]))
        collate = lambda key: torch.tensor([dat[key] for dat in dats])
        pack = lambda key: pack_sequence([torch.from_numpy(dat[key]) for dat in dats], enforce_sorted=False)

        bat = {"index" : torch.tensor([d["index"] for d in ds])}

        seq = bat["seq"] = pack("seq")
        msk, _ = bat["msk"], _ = pad_packed_sequence(utils.pack_as(torch.ones_like(seq.data), seq), batch_first=True)

        bat["mem"] = hstack("mem")
        n_mem = hstack("n_mem")
        bat["grp"] = torch.arange(len(n_mem)).repeat_interleave(n_mem)
        bat["pos2grp"] = msk.flatten().cumsum(0).sub(1).view(*msk.shape)

        n = hstack("n")
        m = n * n
        '''
        disp = n.cumsum(0).sub(n[0]).repeat_interleave(m)
        bat["src"], bat["dst"] = hstack("src") + disp, hstack("dst") + disp
        '''
        bat["src"], bat["dst"] = hstack("src"), hstack("dst")
        bat["typ"] = vstack("typ")
        bat["idx"] = torch.arange(len(m)).repeat_interleave(m)
        bat["m"] = m

        bat["seq_tag"] = pack("seq_tag")
        bat["seq_noun"] = pack_sequence(list(chain(*((torch.from_numpy(seq) for seq in dat["seq_noun"]) for dat in dats))), enforce_sorted=False)
        len_np = collate("len_np")
        bat["seq_np"] = pack_sequence(torch.arange(len_np.sum()).split(len_np.tolist()), enforce_sorted=False)
        bat["idx_np"] = torch.arange(len(dats)).repeat_interleave(len_np)
        bat["pos_np"] = torch.cat([torch.tensor(dat["pos_np"]) for dat in dats])

        bat["isnoun"], bat["pos_noun"], bat["pos_tag"] = pack("isnoun"), pack("pos_noun"), pack("pos_tag")
        isnoun, len_isnoun = pad_packed_sequence(bat["isnoun"], batch_first=True)
        bat["idx_tag"] = pack_sequence([torch.zeros(len(dat["pos_tag"])).long() for dat in dats], enforce_sorted=False)
        idx_noun, lengths = pad_packed_sequence(bat["idx_tag"], batch_first=True)
        len_noun = hstack("len_noun")
        idx_noun[isnoun] = torch.arange(len(len_noun)).repeat_interleave(len_noun)
        bat["idx_noun"] = pack_padded_sequence(idx_noun, lengths, batch_first=True, enforce_sorted=False)

        return bat
"""


class CollateFunction:
    def collate_fn(self, ds):
        dats = ds

        hstack = lambda k: torch.from_numpy(np.hstack([d[k] for d in ds]))
        vstack = lambda k: torch.from_numpy(np.vstack([d[k] for d in ds]))
        collate = lambda key: torch.tensor([dat[key] for dat in dats])
        pack = lambda key: pack_sequence([torch.from_numpy(dat[key]) for dat in dats], enforce_sorted=False)
        pad = lambda key: pad_packed_sequence(pack(key), batch_first=True)[0]

        bat = {"index" : torch.tensor([d["index"] for d in ds])}

        n = hstack("n")
        m = n * n
        bat["src"], bat["dst"] = hstack("src"), hstack("dst")
        bat["typ"] = vstack("typ")
        bat["idx"] = torch.arange(len(m)).repeat_interleave(m)
        bat["m"] = m

        seq_tag = bat["seq_tag"] = pack("seq_tag")
        bat["seq_noun"] = pack_sequence(list(chain(*((torch.from_numpy(seq) for seq in dat["seq_noun"]) for dat in dats))), enforce_sorted=False)
        n_np = collate("n_np")
        bat["seq_np"] = pack_sequence(torch.arange(n_np.sum()).split(n_np.tolist()), enforce_sorted=False)
        bat["seq_var"] = pack("seq_var")

        bat["idx_np"] = torch.arange(len(dats)).repeat_interleave(collate("n_np"))
        bat["pos_np"] = hstack("pos_np")

        istag, isnoun, isvar = bat["istag"], bat["isnoun"], bat["isvar"] = list(map(pad, ["istag", "isnoun", "isvar"]))

        pos_all = bat["pos_all"] = pad("pos_all")
        idx_all = bat["idx_all"] = torch.zeros_like(pos_all)
        idx_all[istag] = torch.arange(len(dats)).repeat_interleave(istag.sum(1))
        idx_noun = idx_all[isnoun] = torch.arange(collate("n_np").sum()).repeat_interleave(hstack("len_noun"))
        idx_all[isvar] = torch.arange(len(dats)).repeat_interleave(isvar.sum(1))

        pad_pack = lambda xs: pad_packed_sequence(pack_sequence(xs, enforce_sorted=False), batch_first=True)[0]
        bat["idx_noun"] = pad_pack(idx_noun.split([sum(dat["len_noun"]) for dat in dats]))
        bat["pos_noun"] = pad_pack([torch.cat([torch.arange(x) for x in dat["len_noun"]]) for dat in dats])
        bat["mask_noun"] = pad_pack([torch.ones(sum(dat["len_noun"]), dtype=bool) for dat in dats])

        bat["pos2grp"] = istag.logical_or(isnoun).long().flatten().cumsum(0).sub(1).view(*istag.shape)
        bat["mem"] = hstack("mem")
        n_mem = hstack("n_mem")
        bat["grp"] = torch.arange(len(n_mem)).repeat_interleave(n_mem)

        grs = []
        for dat in dats:
            for symb, isleaf, src, dst in zip(dat["symb"], dat["isleaf"], dat["src_tree"], dat["dst_tree"]):
                gr = dgl.DGLGraph()
                isroot = torch.tensor([True] + (len(symb) - 1) * [False])
                gr.add_nodes(len(symb), {"symb" : torch.from_numpy(symb), "isroot" : isroot, "isleaf" : torch.from_numpy(isleaf)})
                gr.add_edges(src, dst)
                grs.append(gr)
        gr = bat["gr"] = dgl.batch(grs)

        n_symb = [sum(map(len, dat["symb"])) for dat in dats]
        bat["idx_node"] = pad_pack(torch.arange(gr.num_nodes()).split(n_symb))
        bat["mask_node"] = pad_pack([torch.ones(n, dtype=bool) for n in n_symb])
        _, n_leaves = torch.unique_consecutive(torch.arange(gr.batch_size).repeat_interleave(gr.batch_num_nodes())[gr.ndata["isleaf"]], return_counts=True)
        bat["idx_leaf"] = pad_pack(torch.arange(gr.num_nodes())[gr.ndata["isleaf"]].split(n_leaves.tolist()))

        return bat


class CFQDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, tok_vocab, tag_vocab, typ_vocab, symb_vocab):
        super().__init__()
        self.batch_size = batch_size
        self.tok_vocab, self.tag_vocab, self.typ_vocab, self.symb_vocab = tok_vocab, tag_vocab, typ_vocab, symb_vocab
        collate_fn = CollateFunction()
        self.data_kwargs = {"num_workers": FLAGS.num_workers, "collate_fn": collate_fn.collate_fn, "pin_memory": True}

    def setup(self, stage=None):
        logger.info(f"Initializing dataset at stage {stage}")
        data = np.load(FLAGS.cfq_data_path)
        split_data_dir_path = Path(FLAGS.cfq_split_data_dir) / f"{FLAGS.cfq_split}.npz"
        logger.info(f"Loading data from {split_data_dir_path}")
        split_data = np.load(split_data_dir_path)
        self.train_dataset = CFQDataset(split_data["trainIdxs"], data, self.tok_vocab, self.tag_vocab, self.typ_vocab)
        self.dev_dataset = CFQDataset(split_data["devIdxs"], data, self.tok_vocab, self.tag_vocab, self.typ_vocab)
#       self.test_dataset = CFQDataset(split_data["trainIdxs"], data, self.tok_vocab, self.tag_vocab, self.typ_vocab)
        self.test_dataset = CFQDataset(split_data["testIdxs"], data, self.tok_vocab, self.tag_vocab, self.typ_vocab)

    def train_step_count_per_epoch(self):
        return len(self.train_dataset)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, **self.data_kwargs)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, **self.data_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, **self.data_kwargs)
