from absl import flags
import dgl
import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data.dataset import Dataset

FLAGS = flags.FLAGS
# FLAGS.seq_model, FLAGS.gr pulled from model.py


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
        self.seq_model = FLAGS.seq_model

        self.tok_vocab, self.rel_vocab = tok_vocab, rel_vocab
        _, tok2idx = self.tok_vocab
        ntok = len(tok2idx)

        self.idx = idx
        get = lambda k: data[k] if type(k) is str else k
        rag = lambda x, y: RaggedArray(get(x), np.cumsum(np.hstack([[0], get(y)])))
        self.data = dict(data.items())
        self.data["seq"] = rag("seq", "n_tok")
        self.data["isconcept"], self.data["isvariable"] = rag("isconcept", "n_tok"), rag("isvariable", "n_tok")
        self.data["n_idx"], self.data["idx"] = rag("n_idx", "n"), rag(rag("idx", "n_idx"), "n")
        self.data["tok"] = rag("tok", "n")
        self.data["src"], self.data["dst"], self.data["rel"] = rag("src", "m"), rag("dst", "m"), rag("rel", "m")

        disp = np.hstack([np.zeros(1, dtype=int), data["n"]]).cumsum()
        disp_ = disp[:-1].repeat(data["m"])
        tok_src = data["tok"][data["src"] + disp_]
        tok_dst = data["tok"][data["dst"] + disp_]
        tok_rel = data["rel"] + ntok
        self.data["q"] = RaggedArray(np.vstack([tok_src, tok_rel, tok_dst]).T, disp)

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
        _, tok2idx = self.tok_vocab
        _, rel2idx = self.rel_vocab
        self.ntok = len(tok2idx)
        self.nrel = len(rel2idx)

    def collate_fn(self, samples):
        b = {}

        max_len = max(len(s["seq"]) for s in samples)
        pad = lambda k, p: torch.from_numpy(np.vstack([np.hstack([s[k], np.full(max_len - len(s[k]), p)]) for s in samples]))
        if self.seq_model == "lstm":
            b["seq"] = pack_sequence([torch.from_numpy(s["seq"]) for s in samples], enforce_sorted=False)
        elif self.seq_model == "transformer":
            _, tok2idx = self.tok_vocab
            b["seq"] = {
                "tok": pad("seq", tok2idx["[PAD]"]).t(),
                "ispad": pad("ispad", True),
                "isconcept": pad("isconcept", False).t(),
                "isvariable": pad("isvariable", False).t(),
            }
        else:
            raise Exception()

        cat = lambda k: torch.from_numpy(np.hstack([s[k] for s in samples]))
        b["n"], b["n_idx"], b["idx"] = cat("n"), cat("n_idx"), cat("idx")
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

        if FLAGS.use_graph:
            b["g"] = dgl.DGLGraph((b["src"], b["dst"]))
            b["g"].ndata["tok"] = b["tok"]
            b["g"].edata["rel_type"] = b["rel"]
            _, inverse, counts = torch.unique(torch.stack([b["dst"], b["rel"]]), dim=1, return_inverse=True, return_counts=True)
            b["g"].edata["norm"] = counts.float().reciprocal().unsqueeze(1)[inverse]

        if self.seq_model == "lstm":
            max_len = b["m"].max() + 2
            hd = self.ntok + self.nrel
            tl = hd + 1
            b["q"] = torch.from_numpy(
                np.vstack(
                    [np.hstack([[hd], s["q"].reshape([-1]), 3 * (max_len - len(s["q"])) * [tok2idx["[PAD]"]], [tl]]) for s in samples]
                )
            )
        return b
