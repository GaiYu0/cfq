import math

from absl import flags
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence, PackedSequence
from torch_scatter import scatter_min, scatter_sum

FLAGS = flags.FLAGS
# global flags
flags.DEFINE_float("gamma", 1.0, "Gamma.", lower_bound=0.0, upper_bound=1.0)
flags.DEFINE_float("w_pos", 1.0, "Positional weight.", lower_bound=0.0, upper_bound=1.0)
flags.DEFINE_float("dropout", 0.0, "Dropout value", lower_bound=0.0)

# sequence model flags
flags.DEFINE_enum("seq_model", "lstm", ["lstm", "transformer"], "Sequence model implementation.")
flags.DEFINE_integer("seq_inp", 64, "Sequence input dimension")
flags.DEFINE_integer("seq_hidden_dim", 64, "Sequence hidden dimension")
flags.DEFINE_integer("seq_nlayers", 1, "Sequence model depth")
flags.DEFINE_integer("seq_nhead", 0, "Transformer number of heads")

# neural tensor layer flags
flags.DEFINE_integer("ntl_inp", 64, "Neural tensor layer input dimension", lower_bound=0)
flags.DEFINE_integer("ntl_hidden_dim", 64, "Neural tensor layer hidden dimension")
flags.DEFINE_boolean("ntl_bilinear", True, "Sequence model is bilinear?")


class LSTMModel(nn.Module):
    def __init__(self, ntok, ninp, nhid, nlayer):
        super().__init__()
        self.ninp = ninp
        self.tok_encoder = nn.Embedding(ntok, ninp)
        self.lstm_encoder = nn.LSTM(ninp, nhid // 2, nlayer, batch_first=True, bidirectional=True)

    def forward(self, seq):
        apply_out = PackedSequence(self.tok_encoder(seq.data), seq.batch_sizes, seq.sorted_indices, seq.unsorted_indices)
        h, _ = self.lstm_encoder(apply_out)
        h, _ = pad_packed_sequence(h, batch_first=True)
        return h


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

        self.concept = nn.Parameter(1e-3 * torch.randn(d_model))
        self.variable = nn.Parameter(1e-3 * torch.randn(d_model))

    def forward(self, x, isvariable, isconcept):
        x = x + self.pe[: x.size(0), :, :]
        return self.dropout(x)


class TransformerModel(nn.Module):
    def __init__(self, ntok, ninp, nhead, nhid, nlayer, dropout):
        super().__init__()
        self.tok_encoder = nn.Embedding(ntok, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayer)
        self.ninp = ninp

    def forward(self, seq):
        src = self.tok_encoder(seq["tok"]) * math.sqrt(self.ninp)
        src = self.pos_encoder(src, seq["isvariable"], seq["isconcept"])
        return self.transformer_encoder(src, src_key_padding_mask=seq["ispad"]).permute(1, 0, 2)


class NeuralTensorLayer(nn.Module):
    def __init__(self, ninp, nhid, nrel, bilinear):
        super().__init__()
        self.nhid = nhid
        self.nrel = nrel
        self.bilinear = bilinear
        if bilinear:
            self.bn = nn.BatchNorm1d(nrel * nhid)

        self.w = nn.Parameter(1e-3 * torch.randn(nrel, nhid, ninp, ninp))
        self.v = nn.Parameter(1e-3 * torch.randn(nrel, nhid, 2 * ninp))
        self.b = nn.Parameter(torch.zeros(nrel, nhid, 1))
        self.u = nn.Parameter(1e-3 * torch.randn(nrel, 1, nhid))

    def forward(self, hd, tl):
        """
        Parameters
        ----------
        hd : (n, d)
        tl : (n, d

        Returns
        -------
         : (n, nrel)
        """
        """
        (nrel, nhid, ninp, ninp) . (ninp, n) -> (nrel, nhid, ninp, n)
        (nrel, nhid, n, 1, ninp) . (n, ninp, 1) -> (nrel, nhid, n, 1, 1)
        """
        linear = self.v.matmul(torch.cat([hd, tl], 1).t())
        if self.bilinear:
            bilinear = self.w.matmul(hd.t()).permute(0, 1, 3, 2).unsqueeze(3).matmul(tl.unsqueeze(2)).squeeze()
            #           return self.u.bmm(torch.tanh(bilinear + linear + self.b)).squeeze(1).t()
            #           return self.u.bmm(torch.tanh(bilinear / hd.size(1) ** 0.5 + linear + self.b)).squeeze(1).t()
            normalize = lambda x: self.bn(x.view(self.nrel * self.nhid, -1).t()).t().view(self.nrel, self.nhid, -1)
            return self.u.bmm(torch.tanh(normalize(bilinear + linear + self.b))).squeeze(1).t()
        else:
            return self.u.bmm(torch.tanh(linear + self.b)).squeeze(1).t()


class Model(nn.Module):
    def __init__(self, tok_vocab, rel_vocab):
        super().__init__()
        self.idx2tok, self.tok2idx = tok_vocab
        self.idx2rel, self.rel2idx = rel_vocab
        ntok, nrel = len(self.idx2tok), len(self.idx2rel)

        if FLAGS.seq_model == "embedding":
            # self.seq_encoder = EmbeddingModel(ntok, FLAGS.ntl_inp)
            raise ValueError()
        elif FLAGS.seq_model == "lstm":
            nout = FLAGS.seq_hidden_dim
            self.seq_encoder = LSTMModel(ntok, FLAGS.seq_inp, FLAGS.seq_hidden_dim, FLAGS.seq_nlayers)
        elif FLAGS.seq_model == "transformer":
            nout = FLAGS.seq_inp
            self.seq_encoder = TransformerModel(
                ntok, FLAGS.seq_inp, FLAGS.seq_nhead, FLAGS.seq_hidden_dim, FLAGS.seq_nlayers, FLAGS.dropout
            )
        else:
            raise Exception()

        self.bn_src = nn.BatchNorm1d(FLAGS.ntl_inp)
        self.bn_dst = nn.BatchNorm1d(FLAGS.ntl_inp)
        self.ntl = NeuralTensorLayer(FLAGS.ntl_inp, FLAGS.ntl_hidden_dim, nrel, FLAGS.ntl_bilinear)
        # TODO(gaiyu0): scale of inner product

        self.linear_src = nn.Linear(nout, FLAGS.ntl_inp)
        self.linear_dst = nn.Linear(nout, FLAGS.ntl_inp)

    def forward(self, seq, n, tok, n_idx, idx, m, u, v, src, dst, mask, rel=None, g=None, **kwargs):
        """
        Parameters
        ----------
        seq : (n, l)
        n : (n,)
        tok : (n.sum(),)
        n_idx : (n.sum(),)
        idx : (n_idx.sum(),)
        m : (n,)
        src : (m.sum(),)
        dst : (m.sum(),)
        rel : (m.sum(),)
        """
        i = torch.arange(len(n)).type_as(tok).repeat_interleave(n).repeat_interleave(n_idx)
        j = torch.arange(n.sum()).type_as(tok).repeat_interleave(n_idx)
        h = scatter_sum(self.seq_encoder(seq)[i, idx, :], j, 0)

        #       logit = self.ntl(self.bn_src(h[src]), self.bn_dst(h[dst]))
        logit = self.ntl(self.bn_src(self.linear_src(h[u])), self.bn_dst(self.linear_dst(h[v])))

        d = {}
        gt = logit.gt(0)
        eq = gt.eq(mask)
        d["acc"] = eq.float().mean()
        em, _ = scatter_min(eq.all(1).int().type_as(tok), torch.arange(len(n)).type_as(tok).repeat_interleave(n * n))
        d["emr"] = em.float().mean()
        if self.training:
            #           d['loss'] = d['nll'] = -logit.log_softmax(1).gather(1, rel.unsqueeze(1)).mean()
            nll_pos = -F.logsigmoid(logit[mask]).sum() / len(n)
            nll_neg = -torch.sum(torch.log(1 + 1e-5 - logit[~mask].sigmoid())) / len(n)
            d["loss"] = d["nll"] = FLAGS.w_pos * nll_pos + nll_neg

            if g is not None:
                h_ref = self.gr_model(g)
                d["norm"] = torch.norm(h - h_ref, p=2, dim=1).mean()
                d["loss"] = d["nll"] + FLAGS.gamma * d["norm"]

        return d, {"cfq_idx" : kwargs['cfq_idx'], "n" : n, "em": em, "rel_true": mask, "rel_pred": gt, "u": tok[u], "v": tok[v], "logit" : logit}


class InvariantModel(nn.Module):
    def __init__(self, tok_vocab, rel_vocab):
        super().__init__()
        self.idx2tok, self.tok2idx = tok_vocab
        self.idx2rel, self.rel2idx = rel_vocab
        ntok, nrel = len(self.idx2tok), len(self.idx2rel)

        self.tok_encoder = nn.Embedding(ntok, FLAGS.seq_inp)
        self.lstm_encoder = nn.LSTM(FLAGS.seq_inp, FLAGS.seq_hidden_dim // 2, FLAGS.seq_nlayers, batch_first=True, bidirectional=True)

        self.bn_src = nn.BatchNorm1d(FLAGS.ntl_inp)
        self.bn_dst = nn.BatchNorm1d(FLAGS.ntl_inp)
        self.ntl = NeuralTensorLayer(FLAGS.ntl_inp, FLAGS.ntl_hidden_dim, nrel, FLAGS.ntl_bilinear)

        nout = FLAGS.seq_hidden_dim
        self.linear_src = nn.Linear(nout, FLAGS.ntl_inp)
        self.linear_dst = nn.Linear(nout, FLAGS.ntl_inp)

    def forward(self, seq, n, tok, n_idx, idx, m, u, v, src, dst, mask, pos2grp, n_grp, rel=None, g=None, **kwargs):
        """
        Parameters
        ----------
        seq : (n, l)
        n : (n,)
        tok : (n.sum(),)
        n_idx : (n.sum(),)
        idx : (n_idx.sum(),)
        m : (n,)
        src : (m.sum(),)
        dst : (m.sum(),)
        rel : (m.sum(),)
        mask :
        pos2grp : PackedSequence
        """
        pack_as = lambda x, packed_seq: PackedSequence(x, packed_seq.batch_sizes,
                                                       packed_seq.sorted_indices, packed_seq.unsorted_indices)
        x_tok, _ = pad_packed_sequence(pack_as(self.tok_encoder(seq.data), seq), batch_first=True)
        disp = torch.cat([torch.zeros(1).type_as(n), n_grp.cumsum(0)[:-1]])
        pos2grp_, _ = pad_packed_sequence(pos2grp, batch_first=True)
        x_grp = scatter_sum(x_tok.view(-1, x_tok.size(-1)), torch.flatten(pos2grp_ + disp.view(-1, 1)), 0)
        x = pack_sequence(x_grp.split(n_grp.cpu().tolist(), 0), enforce_sorted=False).to(n.device)  # TODO(GaiYu0): type_as?
        h_grp, _ = pad_packed_sequence(self.lstm_encoder(x)[0], batch_first=True)
        i = torch.arange(len(n)).type_as(n).repeat_interleave(n).repeat_interleave(n_idx)
        j = torch.arange(n.sum()).type_as(n).repeat_interleave(n_idx)
        h = scatter_sum(h_grp[i, pos2grp_[i, idx], :], j, 0)

        logit = self.ntl(self.bn_src(self.linear_src(h[u])), self.bn_dst(self.linear_dst(h[v])))

        d = {}
        gt = logit.gt(0)
        eq = gt.eq(mask)
        d["acc"] = eq.float().mean()
        em, _ = scatter_min(eq.all(1).int().type_as(tok), torch.arange(len(n)).type_as(tok).repeat_interleave(n * n))
        d["emr"] = em.float().mean()
        if self.training:
            #           d['loss'] = d['nll'] = -logit.log_softmax(1).gather(1, rel.unsqueeze(1)).mean()
            nll_pos = -F.logsigmoid(logit[mask]).sum() / len(n)
            nll_neg = -torch.sum(torch.log(1 + 1e-5 - logit[~mask].sigmoid())) / len(n)
            d["loss"] = d["nll"] = FLAGS.w_pos * nll_pos + nll_neg

        return d, {"cfq_idx" : kwargs['cfq_idx'], "n" : n, "em": em, "rel_true": mask, "rel_pred": gt, "u": tok[u], "v": tok[v], "logit" : logit}
