from absl import flags
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch_scatter import scatter_min, scatter_sum
from cfq import utils

FLAGS = flags.FLAGS


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

    def forward(self, x):
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
        src = self.pos_encoder(src)
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
            normalize = lambda x: self.bn(x.view(self.nrel * self.nhid, -1).t()).t().view(self.nrel, self.nhid, -1)
            return self.u.bmm(torch.tanh(normalize(bilinear + linear + self.b))).squeeze(1).t()
        else:
            return self.u.bmm(torch.tanh(linear + self.b)).squeeze(1).t()


class Model(nn.Module):
    def __init__(self, tok_vocab, typ_vocab):
        super().__init__()
        self.idx2tok, self.tok2idx = tok_vocab
        self.idx2rel, self.rel2idx = typ_vocab
        ntok, nrel = len(self.idx2tok), len(self.idx2rel)

        if FLAGS.seq_model == "embedding":
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

        logit = self.ntl(self.bn_src(self.linear_src(h[u])), self.bn_dst(self.linear_dst(h[v])))

        d = {}
        gt = logit.gt(0)
        eq = gt.eq(mask)
        d["acc"] = eq.float().mean()
        em, _ = scatter_min(eq.all(1).int().type_as(tok), torch.arange(len(n)).type_as(tok).repeat_interleave(n * n))
        d["emr"] = em.float().mean()
        if self.training:
            nll_pos = -F.logsigmoid(logit[mask]).sum() / len(n)
            nll_neg = -torch.sum(torch.log(1 + 1e-5 - logit[~mask].sigmoid())) / len(n)
            d["loss"] = d["nll"] = FLAGS.w_pos * nll_pos + nll_neg
        return d, {"cfq_idx" : kwargs['cfq_idx'], "n" : n, "em": em, "rel_true": mask, "rel_pred": gt, "u": tok[u], "v": tok[v], "logit" : logit}


class InvariantModel(nn.Module):
    def __init__(self, tok_vocab, typ_vocab):
        super().__init__()
        self.idx2tok, self.tok2idx = tok_vocab
        self.idx2rel, self.rel2idx = typ_vocab
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


class AttentionModel(nn.Module):
    def __init__(self, tok_vocab, tag_vocab, typ_vocab):
        super().__init__()
        self.idx2tok, self.tok2idx = tok_vocab
        self.idx2tag, self.tag2idx = tag_vocab
        self.idx2typ, self.typ2idx = typ_vocab
        n_tok, n_tag, n_typ = len(self.idx2tok), len(self.idx2tag), len(self.idx2typ)

        dx, dh = FLAGS.seq_inp, FLAGS.seq_hidden_dim
        self.tok_encoder = nn.Embedding(n_tok, dx)
        self.tag_encoder = nn.Embedding(n_tag, dx)
        if FLAGS.recurrent_layer == "LSTM":
            self.lstm_encoder = nn.LSTM(
                dx, dh // 2,
                FLAGS.seq_nlayers,
                batch_first=True,
                bidirectional=True,
                dropout=FLAGS.dropout)
        elif FLAGS.recurrent_layer == "GRU":
            self.lstm_encoder = nn.GRU(
                dx, dh // 2,
                FLAGS.seq_nlayers,
                batch_first=True,
                bidirectional=True,
                dropout=FLAGS.dropout)
        else:
            raise ValueError()
        
        self.q = nn.Linear(2 * dh, dh)
        self.k = nn.Linear(dh, dh)
        self.rel = nn.Linear(2 * dh + dx, n_typ)
        self.bn = nn.BatchNorm1d(dh)

    def attention(self, nq, q, k, v, m):
        """
        Parameters
        ----------
        nq : (n,)
        q : (nq.sum(), d)
        k : (n, l, d)
        v : (n, l, d)
        m : (n, l, d)

        Returns
        -------
        (nq.sum(), d)
        """
        n, l, d = k.shape
        q = self.q(q).view(-1, 1, d, 1)
        reshape = lambda x: x.view(n, l, 1, -1).repeat_interleave(nq, 0)
        k, v, m = map(reshape, [self.k(k), v, m])
        s = k.matmul(q).div(d**0.5).masked_fill(~m.bool(), float('-inf'))
        return v.mul(s.softmax(1)).sum(1).squeeze()

    def forward(self, seq, mem, grp, pos2grp, msk, src, dst, typ, idx, m, **kwargs):
        """
        Parameters
        ----------
        seq : PackedSequence
        msk :
        mem : (n,)
        grp : (n,)
        idx : (m,)
        u : (m,)
        v : (m,)
        typ : (m, n_typ)
        """
        z_grp = scatter_sum(self.tok_encoder(mem), grp, 0)[pos2grp]
        x_grp = utils.pack_as(self.tag_encoder(seq.data), seq)
        h_grp, _ = pad_packed_sequence(self.lstm_encoder(x_grp)[0], batch_first=True)
        q = torch.cat([h_grp[idx, src], h_grp[idx, dst]], -1)
        logit = self.rel(torch.cat([q, self.attention(m, q, h_grp, z_grp, msk)], -1))

        d = {}
        gt = logit.gt(0)
        eq = gt.eq(typ)
        d["acc"] = eq.float().mean()
        em, _ = scatter_min(eq.all(1).int(), idx)
        d["emr"] = em.float().mean()
        if self.training:
            d["loss"] = d["nll"] = F.binary_cross_entropy_with_logits(logit, typ.float(), reduction='sum') / len(h_grp)

        return d, {"index" : kwargs["index"], "em": em, "typ_true": typ, "typ_pred": gt, "logit" : logit}
