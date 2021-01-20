import math

from absl import flags
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch_scatter import scatter_min, scatter_sum

from tree_lstm import TreeLSTM
import utils

FLAGS = flags.FLAGS
# global flags
flags.DEFINE_float("gamma", 1.0, "Gamma.", lower_bound=0.0, upper_bound=1.0)
flags.DEFINE_float("w_pos", 1.0, "Positional weight.", lower_bound=0.0, upper_bound=1.0)
flags.DEFINE_float("dropout", 0.0, "Dropout value", lower_bound=0.0)

# sequence model flags
flags.DEFINE_enum("seq_model", "lstm", ["lstm", "transformer"], "Sequence model implementation.")
flags.DEFINE_integer("seq_inp", 64, "Sequence input dimension")
flags.DEFINE_integer("seq_hidden_dim", 256, "Sequence hidden dimension")
flags.DEFINE_integer("seq_nlayers", 2, "Sequence model depth")
flags.DEFINE_integer("seq_nhead", 0, "Transformer number of heads")

# neural tensor layer flags
flags.DEFINE_integer("ntl_inp", 64, "Neural tensor layer input dimension", lower_bound=0)
flags.DEFINE_integer("ntl_hidden_dim", 64, "Neural tensor layer hidden dimension")
flags.DEFINE_boolean("ntl_bilinear", True, "Sequence model is bilinear?")

flags.DEFINE_integer("arity", 3, "Parse tree arity")

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
    def __init__(self, tok_vocab, typ_vocab):
        super().__init__()
        self.idx2tok, self.tok2idx = tok_vocab
        self.idx2rel, self.rel2idx = typ_vocab
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
        self.lstm_encoder = nn.LSTM(dx, dh // 2, FLAGS.seq_nlayers,
                                    batch_first=True, bidirectional=True)
        self.q = nn.Linear(2 * dh, dh)
        self.k = nn.Linear(dh, dh)
        self.rel = nn.Linear(2 * dh + dx, n_typ)

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
        s = k.matmul(q).div(d**0.5).sub(torch.exp(50 * (1 - m.float())) - 1)
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
        '''
        _, lens = pad_packed_sequence(seq)
        x_grp = pack_padded_sequence(z_grp, lens, batch_first=True, enforce_sorted=False)
        '''

        x_grp = utils.pack_as(self.tag_encoder(seq.data), seq)
        h_grp, _ = pad_packed_sequence(self.lstm_encoder(x_grp)[0], batch_first=True)
        h_grp = h_grp.view(-1, h_grp.size(-1)).view(*h_grp.shape)
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


class LSTMEncoder(nn.Module):
    def __init__(self, n_tok, d_x, d_h, n_lay):
        super().__init__()
        self.embedding = None if n_tok is None else nn.Embedding(n_tok, d_x)
        self.lstm = nn.LSTM(d_x, d_h // 2, n_lay, batch_first=True, bidirectional=True)

    def forward(self, seq, z=None, view=None):
        seq = seq if self.embedding is None else utils.pack_as(self.embedding(seq.data), seq)
        h, (z, _) = self.lstm(seq, *([] if z is None else [[z, torch.zeros_like(z)]]))

        if view is not None:
            views = [view] if isinstance(view, str) else view
            h, n = pad_packed_sequence(h, batch_first=True)
            def view_as(view):
                if view == "padded":
                    return h
                elif view == "flat":
                    trues = torch.ones_like(seq.data, dtype=bool)
                    m, _ = pad_packed_sequence(utils.pack_as(trues, seq), batch_first=True)
                    return h[m[:, :, 0]]
            h = [view_as(view) for view in views] + [n]

        return h, z


class NounPhraseModel(nn.Module):
    def __init__(self, tok_vocab, tag_vocab, typ_vocab):
        super().__init__()
        self.idx2tok, self.tok2idx = tok_vocab
        self.idx2tag, self.tag2idx = tag_vocab
        self.idx2typ, self.typ2idx = typ_vocab
        n_tok, n_tag, n_typ = len(self.idx2tok), len(self.idx2tag), len(self.idx2typ)

        d_x, d_h, n_lay = _, self.d_h, self.n_lay = FLAGS.seq_inp, FLAGS.seq_hidden_dim, FLAGS.seq_nlayers
        self.tok_embedding = nn.Embedding(n_tok, d_x)
        self.tag_encoder = LSTMEncoder(n_tag + 1, d_x, d_h, n_lay)
        self.noun_encoder = LSTMEncoder(n_tag, d_x, d_h, n_lay)
        self.np_encoder = LSTMEncoder(None, d_x, d_h, n_lay)
        self.encoder = LSTMEncoder(None, d_x, d_h, n_lay)
        self.tag2noun = nn.Linear(d_h, n_lay * d_h)
        self.noun2np = nn.Linear(n_lay * d_h, d_x)

        self.q = nn.Linear(2 * d_h, d_h)
        self.k = nn.Linear(d_h, d_h)
        self.rel = nn.Linear(2 * d_h + d_x, n_typ)

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
        s = k.matmul(q).div(d**0.5).sub(torch.exp(50 * (1 - m.float())) - 1)
        return v.mul(s.softmax(1)).sum(1).squeeze()

    def forward(self, seq_tag, seq_noun,
                idx_np, pos_np,
                isnoun, idx_noun, pos_noun, idx_tag, pos_tag,
                mem, grp, pos2grp,
                m, idx, src, dst, typ, msk, **kwargs):
        h_tag, _ = self.tag_encoder(seq_tag)
        z_tag = F.relu(self.tag2noun(h_tag[idx_np, pos_np]))
        z_tag = z_tag.view(-1, 2 * self.n_lay, self.d_h // 2).permute(1, 0, 2)
        h_noun, _ = self.noun_encoder(seq_noun, z_tag)
        h_grp = h_noun[idx_noun.data, pos_noun.data].where(isnoun.data[:, None], h_tag[idx_tag.data, pos_tag.data])
        h_grp, _ = pad_packed_sequence(utils.pack_as(h_grp, isnoun), batch_first=True)
        q = torch.cat([h_grp[idx, src], h_grp[idx, dst]], -1)
        x_grp = scatter_sum(self.tok_embedding(mem), grp, 0)[pos2grp]
        logit = self.rel(torch.cat([q, self.attention(m, q, h_grp, x_grp, msk)], -1))

        d = {}
        gt = logit.gt(0)
        eq = gt.eq(typ)
        d["acc"] = eq.float().mean()
        em, _ = scatter_min(eq.all(1).int(), idx)
        d["emr"] = em.float().mean()
        if self.training:
            d["loss"] = d["nll"] = F.binary_cross_entropy_with_logits(logit, typ.float(), reduction='sum') / len(h_grp)

        return d, {"index" : kwargs["index"], "em": em, "typ_true": typ, "typ_pred": gt, "logit" : logit}


class NounPhraseModelV2(nn.Module):
    def __init__(self, tok_vocab, tag_vocab, typ_vocab):
        super().__init__()
        self.idx2tok, self.tok2idx = tok_vocab
        self.idx2tag, self.tag2idx = tag_vocab
        self.idx2typ, self.typ2idx = typ_vocab
        n_tok, n_tag, n_typ = len(self.idx2tok), len(self.idx2tag), len(self.idx2typ)

        d_x, d_h, n_lay = _, self.d_h, self.n_lay = FLAGS.seq_inp, FLAGS.seq_hidden_dim, FLAGS.seq_nlayers
        self.tok_embedding = nn.Embedding(n_tok, d_x)
        self.sep_embedding = nn.Embedding(1, d_h)
        self.tag_encoder = LSTMEncoder(n_tag + 2, d_x, d_h, n_lay)
        self.noun_encoder = LSTMEncoder(n_tag, d_x, d_h, n_lay)
        self.np_encoder = LSTMEncoder(None, d_h, d_h, n_lay)
        self.encoder = LSTMEncoder(None, d_x, d_h, n_lay)
        self.tag2noun = nn.Linear(d_h, n_lay * d_h)
        self.noun2np = nn.Linear(n_lay * d_h, d_x)

        self.q = nn.Linear(2 * d_h, d_h)
        self.k = nn.Linear(d_h, d_h)
        self.rel = nn.Linear(2 * d_h + d_x, n_typ)

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

    def forward(self, seq_tag, seq_noun,
                idx_np, pos_np,
                issep, idx_noun, pos_noun,
                idx_nps, pos_nps, isnp, idx_tag, pos_tag,
                mem, grp, pos2grp,
                m, idx, src, dst, typ, msk, **kwargs):
        h_tag, _ = self.tag_encoder(seq_tag)
        z_tag = F.relu(self.tag2noun(h_tag[idx_np, pos_np]))
        z_tag = z_tag.view(-1, 2 * self.n_lay, self.d_h // 2).permute(1, 0, 2)
        h_noun, _ = self.noun_encoder(seq_noun, z_tag)
        x_sep = self.sep_embedding(torch.zeros_like(issep.data, dtype=torch.long))
        x_np = x_sep.where(issep.data[:, None], h_noun[idx_noun.data, pos_noun.data])
        h_np, _ = self.np_encoder(utils.pack_as(x_np, issep))
        h_grp = h_np[idx_nps.data, pos_nps.data].where(isnp.data[:, None], h_tag[idx_tag.data, pos_tag.data])
        h_grp, _ = pad_packed_sequence(utils.pack_as(h_grp, isnp), batch_first=True)
        q = torch.cat([h_grp[idx, src], h_grp[idx, dst]], -1)
        x_grp = scatter_sum(self.tok_embedding(mem), grp, 0)[pos2grp]
        logit = self.rel(torch.cat([q, self.attention(m, q, h_grp, x_grp, msk)], -1))

        d = {}
        gt = logit.gt(0)
        eq = gt.eq(typ)
        d["acc"] = eq.float().mean()
        em, _ = scatter_min(eq.all(1).int(), idx)
        d["emr"] = em.float().mean()
        if self.training:
            d["loss"] = d["nll"] = F.binary_cross_entropy_with_logits(logit, typ.float(), reduction='sum') / len(h_grp)

        return d, {"index" : kwargs["index"], "em": em, "typ_true": typ, "typ_pred": gt, "logit" : logit}


class Attention(nn.Module):
    def __init__(self, d_q, d_k, d_z):
        super().__init__()
        self.d_z = d_z
        self.q = nn.Linear(d_q, d_z)
        self.k = nn.Linear(d_k, d_z)

    def forward(self, nq, q, k, v, m):
        """
        Parameters
        ----------
        nq : (n,)
        q : (nq.sum(), d_q)
        k : (n, l, d_k)
        v : (n, l, d)
        m : (n, l)

        Returns
        -------
        (nq.sum(), d)
        """
        n, l = m.shape
        q = self.q(q).view(-1, 1, self.d_z, 1)
        reshape = lambda x: x.view(n, l, 1, -1).repeat_interleave(nq, 0)
        k, v, m = map(reshape, [self.k(k), v, m])
        s = k.matmul(q).div(self.d_z**0.5).masked_fill(~m, float('-inf'))
        return v.mul(s.softmax(1)).sum(1).view(-1, v.size(-1))


class NounPhraseModelV3(nn.Module):
    def __init__(self, tok_vocab, tag_vocab, typ_vocab):
        super().__init__()
        self.idx2tok, self.tok2idx = tok_vocab
        self.idx2tag, self.tag2idx = tag_vocab
        self.idx2typ, self.typ2idx = typ_vocab
        n_tok, n_tag, n_typ = len(self.idx2tok), len(self.idx2tag), len(self.idx2typ)

        d_x, d_h, n_lay = _, self.d_h, self.n_lay = FLAGS.seq_inp, FLAGS.seq_hidden_dim, FLAGS.seq_nlayers
        self.tok_embedding = nn.Embedding(n_tok, d_x)
        self.tag_encoder = LSTMEncoder(n_tag + 2, d_x, d_h, n_lay)
        self.noun_encoder = LSTMEncoder(n_tag, d_x, d_h, n_lay)
        self.np_encoder = LSTMEncoder(None, n_lay * d_h, d_h, n_lay)
        self.var_encoder = LSTMEncoder(n_tag, d_x, d_h, n_lay)
        self.tag2np = nn.Linear(d_h, n_lay * d_h)
        self.att_noun = Attention(d_h, d_h, d_h)
        self.att_all = Attention(2 * d_h, d_h, d_h)
        self.rel = nn.Linear(2 * d_h + d_x, n_typ)

        self.encoder = LSTMEncoder(None, d_h, d_h, n_lay)

    def forward(self, seq_tag, seq_noun, seq_var, seq_np,
                idx_np, pos_np,
                idx_noun, pos_noun, mask_noun,
                idx_all, pos_all, isvar, isnoun, istag,
                m, idx, src, dst, typ, mem, grp, pos2grp, **kwargs):
        [h_tag, _], z_tag = self.tag_encoder(seq_tag, view="padded")

        z_np = F.relu(self.tag2np(h_tag[idx_np, pos_np]))
        z_np = z_np.view(-1, 2 * self.n_lay, self.d_h // 2).permute(1, 0, 2)
        [h_noun, _], z_noun = self.noun_encoder(seq_noun, z_np, view="padded")

        _, z = self.np_encoder(utils.pack_as(z_noun.permute(1, 0, 2).reshape(z_noun.size(1), -1)[seq_np.data], seq_np))
        [h_var, h_var_, n_var], _ = self.var_encoder(seq_var, z, view=["flat", "padded"])
        h_noun_ = h_noun[idx_noun, pos_noun]
        h_var = self.att_noun(n_var.cuda(), h_var, h_noun_, h_noun_, mask_noun)

        where = lambda x, m: x.where(m, torch.zeros_like(x))
        index = lambda h, m: h[where(idx_all, m), where(pos_all, m)]
        h_all = index(h_var_, isvar).where(isvar[:, :, None], index(h_noun, isnoun).where(isnoun[:, :, None], index(h_tag, istag)))

        q = torch.cat([h_all[idx, src], h_all[idx, dst]], -1)
        x_all = scatter_sum(self.tok_embedding(mem), grp, 0)[pos2grp]
        crop = lambda x: x[:, :x_all.size(1)]
        logit = self.rel(torch.cat([q, self.att_all(m, q, crop(h_all), x_all, crop(istag | isnoun))], -1))

        d = {}
        gt = logit.gt(0)
        eq = gt.eq(typ)
        d["acc"] = eq.float().mean()
        em, _ = scatter_min(eq.all(1).int(), idx)
        d["emr"] = em.float().mean()
        if self.training:
            d["loss"] = d["nll"] = F.binary_cross_entropy_with_logits(logit, typ.float(), reduction='sum') / len(m)

        return d, {"index" : kwargs["index"], "em": em, "typ_true": typ, "typ_pred": gt, "logit" : logit}


class NounPhraseModelV4(nn.Module):
    def __init__(self, tok_vocab, tag_vocab, typ_vocab, symb_vocab):
        super().__init__()
        self.idx2tok, self.tok2idx = tok_vocab
        self.idx2tag, self.tag2idx = tag_vocab
        self.idx2typ, self.typ2idx = typ_vocab
        self.idx2symb, self.symb2idx = symb_vocab
        n_tok, n_tag, n_typ, n_symb = len(self.idx2tok), len(self.idx2tag), len(self.idx2typ), len(self.idx2symb)

        d_x, d_h, n_lay = _, self.d_h, self.n_lay = FLAGS.seq_inp, FLAGS.seq_hidden_dim, FLAGS.seq_nlayers
        self.tok_embedding = nn.Embedding(n_tok, d_x)
        self.tag_encoder = LSTMEncoder(n_tag + 2, d_x, d_h, n_lay)
        self.noun_encoder = TreeLSTM(n_symb, d_x, d_h, FLAGS.arity)
        self.encoder = LSTMEncoder(None, d_h, d_h, n_lay)
        self.var_encoder = LSTMEncoder(n_tag, d_x, d_h, n_lay)
        self.tag2np = nn.Linear(d_h, n_lay * d_h)
        self.att_node = Attention(d_h, d_h, d_h)
        self.att_all = Attention(2 * d_h, d_h, d_h)
        self.rel = nn.Linear(2 * d_h + d_x, n_typ)

    def forward(self, seq_tag, seq_noun, seq_var, seq_np,
                idx_np, pos_np,
                gr, idx_node, mask_node, idx_leaf,
                idx_all, pos_all, isvar, isnoun, istag,
                m, idx, src, dst, typ, mem, grp, pos2grp, **kwargs):
        [h_tag, _], z_tag = self.tag_encoder(seq_tag, view="padded")

        z_np = h_tag[idx_np, pos_np]
#       z_np = F.relu(self.tag2np(h_tag[idx_np, pos_np]))

        h_node, h_root = self.noun_encoder(gr, z_np)

        _, z_root = self.encoder(utils.pack_as(h_root[seq_np.data], seq_np))
        [h_var, h_var_, n_var], _ = self.var_encoder(seq_var, z_root, view=["flat", "padded"])
        h_node_ = h_node[idx_node]
        h_var = self.att_node(n_var.cuda(), h_var, h_node_, h_node_, mask_node)  # TODO .cuda()

        where = lambda x, m: x.where(m, torch.zeros_like(x))
        index = lambda h, m: h[where(idx_all, m), where(pos_all, m)]
        h_all = index(h_var_, isvar).where(isvar[:, :, None], index(h_node[idx_leaf], isnoun).where(isnoun[:, :, None], index(h_tag, istag)))

        q = torch.cat([h_all[idx, src], h_all[idx, dst]], -1)
        x_all = scatter_sum(self.tok_embedding(mem), grp, 0)[pos2grp]
        crop = lambda x: x[:, :x_all.size(1)]
        logit = self.rel(torch.cat([q, self.att_all(m, q, crop(h_all), x_all, crop(istag | isnoun))], -1))

        d = {}
        gt = logit.gt(0)
        eq = gt.eq(typ)
        d["acc"] = eq.float().mean()
        em, _ = scatter_min(eq.all(1).int(), idx)
        d["emr"] = em.float().mean()
        if self.training:
            d["loss"] = d["nll"] = F.binary_cross_entropy_with_logits(logit, typ.float(), reduction='sum') / len(m)

        return d, {"index" : kwargs["index"], "em": em, "typ_true": typ, "typ_pred": gt, "logit" : logit}
