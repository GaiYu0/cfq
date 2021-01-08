import math

from absl import flags
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence
from transformers import AutoConfig, AutoModel
from torch_scatter import scatter_min, scatter_sum

FLAGS = flags.FLAGS


class LSTMModel(nn.Module):
    def __init__(self, ntok, ninp, nhid, nlayer):
        super().__init__()
        self.ninp = ninp
        self.tok_encoder = nn.Embedding(ntok, ninp)
        self.lstm_encoder = nn.LSTM(ninp, nhid // 2, nlayer, batch_first=True, bidirectional=True)
        self.pos_encoder = PositionalEncoding(ninp, 0.0)

    def forward(self, tok):
        apply_out = PackedSequence(self.tok_encoder(tok.data), tok.batch_sizes, tok.sorted_indices, tok.unsorted_indices)
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

    def forward(self, tok, ispad):
        src = self.pos_encoder(self.tok_encoder(tok) * math.sqrt(self.ninp))
        return self.transformer_encoder(src, src_key_padding_mask=ispad).permute(1, 0, 2)


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
            raise ValueError()
        elif FLAGS.seq_model == "lstm":
            nout = FLAGS.seq_hidden_dim
            self.seq_encoder = LSTMModel(ntok, FLAGS.seq_inp, FLAGS.seq_hidden_dim, FLAGS.seq_nlayers)
        elif FLAGS.seq_model == "transformer":
            nout = FLAGS.seq_inp
            self.seq_encoder = TransformerModel(
                ntok, FLAGS.seq_inp, FLAGS.seq_nhead, FLAGS.seq_hidden_dim, FLAGS.seq_nlayers, FLAGS.dropout
            )
        elif FLAGS.seq_model == "bert":
            self.seq_encoder = AutoModel.from_pretrained(FLAGS.bert_model_version)
            self.bert_config = AutoConfig.from_pretrained(FLAGS.bert_model_version)
            nout = self.bert_config.hidden_size
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
        if FLAGS.seq_model == "bert":
            encoder_out = self.seq_encoder(**seq, return_dict=True)['last_hidden_state']  # or pooler_output
        else:
            encoder_out = self.seq_encoder(**seq)
        h = scatter_sum(encoder_out[i, idx, :], j, 0)
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
        return d, {"em": em, "rel_true": mask, "rel_pred": gt, "u": tok[u], "v": tok[v]}
