import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer


if torch.cuda.is_available():
    [arange,
     ones,
     zeros] = map(partial(device=torch.device('cuda:0')), [torch.arange,
                                                           torch.ones,
                                                           torch.zeros])


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

        self.concept = nn.Parameter(1e-3 * torch.randn(d_model))
        self.variable = nn.Parameter(1e-3 * torch.randn(d_model))

    def forward(self, x, isvariable, isconcept):
        x = x + self.variable.where(isvariable, self.concept.where(isconcept, self.pe[:x.size(0), :]))
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayer, nout, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.tok_encoder = nn.Embedding(ntoken, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayer)
        self.linear = nn.Linear(nhid, nout)
        self.ninp = ninp

    def forward(self, src, masks):
        src = self.tok_encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src, *masks)
        return self.linear(self.transformer_encoder(src))


class NeuralTensorLayer(nn.Module):

    def __init__(self, ninp, nhid, nrel):
        self.w = nn.Parameter(1e-3 * torch.randn(nrel, nhid, ninp, ninp))
        self.v = nn.Parameter(1e-3 * torch.randn(nrel, nhid, 2 * ninp))
        self.b = nn.Parameter(torch.zeros(nrel, nhid, 1))
        self.u = nn.Parameter(1e-3 * torch.randn(nrel, 1, nhid))

    def forward(self, hd, tl):
        """
        Parameters
        ----------
        hd : (n, d)
        tl : (n, d)

        Returns
        -------
         : (n, nrel)
        """
        bilinear = self.w.matmul(hd.t()).permute(0, 1, 3, 2).matmult(tl.t())
        linear = self.v.matmul(torch.cat([hd, tl], dim=1).t())
        return self.u.bmm(torch.tanh(bilinear + linear + self.b)).squeeze(1).t()


class Model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.seq_encoder = TransformerModel(args.ntoken,
                                            args.seq_ninp,
                                            args.nhead,
                                            args.seq_nhid,
                                            args.seq_nlayer,
                                            args.ntl_ninp,
                                            args.dropout)
        self.ntl = NeuralTensorLayer(args.ntl_ninp, args.ntl_nhid, args.nrel)

    def forward(self, seq, masks, n, n_idx, idx, m, src, dst, rel=None):
        """
        Parameters
        ----------
        seq : (n, l)
        n : (n,)
        n_idx : (n.sum(),)
        idx : (n_idx.sum(),)
        m : (n,)
        src : (m.sum(),)
        dst : (m.sum(),)
        rel : (m.sum(),)
        """
        i = arange(len(seq)).repeat_interleave(n).repeat_interleave(n_idx)
        j = arange(n.sum()).repeat_interleave(n_idx)
        h = scatter_sum(self.seq_encoder(seq, masks)[i, idx, :], j, dim=0)

        logit = self.ntl(h[src], h[dst])

        d = {}
        eq = rel.eq(logit.max(1)[1])
        d['acc'] = eq.float().mean()
        d['emr'] = scatter_min(eq, arange(len(m)).repeat_interleave(m)).eq(1).float().mean()
        if self.training:
            d['logp'] = logit.log_softmax(1).gather(rel.unsqueeze(1)).sum() / len(seq)

        return d
