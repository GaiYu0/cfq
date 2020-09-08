import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence
from torch_scatter import scatter_min, scatter_sum

from rgcn import *
import utils


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
[arange,
 ones,
 zeros] = map(partial(partial, device=device), [torch.arange,
                                                torch.ones,
                                                torch.zeros])


class LSTMModel(nn.Module):

    def __init__(self, ntok, ninp, nhid, nlayer):
        super().__init__()
        self.tok_encoder = nn.Embedding(ntok, ninp)
        self.lstm_encoder = nn.LSTM(ninp, nhid // 2, nlayer, batch_first=True, bidirectional=True)

        self.ninp = ninp
        self.pos_encoder = PositionalEncoding(ninp, 0.0)

    def forward(self, seq):
        h, _ = self.lstm_encoder(utils.apply(self.tok_encoder, seq))
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
        self.register_buffer('pe', pe)

        self.concept = nn.Parameter(1e-3 * torch.randn(d_model))
        self.variable = nn.Parameter(1e-3 * torch.randn(d_model))

    def forward(self, x, isvariable, isconcept):
        x = x + self.pe[:x.size(0), :, :]
        '''
        x = x + self.variable.where(isvariable.unsqueeze(2),
                                    self.concept.where(isconcept.unsqueeze(2),
                                                       self.pe[:x.size(0), :, :]))
        '''
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
        src = self.tok_encoder(seq['tok']) * math.sqrt(self.ninp)
        src = self.pos_encoder(src, seq['isvariable'], seq['isconcept'])
        return self.transformer_encoder(src, src_key_padding_mask=seq['ispad']).permute(1, 0, 2)


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
        tl : (n, d)

        Returns
        -------
         : (n, nrel)
        """
        '''
        (nrel, nhid, ninp, ninp) . (ninp, n) -> (nrel, nhid, ninp, n)
        (nrel, nhid, n, 1, ninp) . (n, ninp, 1) -> (nrel, nhid, n, 1, 1)
        '''
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

    def __init__(self, args, vocab, rel_vocab):
        super().__init__()
        self.idx2tok, self.tok2idx = vocab
        self.idx2rel, self.rel2idx = rel_vocab
        ntok, nrel = len(self.idx2tok), len(self.idx2rel)

        if args.seq_model == 'embedding':
            self.seq_encoder = EmbeddingModel(ntok, args.ntl_ninp)
        elif args.seq_model == 'lstm':
            nout = args.seq_nhid
            self.seq_encoder = LSTMModel(ntok, args.seq_ninp, args.seq_nhid, args.seq_nlayer)
        elif args.seq_model == 'transformer':
            nout = args.seq_ninp
            self.seq_encoder = TransformerModel(ntok,
                                                args.seq_ninp,
                                                args.nhead,
                                                args.seq_nhid,
                                                args.seq_nlayer,
                                                args.dropout)
        else:
            raise Exception()

        self.bn_src = nn.BatchNorm1d(args.ntl_ninp)
        self.bn_dst = nn.BatchNorm1d(args.ntl_ninp)
        self.ntl = NeuralTensorLayer(args.ntl_ninp, args.ntl_nhid, nrel, args.bilinear)  # TODO scale of inner product

        self.linear_src = nn.Linear(nout, args.ntl_ninp)
        self.linear_dst = nn.Linear(nout, args.ntl_ninp)

        if args.gr:
            self.gamma = args.gamma
            if args.gr_model == 'rgcn':
                self.gr_model = RGCN(ntok, args.gr_ninp, args.gr_nhid, args.ntl_ninp, nrel, num_hidden_layers=args.gr_nlayer)
            else:
                raise Exception()

        self.args = args

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
        i = arange(len(n)).repeat_interleave(n).repeat_interleave(n_idx)
        j = arange(n.sum()).repeat_interleave(n_idx)
        h = scatter_sum(self.seq_encoder(seq)[i, idx, :], j, 0)

#       logit = self.ntl(self.bn_src(h[src]), self.bn_dst(h[dst]))
        logit = self.ntl(self.bn_src(self.linear_src(h[u])), self.bn_dst(self.linear_dst(h[v])))

        d = {}
        gt = logit.gt(0)
        eq = gt.eq(mask)
        d['acc'] = eq.float().mean()
        em, _ = scatter_min(eq.all(1).int(), arange(len(n)).repeat_interleave(n * n))
        d['emr'] = em.float().mean()
        if self.training:
#           d['loss'] = d['nll'] = -logit.log_softmax(1).gather(1, rel.unsqueeze(1)).mean()
            nll_pos = -F.logsigmoid(logit[mask]).sum() / len(n)
            nll_neg = -torch.sum(torch.log(1 + 1e-5 - logit[~mask].sigmoid())) / len(n)
            d['loss'] = d['nll'] = self.args.w_pos * nll_pos + nll_neg

            if g is not None:
                h_ref = self.gr_model(g)
                d['norm'] = torch.norm(h - h_ref, p=2, dim=1).mean()
                d['loss'] = d['nll'] + self.gamma * d['norm']

        return d, {'em' : em, 'rel_true' : mask, 'rel_pred' : gt, 'u' : tok[u], 'v' : tok[v]}

        '''
        logit = self.linear(self.seq_encoder(seq, masks).sum(1))
        mask = zeros(m.sum(), len(self.idx2rel))
        idx = arange(len(m)).repeat_interleave(m)
        mask[idx, rel] = 1
        mask = scatter_sum(mask, idx.unsqueeze(1), 0).clamp_max(1).bool()
        d = {}
        d['loss'] = d['nll'] = -F.logsigmoid(logit[mask]).mean() - (1 + 1e-5 - logit[~mask].sigmoid()).log().mean()
        d['acc'] = logit.gt(0.5).eq(mask).float().mean()

        return d, []
        '''
