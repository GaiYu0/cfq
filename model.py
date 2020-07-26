import math
from functools import partial

from sklearn.metrics import precision_recall_fscore_support
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_min, scatter_sum


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
[arange,
 ones,
 zeros] = map(partial(partial, device=device), [torch.arange,
                                                torch.ones,
                                                torch.zeros])


def print_(*args):
    print(*[arg.item() for arg in args])


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
#       x = x + self.pe[:x.size(0), :, :]
        x = x + self.variable.where(isvariable.unsqueeze(2),
                                    self.concept.where(isconcept.unsqueeze(2),
                                                       self.pe[:x.size(0), :, :]))
        return self.dropout(x)


class TransformerModel(nn.Module):

    def __init__(self, ntoken, ninp, nhead, nhid, nlayer, nout, dropout=0.5):
        super(TransformerModel, self).__init__()
        self.tok_encoder = nn.Embedding(ntoken, ninp)
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = nn.TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayer)
        self.linear = nn.Linear(nhid, nout)
        self.ninp = ninp

    def forward(self, src, masks):
        src = self.tok_encoder(src.t()) * math.sqrt(self.ninp)
        src = self.pos_encoder(src, **{k : v.t() for k, v in masks.items()})
        return self.linear(self.transformer_encoder(src)).permute(1, 0, 2)


class NeuralTensorLayer(nn.Module):

    def __init__(self, ninp, nhid, nrel):
        super().__init__()
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
        bilinear = self.w.matmul(hd.t()).permute(0, 1, 3, 2).unsqueeze(3).matmul(tl.unsqueeze(2)).squeeze()
        linear = self.v.matmul(torch.cat([hd, tl], 1).t())
        return self.u.bmm(torch.tanh(bilinear + linear + self.b)).squeeze(1).t()


class LSTMModel(nn.Module):

    def __init__(self, ntoken, ninp, nhid, nlayer, nout):
        super().__init__()
        self.tok_encoder = nn.Embedding(ntoken, ninp)
        self.lstm_encoder = nn.LSTM(ninp, nhid // 2, nlayer, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(nhid, nout)

        self.ninp = ninp
        self.pos_encoder = PositionalEncoding(ninp)

    def forward(self, seq, *args):
        h, _ = self.lstm_encoder(self.tok_encoder(seq))
#       h, _ = self.lstm_encoder(self.pos_encoder(self.tok_encoder(seq) * math.sqrt(self.ninp), **args[0]))

        return self.linear(h.view(seq.size(0), seq.size(1), -1))


class Model(nn.Module):

    def __init__(self, args, vocab, rel_vocab):
        super().__init__()
#       self.seq_encoder = LSTMModel(args.ntoken, args.seq_ninp, args.seq_nhid, args.seq_nlayer, args.ntl_ninp)
        self.seq_encoder = TransformerModel(args.ntoken,
                                            args.seq_ninp,
                                            args.nhead,
                                            args.seq_nhid,
                                            args.seq_nlayer,
                                            args.ntl_ninp,
                                            args.dropout)

        self.bn_src = nn.BatchNorm1d(args.ntl_ninp)
        self.bn_dst = nn.BatchNorm1d(args.ntl_ninp)
        self.ntl = NeuralTensorLayer(args.ntl_ninp, args.ntl_nhid, args.nrel)
        self.idx2tok, self.tok2idx = vocab
        self.idx2rel, self.rel2idx = rel_vocab

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
        h = self.seq_encoder(seq, masks)[i, idx, :]
#       h = scatter_sum(self.seq_encoder(seq, masks)[i, idx, :], j, 0)

        logit = self.ntl(self.bn_src(h[src]), self.bn_dst(h[dst]))

        d = {}
        _, argmax = logit.max(1)
        eq = rel.eq(argmax)
        d['acc'] = eq.float().mean()
        d['emr'] = scatter_min(eq.int(), arange(len(m)).repeat_interleave(m))[0].eq(1).float().mean()
        if self.training:
            d['logp'] = logit.log_softmax(1).gather(1, rel.unsqueeze(1)).sum() / len(seq)

        toks = seq[i, idx]
#       print(set(self.idx2tok[src_] for src_ in toks[src][~eq]), set(self.idx2tok[src_] for src_ in toks[src][~eq]))
#       print(set(self.idx2rel[rel_] for rel_ in rel[~eq]), set(self.idx2rel[rel_] for rel_ in rel[~eq]))
        '''
        print(*[[self.idx2tok[src_], self.idx2rel[rel_], self.idx2rel[argmax_], self.idx2tok[dst_]] for src_, rel_, argmax_, dst_ in zip(toks[src][~eq], rel[~eq], argmax[~eq], toks[dst][~eq])], sep='\n')
        '''

        return d, [rel.cpu(), argmax.cpu()]
