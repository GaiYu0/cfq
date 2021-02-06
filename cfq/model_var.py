import math

from absl import flags
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, pack_padded_sequence, pad_packed_sequence, PackedSequence
from torch_scatter import scatter_min, scatter_sum

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


class Model(nn.Module):
    def __init__(self, n_max, tok_vocab, tag_vocab, typ_vocab):
        super().__init__()
        self.idx2tok, self.tok2idx = tok_vocab
        self.idx2tag, self.tag2idx = tag_vocab
        self.idx2typ, self.typ2idx = typ_vocab
        n_tok, n_tag, n_typ = len(self.idx2tok), len(self.idx2tag), len(self.idx2typ)

        d_x, d_h = FLAGS.seq_inp, FLAGS.seq_hidden_dim
        self.tok_encoder = nn.Embedding(n_tag, d_x)
        self.seq_encoder = nn.LSTM(d_x, d_h // 2, FLAGS.seq_nlayers, bidirectional=True)
        self.linear = nn.Linear(FLAGS.seq_nlayers * d_h, n_max)

    def forward(self, seq, n, **kwargs):
        x = utils.pack_as(self.tok_encoder(seq.data), seq)
        _, [h, _] = self.seq_encoder(x)
        logit = self.linear(h.permute(1, 0, 2).reshape(len(n), -1))
        ret = {}
        ret["acc"] = logit.max(1)[1].eq(n).float().mean()
        ret["loss"] = ret["nll"] = F.cross_entropy(logit, n)
        return ret, {}


class LSTMEncoder(nn.Module):
    def __init__(self, n_tok, d_x, d_h, n_lay):
        super().__init__()
        self.embedding = None if n_tok is None else nn.Embedding(n_tok, d_x)
        self.lstm = nn.LSTM(d_x, d_h // 2, n_lay, batch_first=True, bidirectional=True)

    def forward(self, seq, z=None):
        seq = seq if self.embedding is None else utils.pack_as(self.embedding(seq.data), seq)
        h, (z, _) = self.lstm(seq, *([] if z is None else [[z, torch.zeros_like(z)]]))
        h, _ = pad_packed_sequence(h, batch_first=True)
        return h, z


class NounPhraseModel(nn.Module):
    def __init__(self, n_var, tok_vocab, tag_vocab, typ_vocab):
        super().__init__()
        self.idx2tok, self.tok2idx = tok_vocab
        self.idx2tag, self.tag2idx = tag_vocab
        self.idx2typ, self.typ2idx = typ_vocab
        n_tok, n_tag, n_typ = len(self.idx2tok), len(self.idx2tag), len(self.idx2typ)

        d_x, d_h, n_lay = _, self.d_h, self.n_lay = FLAGS.seq_inp, FLAGS.seq_hidden_dim, FLAGS.seq_nlayers
        self.tag_encoder = LSTMEncoder(n_tag + 1, d_x, d_h, n_lay)
        self.noun_encoder = LSTMEncoder(n_tag, d_x, d_h, n_lay)
        self.np_encoder = LSTMEncoder(None, d_x, d_h, n_lay)
        self.tag2noun = nn.Linear(d_h, n_lay * d_h)
        self.noun2np = nn.Linear(n_lay * d_h, d_x)
        self.linear = nn.Linear(n_lay * d_h, n_var)

    def forward(self, seq_tag, seq_noun, seq_np, idx_np, pos_np, n_var, **kwargs):
        h_tag, z_tag = self.tag_encoder(seq_tag)
#       h_tag, _ = self.tag_encoder(seq_tag)
        h_tag = F.relu(self.tag2noun(h_tag[idx_np, pos_np]))
        h_tag = h_tag.view(-1, 2 * self.n_lay, self.d_h // 2).permute(1, 0, 2)
        _, z_noun = self.noun_encoder(seq_noun, h_tag)
        z_noun = z_noun.permute(1, 0, 2).reshape(z_noun.size(1), -1)
        z_noun = F.relu(self.noun2np(z_noun))
        _, z_np = self.np_encoder(utils.pack_as(z_noun[seq_np.data], seq_np))
        logit = self.linear(z_tag.permute(1, 0, 2).reshape(z_tag.size(1), -1))
#       logit = self.linear(z_np.permute(1, 0, 2).reshape(z_np.size(1), -1))
        ret = {}
        ret["acc"] = logit.max(1)[1].eq(n_var).float().mean()
        ret["loss"] = ret["nll"] = F.cross_entropy(logit, n_var)
        return ret, {}
