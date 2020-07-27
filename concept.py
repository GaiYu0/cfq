from argparse import ArgumentParser
from collections import defaultdict
import pickle

import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_sequence, PackedSequence
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from data import RaggedArray


def get_data_loaders(args):
    class CFQDataset(Dataset):

        def __init__(self, idx, data):
            super().__init__()
            self.idx = idx
            rag = lambda x, y: RaggedArray(data[x], np.cumsum(np.hstack([[0], data[y]])))
            self.seq = rag('seq', 'n_tok')
            self.con = rag('con', 'n_con')

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, key):
            idx = self.idx[key]
            return self.seq[idx], self.con[idx]

    def collate_fn(samples):
        seq, con = zip(*samples)
        seq = pack_sequence(list(map(torch.from_numpy, seq)), enforce_sorted=False)

        con_ = torch.zeros(len(con), ncon, dtype=torch.bool)
        for i, jj in enumerate(con):
            for j in jj:
                con_[i][j] = True

        return seq, con_

    _, tok2idx = pickle.load(open(args.vocab, 'rb'))
    ntok = len(tok2idx)

    data = np.load(args.data)
    ncon = max(data['con']) + 1

    split = np.load(args.split)
    train_dataset = CFQDataset(split['trainIdxs'], data)
    dev_dataset = CFQDataset(split['devIdxs'], data)
    test_dataset = CFQDataset(split['testIdxs'], data)

    kwargs = {'num_workers' : args.num_workers,
              'collate_fn' : collate_fn,
              'pin_memory' : True}
    train_data_loader = DataLoader(train_dataset, args.train_batch_size, shuffle=False, drop_last=True, **kwargs)
    dev_data_loader = DataLoader(dev_dataset, args.eval_batch_size, **kwargs)
    test_data_loader = DataLoader(test_dataset, args.eval_batch_size, **kwargs)

    return train_data_loader, dev_data_loader, test_data_loader, ntok, ncon


class Model(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.tok_encoder = nn.Embedding(args.ntok, args.ninp)
        self.lstm_encoder = nn.LSTM(args.ninp, args.nhid // 2, args.nlayer, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(args.nlayer * args.nhid, args.ncon)

    @staticmethod
    def apply(module, packed_seq):
        return PackedSequence(module(packed_seq.data), packed_seq.batch_sizes,
                              packed_seq.sorted_indices, packed_seq.unsorted_indices)

    def forward(self, seq, con):
        _, [h, _] = self.lstm_encoder(self.apply(self.tok_encoder, seq))
        logit = self.linear(h.permute(1, 0, 2).reshape(len(con), -1))
        d = {}
        d['loss'] = d['nll'] = -F.logsigmoid(logit[con]).mean() - (1 + 1e-5 - logit[~con].sigmoid()).log().mean()
        d['acc'] = logit.gt(0.5).eq(con).float().mean()
        return d, [logit]


def main(args):
    train_data_loader, dev_data_loader, test_data_loader, ntok, ncon = get_data_loaders(args)

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    args.ntok, args.ncon = ntok, ncon
    model = Model(args).to(device)
    opt = getattr(optim, args.optim)(model.parameters(), args.lr)

    def eval_(data_loader, desc, global_step=None, dump_pred=False):
        d = defaultdict(lambda: 0)
        y_true, y_pred = [], []
        for j, [seq, con] in enumerate(dev_data_loader):
            seq, con = seq.to(device), con.to(device)
            d_, [con_] = model(seq, con)
            y_true.append(con.cpu())
            y_pred.append(con_.cpu())
            for k, v in d_.items():
                d[k] += float(v)

        d = {k : v / (j + 1) for k, v in d.items()}
        '''
        y_true, y_pred = np.hstack(y_true), np.hstack(y_pred)
        d['p'], d['r'], d['f'], _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        '''

        print(f"[{i + 1}-{desc}]{' | '.join(f'{k}: {round(d[k], 3)}' for k in sorted(d))}")

        for k, v in d.items():
            writer.add_scalar(f'{desc} {k}', v, global_step)

    nbat = len(train_data_loader)
    writer = SummaryWriter(args.logdir)
    for i in range(args.num_epochs):
        for j, [seq, con] in enumerate(train_data_loader):
            seq, con = seq.to(device), con.to(device)
            d, _ = model(seq, con)
            opt.zero_grad()
            d['loss'].backward()
            opt.step()

            if (j + 1) % 100 == 0:
                print(f"[{i}-{j + 1}]{' | '.join(f'{k}: {round(float(d[k]), 3)}' for k in sorted(d))}")

            for k, v in d.items():
                writer.add_scalar(k, float(v), i * nbat + j + 1)

        eval_(train_data_loader, 'train', i + 1)
        eval_(dev_data_loader, 'dev', i + 1, dump_pred=True)

    eval_(test_data_loader, 'test', i + 1, dump_pred=True)

if __name__ == '__main__':
    parser = ArgumentParser()

    # data
    parser.add_argument('--data', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--train-batch-size', type=int)
    parser.add_argument('--eval-batch-size', type=int)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--num-workers', type=int)

    # model
    parser.add_argument('--model', type=str)
    parser.add_argument('--ninp', type=int)
    parser.add_argument('--nhead', type=int)
    parser.add_argument('--nhid', type=int)
    parser.add_argument('--nlayer', type=int)
    parser.add_argument('--dropout', type=float)

    # training
    parser.add_argument('--optim', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--logdir', type=str, default=None)

    args = parser.parse_args()

    main(args)
