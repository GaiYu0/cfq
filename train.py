from argparse import ArgumentParser
from collections import defaultdict
from itertools import islice

import numpy as np
from tensorboardX import SummaryWriter
import torch.optim as optim

from data import *
from model import *


def main(args):
    [train_data_loader,
     dev_data_loader,
     test_data_loader,
     ntok, nrel, vocab, rel_vocab] = get_data_loaders(args)

    args.ntoken = ntok
    args.nrel = nrel
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = Model(args, vocab, rel_vocab).to(device)
    opt = getattr(optim, args.optim)(model.parameters(), args.lr)

    def place(b):
        for k, v in b.items():
            if isinstance(v, torch.Tensor):
                b[k] = v.to(device)
            elif type(v) is dict:
                place(v)
            else:
                raise TypeError()

    def eval_(data_loader, desc, global_step=None):
        d = defaultdict(lambda: 0)
        y_true, y_pred = [], []
        for j, b in enumerate(dev_data_loader):
            place(b)
            d_, [y_true_, y_pred_] = model(**b)
            y_true.append(y_true_)
            y_pred.append(y_pred_)
            for k, v in d_.items():
                d[k] += float(v)

        d = {k : v / (j + 1) for k, v in d.items()}
        d['p'], d['r'], d['f'], _ = precision_recall_fscore_support(np.hstack(y_true), np.hstack(y_pred), average='macro')

        print(f"[{i + 1}-{desc}]{' | '.join(f'{k}: {round(d[k], 3)}' for k in sorted(d))}")

        for k, v in d.items():
            writer.add_scalar(f'{desc} {k}', v, global_step)

    nbat = len(train_data_loader)
    writer = SummaryWriter(args.logdir)
    for i in range(args.num_epochs):
        for j, b in enumerate(train_data_loader):
            place(b)
            d, _ = model(**b)
            opt.zero_grad()
            d['logp'].neg().backward()
            opt.step()

            if (j + 1) % 10 == 0:
                print(f"[{i}-{j + 1}]{' | '.join(f'{k}: {round(float(d[k]), 3)}' for k in sorted(d))}")

            for k, v in d.items():
                writer.add_scalar(k, float(v), i * nbat + j + 1)

        eval_(train_data_loader, 'train', i + 1)
        eval_(dev_data_loader, 'dev', i + 1)
        eval_(test_data_loader, 'test', i + 1)


if __name__ == '__main__':
    parser = ArgumentParser()

    # data
    parser.add_argument('--data', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--train-batch-size', type=int)
    parser.add_argument('--eval-batch-size', type=int)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--rel-vocab', type=str)
    parser.add_argument('--num-workers', type=int)

    # model
    parser.add_argument('--seq-ninp', type=int)
    parser.add_argument('--nhead', type=int)
    parser.add_argument('--seq-nhid', type=int)
    parser.add_argument('--seq-nlayer', type=int)
    parser.add_argument('--ntl-ninp', type=int)
    parser.add_argument('--ntl-nhid', type=int)
    parser.add_argument('--nrel', type=int)
    parser.add_argument('--dropout', type=float)

    # training
    parser.add_argument('--optim', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--logdir', type=str, default=None)

    args = parser.parse_args()

    main(args)
