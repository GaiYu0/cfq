from argparse import ArgumentParser

import numpy as np
import torch.nn.optim as optim

from data import *
from model import *


def main(args):
    [train_data_loader,
     dev_data_loader,
     test_data_loader] = get_data_loaders(args.vocab)

    model = Model(args)
    opt = getattr(optim, args.optim)(model.parameters(), args.lr)

    for i in range(args.num_epochs):
        for j, b in enumerate(train_data_loader):
            d = model(**b)
            opt.zero_grad()
            d['logp'].neg().backward()
            opt.step()
            print(f"[{i}-{j}]{' | '.join(round(d[k].item(), 3) for k in sorted(d))}")

        for b in enumerate(dev_data_loader):
            model(**b)


if __name__ == '__main__':
    parser = ArgumentParser()

    # data
    parser.add_argument('--data', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--train-batch-size', type=int)
    parser.add_argument('--eval-batch-size', type=int)
    parser.add_argument('--num-workers', type=int)

    # model
    parser.add_argument('--ntoken', type=int)
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

    args = parser.parse_args()

    main(args)
