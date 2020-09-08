from argparse import ArgumentParser
from collections import defaultdict
from datetime import datetime
from itertools import islice
import os
import socket

import dgl
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tensorboardX import SummaryWriter
import torch
import torch.optim

from data import *
from model import *
from utils import *


def main(args):
    [train_data_loader,
     dev_data_loader,
     test_data_loader,
     vocab, rel_vocab] = get_data_loaders(args)

    model = Model(args, vocab, rel_vocab).to(device)
    lr = 0 if args.num_warmup_steps > 0 else args.lr
    optim = getattr(torch.optim, args.optim)(model.parameters(), lr)

    nbat = len(train_data_loader)
    writer = SummaryWriter(f'runs/{args.id}')
    os.makedirs(f'{args.output_dir}/{args.id}')
    for i in range(args.num_epochs):
        model.train()
        d = defaultdict(lambda: 0)
        for j, b in enumerate(train_data_loader):
            for param_group in optim.param_groups:
                if param_group['lr'] < args.lr:
                    param_group['lr'] += args.lr / args.num_warmup_steps

            place(b)
            d_, _ = model(**b)
            optim.zero_grad()
            d_['loss'].backward()
            optim.step()

            if (j + 1) % 10 == 0:
                print(f"[{i}-{j + 1}]{metrics(d_)}")

            for k, v in d_.items():
                writer.add_scalar(k, float(v), i * nbat + j + 1)

            for k, v in d_.items():
                d[k] += float(v)

        # TODO: running mean far from mean
        print(f"[{i + 1}]{metrics({k : v / (j + 1) for k, v in d.items()})}")

        for x in ['train', 'dev', 'test']:
            data_loader = locals()[f'{x}_data_loader']
            d, z = eval_(data_loader, model, ['n', 'cfq_idx'])
            d['p'], d['r'], d['f'], _ = precision_recall_fscore_support(z['rel_true'], z['rel_pred'], average='macro')

            print(f"[{i + 1}-{x}]{metrics(d)}")

            for k, v in d.items():
                writer.add_scalar(f'{x} {k}', v, i + 1)

            if args.save_pred:
                np.savez(f'{i + 1}-{x}', **z)

        if (i + 1) % 25 == 0:
            torch.save({'model_state_dict': model.state_dict(),
                        'optim_state_dict': optim.state_dict()}, f'{args.output_dir}/{args.id}/{i}.ckpt')


if __name__ == '__main__':
    parser = ArgumentParser()

    # data
    parser.add_argument('--input-dir', type=str)
    parser.add_argument('--output-dir', type=str)

    parser.add_argument('--data', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--train-batch-size', type=int)
    parser.add_argument('--eval-batch-size', type=int)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--rel-vocab', type=str)
    parser.add_argument('--num-workers', type=int)

    # model
    parser.add_argument('--seq-model', type=str)
    parser.add_argument('--seq-ninp', type=int)
    parser.add_argument('--nhead', type=int)
    parser.add_argument('--seq-nhid', type=int)
    parser.add_argument('--seq-nlayer', type=int)
    parser.add_argument('--dropout', type=float)

    parser.add_argument('--gr', action='store_true')
    parser.add_argument('--gr-model', type=str)
    parser.add_argument('--gr-ninp', type=int)
    parser.add_argument('--gr-nhid', type=int)
    parser.add_argument('--gr-nlayer', type=int)

    parser.add_argument('--ntl-ninp', type=int)
    parser.add_argument('--ntl-nhid', type=int)
    parser.add_argument('--bilinear', action='store_true')

    parser.add_argument('--w-pos', type=float)
    parser.add_argument('--gamma', type=float)

    # training
    parser.add_argument('--optim', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--num-warmup-steps', type=int)
    parser.add_argument('--save-pred', action='store_true')

    args = parser.parse_args()
    args.id = datetime.now().strftime('%Y-%m-%d@%H:%M:%S') + f"@{socket.gethostname()}#{os.getpid()}"
    for k, v in vars(args).items():
        print(f'args.{k} = {repr(v)}')

    main(args)
