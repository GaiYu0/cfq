from argparse import ArgumentParser
from collections import defaultdict
from itertools import islice

import dgl
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from tensorboardX import SummaryWriter
from torch.nn.utils.rnn import PackedSequence
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
            if isinstance(v, (torch.Tensor, PackedSequence, dgl.DGLGraph)):
                b[k] = v.to(device)
            elif type(v) is dict:
                place(v)
            else:
                raise TypeError()

    def eval_(data_loader, desc, global_step=None, dump_pred=False):
        d = defaultdict(lambda: 0)
        rel_true, rel_pred = [], []
        if dump_pred:
            m, cfq_idx, src, dst = [], [], [], []
        for j, b in enumerate(dev_data_loader):
            place(b)
            d_, [rel_true_, rel_pred_, src_, dst_] = model(**b)
            rel_true.append(rel_true_)
            rel_pred.append(rel_pred_)
            for k, v in d_.items():
                d[k] += float(v)

            if dump_pred:
                cfq_idx.append(b['cfq_idx'].cpu().numpy())
                m.append(b['m'].cpu().numpy())
                src.append(src_)
                dst.append(dst_)

        d = {k : v / (j + 1) for k, v in d.items()}
        rel_true, rel_pred = np.hstack(rel_true), np.hstack(rel_pred)
        d['p'], d['r'], d['f'], _ = precision_recall_fscore_support(rel_true, rel_pred, average='macro')

        print(f"[{i + 1}-{desc}]{' | '.join(f'{k}: {round(d[k], 3)}' for k in sorted(d))}")

        for k, v in d.items():
            writer.add_scalar(f'{desc} {k}', v, global_step)

        if dump_pred:
            np.savez(f'{desc}-{global_step}', rel_true=rel_true, rel_pred=rel_pred, src=np.hstack(src), dst=np.hstack(dst), m=np.hstack(m), cfq_idx=np.hstack(cfq_idx))

    nbat = len(train_data_loader)
    writer = SummaryWriter(args.logdir)
    for i in range(args.num_epochs):
        acc, emr = 0, 0
        for j, b in enumerate(train_data_loader):
            place(b)
            d, _ = model(**b)
            opt.zero_grad()
            d['loss'].backward()
            opt.step()

            if (j + 1) % 10 == 0:
                print(f"[{i}-{j + 1}]{' | '.join(f'{k}: {round(float(d[k]), 3)}' for k in sorted(d))}")

            for k, v in d.items():
                writer.add_scalar(k, float(v), i * nbat + j + 1)

            acc += float(d['acc'])
            emr += float(d['emr'])

        print(acc / (j + 1), emr / (j + 1))

        eval_(train_data_loader, 'train', i + 1)
#       eval_(dev_data_loader, 'dev', i + 1, dump_pred=True)
        eval_(test_data_loader, 'test', i + 1, dump_pred=True)

#   eval_(test_data_loader, 'test', i + 1, dump_pred=True)


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

    parser.add_argument('--gamma', type=float)

    # training
    parser.add_argument('--optim', type=str)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--num-epochs', type=int)
    parser.add_argument('--logdir', type=str, default=None)

    args = parser.parse_args()

    main(args)
