import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader

from data import RaggedArray


def get_data_loaders(args):
    class CFQDataset(Dataset):

        def __init__(self, idx, data):
            super().__init__()
            self.idx = idx
            self.seq = RaggedArray(data['seq'], np.cumsum(np.hstack([[0], data['n_tok']])))
            self.nvar = data['nvar']

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, key):
            idx = self.idx[key]
            return self.seq[idx], self.nvar[idx]


    def collate_fn(samples):
        seq, nvar = zip(*samples)
        max_len = max(map(len, seq))
        seq = torch.from_numpy(np.vstack([np.hstack([s, np.full(max_len - len(s), tok2idx['[PAD]'])]) for s in seq]))
        nvar = torch.tensor(nvar)
        return seq, nvar

    _, tok2idx = pickle.load(open(args.vocab, 'rb'))
    ntok = len(tok2idx)

    data = np.load(args.data)
    max_nvar = max(data['nvar'])

    split = np.load(args.split)
    train_dataset = CFQDataset(split['trainIdxs'], data)
    dev_dataset = CFQDataset(split['devIdxs'], data)
    test_dataset = CFQDataset(split['testIdxs'], data)

    kwargs = {'num_workers' : args.num_workers,
              'collate_fn' : collate_fn,
              'pin_memory' : True}
    train_data_loader = DataLoader(train_dataset, args.train_batch_size, shuffle=True, drop_last=True, **kwargs)
    dev_data_loader = DataLoader(dev_dataset, args.eval_batch_size, **kwargs)
    test_data_loader = DataLoader(test_dataset, args.eval_batch_size, **kwargs)

    return train_data_loader, dev_data_loader, test_data_loader, ntok, max_nvar


class Model(nn.Module):

    def __init__(self, ntok, ninp, nhid, max_nvar):
        super().__init__()
        self.tok_encoder = nn.Embedding(ntok, ninp)
        self.lstm_encoder = nn.LSTM(ninp, nhid // 2, nlayer, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(nhid, max_nvar)

    def forward(self, seq, nvar):
        logit = self.linear(self.lstm_encoder(self.tok_encoder(seq))[0])
        d['loss'] = d['nll'] = -logit.log_softmax(1).gather(1, nvar.unsqueeze(1)).sum() / len(seq)
        _, y_pred = logit.max(1)
        d['acc'] = pred.eq(nvar).float().mean()
        return d, [y_pred]


def main(args):

    def place(b):
        for k, v in b.items():
            if isinstance(v, (torch.Tensor, dgl.DGLGraph)):
                b[k] = v.to(device)
            elif type(v) is dict:
                place(v)
            else:
                raise TypeError()


    def eval_(data_loader, desc, global_step=None, dump_pred=False):
        d = defaultdict(lambda: 0)
        y_true, y_pred = [], []
        for j, [] in enumerate(dev_data_loader):
            place(b)
            d_, [y_true_, y_pred_, src_, dst_] = model(**b)
            y_true.append(y_true_)
            y_pred.append(y_pred_)
            for k, v in d_.items():
                d[k] += float(v)

            if dump_pred:
                src.append(src_)
                dst.append(dst_)

        d = {k : v / (j + 1) for k, v in d.items()}
        rel_true, rel_pred = np.hstack(rel_true), np.hstack(rel_pred)
        d['p'], d['r'], d['f'], _ = precision_recall_fscore_support(rel_true, rel_pred, average='macro')

        print(f"[{i + 1}-{desc}]{' | '.join(f'{k}: {round(d[k], 3)}' for k in sorted(d))}")

        for k, v in d.items():
            writer.add_scalar(f'{desc} {k}', v, global_step)

        if dump_pred:
            np.savez(f'{desc}-{global_step}', rel_true=rel_true, rel_pred=rel_pred, src=np.hstack(src), dst=np.hstack(dst))


    train_data_loader, dev_data_loader, test_data_loader, ntok, max_nvar = get_data_loaders(args)

    nbat = len(train_data_loader)
    writer = SummaryWriter(args.logdir)
    for i in range(args.num_epochs):
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

        eval_(train_data_loader, 'train', i + 1)
        eval_(dev_data_loader, 'dev', i + 1, dump_pred=True)


if __name__ == '__main__':
    # data
    parser.add_argument('--data', type=str)
    parser.add_argument('--split', type=str)
    parser.add_argument('--train-batch-size', type=int)
    parser.add_argument('--eval-batch-size', type=int)
    parser.add_argument('--vocab', type=str)
    parser.add_argument('--num-workers', type=int)

    # model
    parser.add_argument('--ninp', type=int)
    parser.add_argument('--nhid', type=int)
    parser.add_argument('--nlayer', type=int)

    main(args)
