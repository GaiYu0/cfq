from itertools import takewhile
import pickle

import dgl
import numpy as np
import torch
from torch.nn.utils.rnn import pack_sequence
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


class RaggedArray:

    def __init__(self, data, indptr):
        self.data = data
        self.indptr = indptr

    def __getitem__(self, key):
        if isinstance(key, (int, np.integer)):
            return self.data[self.indptr[key] : self.indptr[key + 1]]
        elif type(key) is slice:
            start, stop, stride = key.indices(len(self.indptr))
            assert stride == 1
            return self.data[self.indptr[start] : self.indptr[stop]]
        else:
            raise RuntimeError()

    def __len__(self):
        return len(self.indptr) - 1


def get_data_loaders(args):
    class CFQDataset(Dataset):

        def __init__(self, idx, data):
            super().__init__()
            self.idx = idx
            get = lambda k: data[k] if type(k) is str else k
            rag = lambda x, y: RaggedArray(get(x), np.cumsum(np.hstack([[0], get(y)])))
            self.data = dict(data.items())
            self.data['seq'] = rag('seq', 'n_tok')
            self.data['isconcept'], self.data['isvariable'] = rag('isconcept', 'n_tok'), rag('isvariable', 'n_tok')
            self.data['n_idx'], self.data['idx'] = rag('n_idx', 'n'), rag(rag('idx', 'n_idx'), 'n')
            self.data['tok'] = rag('tok', 'n')
            self.data['src'], self.data['dst'], self.data['rel'] = rag('src', 'm'), rag('dst', 'm'), rag('rel', 'm')

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, key):
            i = self.idx[key]
            d = {k : v[i] for k, v in self.data.items()}
            d['cfq_idx'] = i
            return d

    def collate_fn(samples):
        b = {}

        max_len = max(len(s['seq']) for s in samples)
        pad = lambda k, p: torch.from_numpy(np.vstack([np.hstack([s[k], np.full(max_len - len(s[k]), p)]) for s in samples]))
        if args.seq_model == 'lstm':
            b['seq'] = pack_sequence([torch.from_numpy(s['seq']) for s in samples], enforce_sorted=False)
        elif args.seq_model == 'transformer':
            b['seq'] = pad('seq', tok2idx['[PAD]'])
            b['masks'] = {'isconcept' : pad('isconcept', False),
                          'isvariable' : pad('isvariable', False)}
        else:
            raise Exception()

        cat = lambda k: torch.from_numpy(np.hstack([s[k] for s in samples]))
        b['n'], b['n_idx'], b['idx'] = cat('n'), cat('n_idx'), cat('idx')
        b['m'], src, dst, b['rel'] = cat('m'), cat('src'), cat('dst'), cat('rel')

        offset = torch.tensor([0] + [s['n'] for s in samples[:-1]]).cumsum(0).repeat_interleave(b['m'])
        b['src'] = src + offset
        b['dst'] = dst + offset

        b['tok'] = cat('tok')
        b['cfq_idx'] = cat('cfq_idx')

        if args.gr:
            b['g'] = dgl.DGLGraph((b['src'], b['dst']))
            idx = torch.arange(len(b['seq'])).repeat_interleave(b['n']).repeat_interleave(b['n_idx'])
            b['g'].ndata['tok'] = b['tok']
            b['g'].edata['rel_type'] = b['rel']
            _, inverse, counts = torch.unique(torch.stack([b['dst'], b['rel']]), dim=1, return_inverse=True, return_counts=True)
            b['g'].edata['norm'] = counts.float().reciprocal().unsqueeze(1)[inverse]

        return b

    data = np.load(args.data)

    split = np.load(args.split)
    train_dataset = CFQDataset(split['trainIdxs'], data)
    dev_dataset = CFQDataset(split['devIdxs'], data)
    test_dataset = CFQDataset(split['testIdxs'], data)

    vocab = _, tok2idx = pickle.load(open(args.vocab, 'rb'))
    rel_vocab = pickle.load(open(args.rel_vocab, 'rb'))
    kwargs = {'num_workers' : args.num_workers,
              'collate_fn' : collate_fn,
              'pin_memory' : True}
    train_data_loader = DataLoader(train_dataset, args.train_batch_size, shuffle=False, drop_last=True, **kwargs)
    dev_data_loader = DataLoader(dev_dataset, args.eval_batch_size, **kwargs)
    test_data_loader = DataLoader(test_dataset, args.eval_batch_size, **kwargs)

    ntok = len(tok2idx)
    nrel = data['rel'].max() + 1

    return train_data_loader, dev_data_loader, test_data_loader, ntok, nrel, vocab, rel_vocab


def decode(b, vocab, rel_vocab):
    idx2tok, tok2idx = vocab
    idx2rel, rel2idx = rel_vocab
    seq = [' '.join(idx2tok[idx] for idx in takewhile(lambda idx: idx != tok2idx['[PAD]'], r)) for r in b['seq'].tolist()]
    '''
    for i, [src, rel, dst] in enumerate(zip(b['src'], b['rel'], b['dst'])):
        for src_, rel_, dst_ in zip(src.split(b['m']), rel.split(b['m']), dst.split(b['m'])):
    '''
