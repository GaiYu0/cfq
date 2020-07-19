import pickle

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


def get_data_loaders(args):
    class CFQDataset(Dataset):

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

        def __init__(self, idx, data):
            super().__init__()
            self.idx = idx
            self.data = dict(data.items())
            get = lambda k: data[k] if type(k) is str else k
            rag = lambda x, y: self.RaggedArray(get(x), np.cumsum(np.hstack([[0], get(y)])))
            self.data['seq'] = rag('seq', 'n_tok')
            self.data['isconcept'], self.data['isvariable'] = rag('isconcept', 'n_tok'), rag('isvariable', 'n_tok')
            self.data['n_idx'], self.data['idx'] = rag('n_idx', 'n'), rag(rag('idx', 'n_idx'), 'n')
            self.data['src'], self.data['dst'], self.data['rel'] = rag('src', 'm'), rag('dst', 'm'), rag('rel', 'm')

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, key):
            i = self.idx[key]
            return {k : v[i] for k, v in self.data.items()}

    def collate_fn(samples):
        b = {}

        max_len = max(len(s['seq']) for s in samples)
        pad = lambda k, p: torch.from_numpy(np.vstack([np.hstack([s[k], np.full(max_len - len(s[k]), p)]) for s in samples]))
        b['seq'] = pad('seq', tok2idx['[PAD]'])
        b['masks'] = {'isconcept' : pad('isconcept', False),
                      'isvariable' : pad('isvariable', False)}

        cat = lambda k: torch.from_numpy(np.hstack([s[k] for s in samples]))
        b['n'], b['n_idx'], b['idx'] = cat('n'), cat('n_idx'), cat('idx')
        b['m'], b['src'], b['dst'], b['rel'] = cat('m'), cat('src'), cat('dst'), cat('rel')

        return b

    data = np.load(args.data)
    split = np.load(args.split)
    train_dataset = CFQDataset(split['trainIdxs'], data)
    dev_dataset = CFQDataset(split['devIdxs'], data)
    test_dataset = CFQDataset(split['testIdxs'], data)

    _, tok2idx = pickle.load(open(args.vocab, 'rb'))
    kwargs = {'num_workers' : args.num_workers,
              'collate_fn' : collate_fn,
              'pin_memory' : True}
    train_data_loader = DataLoader(train_dataset, args.train_batch_size, drop_last=True, **kwargs)
    dev_data_loader = DataLoader(dev_dataset, args.eval_batch_size, **kwargs)
    test_data_loader = DataLoader(test_dataset, args.eval_batch_size, **kwargs)

    ntok = data['seq'].max() + 2  # [PAD]
    nrel = data['rel'].max() + 1

    return train_data_loader, dev_data_loader, test_data_loader, ntok, nrel
