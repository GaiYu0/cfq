import pickle

from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader


def get_data_loaders(args):
    class Dataset(Dataset):

        class RaggedArray:

            def __init__(self, data, indptr):
                self.data = data
                self.indptr = indptr

            def __getitem__(self, key):
                i = self.indptr[key]
                j = self.indptr[key + 1]
                return self.data[i : j]

        def __init__(self, idx, data):
            super().__init__()
            self.idx = idx
            self.data = data.copy()
            rag = lambda k, l: RaggedArray(data[k], np.hstack([0], data[l]))
            self.data['seq'] = rag('seq', 'n_tok')
            self.data['isconcept'], self.data['isvariable'] = rag('isconcept', 'n_tok'), rag('isvariable', 'n_tok')
            self.data['n_idx'], self.data['idx'] = rag('n_idx', 'n'), rag('idx', 'n_idx')
            self.data['src'], self.data['dst'], self.data['rel'] = rag('src', 'm'), rag('dst', 'm'), rag('rel', 'm')

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, key):
            i = self.idx[key]
            return {k : v[i] for k, v in self.data.items()}

    def collate_fn(samples):
        b = {}

        max_len = max(len(s['seq']) for s in samples)
        pad = lambda k, p: torch.tensor([s[k] + (max_len - len(s[k])) * [p] for s in samples])
        b['seq'] = pad('seq', tok2idx['[PAD]'])
        b['isconcept'] = pad('isconcept', False)
        b['isvariable'] = pad('isvariable', False)

        cat = lambda k: torch.tensor([s[k] for s in samples])
        b['n'], b['n_idx'], b['idx'] = cat('n'), cat('n_idx'), cat('idx')
        b['m'], b['src'], b['dst'], b['rel'] = cat('m'), cat('src'), cat('dst'), cat('rel')

        return b

    data = np.load(args.data)
    split = np.load(args.split)
    train_dataset = Dataset(split['train_idx'], data)
    dev_dataset = Dataset(split['dev_idx'], data)
    test_dataset = Dataset(split['test_idx'], data)

    _, tok2idx = pickle.load(open(vocab, 'rb'))
    kwargs = {'num_worker' : args.num_worker,
              'collate_fn' : collate_fn,
              'pin_memory' : True}
    train_data_loader = DataLoader(train_dataset, args.train_batch_size, drop_last=True, **kwargs)
    dev_data_loader = DataLoader(dev_dataset, args.eval_batch_size, **kwargs)
    test_data_loader = DataLoader(test_dataset, args.eval_batch_size, **kwargs)

    return train_data_loader, dev_data_loader, test_data_loader
