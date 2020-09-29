from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_packed_sequence, PackedSequence


device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')


def apply(module, seq):
    return PackedSequence(module(seq.data), seq.batch_sizes, seq.sorted_indices, seq.unsorted_indices)


def place(b):
    for k, v in b.items():
        if type(v) is dict:
            place(v)
        else:
            b[k] = v.to(device)


def eval_(data_loader, model, keys):
    model.eval()
    d, z = defaultdict(lambda: 0), defaultdict(lambda: [])
    for j, b in enumerate(data_loader):
        place(b)
        with torch.no_grad():
            d_, z_ = model(**b)

        for k, v in d_.items():
            d[k] += float(v)

        for k, v in z_.items():
            z[k].append(v.cpu())

        for key in keys:
            z[key].append(b[key].cpu())

    d = {k : v / (j + 1) for k, v in d.items()}
    z = {k : torch.cat(v).numpy() for k, v in z.items()}

    return d, z


metrics = lambda d, n=3: ' | '.join(f'{k}: {round(float(d[k]), n)}' for k in sorted(d))
