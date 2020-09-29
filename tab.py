from argparse import Namespace
from functools import reduce
import glob
from multiprocessing import Pool
import pickle
import re
import time

import pandas as pd


attrs = [
    'split',
    'train_batch_size',
    'seq_model',
    'seq_ninp',
    'nhead',
    'seq_nhid',
    'seq_nlayer',
    'dropout',
    'lr',
]

splits = ['train', 'dev', 'test']
fields = ['acc', 'emr']

def mapper(f):
    lines = open(f).readlines()

    args = Namespace()
    for line in lines:
        if line.startswith('args.'):
            exec(line)

    if hasattr(args, 'seq2seq') and args.seq2seq:
        return []

    row = {'id' : f}
    row.update({attr : getattr(args, attr) for attr in attrs})
    row.update({split : {field : [] for field in fields} for split in splits})

    for line in lines:
        for split in splits:
            match = re.match(f'\\[[0-9]+-{split}\\]', line)
            if match:
                for field, value in zip(fields, line[match.end():].split(' | ')):
                    row[split][field].append(float(value.replace(f'{field}: ', '')))

    lens = reduce(set.intersection, [{len(row[split][field]) for field in fields} for split in splits])
    assert len(lens) == 1 and lens.pop() > 0, f

    for field in fields:
        row[f'train-{field}'] = max(row['train'][field])
        argmax = lambda xs: max(enumerate(xs), key=lambda x: x[0])
        index, row[f'dev-{field}'] = argmax(row['dev'][field])
        row[f'test-{field}'] = row['test'][field][index]

    return row

t = time.time()
with Pool() as pool:
    rows = filter(bool, pool.map(mapper, glob.glob('slurm-*.out')))
print(f'{round(time.time() - t)}s')

rows = sorted(rows, key=lambda row: row['id'])
pickle.dump(rows, open('tab.pickle', 'wb'))

pd.DataFrame([{k : v for k, v in row.items() if k not in splits} for row in rows]).to_pickle('df.pickle')
