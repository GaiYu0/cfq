from argparse import ArgumentParser, Namespace
from collections import defaultdict

from sklearn.metrics import precision_recall_fscore_support
import torch

from data import *
from model import *
from utils import *


def main(args_):
    args = Namespace()
    for line in open(args_.hist):
        if line.startswith('args.'):
            exec(line)

    [train_data_loader,
     dev_data_loader,
     test_data_loader,
     vocab, rel_vocab] = get_data_loaders(args)

    model = Model(args, vocab, rel_vocab).to(device)
    ckpt_path = f'{args_.input_dir}/{args.id}/{args_.epoch}.ckpt'
    model.load_state_dict(torch.load(ckpt_path)['model_state_dict'])

    for x in ['train', 'dev', 'test']:
        data_loader = locals()[f'{x}_data_loader']
        d, z = eval_(data_loader, model, ['n', 'cfq_idx'])
        d['p'], d['r'], d['f'], _ = precision_recall_fscore_support(z['rel_true'], z['rel_pred'], average='macro')
        print(f"[{args.id}][{args_.epoch}-{x}]{' | '.join(f'{k}: {round(d[k], 3)}' for k in sorted(d))}")
        np.savez(f'{args_.output_dir}/{args.id}/{args_.epoch}-{x}', **z)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--epoch', type=int)
    parser.add_argument('--hist', type=str)
    parser.add_argument('--input-dir', type=str)
    parser.add_argument('--output-dir', type=str)
    args = parser.parse_args()

    main(args)
