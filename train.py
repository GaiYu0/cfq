from argparse import ArgumentParser

import numpy as np


def main(args):
    
if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--ntoken', type=int)
    parser.add_argument('--seq-ninp', type=int)
    parser.add_argument('--nhead', type=int)
    parser.add_argument('--seq-nhid', type=int)
    parser.add_argument('--seq-nlayer', type=int)
    parser.add_argument('--ntl-ninp', type=int)
    parser.add_argument('--ntl-nhid', type=int)
    parser.add_argument('--nrel', type=int)
    parser.add_argument('--dropout', type=float)
    args = parser.parse_args()

    main(args)
