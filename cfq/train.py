import pickle

from absl import app
from absl import flags

import numpy as np
from pathlib import Path

from cfq.model import Model
from cfq import DATA_DIR
from cfq.data import CFQDataset, CollateFunction

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_lightning as pl

FLAGS = flags.FLAGS
flags.DEFINE_integer("num_workers", 8, "Total number of workers.", lower_bound=1)
flags.DEFINE_integer("seed", 2, "Random seed.", lower_bound=0)
flags.DEFINE_enum("optimizer_name", "Adam", ["Adam", "SGD"], "Optimizer name.")
flags.DEFINE_integer("batch_size", 64, "Total batch size.", lower_bound=1)
flags.DEFINE_float("lr", 1e-3, "Learning rate.", lower_bound=0)
flags.DEFINE_integer("num_warmup_steps", 0, "Warmup steps to peak.", lower_bound=0)

flags.DEFINE_string("tok_vocab_path", str(DATA_DIR / "cfq" / "vocab.pickle"), "Token vocab path")
flags.DEFINE_string("rel_vocab_path", str(DATA_DIR / "cfq" / "rel-vocab.pickle"), "Rel. vocab path")
flags.DEFINE_enum(
    "cfq_split",
    "random_split",
    [
        "mcd1",
        "mcd2",
        "mcd3",
        "question_pattern_split",
        "question_complexity_split",
        "query_pattern_split",
        "query_complexity_split",
        "random_split",
    ],
    "CFQ data split to train with.",
)
flags.DEFINE_string("cfq_data_path", str(DATA_DIR / "cfq" / "data.npz"), "Main CFQ data path")
flags.DEFINE_string("cfq_split_data_dir", str(DATA_DIR / "cfq" / "splits"), "CFQ split data path directory.")


class CFQDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, tok_vocab, rel_vocab):
        super().__init__()
        self.batch_size = batch_size
        self.tok_vocab = tok_vocab
        self.rel_vocab = rel_vocab
        collate_fn = CollateFunction()
        self.data_kwargs = {"num_workers": FLAGS.num_workers, "collate_fn": collate_fn.collate_fn, "pin_memory": True}

    def setup(self, stage=None):
        _, tok2idx = self.tok_vocab
        _, rel2idx = self.rel_vocab
        data = np.load(FLAGS.cfq_data_path)
        split_data = np.load(Path(FLAGS.cfq_split_data_dir) / f"{FLAGS.split}.npz")
        self.train_dataset = CFQDataset(split_data["trainIdxs"], data)
        self.dev_dataset = CFQDataset(split_data["devIdxs"], data)
        self.test_dataset = CFQDataset(split_data["testIdxs"], data)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, **self.data_kwargs)

    def val_dataloader(self):
        return DataLoader(self.dev_dataset, batch_size=self.batch_size, **self.data_kwargs)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, **self.data_kwargs)


class CFQTrainer(pl.LightningModule):
    def __init__(self, tok_vocab, rel_vocab):
        super().__init__()
        self.model = Model(tok_vocab, rel_vocab)

    def forward(self, x):
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optim_def = getattr(torch.optim, FLAGS.optimzer_name)
        optimizer = optim_def(self.parameters(), lr=FLAGS.lr)
        return optimizer


def main(argv):
    pl.seed_everything(FLAGS.seed)

    # load vocab
    tok_vocab = pickle.load(open(FLAGS.tok_vocab_path, "rb"))
    rel_vocab = pickle.load(open(FLAGS.rel_vocab_path, "rb"))

    # load data
    data_module = CFQDataModule(FLAGS.batch_size, tok_vocab, rel_vocab)
    model = CFQTrainer(tok_vocab, rel_vocab)
    trainer = pl.Trainer()

    # train
    trainer.fit(model, datamodule=data_module)
    result = trainer.test(datamodule=data_module)
    print(result)


if __name__ == "__main__":
    app.run(main)
