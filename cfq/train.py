from absl import app
from absl import flags
from datetime import datetime
from loguru import logger
import numpy as np
from pathlib import Path
import pickle
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support
import torch
from torch.utils.data import DataLoader
import transformers

from cfq.model import Model
from cfq import DATA_DIR, RUN_DIR_ROOT
from cfq.data import CFQDataset, CollateFunction


FLAGS = flags.FLAGS
flags.DEFINE_boolean("sweep_mode", False, "Set to true to enable wandb sweep mode.")
flags.DEFINE_string("run_dir_root", str(RUN_DIR_ROOT), "Output run directory (root)")
flags.DEFINE_string("run_dir_name", None, "Name of the run dir (defaults to run_name).")
flags.DEFINE_string("run_name", None, "Unique run ID")
flags.DEFINE_string("wandb_project", "cfq_pl_nopretrain", "wandb project for logging (cfq split automatically appended).")

flags.DEFINE_string("gpus", "-1", "GPU assignment (from pytorch-lightning).")
flags.DEFINE_integer("num_workers", 8, "Total number of workers.", lower_bound=1)
flags.DEFINE_integer("seed", 2, "Random seed.", lower_bound=0)
flags.DEFINE_enum("precision", "32", ["32", "16"], "FP precision to use.")
flags.DEFINE_string("resume_from_checkpoint", None, "Path to checkpoint, if resuming.")
flags.DEFINE_boolean("debug", False, "Use pytorch-lighting quick smoke test (fast_dev_run).")

flags.DEFINE_enum("optimizer_name", "Adam", ["Adam", "SGD"], "Optimizer name.")
flags.DEFINE_integer("batch_size", 64, "Total batch size.", lower_bound=1)
flags.DEFINE_float("gradient_clip_val", 0.0, "Gradient scale value.")
flags.DEFINE_float("lr", 1e-3, "Learning rate.", lower_bound=0)
flags.DEFINE_integer("warmup_epochs", 5, "Warmup steps to peak, set to None to disable LR scheduler.", lower_bound=0)
flags.DEFINE_integer("num_epochs", 100, "Number of training epochs.", lower_bound=1)
flags.DEFINE_float("cosine_lr_period", 0.5, "Cosine learning rate schedule.", lower_bound=0)
#  0 = constant, 0.5 = cosine decay, 1.5 = two cycle cosine LR schedule

flags.DEFINE_string("tok_vocab_path", str(DATA_DIR / "cfq" / "vocab.pickle"), "Token vocab path")
flags.DEFINE_string("rel_vocab_path", str(DATA_DIR / "cfq" / "rel-vocab.pickle"), "Rel. vocab path")
flags.DEFINE_string("cfq_data_path", str(DATA_DIR / "cfq" / "data.npz"), "Main CFQ data path")
flags.DEFINE_string("cfq_split_data_dir", str(DATA_DIR / "cfq" / "splits"), "CFQ split data path directory.")

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


class CFQDataModule(pl.LightningDataModule):
    def __init__(self, batch_size: int, tok_vocab, rel_vocab):
        super().__init__()
        self.batch_size = batch_size
        self.tok_vocab = tok_vocab
        self.rel_vocab = rel_vocab
        collate_fn = CollateFunction(tok_vocab, rel_vocab)
        self.data_kwargs = {"num_workers": FLAGS.num_workers, "collate_fn": collate_fn.collate_fn, "pin_memory": True}

    def setup(self, stage=None):
        _, tok2idx = self.tok_vocab
        _, rel2idx = self.rel_vocab
        data = np.load(FLAGS.cfq_data_path)
        split_data_dir_path = Path(FLAGS.cfq_split_data_dir) / f"{FLAGS.cfq_split}.npz"
        logger.info(f"Loading data from {split_data_dir_path}")
        split_data = np.load(split_data_dir_path)
        self.train_dataset = CFQDataset(split_data["trainIdxs"], data, self.tok_vocab, self.rel_vocab)
        self.dev_dataset = CFQDataset(split_data["devIdxs"], data, self.tok_vocab, self.rel_vocab)
        self.test_dataset = CFQDataset(split_data["testIdxs"], data, self.tok_vocab, self.rel_vocab)

    def train_step_count_per_epoch(self):
        return len(self.train_dataset)

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
        raise NotImplementedError()

    def place_batch(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

    def compute_f1(self, rel_true, rel_pred):
        p, r, f1, _ = precision_recall_fscore_support(rel_true, rel_pred, average='macro')
        return p, r, f1

    def training_step(self, batch, batch_idx):
        placed_batch = self.place_batch(batch)
        out_d, out_dict = self.model(**placed_batch)
        self.log_dict({"train/{}".format(k): v for k, v in out_d.items()})
        return out_d["loss"]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        key = "valid" if dataloader_idx == 0 else "test"
        with torch.no_grad():
            out_d, out_dict = self.model(**self.place_batch(batch))
        out_d['precision'], out_d['recall'], out_d['f1'] = self.compute_f1(out_dict['rel_true'].cpu().numpy(), out_dict['rel_pred'].cpu().numpy())
        self.log_dict({"{}/{}".format(key, k): v for k, v in out_d.items()}, on_epoch=True, prog_bar=True)
        return out_d["emr"]

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            out_d, out_dict = self.model(**self.place_batch(batch))
        self.log_dict({"test/{}".format(k): v for k, v in out_d.items()}, on_epoch=True)
        return out_d["acc"]

    def configure_optimizers(self):
        optim_def = getattr(torch.optim, FLAGS.optimizer_name)
        optimizer = optim_def(self.parameters(), lr=FLAGS.lr)
        if FLAGS.warmup_epochs is None:
            return optimizer
        else:
            scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, FLAGS.warmup_epochs, FLAGS.num_epochs + 1)
            return [optimizer], [scheduler]


def main(argv):
    pl.seed_everything(FLAGS.seed)
    rundir_name = FLAGS.run_dir_name if FLAGS.run_dir_name is not None else FLAGS.run_name
    timestamp = str(datetime.now().strftime("%m%d%y_%H%M%S"))
    log_dir = Path(FLAGS.run_dir_root) / "{}_{}".format(rundir_name, timestamp)
    logger.info(f"Saving logs to {log_dir}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # load vocab
    tok_vocab = pickle.load(open(FLAGS.tok_vocab_path, "rb"))
    rel_vocab = pickle.load(open(FLAGS.rel_vocab_path, "rb"))

    # load data
    data_module = CFQDataModule(FLAGS.batch_size, tok_vocab, rel_vocab)
    model = CFQTrainer(tok_vocab, rel_vocab)

    # configure loggers and checkpointing
    checkpoint_callback = ModelCheckpoint(monitor="valid/emr", save_top_k=5, save_last=True, mode="max")
    lr_logger = LearningRateMonitor(logging_interval="step")
    wandb_logger = WandbLogger(
        entity="cfq",
        project=f"{FLAGS.wandb_project}_{FLAGS.cfq_split}",
        name=FLAGS.run_name if not FLAGS.sweep_mode else None,  # wandb will autogenerate a sweep name
        save_dir=str(FLAGS.run_dir_root),
        log_model=True,
    )
    wandb_logger.watch(model, log="all", log_freq=100)
    for k, v in FLAGS.flag_values_dict().items():
        model.hparams[k] = v

    # create trainer
    trainer = pl.Trainer(
        # GPU training args
        gpus=FLAGS.gpus,
        accelerator="ddp" if torch.cuda.device_count() > 1 else None,
        benchmark=not FLAGS.debug,
        # logging
        log_every_n_steps=10,
        flush_logs_every_n_steps=250,
        default_root_dir=log_dir,
        logger=wandb_logger,
        callbacks=[lr_logger],
        checkpoint_callback=checkpoint_callback,
        # training flags
        max_epochs=FLAGS.num_epochs,
        gradient_clip_val=FLAGS.gradient_clip_val,
        fast_dev_run=FLAGS.debug,
        precision=int(FLAGS.precision),
    )

    # train
    trainer.fit(model, datamodule=data_module)
    result = trainer.test(datamodule=data_module)
    print(result)


if __name__ == "__main__":
    app.run(main)
