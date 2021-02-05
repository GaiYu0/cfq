from absl import app
from absl import flags
from datetime import datetime
from loguru import logger
import numpy as np
import os
from pathlib import Path
import pickle
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support
import sys
import torch
import transformers

from cfq.model import AttentionModel as Model
from cfq import CFQ_SPLITS, DATA_DIR, RUN_DIR_ROOT
from cfq.data import CFQDataModule


FLAGS = flags.FLAGS
flags.DEFINE_enum("mode", "train", ["train", "test"], "Train or test an existing model")
flags.DEFINE_boolean("sweep_mode", False, "Set to true to enable wandb sweep mode.")
flags.DEFINE_boolean("debug", False, "Use pytorch-lighting quick smoke test (fast_dev_run).")
flags.DEFINE_bool("dump_test_pred", False, "Whether to dump test predictions.")

flags.DEFINE_enum("cfq_split", "random_split", CFQ_SPLITS, "CFQ data split to train with.")
flags.DEFINE_string("run_name", None, "Unique run ID")
flags.DEFINE_string("wandb_project", "cfq_pl_nopretrain", "wandb project for logging (cfq split automatically appended).")
flags.DEFINE_string("data_root_path", str(DATA_DIR), "Root base path")
flags.DEFINE_string("run_dir_root", str(RUN_DIR_ROOT), "Output run directory (root)")
flags.DEFINE_string("tok_vocab_path", "cfq/tok-vocab.pickle", "Token vocab path")
flags.DEFINE_string("tag_vocab_path", "cfq/tag-vocab.pickle", "Tag vocab path")
flags.DEFINE_string("typ_vocab_path", "cfq/rel-vocab.pickle", "Rel. vocab path")
flags.DEFINE_string("cfq_data_path", "cfq/data.npz", "Main CFQ data path")
flags.DEFINE_string("cfq_split_data_dir", "cfq/splits", "CFQ split data path directory.")

flags.DEFINE_string("gpus", "-1", "GPU assignment (from pytorch-lightning).")
flags.DEFINE_integer("num_workers", 8, "Total number of workers.", lower_bound=1)
flags.DEFINE_integer("seed", 2, "Random seed.", lower_bound=0)
flags.DEFINE_enum("precision", "32", ["32", "16"], "FP precision to use.")
flags.DEFINE_string("resume_from_checkpoint", None, "Path to checkpoint, if resuming.")
flags.DEFINE_integer("checkpoint_epoch_idx", -1, "Epoch ID for checkpoint (-1 = infer from checkpoint).")

# optimizer configuration
flags.DEFINE_enum("optimizer_name", "Adam", ["AdamW", "Adam", "SGD"], "Optimizer name.")
flags.DEFINE_integer("batch_size", 64, "Total batch size.", lower_bound=1)
flags.DEFINE_float("gradient_clip_val", 0, "Gradient scale value.")
flags.DEFINE_float("lr", 1e-3, "Learning rate.", lower_bound=0)
flags.DEFINE_integer("warmup_epochs", 5, "Warmup steps to peak, set to None to disable LR scheduler.", lower_bound=0)
flags.DEFINE_integer("num_epochs", 100, "Number of training epochs.", lower_bound=1)
flags.DEFINE_float("cosine_lr_period", 0.5, "Cosine learning rate schedule.", lower_bound=0)
#  0 = constant, 0.5 = cosine decay, 1.5 = two cycle cosine LR schedule

# global architecture flags
flags.DEFINE_float("gamma", 1.0, "Gamma.", lower_bound=0.0, upper_bound=1.0)
flags.DEFINE_float("w_pos", 1.0, "Positional weight.", lower_bound=0.0, upper_bound=1.0)
flags.DEFINE_float("dropout", 0.0, "Dropout value", lower_bound=0.0)
flags.DEFINE_enum("recurrent_layer", "LSTM", ["LSTM", "GRU"], "Recurrent unit to use for attention model.")

# sequence model flags
flags.DEFINE_enum("seq_model", "lstm", ["lstm", "transformer"], "Sequence model implementation.")
flags.DEFINE_integer("seq_inp", 64, "Sequence input dimension")
flags.DEFINE_integer("seq_hidden_dim", 256, "Sequence hidden dimension")
flags.DEFINE_integer("seq_nlayers", 2, "Sequence model depth")
flags.DEFINE_integer("seq_nhead", 0, "Transformer number of heads")

# neural tensor layer flags
flags.DEFINE_integer("ntl_inp", 64, "Neural tensor layer input dimension", lower_bound=0)
flags.DEFINE_integer("ntl_hidden_dim", 64, "Neural tensor layer hidden dimension")
flags.DEFINE_boolean("ntl_bilinear", True, "Sequence model is bilinear?")


class CFQTrainer(pl.LightningModule):
    def __init__(self, tok_vocab, tag_vocab, typ_vocab, last_epoch=-1):
        super().__init__()
        self.last_epoch = last_epoch
        self.model = Model(tok_vocab, tag_vocab, typ_vocab)
        self.out = {}

    def forward(self, x):
        raise NotImplementedError()

    def place_batch(self, batch):
        out = {}
        for k, v in batch.items():
            if type(v) is dict:
                out[k] = self.place_batch(v)
            elif type(v) is tuple:
                out[k] = [x.to(self.device) for x in v]
            else:
                out[k] = v.to(self.device)
        return out

    def compute_f1(self, rel_true, rel_pred):
        p, r, f1, _ = precision_recall_fscore_support(rel_true, rel_pred, average="macro")
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
        out_d["precision"], out_d["recall"], out_d["f1"] = self.compute_f1(
            out_dict["typ_true"].cpu().numpy(), out_dict["typ_pred"].cpu().numpy()
        )
        self.log_dict({"{}/{}".format(key, k): v for k, v in out_d.items()}, on_epoch=True, prog_bar=True)
        return out_d["emr"]

    def test_step(self, batch, batch_idx):
        with torch.no_grad():
            out_d, out_dict = self.model(**self.place_batch(batch))
        self.log_dict({"test/{}".format(k): v for k, v in out_d.items()}, on_epoch=True)

        if FLAGS.dump_test_pred:
            if self.out:
                self.out = {k : torch.cat([self.out[k], v.cpu()], 0) for k, v in out_dict.items()}
            else:
                self.out = {k : v.cpu() for k, v in out_dict.items()}

        return out_d["acc"]

    def configure_optimizers(self):
        optim_def = getattr(torch.optim, FLAGS.optimizer_name)
        optimizer = optim_def(self.parameters(), lr=FLAGS.lr)
        if FLAGS.warmup_epochs is None:
            return optimizer
        else:
            scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, FLAGS.warmup_epochs, FLAGS.num_epochs + 1, num_cycles=FLAGS.cosine_lr_period, last_epoch=self.last_epoch)
            return [optimizer], [scheduler]


def main(argv):
    pl.seed_everything(FLAGS.seed)
    print(FLAGS.run_dir_root, FLAGS.data_root_path)
    log_dir = Path(FLAGS.run_dir_root) / "{}_{}".format(FLAGS.run_name, str(datetime.now().strftime("%m%d%y_%H%M%S")))
    print(log_dir)
    logger.info(f"Saving logs to {log_dir}")
    log_dir.mkdir(parents=True, exist_ok=True)

    # load vocab
    tok_vocab = pickle.load(open(os.path.join(FLAGS.data_root_path, FLAGS.tok_vocab_path), "rb"))
    tag_vocab = pickle.load(open(os.path.join(FLAGS.data_root_path, FLAGS.tag_vocab_path), "rb"))
    typ_vocab = pickle.load(open(os.path.join(FLAGS.data_root_path, FLAGS.typ_vocab_path), "rb"))

    # load data
    data_module = CFQDataModule(FLAGS.batch_size, tok_vocab, tag_vocab, typ_vocab)
    data_module.setup()
    if FLAGS.resume_from_checkpoint is None:
        model = CFQTrainer(tok_vocab, tag_vocab, typ_vocab)
    else:
        ckpt = torch.load(FLAGS.resume_from_checkpoint)
        model = CFQTrainer.load_from_checkpoint(FLAGS.resume_from_checkpoint, tok_vocab=tok_vocab, tag_vocab=tag_vocab, typ_vocab=typ_vocab, last_epoch=ckpt['epoch'])

    # configure loggers and checkpointing
    if FLAGS.mode == "train":
        lr_logger = [LearningRateMonitor(logging_interval="step")]
        checkpoint_callback = ModelCheckpoint(monitor="valid/emr/dataloader_idx_0", save_top_k=5, save_last=True, mode="max")
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
        flag_save_file = log_dir / "flags.txt"
        with flag_save_file.open('w') as f:
            key_flags = FLAGS.get_key_flags_for_module(sys.argv[0])
            f.write('\n'.join(f.serialize() for f in key_flags))
        wandb_logger.experiment.save(str(flag_save_file.resolve()))
    else:
        lr_logger = []
        checkpoint_callback = None
        wandb_logger = None

    # create trainer
    trainer = pl.Trainer(
        # GPU training args
        gpus=FLAGS.gpus,
        accelerator="ddp" if torch.cuda.device_count() > 1 else None,
        benchmark=not FLAGS.debug,
        deterministic=True,
        # logging
        log_every_n_steps=10,
        flush_logs_every_n_steps=250,
        default_root_dir=log_dir,
        logger=wandb_logger,
        callbacks=lr_logger,
        checkpoint_callback=checkpoint_callback,
        # training flags
        max_epochs=FLAGS.num_epochs,
        gradient_clip_val=FLAGS.gradient_clip_val,
        fast_dev_run=FLAGS.debug,
        precision=int(FLAGS.precision),
    )

    # train
    if FLAGS.mode == "train":
        trainer.fit(model, datamodule=data_module)
    result = trainer.test(datamodule=data_module)
    logger.info(f"Result of evaluation: {result}")

    if FLAGS.dump_test_pred:
        np.savez(log_dir / "test_dict.npz", **{k : v.numpy() for k, v in model.out.items()})

if __name__ == "__main__":
    app.run(main)
