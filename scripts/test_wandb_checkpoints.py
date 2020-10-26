import json
import subprocess

from absl import app
from absl import flags
from loguru import logger
from pathlib import Path
from tqdm import tqdm
import wandb


FLAGS = flags.FLAGS
flags.DEFINE_string('entity', 'cfq', 'wandb entity ID')
flags.DEFINE_string('project', None, 'wandb project ID')
flags.DEFINE_string('run_id', None, 'wandb run ID')
flags.mark_flag_as_required('project')
flags.mark_flag_as_required('run_id')


def download_all_runs(entity, project, run_id):
    root = Path("/tmp/cfq_checkpoints")
    download_dir = root / project / run_id
    download_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    checkpoints = [file for file in run.files() if file.name.endswith(".ckpt")]
    out_files = []
    for file in tqdm(checkpoints, desc="Download checkpoints"):
        file.download(root=str(root.resolve()), replace=True)
        out_files.append(root / file.name)
    return run.config, out_files


def main(unused_argv):
    root_dir = str((Path(__file__).parent.absolute() / '..' / 'cfq' / 'train.py').resolve())
    skipped_hparams = ['mode', 'resume_from_checkpoint']
    config, outputs = download_all_runs(FLAGS.entity, FLAGS.project, FLAGS.run_id)
    results = {}
    for checkpoint in tqdm(outputs, desc="Evaluating checkpoints"):
        logger.info(f"Evaluating checkpoint {checkpoint}")
        args = ' '.join([f"--{k} {v}" for k, v in config.items() if k not in skipped_hparams])
        cmd = f"python3 {root_dir} --mode test --resume_from_checkpoint '{checkpoint}' {args}"
        results[checkpoint] = subprocess.check_output(cmd, shell=True)
    with open('/tmp/results_out.json', 'wb') as f:
        json.dump(results, f)

if __name__ == "__main__":
    app.run(main)