import argparse
from pathlib import Path
import os

from tqdm import tqdm

REMOTE_BASE = "https://cfq.s3.amazonaws.com"
DEFAULT_SHARED_BASE = Path("/work/paras/cfq/data").resolve()
DEFAULT_LOCAL_BASE = str((Path(__file__).parent.parent / "data").resolve())

def dl_cmds(dataset_path: str, extract=False, local_base=DEFAULT_LOCAL_BASE, shared_base=DEFAULT_SHARED_BASE):
    remote_path = os.path.join(REMOTE_BASE, dataset_path)
    cache_path = (shared_base / dataset_path).resolve()
    local_path = (local_base / dataset_path).resolve()
    local_path.parent.mkdir(parents=True, exist_ok=True)

    cmds = []
    if not local_path.exists():
        if cache_path.exists():
            cmds.append("rsync -avhW --no-compress --progress {} {}".format(cache_path, local_path))
        else:
            cmds.append("wget -nc -O {} {}".format(local_path, remote_path))
        if dataset_path.endswith(".tar.gz") and extract:
            cmds.append("(cd {} && tar -xzf {})".format(local_path.parent, local_path))
        elif dataset_path.endswith(".gz") and extract:
            cmds.append("gunzip -d -k {}".format(local_path))
    return cmds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ContraCode data")
    parser.add_argument("--path", type=str, default=DEFAULT_LOCAL_BASE, help="Path to save output to")
    args = parser.parse_args()
    local_path = Path(args.path)
    cmds = []

    cmds.extend(dl_cmds("cfq.tar.gz", True, local_path))
    print("\n".join(cmds))

    t = tqdm(cmds)
    for cmd in t:
        t.set_description("Running command: {}".format(cmd))
        t.refresh()
        os.system(cmd)