def get_package_root():
    import os
    from pathlib import Path

    return Path(os.path.dirname(os.path.abspath(__file__))) / ".."


PACKAGE_ROOT = get_package_root()
DATA_DIR = get_package_root() / "data"
RUN_DIR_ROOT = get_package_root() / "data" / "runs"
CFQ_DATA_DIR = DATA_DIR / "webq"
