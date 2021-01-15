def get_package_root():
    import os
    from pathlib import Path
    return Path(os.path.dirname(os.path.abspath(__file__))) / ".."


PACKAGE_ROOT = get_package_root()
DATA_DIR = get_package_root() / "data"
RUN_DIR_ROOT = get_package_root() / "data" / "runs"
CFQ_DATA_DIR = DATA_DIR / "cfq"
CFQ_SPLITS = [
    "mcd1",
    "mcd2",
    "mcd3",
    "question_pattern_split",
    "question_complexity_split",
    "query_pattern_split",
    "query_complexity_split",
    "random_split",
]
