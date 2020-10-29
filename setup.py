from setuptools import setup

setup(
    name='cfq',
    version='1.0',
    packages=["cfq"],
    python_requires=">=3.7",
    install_requires=[
        "absl-py",
        "tqdm",
        "loguru",

        # Data
        "matplotlib",
        "numpy",
        "pandas",
        "plotnine",
        "pyarrow",
        "seaborn",
        "scikit-learn",

        # PyTorch
        "datasets",
        "pytorch-lightning>=0.10.0",
        "sacremoses",
        "sentencepiece",
        "tokenizers",
        "torch>=1.6.0",
        "torchtext",
        "transformers>=3.3.1",
        "wandb",
        "torch-scatter>=2.0.5"
    ],
    extras_require={"test": ["pytest","ipython", "jupyter_console"]}
)
