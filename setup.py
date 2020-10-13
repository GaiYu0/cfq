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
        "tensorflow-gpu>=2.1.0",
        "tokenizers",
        "torch>=1.6.0",
        "torchtext",
        "transformers>=3.3.1",
        "wandb",
    ],
    extras_require={"test": ["pytest"]}
)
