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
        "pytorch-lightning>=0.9.0",
        "sacremoses",
        "sentencepiece",
        "tensorflow-gpu>=2.1.0",
        "tokenizers",
        "torch",
        "torchtext",
        "transformers>=3.2.0",
        "wandb",
    ],
    extras_require={"test": ["pytest"]}
)
