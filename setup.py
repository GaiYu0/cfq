from setuptools import setup

setup(
    name='cfq',
    version='1.0',
    packages=["cfq"],
    python_requires=">=3.8",
    install_requires=[
        "absl-py",
        "tqdm",
        "loguru",
        "certifi",
        "requests",

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
        "pytorch-lightning>=1.1.0",
        "sacremoses",
        "sentencepiece",
        "torch>=1.7.0",
        "torchtext",
        "transformers",
        "wandb>=0.10.12",
        "torch-scatter>=2.0.5"
    ],
    extras_require={"test": ["pytest","ipython", "jupyter_console"]}
)
