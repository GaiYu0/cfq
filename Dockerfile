FROM pytorch/pytorch:1.7.1-cuda11.0-cudnn8-runtime

# install common dependencies
RUN apt-get update && apt-get install -y --no-install-recommends sudo git bzip2 build-essential curl ca-certificates && rm -rf /var/lib/apt/lists/*

# install dependencies
ARG CUDA=cu110
RUN pip install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.7.0+${CUDA}.html
RUN pip install absl-py tqdm loguru certifi requests matplotlib numpy pandas plotnine pyarrow \
  seaborn scikit-learn datasets pytorch-lightning>=1.1.0 sacremoses sentencepiece torchtext \
  transformers wandb>=0.10.12

# Create a non-root user and switch to it
WORKDIR /app
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

COPY --chown=user:user . /app
RUN pip install -e .
RUN chmod -R 777 /app

VOLUME /data_cache
VOLUME /run_cache
CMD "/bin/bash"