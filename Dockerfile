FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04
MAINTAINER Jonathan Cook
RUN apt-get clean && apt-get update && DEBIAN_FRONTEND="noninteractive" TZ="Europe/London" apt install -y python3-pip git tmux openssh-client ffmpeg
RUN pip install --upgrade pip

RUN  pip install jax[cuda12_pip]==0.4.11 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html numpy matplotlib wandb flax orbax-checkpoint optax distrax chex gymnax hydra-core brax uuid ml_dtypes==0.2.0
ARG UID
RUN useradd -u $UID --create-home duser && \
    echo "duser:duser" | chpasswd && \
    adduser duser sudo

USER duser
WORKDIR /home/duser/SMAXbaselines
