FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
MAINTAINER Jonathan Cook
RUN apt-get clean && apt-get update && DEBIAN_FRONTEND="noninteractive" TZ="Europe/London" apt install -y python3-pip git tmux openssh-client ffmpeg
RUN pip install --upgrade pip

RUN  pip install jax[cuda11_cudnn82]==0.4.7 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html jaxlib==0.4.7+cuda11.cudnn82  numpy matplotlib wandb flax orbax-checkpoint optax distrax chex gymnax hydra-core brax uuid
ARG UID
RUN useradd -u $UID --create-home duser && \
    echo "duser:duser" | chpasswd && \
    adduser duser sudo

USER duser
WORKDIR /home/duser/SMAXbaselines