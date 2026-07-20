FROM nvcr.io/nvidia/jax:26.04-py3

# Create user
ARG UID=1000
ARG MYUSER=myuser
RUN useradd -u $UID -o --create-home ${MYUSER}

# Install system deps before COPY so this layer is cached across source changes
RUN apt-get update && \
    apt-get install -y --no-install-recommends tmux && \
    rm -rf /var/lib/apt/lists/*

USER ${MYUSER}

# default workdir
WORKDIR /home/${MYUSER}/
COPY --chown=${MYUSER} . .

USER root

# Install jaxmarl and dependencies, pinning jax/jaxlib to the base image versions
RUN pip freeze | grep -iE '^(jax|jaxlib|jax-cuda13)' > /tmp/jax-pins.txt && \
    pip install --no-cache-dir --constraint /tmp/jax-pins.txt -e .[algs,dev] && \
    git config --global --add safe.directory /home/${MYUSER} && \
    git config --global --add safe.directory /home/${MYUSER}/jaxmarl/environments/robotarium && \
    git config --global --add safe.directory /home/${MYUSER}/jaxmarl/environments/robotarium/jaxrobotarium/robotarium_python_simulator && \
    git submodule update --init --recursive --depth 1 && \
    pip install --no-cache-dir --constraint /tmp/jax-pins.txt -e jaxmarl/environments/robotarium && \
    pip install --no-cache-dir --constraint /tmp/jax-pins.txt -e jaxmarl/environments/robotarium/jaxrobotarium/robotarium_python_simulator && \
    rm /tmp/jax-pins.txt

USER ${MYUSER}

ENV XLA_PYTHON_CLIENT_PREALLOCATE=false TF_FORCE_GPU_ALLOW_GROWTH=true

# Uncomment below if you want jupyter
# RUN pip install jupyterlab

ENV WANDB_API_KEY="" WANDB_ENTITY=""
