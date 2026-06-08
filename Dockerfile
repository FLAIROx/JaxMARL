FROM nvcr.io/nvidia/jax:26.05-py3

# Create user
ARG UID=1000
ARG MYUSER=myuser
ARG INSTALL_ROBOTARIUM=false
RUN useradd -u $UID -o --create-home ${MYUSER}
USER ${MYUSER}

# default workdir
WORKDIR /home/${MYUSER}/
COPY --chown=${MYUSER} --chmod=755 . .

USER root

# install tmux
RUN apt-get update && \
    apt-get install -y --no-install-recommends tmux && \
    rm -rf /var/lib/apt/lists/*

#jaxmarl from source if needed, all the requirements
RUN pip install --no-cache-dir -e .[algs,dev]

RUN git config --global --add safe.directory /home/${MYUSER} && \
    if [ "$INSTALL_ROBOTARIUM" = "true" ]; then \
        git config --global --add safe.directory /home/${MYUSER}/jaxmarl/environments/robotarium && \
        git config --global --add safe.directory /home/${MYUSER}/jaxmarl/environments/robotarium/jaxrobotarium/robotarium_python_simulator && \
        git submodule update --init --recursive && \
        pip install --no-cache-dir -e jaxmarl/environments/robotarium && \
        pip install --no-cache-dir -e jaxmarl/environments/robotarium/jaxrobotarium/robotarium_python_simulator ; \
    fi

USER ${MYUSER}

ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV TF_FORCE_GPU_ALLOW_GROWTH=true

# Uncomment below if you want jupyter
# RUN pip install jupyterlab

#for secrets and debug
ENV WANDB_API_KEY=""
ENV WANDB_ENTITY=""
