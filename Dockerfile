FROM nvcr.io/nvidia/jax:23.10-py3

# default workdir
WORKDIR /home/workdir
COPY . .

#jaxmarl from source if needed, all the requirements
RUN pip install -e .

#disabling preallocation
RUN export XLA_PYTHON_CLIENT_PREALLOCATE=false
#safety measures
RUN export XLA_PYTHON_CLIENT_MEM_FRACTION=0.25 
RUN export TF_FORCE_GPU_ALLOW_GROWTH=true

# if you want jupyter 
RUN pip install pip install jupyterlab

#for secrets and debug
ENV WANDB_API_KEY=""
ENV WANDB_ENTITY=""
RUN git config --global --add safe.directory /home/workdir