#!/bin/bash

# Check if nvcc is installed
if which nvcc > /dev/null 2>&1; then
    GPUS="--gpus all"
else
    GPUS=""
fi

# Set environment variables
MYUSER="myuser"
BASE_FLAGS="-it --rm -v ${PWD}:/home/${MYUSER} --shm-size 20G"
RUN_FLAGS="${GPUS} ${BASE_FLAGS}"
DOCKER_IMAGE_NAME="jaxmarl"
IMAGE="${DOCKER_IMAGE_NAME}:latest"

# Run the Docker container
docker run ${RUN_FLAGS} ${IMAGE} /bin/bash