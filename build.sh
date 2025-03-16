#!/bin/bash

# Check if nvcc is installed
if which nvcc > /dev/null 2>&1; then
    GPUS="--gpus all"
    USE_CUDA="true"
else
    GPUS=""
    USE_CUDA="false"
fi

# Set environment variables
MYUSER="myuser"
BASE_FLAGS="-it --rm -v ${PWD}:/home/${MYUSER} --shm-size 20G"
RUN_FLAGS="${GPUS} ${BASE_FLAGS}"
DOCKER_IMAGE_NAME="jaxmarl"
IMAGE="${DOCKER_IMAGE_NAME}:latest"
ID=$(id -u)

# Build the Docker image
DOCKER_BUILDKIT=1 docker build \
    --build-arg USE_CUDA=${USE_CUDA} \
    --build-arg MYUSER=${MYUSER} \
    --build-arg UID=${ID} \
    --tag ${IMAGE} \
    --progress=plain ${PWD}/.