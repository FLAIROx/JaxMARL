#!/bin/bash

WANDB_API_KEY=$(cat $HOME/.oxwhirl_wandb_api_key)

script_and_args="${@:2}"

if [ $1 == "all" ]; then
    gpus="0 1 2 3 4 5 6 7"
else
    gpus=$1
fi

for gpu in $gpus; do
    echo "Launching container"
    docker run \
        --gpus device=$gpu \
        -e WANDB_API_KEY=$WANDB_API_KEY \
        -v $(pwd):/home/workdir \
        --shm-size 20G \
        --rm \
        -d \
        -t jaxmarl:latest \
        /bin/bash -c "$script_and_args"
done
