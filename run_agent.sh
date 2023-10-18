#!/bin/bash
WANDB_API_KEY=$(cat $HOME/.oxwhirl_wandb_api_key)
script_and_args="${@:2}"
if [ $1 == "all" ]; then
    gpus="0 1 2 3 4 5 6 7"
else
    gpus=$1
fi
for gpu in $gpus; do
    echo "Launching container SMAX_benlis_$gpu on GPU $gpu"
    docker run \
        --gpus device=$gpu \
        -e WANDB_API_KEY=$WANDB_API_KEY \
        -e XLA_PYTHON_CLIENT_PREALLOCATE=false \
        -e TF_CUDNN_DETERMINISTIC=1 \
        -e PYTHONPATH=/home/duser/SMAXbaselines/baselines \
        -v $(pwd):/home/duser/SMAXbaselines \
        --name SMAXbaselines_${user}_${gpu} \
        --user $(id -u) \
        -d \
        --rm \
        -t smaxbaselines:benlis_minismac \
        /bin/bash -c "$script_and_args"
done