WANDB_API_KEY=$(cat $HOME/.oxwhirl_wandb_api_key)

docker run -it --gpus device=$1 \
    -v $(pwd):/home/duser/SMAXbaselines -e PYTHONPATH=/home/duser/SMAXbaselines/smaxbaselines \
    -e WANDB_API_KEY=$WANDB_API_KEY -e XLA_PYTHON_CLIENT_PREALLOCATE=false -e TF_CUDNN_DETERMINISTIC=1 smaxbaselines:benlis_minismac \
    /bin/bash
    # python3 smaxbaselines/IPPO/ippo_homogenous.py ${@:2}