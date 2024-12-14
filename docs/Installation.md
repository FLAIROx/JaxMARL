# Installation

## Environments 🌍

Before installing, ensure you have the correct [JAX installation](https://github.com/google/jax#installation) for your hardware accelerator. We have tested up to JAX version 0.4.25. The JaxMARL environments can be installed directly from PyPi:

``` sh
pip install jaxmarl 
```

## Algorithms 🦉

If you would like to also run the algorithms, install the source code as follows:

1. Clone the repository:
    ``` sh
    git clone https://github.com/FLAIROx/JaxMARL.git && cd JaxMARL
    ```
2. Install requirements:
    ``` sh
    pip install -e .[algs] && export PYTHONPATH=./JaxMARL:$PYTHONPATH
    ```
3. For the fastest start, we reccoment using our Dockerfile, the usage of which is outlined below.

## Development

If you would like to run our test suite, install the additonal dependencies with:
 `pip install -e .[dev]`, after cloning the repository.


## Dockerfile 🐋

To help get experiments up and running we include a [Dockerfile](https://github.com/FLAIROx/JaxMARL/blob/main/Dockerfile) and its corresponding [Makefile](https://github.com/FLAIROx/JaxMARL/blob/main/Makefile). With Docker and the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) installed, the container can be built with:
``` sh
make build
```
The built container can then be run:
``` sh
make run
```