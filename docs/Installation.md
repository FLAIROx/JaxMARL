# Installation

## Environments 🌍

Before installing, ensure you have JAX installed for your hardware:

GPU (CUDA 13):
```sh
pip install "jax[cuda13]"
```
CPU only:
```sh
pip install jax
```
See the [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html) for TPU and other configurations.

The JaxMARL environments can then be installed directly from PyPi:

``` bash
pip install jaxmarl 
```

## Algorithms 🦉

If you would like to also run the algorithms, install the source code as follows:

``` bash
git clone https://github.com/FLAIROx/JaxMARL.git && cd JaxMARL
pip install -e .[algs]
```

For the fastest start, **we recommend using our Dockerfile**, the usage of which is outlined below.

## Development 🔨

If you would like to run our test suite, install the additonal dependencies as follows after cloning the repository:
``` sh
pip install -e .[dev]
```

## Dockerfile 🐋

To help get experiments up and running we include a [Dockerfile](https://github.com/FLAIROx/JaxMARL/blob/main/Dockerfile) and its corresponding [Makefile](https://github.com/FLAIROx/JaxMARL/blob/main/Makefile). With Docker and the [Nvidia Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/index.html) installed, the container can be built with:
``` sh
make build
```
The built container can then be run:
``` sh
make run
```