NVIDIA_SMI_RESULT := $(shell which nvidia-smi 2> NULL; rm NULL)
NVIDIA_SMI_TEST := $(notdir $(NVIDIA_SMI_RESULT))
ifeq ($(NVIDIA_SMI_TEST),nvidia-smi)
GPUS=--gpus all
else
GPUS=
endif


# Set flag for docker run command
MYUSER=myuser
BASE_FLAGS=-it --rm -v ${PWD}:/home/$(MYUSER) --shm-size 20G
RUN_FLAGS=$(GPUS) $(BASE_FLAGS)

DOCKER_IMAGE_NAME = jaxmarl
IMAGE = $(DOCKER_IMAGE_NAME):latest
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE)
DOCKER_RUN_DETACHED=docker run -d --rm $(GPUS) -v ${PWD}:/home/$(MYUSER) --shm-size 20G -e WANDB_API_KEY $(IMAGE)
USE_CUDA = $(if $(GPUS),true,false)
ID = $(shell id -u)

.PHONY: build run test local-test workflow-test run-baseline-set

# make file commands
build:
	DOCKER_BUILDKIT=1 docker build --build-arg USE_CUDA=$(USE_CUDA) --build-arg MYUSER=$(MYUSER) --build-arg UID=$(ID) --tag $(IMAGE) --progress=plain ${PWD}/.

run:
	$(DOCKER_RUN) /bin/bash

test:
	$(DOCKER_RUN) /bin/bash -c "pytest ./tests/"

local-test:
	pytest ./tests/ -v

workflow-test:
	# without -it flag; JAX_PLATFORMS=cpu prevents the CUDA plugin from segfaulting when no GPU driver is present
	docker run --rm -e JAX_PLATFORMS=cpu -v ${PWD}:/home/workdir --shm-size 20G $(IMAGE) /bin/bash -c "pytest ./tests/"

run-baseline-set:
	$(DOCKER_RUN_DETACHED) python baselines/run_minimal_baseline_set.py $(ARGS)
