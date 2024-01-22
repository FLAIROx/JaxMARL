FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# install python
ARG DEBIAN_FRONTEND=noninteractive
ARG PYTHON_VERSION=3.10

# ENV TZ=Europe/London

RUN apt-get update && \
  DEBIAN_FRONTEND=noninteractive apt-get -qq -y install \
  software-properties-common \
  build-essential \
  curl \
  ffmpeg \
  git \
  htop \
  vim \
  nano \
  rsync \
  wget \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*

  RUN add-apt-repository ppa:deadsnakes/ppa
  RUN apt-get update && apt-get install -y -qq python${PYTHON_VERSION} \
      python${PYTHON_VERSION}-dev \
      python${PYTHON_VERSION}-distutils

# Set python aliases
RUN update-alternatives --install /usr/bin/python python /usr/bin/python${PYTHON_VERSION} 1
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python get-pip.py

# default workdir
WORKDIR /home/workdir

# dev: install from source
COPY . .

#jaxmarl from source if needed, all the requirements
RUN pip install --ignore-installed -e '.[qlearning, dev]'

# install jax from to enable cuda
RUN pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#disabling preallocation
ARG XLA_PYTHON_CLIENT_PREALLOCATE=false

#for jupyter
EXPOSE 9999 
CMD ["/bin/bash"]
