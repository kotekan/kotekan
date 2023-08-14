FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt -y update && apt install -y apt-utils

RUN apt install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    git \
    file \
    pkg-config \
    cmake \
    wget \
    less \
    ca-certificates \
    libevent-dev \
    libnuma-dev \
    libboost-test-dev \
    libssl-dev \
    clang-format \
    iwyu \
    && apt clean

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt -y update
RUN apt install -y --no-install-recommends cuda && apt clean
#RUN apt install -y --no-install-recommends cuda-libraries-12-1 cuda-libraries-dev-12-1 cuda-nvcc-12-1

ENV PATH /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/cuda/bin

RUN git clone https://github.com/kmsmith137/gputils.git \
    && cd gputils \
    && make install

RUN git clone https://github.com/kmsmith137/n2k.git \
  && cd n2k \
  && make install

RUN apt install -y --no-install-recommends libdpdk-dev && apt clean

RUN apt install -y --no-install-recommends \
  libopencl-clang-dev opencl-headers ocl-icd-opencl-dev \
  libopenblas-dev liblapack-dev liblapacke-dev \
  libfftw3-dev \
  libairspy-dev \
  libhdf5-serial-dev \
  doxygen graphviz sphinx-doc sphinx-common python3-breathe \
  && apt clean