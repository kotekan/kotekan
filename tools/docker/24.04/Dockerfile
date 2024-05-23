FROM ubuntu:24.04

# So tzdata package doesn't ask for user interaction
ARG DEBIAN_FRONTEND=noninteractive

# Set architecture for all builds done below to haswell, currently the oldest architecture used
# by github workflows
# (see https://help.github.com/en/actions/reference/virtual-environments-for-github-hosted-runners).
# This is to prevent the docker image getting build on a newer architecture and then failing when it
# is loaded on an older one.
ENV CFLAGS "-march=haswell"
ENV CXXFLAGS "-march=haswell"

RUN mkdir -p /code/build
WORKDIR /code/build

# Install any needed packages to run cmake with full CHORD build options
RUN apt update && \
    apt install -y software-properties-common && \
    apt upgrade -y && \
    apt install -y build-essential pkg-config && \
    apt install -y git \
                   file \
                   wget \
                   coreutils && \
    apt install -y ccache \
                   gcc-12 \
                   g++-12 \
                   make \
                   cmake \
                   clang \
                   clang-format \
                   flex \
                   bison \
                   iwyu \
                   gdb && \
    apt install -y dpdk \
                   dpdk-dev \
                   libdpdk-dev \
                   libhdf5-dev \
                   libhdf5-serial-dev \
                   libfftw3-dev \
                   libairspy-dev \
                   libboost-test-dev \
                   libevent-dev \
                   libnuma-dev \
                   libssl-dev \
                   libyaml-cpp-dev \
                   libsystemd-dev && \
    apt install -y python3 \
                   python3-dev && \
    apt clean && apt autoclean

# Install pip and python packages
# Overwrite system install of python/pip if needed
RUN rm /usr/lib/python*/EXTERNALLY-MANAGED
# Disable pip caching (keep docker image size smaller)
ENV PIP_NO_CACHE_DIR=1
RUN wget https://bootstrap.pypa.io/get-pip.py && python3 get-pip.py
RUN python3 -m pip install --upgrade pip \
                                     distro-info \
                                     setuptools \
                                     wheel \
                                     pkgconfig
# We want the `python` command to point to python3
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install kotekan python dependencies
RUN python3 -m pip install cython \
                           numpy \
                           cmake_format \
                           click==8.0.2 \
                           black==19.10b0 \
                           flask
RUN python3 -m pip install h5py \
                           hdf5plugin \
                           bitshuffle
RUN python3 -m pip install msgpack \
                           future \
                           requests \
                           pyyaml \
                           tabulate \
                           pytest \
                           pytest-xdist \
                           pytest-cpp \
                           pytest-localserver \
                           pytest-timeout \
                           posix_ipc

# CUDA / GPU tools
#RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.0-1_all.deb
#RUN dpkg -i cuda-keyring_1.0-1_all.deb
#RUN apt-get -y update
#RUN apt-get install -y --no-install-recommends cuda && apt-get clean

#ENV PATH /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/cuda/bin
#
#RUN apt-get install -y --no-install-recommends \
#  libopencl-clang-dev opencl-headers ocl-icd-opencl-dev

#RUN git clone https://github.com/kmsmith137/gputils.git \
#    && cd gputils \
#    && make install
#
#RUN git clone https://github.com/kmsmith137/n2k.git \
#  && cd n2k \
#  && make install

# Install OpenBLAS and clone Blaze for the eigenvalue processes
RUN apt update && \
    apt install -y libopenblas-dev \
                   liblapack-dev \
                   liblapacke-dev \
                   && \
    apt clean && apt autoclean && \
    git clone https://bitbucket.org/blaze-lib/blaze.git blaze && \
    cd blaze && git checkout v3.4 && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/ && make install && cd ..

# Install Blosc2 (a prerequisite for ASDF)
RUN apt update && apt install -y libbz2-dev liblz4-dev && \
    apt clean && apt autoclean
RUN mkdir src && \
    (cd src && \
    wget https://github.com/Blosc/c-blosc2/archive/refs/tags/v2.14.4.tar.gz && \
    tar xzf v2.14.4.tar.gz && \
    cd c-blosc2-2.14.4 && \
    cmake -B build -S . \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DBUILD_BENCHMARKS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_FUZZERS=OFF \
        -DBUILD_TESTS=OFF \
        && \
    cmake --build build --parallel $(nproc) && \
    cmake --install build && \
    true) && \
    rm -rf src

# Install ASDF (depends on Blosc2)
RUN mkdir src && \
    ( \
        cd src && \
        git clone -n https://github.com/eschnett/asdf-cxx && \
        cd asdf-cxx && \
        git checkout 64399d9b93c5e163c395a4a4440ef16d415fd4f6 && \
        cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && \
        cmake --build build --parallel $(nproc) && \
        cmake --install build && \
    true) && \
    rm -rf src

# Clone HighFive
RUN git clone --single-branch --branch extensible-datasets https://github.com/jrs65/HighFive.git && \
 cd HighFive && git pull && cd ..
ENV HIGHFIVE_PATH /code/build/HighFive

# Documentation libs
RUN apt update && \
    apt install -y doxygen graphviz default-jre && \
    apt clean && apt autoclean
RUN python3 -m pip install breathe==4.35.* \
                           sphinx==6.2.* \
                           sphinx_rtd_theme==2.0.* \
                           sphinxcontrib-plantuml==0.25

# Julia
RUN apt-get update && apt-get install -y curl \
    && curl -fsSL https://install.julialang.org | sh -s -- -y
ENV PATH="$PATH:/root/.juliaup/bin"

# Update Julia...
RUN juliaup self update && juliaup update

# Install Xtensor (for Zarr file format)
RUN apt-get update && \
    apt-get -y install libxtensor-dev xtensor-dev && \
    apt-get clean && apt-get autoclean


# Uncomment this to check what version got installed
# RUN python3 -m pip show <package>
# RUN apt-cache policy <package>

# Clean any unwanted caches
RUN apt-get clean && apt-get autoclean

# Set ccache to store things sensibly
ENV CCACHE_NOHASHDIR 1
ENV CCACHE_BASEDIR /code/kotekan/
ENV CCACHE_DIR /code/kotekan/.ccache/
ENV CCACHE_COMPRESS 1
ENV CCACHE_MAXSIZE 1G

# Set the plugin path so kotekan can find bitshuffle
# Ideally this would be determined dynamically, but for now...
ENV HDF5_PLUGIN_PATH /usr/local/lib/python3.12/dist-packages/hdf5plugin/plugins

ENV KOTEKAN_BUILD_DIRNAME build-2404

# Do nothing when the container launches
CMD ["/bin/bash"]
