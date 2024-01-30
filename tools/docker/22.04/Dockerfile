FROM ubuntu:22.04

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


# Install any needed packages to run cmake with full CHIME build options
RUN apt-get -y update && \
    apt-get install -y software-properties-common && \
    apt-get upgrade -y && \
    apt-get install -y python3=3.10.* \
                       python3-setuptools \
                       python3-pip \
                       python3-distutils \
                       python3-dev \
                       build-essential \
                       git \
                       coreutils \
                       ccache \
                       pkg-config \
                       gcc-12 \
                       g++-12 \
                       make \
                       file \
                       gdb \
                       cmake \
                       clang \
                       clang-format \
                       dpdk \
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
                       wget \
                       flex \
                       bison \
                       iwyu \
                       && \
    apt-get clean && apt-get autoclean


# Install kotekan python dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade distro-info && \
    python3 -m pip install --upgrade --force-reinstall setuptools && \
    python3 -m pip install --upgrade wheel && \
    python3 -m pip install --no-cache-dir numpy && \
    python3 -m pip install --no-cache-dir pkgconfig && \
    python3 -m pip install --no-cache-dir --upgrade cython && \
    python3 -m pip install --no-cache-dir click==8.0.2 && \
    python3 -m pip install --no-cache-dir black==19.10b0 && \
    python3 -m pip install --no-cache-dir cmake_format

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
RUN python3 -m pip install --ignore-installed flask


# Install comet, mysql, redis for comet tests
RUN apt-get update && \
    apt-get install -y mysql-client \
        libmysqlclient-dev && \
    apt-get clean && apt-get autoclean
RUN python3 -m pip install git+https://github.com/chime-experiment/comet.git
RUN apt-get update && \
    apt-get install -y redis && \
    apt-get clean && apt-get autoclean


# CUDA / GPU tools
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
RUN dpkg -i cuda-keyring_1.0-1_all.deb
RUN apt -y update
RUN apt install -y --no-install-recommends cuda && apt clean

ENV PATH /usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/cuda/bin

RUN git clone https://github.com/kmsmith137/gputils.git \
    && cd gputils \
    && make install

RUN git clone https://github.com/kmsmith137/n2k.git \
  && cd n2k \
  && make install

RUN apt install -y --no-install-recommends \
  libopencl-clang-dev opencl-headers ocl-icd-opencl-dev


# Install OpenBLAS and clone Blaze for the eigenvalue processes
RUN apt-get update && \
    apt-get -y install libopenblas-dev \
                       liblapack-dev \
                       liblapacke-dev \
                       && \
    apt-get clean && apt-get autoclean && \
    git clone https://bitbucket.org/blaze-lib/blaze.git blaze && \
    cd blaze && git checkout v3.4 && \
    cmake -DCMAKE_INSTALL_PREFIX=/usr/local/ && make install && cd ..

# Install Blosc2 (a prerequisite for ASDF)
RUN mkdir src && \
    (cd src && \
    wget https://github.com/Blosc/c-blosc2/archive/refs/tags/v2.13.1.tar.gz && \
    tar xzf v2.13.1.tar.gz && \
    cd c-blosc2-2.13.1 && \
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
# RUN mkdir src && \
#     ( \
#         cd src && \
#         wget https://github.com/eschnett/asdf-cxx/archive/refs/tags/version/7.3.1.tar.gz && \
#         tar xzf 7.3.1.tar.gz && \
#         cd asdf-cxx-version-7.3.1 && \
#         cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && \
#         cmake --build build && \
#         cmake --install build && \
#     true) && \
#     rm -rf src
RUN mkdir src && \
    ( \
        cd src && \
        git clone -n https://github.com/eschnett/asdf-cxx && \
        cd asdf-cxx && \
        git checkout 205182f47ad3fe13ec163105dd76c5f3ea72e51c && \
        cmake -B build -S . -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && \
        cmake --build build --parallel $(nproc) && \
        cmake --install build && \
    true) && \
    rm -rf src

# Clone HighFive
RUN git clone --single-branch --branch extensible-datasets https://github.com/jrs65/HighFive.git && \
 cd HighFive && git pull && cd ..
ENV HIGHFIVE_PATH /code/build/HighFive

# Install documentation dependencies
# Install newer version of doxygen from source. This is slow, but older versions had bugs
RUN wget https://www.doxygen.nl/files/doxygen-1.9.7.src.tar.gz && ls && \
    tar -xf doxygen-1.9.7.src.tar.gz && \
    cd doxygen-1.9.7 && \
    mkdir -p build && cd build && pwd && \
    cmake -G "Unix Makefiles" .. && make -j 2 && make install && \
    doxygen --version
RUN apt-get update && \
    apt-get -y install graphviz \
                       default-jre \
                       && \
    apt-get clean && apt-get autoclean
RUN python3 -m pip install --no-cache-dir breathe==4.35.* \
                                          sphinx==6.2.* \
                                          sphinx_rtd_theme==1.2.* \
                                          sphinxcontrib-plantuml==0.25

# Julia
RUN apt-get update && apt-get install -y curl \
    && curl -fsSL https://install.julialang.org | sh -s -- -y
ENV PATH="$PATH:/root/.juliaup/bin"

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

# We need the `python` command to point to python3
RUN ln -s /bin/python3 /bin/python

# Use G++12 as the default
RUN rm /usr/bin/gcc /usr/bin/g++ && \
    ln -s g++-12 /usr/bin/g++ && \
    ln -s gcc-12 /usr/bin/gcc

# Set the plugin path so kotekan can find bitshuffle
# Ideally this would be determined dynamically, but for now...
ENV HDF5_PLUGIN_PATH /usr/local/lib/python3.10/dist-packages/hdf5plugin/plugins

ENV KOTEKAN_BUILD_DIRNAME build-2204

# Update Julia...
RUN juliaup update && echo 2

# Do nothing when the container launches
CMD ["/bin/bash"]
