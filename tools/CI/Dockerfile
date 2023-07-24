FROM ubuntu:22.04

# So tzdata package doesn't ask for user interaction
ARG DEBIAN_FRONTEND=noninteractive

# Install any needed packages to run cmake with full CHIME build options
RUN apt-get -y update && \
    apt-get install -y software-properties-common=0.99.* && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3.11=3.11.* \
                       python3-setuptools=59.6.* \
                       python3-pip=22.* \
                       python3.11-distutils \
                       python3.11-dev=3.11.* \
                       build-essential=12.* \
                       git=1:2.34.* \
                       coreutils=8.32-4.* \
                       ccache=4.5.* \
                       pkg-config=0.29.* \
                       gcc=4:11.2.* \
                       gdb=12.* \
                       cmake=3.22.* \
                       clang=1:14.* \
                       clang-format=1:14.* \
                       dpdk=21.11.* \
                       dpdk-dev=21.11.* \
                       libhdf5-dev=1.10.* \
                       libboost-test-dev=1.74.* \
                       libevent-dev=2.1.* \
                       libssl-dev=3.0.* \
                       wget=1.21.* \
                       flex=2.6.* \
                       bison=2:3.8.* \
                       && \
    apt-get clean && apt-get autoclean

RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade distro-info && \
    python3.11 -m pip install --upgrade --force-reinstall setuptools && \
    python3.11 -m pip install --upgrade wheel && \
    python3.11 -m pip install --no-cache-dir numpy && \
    python3.11 -m pip install --no-cache-dir pkgconfig==1.5.* && \
    python3.11 -m pip install --no-cache-dir --upgrade cython && \
    python3.11 -m pip install --no-cache-dir click==8.0.2 && \
    python3.11 -m pip install --no-cache-dir black==19.10b0 && \
    python3.11 -m pip install --no-cache-dir cmake_format

RUN mkdir -p /code/build
WORKDIR /code/build

# Set architecture for all builds done below to haswell, currently the oldest architecture used
# by github workflows
# (see https://help.github.com/en/actions/reference/virtual-environments-for-github-hosted-runners).
# This is to prevent the docker image getting build on a newer architecture and then failing when it
# is loaded on an older one.
ENV CFLAGS "-march=haswell"
ENV CXXFLAGS "-march=haswell"

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
    apt-get clean && apt-get autoclean && \
    python3.11 -m pip install --no-cache-dir breathe==4.35.* sphinx==6.2.* \
                                             sphinx_rtd_theme==1.2.* \
                                             sphinxcontrib-plantuml==0.25

# Uncomment this to check what version got installed
# RUN python3.11 -m pip show <package>
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
ENV HDF5_PLUGIN_PATH /usr/local/hdf5/lib/plugin

# Do nothing when the container launches
CMD ["/bin/bash"]
