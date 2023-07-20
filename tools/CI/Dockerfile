FROM rocm/dev-ubuntu-22.04

# So tzdata package doesn't ask for user interaction
ARG DEBIAN_FRONTEND=noninteractive

# Install any needed packages to run cmake with full CHIME build options
RUN apt-get -y update && \
    apt-get install -y software-properties-common=0.99.* && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3.11 \
                       python3-setuptools \
                       python3-pip \
                       python3.11-distutils \
                       python3.11-dev \
                       build-essential \
                       git \
                       coreutils \
                       ccache \
                       pkg-config \
                       gcc \
                       gdb \
                       cmake \
                       clang-15 \
                       clang-format-15 \
                       dpdk \
                       dpdk-dev \
                       libhdf5-serial-dev \
                       libboost-test-dev \
                       libevent-dev \
                       libssl-dev \
                       wget \
                       && \
    apt-get clean && apt-get autoclean

RUN apt-cache policy python3.11 python3-setuptools python3-pip python3.11-distutils python3.11-dev build-essential git coreutils ccache pkg-config gcc gdb cmake clang-15 clang-format-15 dpdk dpdk-dev libhdf5-serial-dev libboost-test-dev libevent-dev libssl-dev wget

RUN python3.11 -m pip install --upgrade pip && \
    python3.11 -m pip install --upgrade distro-info && \
    python3.11 -m pip install --upgrade --force-reinstall setuptools && \
    python3.11 -m pip install --upgrade wheel && \
    python3.11 -m pip install --no-cache-dir numpy && \
    python3.11 -m pip install --no-cache-dir pkgconfig && \
    python3.11 -m pip install --no-cache-dir --upgrade cython && \
    python3.11 -m pip install --no-cache-dir black && \
    python3.11 -m pip install --no-cache-dir cmake_format

RUN python3.11 -m pip show h5py pip setuptools wheel numpy pkgconfig cython black cmake_format

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
RUN apt-get update && \
    apt-get -y install doxygen \
                       graphviz \
                       python-sphinx \
                       default-jre \
                       && \
    apt-get clean && apt-get autoclean && \
    python3.11 -m pip install --no-cache-dir breathe \
                                             sphinx_rtd_theme \
                                             sphinxcontrib-plantuml \

RUN python3.11 -m pip show breathe sphinx_rtd_theme sphinxcontrib-plantuml
RUN apt-cache policy doxygen graphviz python-sphinx default-jre redis mysql-client libmysqlclient-dev

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
