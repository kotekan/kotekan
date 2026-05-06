# tools/docker/24.04/Dockerfile
# check=skip=UndefinedVar

###################
# Base Docker Image
# From Ubuntu 24.04
###################

ARG BASE_IMAGE=ubuntu:24.04
FROM ${BASE_IMAGE} AS base

#############################################################
# CPU-ONLY IMAGE
# An image without GPU functionality can be built as a target
# from this Dockerfile, i.e. add `--target cpu` when building
# the docker image.
#############################################################

FROM base AS cpu

# So tzdata package doesn't ask for user interaction
ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /code/build
WORKDIR /code/build

# Install any needed packages to run cmake with full CHORD build options
RUN apt-get update && \
    apt-get install -y \
        bison \
        build-essential \
        ccache \
        clang \
        clang-format \
        cmake \
        coreutils \
        curl \
        default-jre \
        doxygen \
        dpdk \
        dpdk-dev \
        file \
        flex \
        g++ \
        gcc \
        gdb \
        git \
        graphviz \
        iwyu \
        libairspy-dev \
        libblosc2-dev \
        libboost-filesystem-dev \
        libboost-system-dev \
        libboost-test-dev \
        libbz2-dev \
        libdpdk-dev \
        libevent-dev \
        libfftw3-dev \
        libgdal-dev \
        libhdf5-dev \
        libhdf5-serial-dev \
        liblapack-dev \
        liblapacke-dev \
        liblz4-dev \
        libnuma-dev \
        libomp-dev \
        libopenblas-dev \
        libssl-dev \
        libstdc++-14-dev \
        libsystemd-dev \
        libxtensor-dev \
        libyaml-cpp-dev \
        lld \
        make \
        pkg-config \
        python3 \
        python3-dev \
        software-properties-common \
        valgrind \
        wget \
        xtensor-dev \
        && \
    apt-get clean && apt-get autoclean && rm -rf /var/lib/apt/lists/*

# Install pip and python packages
# Disable pip caching (keep docker image size smaller)
ENV PIP_NO_CACHE_DIR=1
# Overwrite system install of python/pip if needed
RUN rm /usr/lib/python*/EXTERNALLY-MANAGED && \
    wget https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    python3 -m pip install --upgrade \
        astropy \
        bitshuffle \
        black==19.10b0 \
        breathe==4.35.* \
        click==8.0.2 \
        cmake_format \
        cython \
        distro-info \
        flask \
        future \
        gmpy2 \
        h5py \
        hdf5plugin \
        mpmath \
        msgpack \
        numpy \
        pip \
        pkgconfig \
        posix_ipc \
        pytest \
        pytest-cpp \
        pytest-localserver \
        pytest-timeout \
        pytest-xdist \
        requests \
        setuptools \
        sphinx==6.2.* \
        sphinx_rtd_theme==2.0.* \
        sphinxcontrib-plantuml==0.25 \
        tabulate \
        wheel

# Uncomment this to check what version got installed
# RUN python3 -m pip show <package>
# RUN apt-cache policy <package>

# We want the `python` command to point to python3
RUN ln -sf /usr/bin/python3 /usr/bin/python

# Install Blaze for the eigenvalue processes
RUN mkdir src && \
    ( \
        git clone -b v3.8.2 https://bitbucket.org/blaze-lib/blaze.git blaze && \
        cd blaze && \
        git checkout && \
        cmake -B cmake-build \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=/usr/local \
            && \
        cmake --build cmake-build --parallel $(nproc) && \
        cmake --install cmake-build && \
    true) && \
    rm -rf src

# Install ASDF (depends on Blosc2)
RUN mkdir src && \
    ( \
        cd src && \
        git clone -n https://github.com/jbmertens/asdf-cxx && \
        cd asdf-cxx && \
        git checkout 13a7112 && \
        cmake -B cmake-build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && \
        cmake --build cmake-build --parallel $(nproc) && \
        cmake --install cmake-build && \
    true) && \
    rm -rf src
# Depends on lz4.pc (called liblz4.pc in apt install)
RUN ln -s /usr/lib/x86_64-linux-gnu/pkgconfig/liblz4.pc /usr/local/lib/pkgconfig/lz4.pc

# Clone HighFive
RUN mkdir src && \
    ( \
        cd src && \
        git clone --recurse-submodules --single-branch --branch v3.2.0 https://github.com/highfive-devs/highfive && \
        cd highfive && \
        cmake -B cmake-build \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=/usr/local \
            -DHIGHFIVE_UNIT_TESTS=OFF \
            -DHIGHFIVE_EXAMPLES=OFF \
            -DHIGHFIVE_BUILD_DOCS=OFF \
            -DHIGHFIVE_TEST_HALF_FLOAT=ON \
            && \
        cmake --build cmake-build --parallel $(nproc) && \
        cmake --install cmake-build && \
    true) && \
    rm -rf src

# Set ccache to store things sensibly
ENV CCACHE_NOHASHDIR=1
ENV CCACHE_BASEDIR=/code/kotekan/
ENV CCACHE_DIR=/code/kotekan/.ccache/
ENV CCACHE_COMPRESS=1
ENV CCACHE_MAXSIZE=1G

# Set the plugin path so kotekan can find bitshuffle
# Ideally this would be determined dynamically, but for now...
ENV HDF5_PLUGIN_PATH=/usr/local/lib/python3.12/dist-packages/hdf5plugin/plugins

ENV KOTEKAN_BUILD_DIRNAME=build-2404

# Code is often cloned into /code/kotekan
# Don't worry about permission issues in here
RUN git config --global --add safe.directory /code/kotekan
# Add this to pythonpath
ENV PYTHONPATH=/code/kotekan/python/

# Do nothing when the container launches
CMD ["/bin/bash"]


##############################################################
# GPU IMAGE
# This image extends the cpu image with NVIDIA drivers and the
# CUDA toolkit.
##############################################################

FROM cpu AS gpu

# CUDA / opencl driver installs and packages
RUN mkdir src && \
    ( cd src &&\
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
        dpkg -i cuda-keyring_1.1-1_all.deb && \
        apt-get -y update && \
        apt-get install -y cuda-toolkit nvidia-open && \
        apt-get install -y --no-install-recommends libopencl-clang-17-dev opencl-headers ocl-icd-opencl-dev && \
        apt-get clean && rm -rf /var/lib/apt/lists/* && \
    true) && \
    rm -rf src

# Install mathdx for CHIME beamformer
RUN mkdir src && \
    ( \
        cd src && \
        wget https://developer.nvidia.com/downloads/compute/cuFFTDx/redist/cuFFTDx/cuda13/nvidia-mathdx-26.03.0-cuda13.tar.gz && \
        tar -xzf nvidia-mathdx-26.03.0-cuda13.tar.gz && \
        cd nvidia-mathdx-26.03.0-cuda13 && \
        mv nvidia/mathdx/26.03/include /usr/local/include/mathdx && \
        cp -a nvidia/mathdx/26.03/lib/* /usr/local/lib/ && \
        cp -a nvidia/mathdx/26.03/src/* /usr/local/src/ && \
    true) && \
    rm -rf src

ENV CUDADIR=/usr/local/cuda
ENV PATH="${PATH:+${PATH}:}/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/local/cuda/bin"
ENV LD_LIBRARY_PATH="${CUDADIR}/lib64:/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
ENV CPLUS_INCLUDE_PATH="${CUDADIR}/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"
