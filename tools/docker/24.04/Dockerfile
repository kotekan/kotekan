# check=skip=UndefinedVar
FROM ubuntu:24.04

LABEL description="Reproducible build environment for CHORD/Kotekan pipeline"
LABEL version="2025.11"

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
ARG DEBIAN_FRONTEND=noninteractive
# Pinned to 13-0 (Requires Host Driver 550+)
ARG CUDA_PKG_VERSION=13-0
ARG JULIA_VER=1.10.10

# Fail fast if a piped command errors
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# PREVENT PYTHON BREAKAGE
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PIP_NO_CACHE_DIR=1

# Create build dir and make it writable for non-root users (via chmod 777)
RUN mkdir -p /code/build && chmod 777 /code/build
WORKDIR /code/build

# -----------------------------------------------------------------------------
# 1. REGISTER NVIDIA REPOSITORY
# -----------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends wget ca-certificates gnupg && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# -----------------------------------------------------------------------------
# 2. INSTALL SYSTEM DEPENDENCIES
# -----------------------------------------------------------------------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        # --- A. NVIDIA / CUDA Stack ---
        cuda-toolkit-${CUDA_PKG_VERSION} \
        libopencl-clang-17-dev \
        opencl-headers \
        ocl-icd-opencl-dev \
        # --- B. Core Compilers & Build Tools ---
        gcc-14 \
        g++-14 \
        clang-18 \
        clang-format-18 \
        build-essential \
        make \
        cmake \
        cmake-format \
        ccache \
        pkg-config \
        bison \
        flex \
        gdb \
        valgrind \
        iwyu \
        # --- C. Python Runtime ---
        python3 \
        python3-dev \
        python3-pip \
        cython3 \
        # --- D. Scientific Math & Numerics ---
        libopenblas-dev \
        liblapack-dev \
        liblapacke-dev \
        libfftw3-dev \
        libgmp-dev \
        libmpc-dev \
        libmpfr-dev \
        # --- E. Libraries (HDF5, Boost, Compression) ---
        libhdf5-dev \
        libhdf5-serial-dev \
        libboost-filesystem-dev \
        libboost-system-dev \
        libboost-test-dev \
        libbz2-dev \
        liblz4-dev \
        libssl-dev \
        libyaml-cpp-dev \
        # --- F. Hardware & Network Drivers ---
        dpdk \
        dpdk-dev \
        libdpdk-dev \
        libairspy-dev \
        libnuma-dev \
        # --- G. Utilities ---
        coreutils \
        curl \
        git \
        default-jre \
        doxygen \
        graphviz \
        file \
        libevent-dev \
        libgdal-dev \
        libomp-dev \
        libstdc++-14-dev \
        libsystemd-dev \
        libxtensor-dev \
        software-properties-common \
        xtensor-dev \
        bitshuffle \
        click \
        distro-info \
        && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Force the system to use our PINNED compilers by default
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-14 100 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-14 100 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-18 100

# Link Python convenience alias
RUN ln -s /usr/bin/python3 /usr/bin/python

# -----------------------------------------------------------------------------
# 3. PYTHON ENVIRONMENT
# -----------------------------------------------------------------------------
WORKDIR /code/build
# Use --ignore-installed to overwrite system 'wheel' package cleanly
RUN python3 -m pip install --upgrade --ignore-installed pip setuptools wheel

COPY requirements.txt /tmp/requirements.txt
# FIX: Added --ignore-installed to prevent h5py uninstall errors
RUN python3 -m pip install --ignore-installed -r /tmp/requirements.txt

# -----------------------------------------------------------------------------
# 4. SOURCE BUILDS (STABLE/LTS)
# -----------------------------------------------------------------------------
WORKDIR /tmp/src

# --- Blaze ---
RUN git clone -b v3.8.2 --depth 1 https://bitbucket.org/blaze-lib/blaze.git blaze && \
    cmake -S blaze -B blaze/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build blaze/build --parallel $(nproc) && \
    cmake --install blaze/build && \
    rm -rf blaze

# --- C-Blosc2 ---
RUN wget -q -O blosc.tar.gz https://github.com/Blosc/c-blosc2/archive/refs/tags/v2.21.3.tar.gz && \
    tar xzf blosc.tar.gz && \
    cmake -S c-blosc2-2.21.3 -B c-blosc2-2.21.3/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DBUILD_BENCHMARKS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_FUZZERS=OFF -DBUILD_TESTS=OFF && \
    cmake --build c-blosc2-2.21.3/build --parallel $(nproc) && \
    cmake --install c-blosc2-2.21.3/build && \
    rm -rf c-blosc2-2.21.3 blosc.tar.gz

# --- ASDF-CXX ---
RUN git clone -n https://github.com/jbmertens/asdf-cxx && \
    cd asdf-cxx && \
    git checkout 13a7112 && \
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && \
    cmake --build build --parallel $(nproc) && \
    cmake --install build && \
    cd .. && rm -rf asdf-cxx

# Link lz4 pkgconfig
RUN ln -s /usr/lib/x86_64-linux-gnu/pkgconfig/liblz4.pc /usr/local/lib/pkgconfig/lz4.pc

# --- HighFive ---
RUN git clone --depth 1 --branch v3.2.0 https://github.com/highfive-devs/highfive && \
    cmake -S highfive -B highfive/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local -DHIGHFIVE_UNIT_TESTS=OFF -DHIGHFIVE_EXAMPLES=OFF -DHIGHFIVE_BUILD_DOCS=OFF -DHIGHFIVE_TEST_HALF_FLOAT=ON && \
    cmake --build highfive/build --parallel $(nproc) && \
    cmake --install highfive/build && \
    rm -rf highfive

# --- Julia ---
RUN wget -q -O julia.tar.gz https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-${JULIA_VER}-linux-x86_64.tar.gz && \
    mkdir -p /opt/julia && \
    tar -xzf julia.tar.gz -C /opt/julia --strip-components=1 && \
    ln -s /opt/julia/bin/julia /usr/local/bin/julia && \
    rm julia.tar.gz

# -----------------------------------------------------------------------------
# 5. FINAL CONFIGURATION
# -----------------------------------------------------------------------------
WORKDIR /code/build

ENV CCACHE_NOHASHDIR=1
ENV CCACHE_BASEDIR=/code/kotekan/
ENV CCACHE_DIR=/code/kotekan/.ccache/
ENV CCACHE_COMPRESS=1
ENV CCACHE_MAXSIZE=1G
ENV HDF5_PLUGIN_PATH=/usr/local/lib/python3.12/dist-packages/hdf5plugin/plugins
ENV KOTEKAN_BUILD_DIRNAME=build-2404
ENV CUDADIR=/usr/local/cuda
ENV PATH="${PATH}:/usr/local/cuda/bin"
ENV LD_LIBRARY_PATH="${CUDADIR}/lib64:/usr/local/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
ENV CPLUS_INCLUDE_PATH="${CUDADIR}/include${CPLUS_INCLUDE_PATH:+:${CPLUS_INCLUDE_PATH}}"

# Use --system so this config applies to ANY user running the container
RUN git config --system --add safe.directory /code/kotekan
ENV PYTHONPATH=/code/kotekan/python/

CMD ["/bin/bash"]