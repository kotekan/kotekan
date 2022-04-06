FROM rocm/dev-ubuntu-18.04:3.7

## The maintainer name and email 
MAINTAINER Rick Nitsche <rick@phas.ubc.ca>

# Install any needed packages to run cmake with full CHIME build options
RUN apt-get update && \
    apt-get install -y software-properties-common=0.96.* && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3.7=3.7.* \
                       python3-setuptools=39.0.* \
                       python3-pip=9.0.* \
                       python3.7-distutils \
                       python3.7-dev=3.7.* \
                       build-essential=12.* \
                       git=1:2.17.* \
                       coreutils=8.* \
                       ccache=3.4.* \
                       pkg-config=0.29.* \
                       gcc=4:7.4.* \
                       gdb=8.* \
                       cmake=3.10.* \
                       clang-9=1:9-* \
                       clang-format-8=1:8-* \
                       dpdk=17.11.* \
                       dpdk-dev=17.11.* \
                       rocm-opencl-dev=3.* \
                       rocm-opencl=3.* \
                       libhdf5-serial-dev=1.10.* \
                       libboost-test-dev=1.65.* \
                       libevent-dev=2.1.* \
                       libssl-dev=1.1.* \
                       wget=1.19.* \
                       && \
    apt-get clean && apt-get autoclean

RUN python3.7 -m pip install --upgrade pip==20.2.2 && \
    python3.7 -m pip install --upgrade --force-reinstall setuptools==49.6.0 && \
    python3.7 -m pip install --upgrade wheel==0.35.1 && \
    python3.7 -m pip install --no-cache-dir numpy==1.19.1 && \
    python3.7 -m pip install --no-cache-dir pkgconfig==1.5.1 && \
    python3.7 -m pip install --no-cache-dir --upgrade cython==0.29.21 && \
    python3.7 -m pip install --no-cache-dir black==19.10b0 &&\
    python3.7 -m pip install --no-cache-dir cmake_format==0.6.13

RUN mkdir -p /code/build
WORKDIR /code/build

# So tzdata package doesn't ask for user interaction
ARG DEBIAN_FRONTEND=noninteractive

# Set architecture for all builds done below to haswell, currently the oldest architecture used
# by github workflows
# (see https://help.github.com/en/actions/reference/virtual-environments-for-github-hosted-runners).
# This is to prevent the docker image getting build on a newer architecture and then failing when it
# is loaded on an older one.
ENV CFLAGS "-march=haswell"
ENV CXXFLAGS "-march=haswell"

# Install h5py from source for bitshuffle, and clone HighFive for kotekan build
RUN git clone https://github.com/h5py/h5py.git h5py && \
    cd h5py && git checkout 2.9.0 && \
    python3.7 setup.py configure --hdf5=/usr/lib/x86_64-linux-gnu/hdf5/serial/ --hdf5-version=1.10.0 && \
    python3.7 setup.py install
RUN git clone --single-branch --branch extensible-datasets https://github.com/jrs65/HighFive.git && \
    cd HighFive && git pull && cd ..  

# Install bitshuffle
RUN git clone https://github.com/kiyo-masui/bitshuffle.git bitshuffle &&\
    cd bitshuffle &&\
    git pull &&\
    python3.7 setup.py build_ext --march=haswell &&\
    python3.7 setup.py install --h5plugin --h5plugin-dir=/usr/local/hdf5/lib/plugin

# Install OpenBLAS and clone Blaze for the eigenvalue processes
RUN apt-get update && \
    apt-get -y install libopenblas-dev=0.2.* \
                       liblapack-dev=3.7.* \
                       liblapacke-dev=3.7.* \
                       && \
    apt-get clean && apt-get autoclean && \
    git clone https://bitbucket.org/blaze-lib/blaze.git blaze && \
    cd blaze && git checkout v3.4 && cd ..

# Install kotekan python dependencies
RUN python3.7 -m pip install msgpack==1.0.0 \
                             click==7.1.2 \
                             future==0.18.2 \
                             requests==2.24.0 \
                             pyyaml==3.12 \
                             tabulate==0.8.7 \
                             pytest==6.0.1 \
                             pytest-xdist==2.1.0 \
                             pytest-cpp==1.4.0 \
                             pytest-localserver==0.5.0 \
                             flask==1.1.2 \
                             pytest-timeout==1.4.2 \
                             posix_ipc==1.0.4 \
                             && \
    apt-get update && \
    apt-get install -y mysql-client=5.7.* \
        libmysqlclient-dev=5.7.* && \
    apt-get clean && apt-get autoclean && \
    python3.7 -m pip install git+https://github.com/chime-experiment/comet.git

# Install redis for comet tests
RUN apt-get update && \
    apt-get install -y redis=5:4.0.* && \
    apt-get clean && apt-get autoclean

# Install documentation dependencies
RUN apt-get update && \
    apt-get -y install doxygen=1.8.* \
                       graphviz=2.40.* \
                       python-sphinx=1.6.* \
                       default-jre=2:1.* \
                       && \
    apt-get clean && apt-get autoclean && \
    python3.7 -m pip install --no-cache-dir breathe==4.20.* \
                                            sphinx_rtd_theme==0.5.* \
                                            sphinxcontrib-plantuml==0.* \
                                            && \
    wget https://phoenixnap.dl.sourceforge.net/project/plantuml/plantuml.jar -P plantuml

# Uncomment this to check what version got installed
# RUN python3.7 -m pip show <package>
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
