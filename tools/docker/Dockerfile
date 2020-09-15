FROM rocm/dev-ubuntu-18.04

## The maintainer name and email
MAINTAINER Rick Nitsche <rick@phas.ubc.ca>

# Install any needed packages to run cmake with full CHIME build options
RUN apt-get update && \
    apt-get install -y software-properties-common=0.96.24.32.14 && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3.7=3.7.9-1+bionic1 \
                       python3-setuptools=39.0.1-2 \
                       python3-pip=9.0.1-2.3~ubuntu1.18.04.2 \
                       python3.7-distutils \
                       python3.7-dev=3.7.9-1+bionic1 \
                       build-essential=12.4ubuntu1 \
                       git=1:2.17.1-1ubuntu0.7 \
                       coreutils=8.28-1ubuntu1 \
                       ccache=3.4.1-1 \
                       pkg-config=0.29.1-0ubuntu2 \
                       gcc=4:7.4.0-1ubuntu2.3 \
                       gdb=8.1-0ubuntu3.2 \
                       cmake=3.10.2-1ubuntu2.18.04.1 \
                       clang-9=1:9-2~ubuntu18.04.2 \
                       clang-format-8=1:8-3~ubuntu18.04.2 \
                       dpdk=17.11.10-0ubuntu0.1 \
                       dpdk-dev=17.11.10-0ubuntu0.1 \
                       rocm-opencl-dev=2.0.0.293-rocm-rel-3.7-20-3f18143 \
                       rocm-opencl=2.0.0.293-rocm-rel-3.7-20-3f18143 \
                       libhdf5-serial-dev=1.10.0-patch1+docs-4 \
                       libboost-test-dev=1.65.1.0ubuntu1 \
                       libevent-dev=2.1.8-stable-4build1 \
                       libssl-dev=1.1.1-1ubuntu2.1~18.04.6 \
                       wget=1.19.4-1ubuntu2.2 \
                       && \
    apt-get clean && apt-get autoclean

RUN python3.7 -m pip install --upgrade pip==20.2.2 && \
    python3.7 -m pip install --upgrade --force-reinstall setuptools==49.6.0 && \
    python3.7 -m pip install --upgrade wheel==0.35.1 && \
    python3.7 -m pip install --no-cache-dir numpy==1.19.1 && \
    python3.7 -m pip install --no-cache-dir pkgconfig==1.5.1 && \
    python3.7 -m pip install --no-cache-dir --upgrade cython==0.29.21 && \
    python3.7 -m pip install --no-cache-dir black==19.10b0

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
    cd h5py && git pull && \
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
    apt-get -y install libopenblas-dev=0.2.20+ds-4 \
                       liblapack-dev=3.7.1-4ubuntu1 \
                       liblapacke-dev=3.7.1-4ubuntu1 \
                       && \
    apt-get clean && apt-get autoclean && \
    git clone https://bitbucket.org/blaze-lib/blaze.git blaze && cd blaze && git pull && cd ..

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
    apt-get install -y mysql-client=5.7.31-0ubuntu0.18.04.1 \
        libmysqlclient-dev=5.7.31-0ubuntu0.18.04.1 && \
    apt-get clean && apt-get autoclean && \
    python3.7 -m pip install git+https://github.com/chime-experiment/comet.git

# Install redis for comet tests
RUN apt-get update && \
    apt-get install -y redis=5:4.0.9-1ubuntu0.2 && \
    apt-get clean && apt-get autoclean

# Install documentation dependencies
RUN apt-get update && \
    apt-get -y install doxygen=1.8.13-10 \
                       graphviz=2.40.1-2 \
                       python-sphinx=1.6.7-1ubuntu1 \
                       default-jre=2:1.11-68ubuntu1~18.04.1 \
                       && \
    apt-get clean && apt-get autoclean && \
    python3.7 -m pip install --no-cache-dir breathe==4.20.0 \
                                            sphinx_rtd_theme==0.5.0 \
                                            sphinxcontrib-plantuml==0.18 \
                                            && \
    wget https://phoenixnap.dl.sourceforge.net/project/plantuml/plantuml.jar -P plantuml


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
