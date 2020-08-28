FROM rocm/dev-ubuntu-18.04

## The maintainer name and email
MAINTAINER Rick Nitsche <rick@phas.ubc.ca>

# Install any needed packages to run cmake with full CHIME build options
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y wget && \
    wget https://www.python.org/ftp/python/3.7.7/Python-3.7.7.tgz && \
    tar xzvf Python-3.7.7.tgz && cd Python-3.7.7 && \
    ./configure && make && make install && \
    python3.7 --version && \
    apt-get install -y python3-setuptools python3-pip python3.7-distutils python3.7-dev \
                       build-essential git coreutils ccache pkg-config \
                       gcc cmake clang-9 clang-format-8 \
                       dpdk dpdk-dev \
                       rocm-opencl-dev rocm-opencl \
                       libhdf5-serial-dev libboost-test-dev libevent-dev libssl-dev \
                       wget && \
    apt-get clean && apt-get autoclean

RUN python3.7 -m pip install --upgrade pip && \
    python3.7 -m pip install --upgrade --force-reinstall setuptools && \
    python3.7 -m pip install --upgrade pip setuptools wheel && \
    python3.7 -m pip install --no-cache-dir "numpy>=1.16.0" && \
    python3.7 -m pip install --no-cache-dir pkgconfig && \
    python3.7 -m pip install --no-cache-dir --upgrade cython && \
    python3.7 -m pip install --no-cache-dir black

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
RUN git clone https://github.com/kiyo-masui/bitshuffle.git bitshuffle && \
    cd bitshuffle && git pull && \
    python3.7 setup.py install --h5plugin --h5plugin-dir=/usr/local/hdf5/lib/plugin

# Install OpenBLAS and clone Blaze for the eigenvalue processes
RUN apt-get update && \
    apt-get -y install libopenblas-dev liblapack-dev liblapacke-dev && \
    apt-get clean && apt-get autoclean && \
    git clone https://bitbucket.org/blaze-lib/blaze.git blaze && cd blaze && git pull && cd ..

# Install kotekan python dependencies
RUN python3.7 -m pip install "msgpack>=0.6.1" click future requests pyyaml tabulate \
                             pytest==5.2.2 pytest-xdist pytest-cpp pytest-localserver flask \
                             pytest-timeout posix_ipc && \
    apt-get update && \
    apt-get install -y mysql-client libmysqlclient-dev && \
    apt-get clean && apt-get autoclean && \
    python3.7 -m pip install git+https://github.com/chime-experiment/comet.git

# Install redis for comet tests
RUN apt-get update && \
    apt-get install -y redis && \
    apt-get clean && apt-get autoclean

# Install documentation dependencies
RUN apt-get update && \
    apt-get -y install doxygen graphviz python-sphinx default-jre && \
    apt-get clean && apt-get autoclean && \
    python3.7 -m pip install --no-cache-dir breathe sphinx_rtd_theme sphinxcontrib-plantuml && \
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
