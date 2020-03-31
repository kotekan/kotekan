FROM rocm/dev-ubuntu-18.04

## The maintainer name and email
MAINTAINER Rick Nitsche <rick@phas.ubc.ca>

# Install any needed packages to run cmake with full CHIME build options
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y python3.7 python3-setuptools python3-pip python3.7-distutils python3.7-dev \
                       build-essential git coreutils ccache pkg-config \
                       gcc cmake clang-9 clang-format-8 \
                       rocm-opencl-dev rocm-opencl \
                       dpdk dpdk-dev \
                       libhdf5-serial-dev libboost-test-dev libevent-dev libssl-dev \
                       wget

RUN python3.7 -m pip install --upgrade pip && \
    python3.7 -m pip install --upgrade --force-reinstall setuptools && \
    python3.7 -m pip install --upgrade pip setuptools wheel && \
    python3.7 -m pip install "numpy>=1.16.0" && \
    python3.7 -m pip install pkgconfig && \
    python3.7 -m pip install --upgrade cython && \
    python3.7 -m pip install black

RUN mkdir -p /code/build
WORKDIR /code/build

# So tzdata package doesn't ask for user interaction
ARG DEBIAN_FRONTEND=noninteractive

# Install h5py from source for bitshuffle, and clone HighFive for kotekan build
RUN git clone https://github.com/h5py/h5py.git h5py && \
    cd h5py && \
    python3.7 setup.py configure --hdf5=/usr/lib/x86_64-linux-gnu/hdf5/serial/ --hdf5-version=1.10.0 && \
    python3.7 setup.py install
RUN git clone --single-branch --branch extensible-datasets https://github.com/jrs65/HighFive.git

# Install bitshuffle
RUN git clone https://github.com/kiyo-masui/bitshuffle.git bitshuffle && \
    cd bitshuffle && \
    python3.7 setup.py install --h5plugin --h5plugin-dir=/usr/local/hdf5/lib/plugin

# Install OpenBLAS and clone Blaze for the eigenvalue processes
RUN apt-get -y install libopenblas-dev liblapack-dev liblapacke-dev && \
    git clone https://bitbucket.org/blaze-lib/blaze.git blaze

# Install kotekan python dependencies
RUN python3.7 -m pip install "msgpack>=0.6.1" click future requests pyyaml \
                 tabulate pytest==5.3.0 pytest-xdist pytest-cpp posix_ipc && \
    apt-get install -y mysql-client libmysqlclient-dev && \
    python3.7 -m pip install git+https://github.com/chime-experiment/comet.git

# Install redis for comet tests
RUN apt-get install -y redis

# Install documentation dependencies
RUN apt-get -y install doxygen graphviz python-sphinx default-jre && \
    python3.7 -m pip install breathe sphinx_rtd_theme sphinxcontrib-plantuml && \
    wget https://phoenixnap.dl.sourceforge.net/project/plantuml/plantuml.jar -P plantuml

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
