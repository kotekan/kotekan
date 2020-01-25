# Use a newer Ubuntu to get iwyu>=0.10
FROM ubuntu:focal

## The maintainer name and email
MAINTAINER Rick Nitsche <rick@phas.ubc.ca>

# Install any needed packages to run cmake with full CHIME build options
RUN apt-get update
RUN apt-get install -y gcc
RUN apt-get install -y cmake
RUN apt-get install -y libhdf5-103 libhdf5-dev h5utils
RUN apt-get install -y libhdf5-serial-dev
RUN apt-get install -y libhdf5-cpp-103
RUN apt-get install -y libboost-all-dev
RUN apt-get install -y libevent-dev
RUN apt-get install -y iwyu
RUN apt-get install -y gfortran
RUN apt-get install -y dpdk dpdk-dev dpdk-igb-uio-dkms
RUN apt-get install -y git
RUN apt-get install -y python3 python3-setuptools python3-pip

# Set compiler to clang (to make cmake choose the right flags for iwyu)
RUN export CC=clang
RUN export CXX=clang++

RUN mkdir -p /code/build
WORKDIR /code/build/

# Clone Highfive
RUN git clone --single-branch --branch extensible-datasets https://github.com/jrs65/HighFive.git

# Install h5py from source for bitshuffle
RUN git clone https://github.com/h5py/h5py.git h5py
WORKDIR /code/build/h5py
RUN python3 setup.py configure --hdf5=/usr/lib/x86_64-linux-gnu/hdf5/serial/ --hdf5-version=1.10.0
RUN python3 setup.py install

WORKDIR /code/build/

# Install bitshuffle
RUN git clone https://github.com/kiyo-masui/bitshuffle.git bitshuffle
WORKDIR /code/build/bitshuffle
RUN export HDF5_PLUGIN_PATH=/usr/local/hdf5/lib/plugin
#RUN python3 setup.py install --h5plugin --h5plugin-dir=/usr/local/hdf5/lib/plugin

WORKDIR /code/build/

# Build OpenBLAS
RUN git clone https://github.com/xianyi/OpenBLAS.git OpenBLAS
WORKDIR /code/build/OpenBLAS
RUN make
RUN make PREFIX=/code/build/OpenBLAS install

WORKDIR /code/build/

#Clone Blaze
RUN git clone https://nickritsche@bitbucket.org/blaze-lib/blaze.git blaze

WORKDIR /code

# copy everything we need
COPY include include/
COPY tools tools/
COPY iwyu.kotekan.imp iwyu.kotekan.imp
COPY CMakeLists.txt CMakeLists.txt
COPY kotekan kotekan/
COPY python python/
COPY tests tests/
COPY lib lib/

WORKDIR /code/build
# Do nothing when the container launches
CMD ["/bin/bash"]

