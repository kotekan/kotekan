FROM rocm/dev-ubuntu-18.04

## The maintainer name and email
MAINTAINER Rick Nitsche <rick@phas.ubc.ca>

# Install any needed packages to run cmake with full CHIME build options
RUN apt-get update
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install -y python3.7 python3-setuptools python3-pip python3.7-distutils python3.7-dev
RUN apt-get install -y gcc
RUN apt-get install -y cmake
RUN apt-get install -y libhdf5-serial-dev
RUN apt-get install -y clang-9
RUN apt-get install -y clang-format-8
RUN apt-get install -y wget
RUN apt-get install -y libssl-dev
RUN apt-get install -y build-essential
RUN apt-get install -y git
RUN apt-get install -y coreutils

WORKDIR /code/build/

# Install h5py from source for bitshuffle
RUN git clone https://github.com/h5py/h5py.git h5py
WORKDIR /code/build/h5py
RUN python3.7 -m pip install --upgrade --force-reinstall setuptools
RUN python3.7 setup.py configure --hdf5=/usr/lib/x86_64-linux-gnu/hdf5/serial/ --hdf5-version=1.10.0

RUN apt-get install -y libboost-test-dev
RUN apt-get install -y libevent-dev
RUN apt-get install -y iwyu
RUN apt-get install -y gfortran
RUN apt-get install -y dpdk dpdk-dev # dpdk-igb-uio-dkms
RUN apt-get install -y libairspy-dev libfftw3-dev

RUN python3.7 -m pip install --upgrade pip
RUN python3.7 -m pip install --upgrade --force-reinstall setuptools
RUN python3.7 -m pip install --upgrade pip setuptools wheel
RUN python3.7 -m pip install numpy
RUN python3.7 -m pip install pkgconfig
RUN python3.7 -m pip install --upgrade cython
RUN python3.7 -m pip install black

#RUN apt-get install -y libevent-2.0-5 libevent-pthreads-2.0-5

RUN mkdir -p /code/build
WORKDIR /code/build/

# Clone Highfive
RUN git clone --single-branch --branch extensible-datasets https://github.com/jrs65/HighFive.git

WORKDIR /code/build/

# Install bitshuffle
RUN git clone https://github.com/kiyo-masui/bitshuffle.git bitshuffle
WORKDIR /code/build/bitshuffle
RUN export HDF5_PLUGIN_PATH=/usr/local/hdf5/lib/plugin
#RUN python3.7 setup.py install --h5plugin --h5plugin-dir=/usr/local/hdf5/lib/plugin

WORKDIR /code/build/

# Build OpenBLAS
RUN git clone https://github.com/xianyi/OpenBLAS.git OpenBLAS
WORKDIR /code/build/OpenBLAS
RUN make
RUN make PREFIX=/code/build/OpenBLAS install

WORKDIR /code/build/

#Clone Blaze
RUN git clone https://nickritsche@bitbucket.org/blaze-lib/blaze.git blaze

# Install iwyu for clang 8.0
RUN wget http://launchpadlibrarian.net/461878080/libstdc++6_9.2.1-25ubuntu1_amd64.deb
RUN wget http://launchpadlibrarian.net/458626653/iwyu_8.0-3_amd64.deb
RUN wget http://launchpadlibrarian.net/461877892/gcc-9-base_9.2.1-25ubuntu1_amd64.deb
RUN wget http://launchpadlibrarian.net/462079080/libllvm9_9.0.1-8build1_amd64.deb
RUN wget http://launchpadlibrarian.net/453587373/libc6_2.30-0ubuntu3_amd64.deb
RUN wget http://launchpadlibrarian.net/461284144/libffi7_3.3-3_amd64.deb
RUN wget http://launchpadlibrarian.net/448587516/libtinfo6_6.1+20191019-1ubuntu1_amd64.deb
RUN dpkg -i libtinfo6_6.1+20191019-1ubuntu1_amd64.deb
RUN dpkg -i libffi7_3.3-3_amd64.deb
RUN dpkg -i libc6_2.30-0ubuntu3_amd64.deb
RUN dpkg -i gcc-9-base_9.2.1-25ubuntu1_amd64.deb
RUN dpkg -i libstdc++6_9.2.1-25ubuntu1_amd64.deb
RUN dpkg -i libllvm9_9.0.1-8build1_amd64.deb
RUN dpkg -i iwyu_8.0-3_amd64.deb

# Set compiler to clang (to make cmake choose the right flags for iwyu)
ENV CC clang-9
ENV CXX clang++-9

# Help iwyu to find builtin headers
ENV CFLAGS "-isystem /usr/lib/llvm-9/lib/clang/9.0.0/include/"
ENV CXXFLAGS "-isystem /usr/lib/llvm-9/lib/clang/9.0.0/include/"

# copy everything we need from kotekan
WORKDIR /code
COPY include include/
COPY tools tools/
COPY iwyu.kotekan.imp iwyu.kotekan.imp
COPY CMakeLists.txt CMakeLists.txt
COPY kotekan kotekan/
COPY python python/
COPY tests tests/
COPY lib lib/
COPY cmake cmake/
COPY scripts scripts/
COPY config config/

# get ready to run cmake
WORKDIR /code/build

RUN cmake -DRTE_SDK=/opt/dpdk -DRTE_TARGET=x86_64-native-linuxapp-gcc  -DUSE_DPDK=ON -DUSE_HSA=ON \
    -DCMAKE_BUILD_TYPE=Debug -DUSE_HDF5=ON -DHIGHFIVE_PATH=/code/build/HighFive \
    -DOPENBLAS_PATH=/code/build/OpenBLAS -DUSE_LAPACK=ON -DBLAZE_PATH=/code/build/blaze \
    -DUSE_OMP=ON -DBOOST_TESTS=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON ..

# Do nothing when the container launches
CMD ["/bin/bash"]
