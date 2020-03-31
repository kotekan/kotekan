..  _compiling:

*********************
Compiling **Kotekan**
*********************

**Kotekan** uses cmake to manage its compile-time settings. A host of options are supported, and will some day be described here.



.. toctree::
   :hidden:

   dpdk
   openblas


.. toctree::
   :caption: Compiling on Specific Platforms
   :maxdepth: 1

   ubuntu
   centos
   macos

Required
=========
* ``pthread``
* ``python`` and ``python-YAML``
* `libevent <http://libevent.org/>`_

Required for some options
==========================

* `DPDK <http://dpdk.org/>`_ (see :ref:`dpdk`)
* `AMD ROCm <https://github.com/RadeonOpenCompute/ROCm>`_ (this also installs AMD OpenCL drivers and SDK)
* `CUDA <https://developer.nvidia.com/cuda-downloads>`_ Required for use with Nvidia cards
* `HDF5 <https://www.hdfgroup.org/HDF5/>`_::

    sudo apt-get install libhdf5-serial-dev

* Branch ``extensible-datasets`` of jrs65's fork of `HighFive <https://github.com/jrs65/HighFive>`_
* Clone HighFive in ``/opt`` with::

    git clone --single-branch --branch extensible-datasets https://github.com/jrs65/HighFive.git

* `h5py <http://docs.h5py.org/en/latest/build.html#source-installation>`_ (Install from source for bitshuffle)::

    python setup.py configure --hdf5=/usr/lib/x86_64-linux-gnu/hdf5/serial/ --hdf5-version=1.10.0
    python setup.py install

* `Bitshuffle <https://github.com/kiyo-masui/bitshuffle>`_ (Needs to be installed after h5py)::

    export HDF5_PLUGIN_PATH=/usr/local/hdf5/lib/plugin
    python setup.py install --h5plugin --h5plugin-dir=/usr/local/hdf5/lib/plugin

* `fftw3 <http://www.fftw.org/>`_
* `libairspy <https://github.com/airspy/airspyone_host/tree/master/libairspy>`_
* `OpenBLAS <http://www.openblas.net/>`_ (see :ref:`openblas`)
* `LAPACK <http://www.netlib.org/lapack/>`_ (Should automatically be installed with OpenBLAS if a Fortran compiler is present on your system)
* `Blaze <https://bitbucket.org/blaze-lib/blaze/src/master/>`_
  * Download and unpack into `/opt`
* OpenSSL (Package is called ``libssl-dev`` on Ubuntu).

Unit tests:
-----------

* `pytest-cpp <https://github.com/pytest-dev/pytest-cpp>`_ [#]_::

    sudo pip3 install pytest-cpp pytest-xdist
    sudo pip3 install atomicwrites pluggy py packaging numpy

* `future <https://pypi.org/project/future/>`_::

    sudo pip3 install future

* `msgpack <https://pypi.org/project/msgpack/>`_::
    
    sudo pip3 install msgpack

* `requests <https://pypi.org/project/requests/>`_::

    sudo pip3 install requests

* `BOOST <https://www.boost.org/>`_::

    sudo apt-get install libboost-all-dev

Code Formatting:
----------------
* `Clang format 8 <https://apt.llvm.org/>`_. For Ubuntu 18.04:

    - Copy these two lines to ``/etc/apt/sources.list``:: 

        deb http://apt.llvm.org/bionic/ llvm-toolchain-bionic-8 main    
        deb-src http://apt.llvm.org/bionic/ llvm-toolchain-bionic-8 main

    - Add the key::

        wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | sudo apt-key add -

    - Install clang-format-8::

        sudo apt update
        sudo apt install clang-format-8

Documentation:
--------------
* Doxygen::

    sudo apt-get install doxygen

* Dot:: 

    sudo apt-get install graphviz

* Sphinx::

    sudo apt-get install python-sphinx
    sudo pip install sphinx_rtd_theme sphinxcontrib-plantuml

* PlantUml::

    sudo wget https://phoenixnap.dl.sourceforge.net/project/plantuml/plantuml.jar -P /opt/plantuml
    sudo apt-get install default-jre

* Breathe::

    sudo pip install breathe

* Black::

    sudo pip3 install black

Hardware
=========

To support the latest HSA builds with full networking stack:

* NIC supporting DPDK, ideally Intel XL710 based
* CPU supporting AVX2, 4 memory channels, and at least 4 real cores. e.g. Intel E5-2620 v3 or i7-5930K
* AMD GPUs R9 Fury (s9300x2), RX 480/580, RX Vega, or later.
* RAM >= 16GB

To support OpenCL builds with the full networking stack:

* NIC supporting DPDK, ideally Intel XL710 based
* CPU supporting AVX2, 4 memory channels, and at least 4 real cores. e.g. Intel E5-2620 v3 or i7-5930K
* AMD GPUs R9 2XX or later.
* RAM >= 16GB


Build Instructions
===================

Base framework
----------------

.. code:: bash

    cd build
    cmake ..
    make

Cmake build options
-------------------

* ``-DCMAKE_BUILD_TYPE=Debug``
    Builds the project with debug symbols.
* ``-DUSE_DPDK=ON``
    Builds with DPDK support, for source installs requires: `-DRTE_SDK=<dir>`
    and `-DRTE_TARGET=x86_64-native-linuxapp-gcc`
    See :ref:`dpdk` for more details.
* ``-DUSE_HSA=ON``
    Build with HSA support if available. Off by default.
* ``-DUSE_OLD_ROCM=ON``
    Build for ROCm versions 2.3 or older. Off by default.
* ``-DUSE_OPENCL=ON``
    Build with OpenCL support.
* ``-DUSE_HCC=ON``
    Build with HCC support, must also set `CXX=hcc`, i.e. `CXX=hcc cmake -DUSE_HCC=ON ..`  This mode has limited support.
* ``-DUSE_CUDA=ON``
    Build support for CUDA kernels and Nvidia GPUs, requires `nvcc`
* ``-DUSE_HDF5=ON``
    Build with HDF5 support. Requires HighFive, Bitshuffle and h5py.
* ``-DHIGHFIVE_PATH=<path>``
    To enable the HDF5 writer
* ``-DUSE_AIRSPY=ON``
    Build the AirSpy producer. Requires libairspy.
* ``-DUSE_FFTW=ON``
    Build an FFTW-based F-engine. Requires FFTW3.
* ``-DUSE_LAPACK=ON``
    Build stages depending on LAPACK. Currently only OpenBLAS built from source is supported (see above).
* ``-DUSE_OMP=ON``
    Build stages using OpenMP. This requires a compiler supporting OpenMP (>= 3.0, see `OpenMP Compilers and Tools <https://www.openmp.org/resources/openmp-compilers-tools/>`).
* ``-DOPENBLAS_PATH=<openblas_prefix>``
    Path to OpenBLAS installation, if not in the ``CMAKE_PREFIX_PATH``
* ``-DCOMPILE_DOCS=ON``
    Build kotekan documentation. Requires doxygen, sphinx (+ sphinx_rtd_theme), and breathe. Note that docs will only compile if explicitly told to, it is not part of the base compile, even when enabled.
* ``-DOPENSSL_ROOT_DIR=<openssl_root_dir>``
    Location of the openssl libs and includes.
* ``-DBOOST_TESTS=ON``
    Build tests using The Boost Test Framework. pytest-cpp needs to be installed for pytest to find them.
* ``-DSUPERDEBUG=ON``
    Add extra debugging info and turn off all optimisation to improve coverage.
* ``-DSANITIZE=ON``
    Turn on extra Clang sanitizers (currently the address sanitizer) for finding issues.

Examples
---------

To build with HSA, DPDK and debug symbols:

.. code:: bash

    cmake -DRTE_SDK=/opt/dpdk-stable-16.11.4/ -DRTE_TARGET=x86_64-native-linuxapp-gcc -DUSE_DPDK=ON -DUSE_HSA=ON -DCMAKE_BUILD_TYPE=Debug ..

To build with OpenCL and DPDK:

.. code:: bash

    cmake -DRTE_SDK=/opt/dpdk-stable-16.11.4/ -DRTE_TARGET=x86_64-native-linuxapp-gcc -DUSE_DPDK=ON -DUSE_OPENCL=ON ..

To install kotekan:

.. code:: bash

    make install

.. [#] To install python packages for all users:
    ``sudo su; cd ~; pip3 install python-module``
