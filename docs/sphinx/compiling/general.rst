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
* pthreads
* python and python-YAML

Required for some options
==========================

* pthread
* `libevent <http://libevent.org/>`_
* `DPDK dpdk-16.11.3 <http://dpdk.org/>`_ (see :ref:`dpdk`), requires hugepage support
* `AMD OpenCL drivers <http://support.amd.com/en-us/download/linux>`_ and `SDK <http://developer.amd.com/amd-accelerated-parallel-processing-app-sdk/>`_
* `AMD ROCm <https://github.com/RadeonOpenCompute/ROCm>`_
* `HDF5 <https://www.hdfgroup.org/HDF5/>`_ and `HighFive <https://github.com/jrs65/HighFive>`_
* `fftw3 <http://www.fftw.org/>`_
* `libairspy <https://github.com/airspy/airspyone_host/tree/master/libairspy>`_
* `OpenBLAS <http://www.openblas.net/>`_ (see :ref:`openblas`)

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
* ``-DRTE_SDK=<dir>``
* ``-DRTE_TARGET=x86_64-native-linuxapp-gcc``
    Includes DPDK support.
* ``-DUSE_HSA=ON``
    Build with HSA support if available. On by default.
* ``-DUSE_PRECOMPILED_OPENCL=ON``
    For HSA, use precompiled OpenCL .hsaco binaries. Otherwise will attempt cloc.sh to compile.
* ``-DUSE_OPENCL=ON``
    Build with OpenCL support.
* ``-DUSE_HCC=ON``
    Build with HCC support, must also set `CXX=hcc`, i.e. `CXX=hcc cmake -DUSE_HCC=ON ..`  This mode has limited support.
* ``-DDPDK_VDIF=ON``
    Adjusts DPDK defines to optimize for single dish VDIF capture mode.
* ``-DUSE_HDF5=ON``
* ``-DHIGHFIVE_PATH=<path>``
    To enable the HDF5 writer
* ``-DUSE_AIRSPY=ON``
    Build the AirSpy producer. Requires libairspy.
* ``-DUSE_FFTW=ON``
    Build an FFTW-based F-engine. Requires FFTW3.
* ``-DUSE_FFTW=ON``
    Build an FFTW-based F-engine. Requires FFTW3.
* ``-DUSE_LAPACK=ON``
    Build processes depending on LAPACK. Currently only OpenBLAS built from source is supported (see above).
* ``-DOPENBLAS_PATH=<openblas_prefix>``
    Path to OpenBLAS installation, if not in the ``CMAKE_PREFIX_PATH``
* ``-DCOMPILE_DOCS=ON``
    Build kotekan documentation. Requires doxygen, sphinx (+ sphinx_rtd_theme), and breathe. Note that docs will only compile if explicitly told to, it is not part of the base compile, even when enabled.

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
