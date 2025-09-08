..  _compiling:

*********************
Compiling **Kotekan**
*********************

**Kotekan** uses cmake to manage its compile-time settings. For a complete list of user‑facing options, see :doc:`cmake_options`.



.. toctree::
   :hidden:

   dpdk


.. toctree::
   :caption: Compiling on Specific Platforms
   :maxdepth: 1

   ubuntu
   macos

Base Requirements
=================

The minimal toolchain for a standard build includes:

* C++ compiler: GCC or Clang (C++17), plus build tools: ``build-essential``, ``cmake``, ``pkg-config``, ``make``, ``git``
* Libraries: ``libevent-dev``, ``libssl-dev``, ``libyaml-cpp-dev``
* Python: ``python3``, ``python3-dev``; Python packages: ``pyyaml``, ``jinja2``, ``requests``, ``tabulate``, ``futures``
* Optional but useful: ``ccache``, ``clang-format``, ``iwyu``

On Ubuntu (24.04 reference), install with:

.. code:: bash

   sudo apt update && sudo apt install \
     build-essential cmake pkg-config git make \
     libevent-dev libssl-dev libyaml-cpp-dev \
     python3 python3-dev \
     python3-yaml python3-jinja2 python3-requests python3-tabulate

Packages by Feature
===================

This section maps common CMake options (see :doc:`cmake_options`) to the packages typically needed on Ubuntu 24.04. For other distros, use the equivalent packages.

- ``USE_CUDA`` (CUDA GPU backend)
  - Ubuntu packages: ``cuda-toolkit`` (provides ``nvcc``)
  - Tip: If not in a standard location, set ``-DCUDAToolkit_ROOT=<path>``.

- ``USE_OPENCL`` (OpenCL GPU backend)
  - Ubuntu packages: ``opencl-headers``, ``ocl-icd-opencl-dev``, ``libopencl-clang-17-dev``

- ``USE_HIP`` (HIP/ROCm GPU backend)
  - Install ROCm per vendor instructions. On Ubuntu, install HIP runtime/toolchain packages for your ROCm version.

- ``USE_HDF5`` (HDF5 output)
  - Ubuntu packages: ``libhdf5-dev`` (and/or ``libhdf5-serial-dev``)
  - HighFive C++ headers are required; on Ubuntu, build and install HighFive from source (e.g., v3.1.1).

- ``USE_ASDF`` (ASDF output)
  - Requires ``asdf-cxx`` and dependencies. On Ubuntu:
    * System packages: ``libbz2-dev``, ``liblz4-dev``, ``libssl-dev``
    * Libraries from source: ``c-blosc2`` (v2.14+) and ``asdf-cxx``
    * Note: On some systems, the pkg-config file name for lz4 is ``liblz4.pc``; ensure ``lz4.pc`` is visible to pkg-config or add a symlink.

- ``USE_GDAL`` (GDAL output)
  - Ubuntu packages: ``libgdal-dev``

- ``USE_AIRSPY`` (Airspy producer)
  - Ubuntu packages: ``libairspy-dev``

- ``USE_FFTW`` (FFTW F-engine)
  - Ubuntu packages: ``libfftw3-dev``

- ``USE_LAPACK_BLAZE`` (LAPACK/OpenBLAS + Blaze stages)
  - Ubuntu packages: ``libopenblas-dev``, ``liblapacke-dev``
  - From source: Blaze headers (e.g., v3.8.2)

- ``USE_OMP`` (OpenMP)
  - Provided by your compiler; no extra package typically required with GCC on Ubuntu.

- ``USE_DPDK`` / DPDK support
  - Ubuntu packages: ``dpdk``, ``libdpdk-dev`` (and ``dpdk-dev``)
  - Default is ``ON`` (auto): prefer NEW (>=19.11) via pkg-config, else fall back to OLD (<19.11). Force a mode with ``-DUSE_DPDK=NEW`` or ``-DUSE_DPDK=OLD``. The legacy ``-DUSE_OLD_DPDK=ON`` is still accepted and maps to ``OLD``.

- ``WITH_TESTS`` (C++ testing helpers in ``lib/testing``)
  - No extra system packages beyond base requirements.

- ``WITH_BOOST_TESTS`` (Boost unit tests under ``tests/boost``)
  - Ubuntu packages: ``libboost-test-dev`` (plus typical Boost dependencies)
  - Note: This configuration disables DPDK to avoid linker issues.

- ``COMPILE_DOCS`` (documentation)
  - Ubuntu packages: ``doxygen``, ``graphviz``
  - Python: ``sphinx``, ``breathe``, ``sphinx_rtd_theme`` (see below)

Developer Python Packages
=========================

For linting, testing, and documentation, install these Python packages (often in a virtualenv):

.. code:: bash

   python3 -m pip install \
     black \
     cmake_format \
     pytest pytest-xdist pytest-timeout \
     pytest-cpp \
     sphinx==6.2.* sphinx_rtd_theme==2.0.* breathe==4.35.* \
     h5py hdf5plugin bitshuffle \
     numpy \
     requests tabulate \
     pyyaml jinja2 \
     msgpack posix_ipc \
     future futures


Notes
=====

- The Dockerfile at ``tools/docker/24.04/Dockerfile`` contains the authoritative package list for a full‑featured build environment on Ubuntu 24.04, including CUDA, OpenCL, DPDK, ASDF, Blaze, and HighFive. Use it as a reference for required versions and extra steps (e.g., building Blosc2/ASDF/Blaze/HighFive from source), since this documentation may get out of date.
- Many features auto‑enable when their dependencies are detected. Disable them explicitly with ``-DUSE_<FEATURE>=OFF``. The configuration summary shows each feature’s toggle flag and status.

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
    Builds the project with asserts, debug logging and debug symbols.
* ``-DCMAKE_BUILD_TYPE=Test``
    Builds the project with asserts and debug logging but without debug symbols.
* ``-DUSE_DPDK=ON|OFF|NEW|OLD``
    Control DPDK support. Default ``ON`` auto-detects: prefer NEW (>=19.11 via pkg-config),
    else fall back to OLD (<19.11 via FindDPDK). For legacy/source installs specify ``OLD``
    and provide `-DRTE_SDK=<dir>` and `-DRTE_TARGET=x86_64-native-linuxapp-gcc`.
    Not needed for newer versions of DPDK on Ubuntu 22.04+.
    The legacy ``-DUSE_OLD_DPDK=ON`` is still accepted and maps to ``OLD``. See :ref:`dpdk`.
* ``-DUSE_OLD_ROCM=ON``
    Build for ROCm versions 2.3 or older. Off by default.
* ``-DUSE_OPENCL=ON``
    Build with OpenCL support.
* ``-DUSE_CUDA=ON``
    Build support for CUDA kernels and Nvidia GPUs, requires `nvcc`
* HDF5/ASDF/GDAL auto‑enable when installed; disable with ``-DUSE_HDF5=OFF``, ``-DUSE_ASDF=OFF``, or ``-DUSE_GDAL=OFF``.
* ``-DUSE_AIRSPY=ON``
    Build the AirSpy producer. Requires libairspy.
* ``-DUSE_FFTW=OFF``
    Disable building the FFTW-based F-engine (FFTW enables automatically if installed).
* ``-DUSE_LAPACK_BLAZE=OFF``
    Disable stages depending on LAPACK and BLAZE (they enable automatically if installed).
* ``-DUSE_OMP=ON``
    Build stages using OpenMP. This requires a compiler supporting OpenMP (>= 3.0, see `OpenMP Compilers and Tools <https://www.openmp.org/resources/openmp-compilers-tools/>`).
* ``-DCOMPILE_DOCS=ON``
    Build kotekan documentation. Requires doxygen, sphinx (+ sphinx_rtd_theme), and breathe. Note that docs will only compile if explicitly told to, it is not part of the base compile, even when enabled.
* ``-DOPENSSL_ROOT_DIR=<openssl_root_dir>``
    Location of the openssl libs and includes.
* ``-DWITH_TESTS=ON``
    Build and link kotekan's C++ testing helper library in ``lib/testing`` (used by some example/QA configs). This does not build Boost unit tests.
* ``-DWITH_BOOST_TESTS=ON``
    Build the C++ unit tests under ``tests/boost`` (Boost.Test). Also disables DPDK to avoid linker issues in this configuration. ``pytest-cpp`` is required for pytest to discover and run them.
* ``-DSUPERDEBUG=ON``
    Add extra debugging info and turn off all optimisation to improve coverage.
* ``-DSANITIZE=ON``
    Turn on extra Clang sanitizers (currently the address sanitizer) for finding issues.

Examples
---------

To build with (old) DPDK and debug symbols:

.. code:: bash

    cmake -DRTE_TARGET=x86_64-native-linuxapp-gcc -DUSE_DPDK=OLD -DCMAKE_BUILD_TYPE=Debug ..

To build with OpenCL and DPDK:

.. code:: bash

    cmake -DRTE_TARGET=x86_64-native-linuxapp-gcc -DUSE_DPDK=OLD -DUSE_OPENCL=ON ..

At the end of configuration, a colorized feature summary lists enabled/disabled features, reasons, and the toggle flag (e.g., ``toggle: -DUSE_CUDA=ON/OFF``). Use ``-D<OPTION>=ON|OFF`` to include or exclude a feature present on your system.

To install kotekan:

.. code:: bash

    make install

.. [#] To install python packages for all users:
    ``sudo su; cd ~; pip3 install python-module``
