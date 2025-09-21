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
  - Default is ``AUTO``: the build enables DPDK when ``libdpdk >= 19.11`` is found via ``pkg-config`` and otherwise prints a warning. Set ``-DUSE_DPDK=ON`` to require it or ``OFF`` to disable it explicitly.

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

Most feature toggles accept ``AUTO``, ``ON``, or ``OFF`` (defaults shown). ``AUTO`` enables the
feature when its dependencies are present and otherwise continues without it. See
:doc:`cmake_options` for the full catalogue and additional notes.

* ``-DCMAKE_BUILD_TYPE=<Debug|Release|Test>`` (default ``Test``)
    Choose the configuration preset. ``Debug`` keeps debug symbols, ``Release`` optimises, and ``Test`` retains asserts/logging without symbols.
* ``-DUSE_CUDA=<AUTO|ON|OFF>`` (default ``AUTO``)
    Enable the CUDA backend when ``nvcc`` and the CUDA toolkit are available; defines ``WITH_CUDA`` on success.
* ``-DUSE_OPENCL=<AUTO|ON|OFF>`` (default ``OFF``)
    Build the OpenCL backend. Set to ``AUTO``/``ON`` when OpenCL headers/libs are installed.
* ``-DUSE_HIP=<AUTO|ON|OFF>`` (default ``OFF``)
    Build the HIP backend when the HIP toolchain is available.
* ``-DUSE_DPDK=<AUTO|ON|OFF>`` (default ``AUTO``)
    Enable DPDK components when ``libdpdk >= 19.11`` is discoverable via ``pkg-config``. Forced ``OFF`` when ``-DWITH_BOOST_TESTS=ON``.
* ``-DUSE_AIRSPY=<AUTO|ON|OFF>`` (default ``AUTO``)
    Build the Airspy capture stages when ``libairspy`` is present.
* ``-DUSE_ASDF``, ``-DUSE_GDAL``, ``-DUSE_HDF5`` (each default ``AUTO``)
    Enable ASDF, GDAL, and HDF5 output stages when their dependencies are installed. Disable explicitly with ``=OFF`` if you do not want the feature even when the libraries are available.
* ``-DUSE_FFTW=<AUTO|ON|OFF>``, ``-DUSE_LAPACK_BLAZE=<AUTO|ON|OFF>`` (defaults ``AUTO``)
    Enable FFTW- and LAPACK/Blaze-based stages when the corresponding math libraries are found.
* ``-DUSE_JULIA=<AUTO|ON|OFF>`` (default ``AUTO``)
    Enable Julia-backed features when the Julia executable and C API are accessible.
* ``-DUSE_OMP=<AUTO|ON|OFF>`` (default ``AUTO``)
    Append OpenMP compiler/linker flags when OpenMP support is detected.
* ``-DUSE_OPENSSL=<AUTO|ON|OFF>`` (default ``AUTO``) with ``-DOPENSSL_ROOT_DIR=<path>`` when needed
    Link OpenSSL for hashing support in the core; point CMake at a custom OpenSSL install via ``OPENSSL_ROOT_DIR``.
* ``-DUSE_NUMA=<AUTO|ON|OFF>`` (default ``ON``) and ``-DNO_MEMLOCK=<AUTO|ON|OFF>`` (default ``OFF``)
    Control NUMA-aware buffering and whether kotekan attempts to ``mlock`` buffers (useful in containers).
* ``-DWITH_TESTS=<AUTO|ON|OFF>``, ``-DWITH_BOOST_TESTS=<AUTO|ON|OFF>`` (defaults ``OFF``)
    Build the helper stages in ``lib/testing`` and the Boost unit tests under ``tests/boost`` (requires ``pytest-cpp``). The Boost tests configuration disables DPDK automatically.
* ``-DCOMPILE_DOCS=<AUTO|ON|OFF>`` (default ``OFF``)
    Build the documentation tree (Sphinx + Doxygen) when the tooling is installed. The docs target must be built explicitly.
* ``-DWERROR=<AUTO|ON|OFF>``, ``-DCCACHE=<AUTO|ON|OFF>``, ``-DIWYU=<AUTO|ON|OFF>`` (defaults ``ON``/``OFF``/``OFF``)
    Treat warnings as errors, enable ``ccache`` as the compiler launcher, or run include-what-you-use during compilation.
* ``-DSUPERDEBUG=<AUTO|ON|OFF>``, ``-DSANITIZE=<AUTO|ON|OFF>`` (defaults ``OFF``)
    Request extra debugging instrumentation (``SUPERDEBUG``) or enable AddressSanitizer (``SANITIZE``) in Debug/Test builds.

Examples
---------

To build with DPDK and debug symbols:

.. code:: bash

    cmake -DUSE_DPDK=ON -DCMAKE_BUILD_TYPE=Debug ..

To probe for GPU backends that are installed locally:

.. code:: bash

    cmake -DUSE_CUDA=AUTO -DUSE_OPENCL=AUTO ..

At the end of configuration, a colorized feature summary lists enabled/disabled features, reasons, and the toggle flag (e.g., ``toggle: -DUSE_CUDA=ON/OFF``). Use ``-D<OPTION>=AUTO|ON|OFF`` to require, auto-detect, or disable individual features.

.. note:: Developers adding new build options should refresh the summary logic
   (``cmake/Summary.cmake``), this guide, and the version metadata in
   ``lib/version/version.c.in``.

To install kotekan:

.. code:: bash

    make install

.. [#] To install python packages for all users:
    ``sudo su; cd ~; pip3 install python-module``
