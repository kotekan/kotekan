****************
CMake Options
****************

This page lists all user‑facing CMake options supported by kotekan, grouped by feature area. Pass options to CMake via ``-D<OPTION>=ON|OFF`` (or a value), for example:

.. code:: bash

   cmake -S . -B build -DUSE_CUDA=ON -DWERROR=ON

General
========

- ``CMAKE_BUILD_TYPE``: One of ``Debug`` (default), ``Release``, or ``Test``.
- ``ARCH``: CPU architecture tuning for ``-march``/``-mtune`` (default: ``native``).

GPU Backends
=============

- ``USE_CUDA``: Build CUDA GPU framework and CUDA stages. Requires ``nvcc`` and CUDAToolkit.
- ``USE_OPENCL``: Build OpenCL GPU framework and stages. Requires OpenCL headers/libs.
- ``USE_HIP``: Build HIP GPU framework and stages.
- ``USE_OLD_ROCM``: Build for older ROCm (<= 2.3). Off by default.

I/O and File Formats
=====================

- ``USE_HDF5``: Enable HDF5 output stages. Auto‑enabled if HDF5 is found; disable with ``OFF``.
- ``USE_ASDF``: Enable ASDF output stages. Auto‑enabled if asdf-cxx is found; disable with ``OFF``.
- ``USE_GDAL``: Enable GDAL output stages. Auto‑enabled if GDAL is found; disable with ``OFF``.

Math Libraries
===============

- ``USE_FFTW``: Enable FFTW F‑engine (auto‑enabled if FFTW is found; disable with ``OFF``).
- ``USE_LAPACK_BLAZE``: Enable LAPACKE/OpenBLAS + Blaze based linear algebra stages (auto‑enabled if found; disable with ``OFF``).

Other Features
===============

- ``USE_AIRSPY``: Enable Airspy producer (requires libairspy).
- ``USE_JULIA``: Enable Julia‑based features if Julia is available.
- ``USE_OMP``: Enable OpenMP in supported stages.
- ``NUMA``: Auto‑detected if ``libnuma`` is present and linked into core; no CMake toggle.
- ``USE_DPDK``: Control DPDK support. Values: ``ON`` (default, auto: prefer NEW >=19.11 via pkg‑config, else OLD), ``OFF`` (disable), ``NEW`` (require >=19.11), ``OLD`` (force legacy <19.11). The legacy ``USE_OLD_DPDK`` is still accepted and maps to ``USE_DPDK=OLD``.
- ``NO_MEMLOCK``: Do not mlock buffer memory (useful in constrained/container environments).

Testing and Tooling
====================

- ``WITH_TESTS``: Build and link the C++ testing helper library in ``lib/testing``. Does not build Boost unit tests.
- ``WITH_BOOST_TESTS``: Build C++ unit tests under ``tests/boost`` (Boost.Test). Disables DPDK to avoid linker issues. ``pytest-cpp`` is required for pytest discovery.
- ``IWYU``: Enable include‑what‑you‑use and print suggestions.
- ``CCACHE``: Use ccache to speed up builds (if available).
- ``WERROR``: Treat warnings as errors. CUDA compiles use ``--warn-no-error``.
- ``SUPERDEBUG``: Extra debug info with no optimization, handy for coverage/diagnostics.
- ``SANITIZE``: Enable AddressSanitizer for C/C++ builds.
- ``COMPILE_DOCS``: Build documentation (Sphinx + doxygen). Not part of the default build.

Common Paths and Legacy Variables
=================================

- ``OPENSSL_ROOT_DIR``: Point to a non‑standard OpenSSL install, e.g., ``-DOPENSSL_ROOT_DIR=/opt/openssl``.
- ``CUDAToolkit_ROOT``: Point to a non‑standard CUDA toolkit, e.g., ``-DCUDAToolkit_ROOT=/usr/local/cuda``.
- ``BLAZE_PATH``: Override Blaze header path if not in the default include locations, e.g., ``-DBLAZE_PATH=/opt/blaze/include``.
- ``RTE_SDK`` and ``RTE_TARGET``: For legacy DPDK (``-DUSE_DPDK=OLD``), specify a source build location, e.g., ``-DRTE_SDK=/opt/dpdk-16.11 -DRTE_TARGET=x86_64-native-linuxapp-gcc``.

Notes
======

- Auto‑detection: Many features auto‑enable when their dependencies are found. Disable them explicitly with ``-DUSE_<FEATURE>=OFF``.
- Summary output: At the end of configuration, CMake prints a colorized summary of features with the toggle flag to use for each.
