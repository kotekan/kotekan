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
- ``USE_OLD_DPDK``: Force support for older DPDK (<19.11). Newer DPDK (>=19.11) is used automatically when present.
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

Notes
======

- Auto‑detection: Many features auto‑enable when their dependencies are found. Disable them explicitly with ``-DUSE_<FEATURE>=OFF``.
- Custom locations: For CUDA use ``-DCUDAToolkit_ROOT=<path>`` if not in a standard location. For OpenSSL use ``-DOPENSSL_ROOT_DIR=<path>``.
- Summary output: At the end of configuration, CMake prints a colorized summary of features with the toggle flag to use for each.

