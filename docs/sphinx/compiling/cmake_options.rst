****************
CMake Options
****************

This page lists the user-facing CMake cache options supported by kotekan. Pass options to CMake via
``-D<OPTION>=<VALUE>``, for example:

.. code:: bash

   cmake -S . -B build -DUSE_CUDA=ON -DWERROR=ON

Most feature switches are tri-state and accept ``AUTO`` (probe dependencies and enable when
available), ``ON`` (require the feature and fail if the dependency is missing), or ``OFF`` (skip the
feature entirely). The default value for each option is shown in parentheses below.


General
========

- ``CMAKE_BUILD_TYPE`` (``Test``): Choose ``Debug`` for full symbols, ``Release`` for an optimised build, or
  ``Test`` for asserts/logging without debug symbols.
- ``ARCH`` (``native``): Override the architecture passed to ``-march``/``-mtune`` if you need to target
  a different CPU.
- ``CMAKE_LINK_WHAT_YOU_USE`` (``OFF``): Request link-time diagnostics for unused link dependencies.
  Useful when hunting for missing ``target_link_libraries`` entries, but slows the link step.


GPU Backends
=============

- ``USE_CUDA`` (``AUTO``): Enable the CUDA GPU backend when ``nvcc`` and the CUDA toolkit are discoverable.
  Adds ``-DWITH_CUDA`` and builds CUDA stages.
- ``USE_OPENCL`` (``OFF``): Enable the OpenCL backend. Set to ``AUTO``/``ON`` when OpenCL headers and
  libraries are installed; the build installs kernel files to ``/var/lib/kotekan/opencl``.
- ``USE_HIP`` (``OFF``): Enable the HIP backend when the HIP toolchain is available.


I/O and File Formats
=====================

- ``USE_HDF5`` (``AUTO``): Build HDF5 output stages when HDF5, HighFive, and the runtime plugin directory
  are all detected. Populates ``KOTEKAN_HDF5_PLUGIN_DIR`` for runtime configuration.
- ``USE_ASDF`` (``AUTO``): Build ASDF output stages when ``asdf-cxx`` is found; defines ``-DWITH_ASDF_CXX``.
- ``USE_GDAL`` (``AUTO``): Build GDAL output stages when GDAL headers/libs are available; defines
  ``-DWITH_GDAL``.


Math Libraries
===============

- ``USE_FFTW`` (``AUTO``): Enable the FFTW F-engine when FFTW is installed; defines ``-DWITH_FFTW``.
- ``USE_LAPACK_BLAZE`` (``AUTO``): Enable LAPACKE/OpenBLAS plus Blaze-based linear algebra stages when
  both dependencies are present. Adds the necessary compile definitions for Blaze/OpenBLAS.


Core & Runtime Features
========================

- ``USE_AIRSPY`` (``AUTO``): Build the Airspy capture stages when ``libairspy`` is present; defines
  ``-DWITH_AIRSPY``.
- ``USE_JULIA`` (``AUTO``): Enable Julia-backed components when the Julia executable and C API are
  available.
- ``USE_OMP`` (``AUTO``): Append OpenMP compile/link flags when OpenMP is supported. Automatically
  switches to ``OFF`` if the compiler lacks OpenMP support.
- ``USE_NUMA`` (``ON``): Link ``libnuma`` and enable NUMA-aware buffer management. Required for DPDK.
- ``USE_DPDK`` (``AUTO``): Enable DPDK features when ``libdpdk >= 19.11`` is discoverable via
  ``pkg-config``. This option is forced ``OFF`` when ``WITH_BOOST_TESTS`` is ``ON`` to avoid linker
  conflicts.
- ``USE_OPENSSL`` (``AUTO``): Link OpenSSL (libcrypto) for hashing support in the core. Combine with
  ``-DOPENSSL_ROOT_DIR=<path>`` for non-standard installations.
- ``NO_MEMLOCK`` (``OFF``): Define ``WITH_NO_MEMLOCK`` to skip memory locking and related policy calls,
  which can be necessary in containerised or restricted environments.


Build & Tooling
================

- ``WERROR`` (``ON``): Treat compiler warnings as errors for C/C++ sources. CUDA compilations still emit
  warnings but do not fail when ``WERROR`` is ``OFF`` thanks to ``--warn-no-error``.
- ``CCACHE`` (``OFF``): Use ``ccache`` as the compiler launcher when it is installed.
- ``COMPILE_DOCS`` (``OFF``): Build the Sphinx + Doxygen documentation tree. Even when enabled, the
  documentation is only generated when the ``docs`` target is built explicitly.
- ``IWYU`` (``OFF``): Run include-what-you-use on C/C++ compilations. Requires the ``include-what-you-use``
  executable and the mapping file (``iwyu.kotekan.imp``).
- ``SUPERDEBUG`` (``OFF``): Force ``-O0`` and keep frame pointers in Debug/Test builds for easier debugging
  and coverage analysis.
- ``SANITIZE`` (``OFF``): Enable the Clang/LLVM AddressSanitizer toolchain flags for Debug/Test builds.
- ``WITH_TESTS`` (``OFF``): Build and link the helper stages in ``lib/testing`` into the kotekan binary.
- ``WITH_BOOST_TESTS`` (``OFF``): Build the Boost.Test unit tests under ``tests/boost`` (requires
  ``pytest-cpp``). Automatically disables DPDK to avoid linker issues during the test build.


Common Paths and Overrides
===========================

- ``OPENSSL_ROOT_DIR``: Point to a non-standard OpenSSL install to help ``USE_OPENSSL`` locate the
  libraries.
- ``BLAZE_PATH``: Provide a custom include path for Blaze headers.
- ``CUDAToolkit_ROOT``: Override the search path for the CUDA toolkit when required.


Notes
======

- Auto-detection: When an option is ``AUTO`` and its dependency is missing, kotekan prints a warning and
  continues without the feature. Setting the option to ``ON`` causes configuration to fail instead.
- Summary output: At the end of configuration, CMake prints a colourised feature summary showing each
  feature’s status and the corresponding toggle flag.
- Developer reminder: When adding a new CMake option, also update this page, the configure summary
  (``cmake/Summary.cmake``), and the version metadata template (``lib/version/version.c.in``).
