# Documentation

Compiled docs are available at https://kotekan.readthedocs.io/en/latest/.

[![Documentation Status](https://readthedocs.org/projects/kotekan/badge/?version=latest)](https://kotekan.readthedocs.io/en/latest/?badge=latest)

# Build Instructions

[![kotekan-ci-tests](https://github.com/kotekan/kotekan/actions/workflows/main.yml/badge.svg)](https://github.com/kotekan/kotekan/actions/workflows/main.yml)

Detailed instructions at https://kotekan.readthedocs.io/en/latest/.
Full list of CMake options: https://kotekan.readthedocs.io/en/latest/compiling/cmake_options.html

This project is built using cmake, so you will need to install cmake
before starting a build.

To build just the base framework:

	cd build
	cmake <options> ..
	make

Building documentation
----------------------

To build both the API (Doxygen) and user docs (Sphinx):

    mkdir -p build-docs
    cd build-docs
    cmake -DCOMPILE_DOCS=ON ..
    make docs

Individual doc targets are also available:

- `make doxygen` – generate API reference only
- `make sphinx` – build Sphinx HTML only

Cmake build options (defaults shown in parentheses; most feature toggles accept `AUTO`, `ON`, or `OFF`, with `AUTO`
probing for dependencies and falling back gracefully):

* `-DCMAKE_BUILD_TYPE=<Debug|Release|Test>` (`Test`) - Choose the configuration preset. `Debug` adds symbols, `Release`
  optimises, `Test` keeps asserts/logging without debug symbols.
* `-DARCH=<arch>` (`native`) - Override the CPU target passed to `-march/-mtune`.
* `-DUSE_CUDA=<AUTO|ON|OFF>` (`AUTO`) - Build the CUDA backend and enable CUDA stages when `nvcc` and the CUDA toolkit
  are available. Adds `-DWITH_CUDA` on success.
* `-DUSE_OPENCL=<AUTO|ON|OFF>` (`OFF`) - Build the OpenCL backend and install kernels to `/var/lib/kotekan/opencl` when
  OpenCL headers/libs are found.
* `-DUSE_HIP=<AUTO|ON|OFF>` (`OFF`) - Build the HIP backend when HIP is detected.
* `-DUSE_DPDK=<AUTO|ON|OFF>` (`AUTO`) - Enable DPDK stages when `libdpdk>=19.11` is present via `pkg-config`. Forced
  `OFF` automatically when `-DWITH_BOOST_TESTS=ON` to avoid linker issues.
* `-DUSE_AIRSPY=<AUTO|ON|OFF>` (`AUTO`) - Build the Airspy capture stages when `libairspy` is available.
* `-DUSE_ASDF=<AUTO|ON|OFF>` (`AUTO`) - Enable ASDF output stages when `asdf-cxx` is found; defines `-DWITH_ASDF_CXX`.
* `-DUSE_GDAL=<AUTO|ON|OFF>` (`AUTO`) - Enable GDAL output stages when GDAL is detected; defines `-DWITH_GDAL`.
* `-DUSE_HDF5=<AUTO|ON|OFF>` (`AUTO`) - Enable HDF5 output stages when HDF5, HighFive, and the runtime plugin directory
  are all available. Populates `KOTEKAN_HDF5_PLUGIN_DIR` for runtime use.
* `-DUSE_FFTW=<AUTO|ON|OFF>` (`AUTO`) - Enable the FFTW F-engine when FFTW is found; adds `-DWITH_FFTW`.
* `-DUSE_LAPACK_BLAZE=<AUTO|ON|OFF>` (`AUTO`) - Enable LAPACKE/OpenBLAS + Blaze linear algebra stages when both
  dependencies are present. Adds the required Blaze/OpenBLAS compile definitions.
* `-DUSE_JULIA=<AUTO|ON|OFF>` (`AUTO`) - Enable Julia-based components when the Julia executable and C API are
  available.
* `-DUSE_OMP=<AUTO|ON|OFF>` (`AUTO`) - Enable OpenMP and append the compiler/linker flags when OpenMP is detected.
  Automatically turns `OFF` if the compiler lacks support.
* `-DUSE_OPENSSL=<AUTO|ON|OFF>` (`AUTO`) - Link OpenSSL for hashing support in the core when libcrypto is available.
  Combine with `-DOPENSSL_ROOT_DIR=<path>` for non-standard installs.
* `-DUSE_NUMA=<AUTO|ON|OFF>` (`ON`) - Link libnuma and enable NUMA-aware buffer handling. Required when DPDK is enabled.
* `-DNO_MEMLOCK=<AUTO|ON|OFF>` (`OFF`) - Define `WITH_NO_MEMLOCK` to skip memory locking and policy calls (useful for
  containers or restricted environments).
* `-DWERROR=<AUTO|ON|OFF>` (`ON`) - Treat warnings as errors for C/C++, with `--warn-no-error` applied to CUDA
  compilations when disabled.
* `-DCCACHE=<AUTO|ON|OFF>` (`OFF`) - Use `ccache` as the compiler launcher when found.
* `-DCMAKE_LINK_WHAT_YOU_USE=<AUTO|ON|OFF>` (`OFF`) - Ask CMake to report link dependencies that are not actually used (
  may slow linking, useful for diagnosing missing link flags).
* `-DCOMPILE_DOCS=<AUTO|ON|OFF>` (`OFF`) - Build the Sphinx/doxygen documentation tree when the tooling is installed.
  Docs are excluded from the normal build even when enabled unless you build the `docs` target.
* `-DIWYU=<AUTO|ON|OFF>` (`OFF`) - Run include-what-you-use during compilation. Requires the `include-what-you-use`
  executable and mapping file.
* `-DSUPERDEBUG=<AUTO|ON|OFF>` (`OFF`) - Force `-O0` and retain frame pointers in Debug/Test configurations for deep
  debugging/coverage work.
* `-DSANITIZE=<AUTO|ON|OFF>` (`OFF`) - Enable the Clang/LLVM AddressSanitizer toolchain flags for Debug/Test builds.
* `-DWITH_TESTS=<AUTO|ON|OFF>` (`OFF`) - Build and link the helper stages from `lib/testing` into the kotekan binary (
  used by QA/example configs). Does not build unit tests.
* `-DWITH_BOOST_TESTS=<AUTO|ON|OFF>` (`OFF`) - Build the Boost.Test unit tests under `tests/boost` (requires
  `pytest-cpp`). Forces `USE_DPDK` to `OFF` while enabled.

Additional helpers:

* `-DOPENSSL_ROOT_DIR=<path>` - Point CMake at a non-default OpenSSL install when `USE_OPENSSL` is enabled.
* `-DBLAZE_PATH=<path>` - Provide a custom include path for Blaze headers if they are not in the standard search path.

**Examples:**

To build with OpenCL and debug symbols and logging:

    cmake -DUSE_OPENCL=ON -DCMAKE_BUILD_TYPE=Debug ..

To build with CUDA:

    cmake -DUSE_CUDA=ON ..

At the end of configuration, CMake prints a colorized feature summary indicating which features were enabled (found) or
disabled (missing/explicitly off). Each feature row shows its toggle flag, e.g.
`CUDA: ON (found, toggle: -DUSE_CUDA=ON/OFF)`. Use `-D<OPTION>=AUTO|ON|OFF` to auto-detect, require, or disable a
feature present on your system.

To install kotekan:

	make install

# Running kotekan

**Using systemd (full install)**

To start kotekan

    sudo systemctl start kotekan

To stop kotekan

    sudo systemctl stop kotekan

**To run in debug mode, run from `ch_gpu/build/kotekan/`**

    sudo ./kotekan -c <config_file>.yaml

For example:

    sudo ./kotekan -c ../../kotekan/kotekan_gpu_replay.yaml

When installed kotekan's config files are located at /etc/kotekan/

If running with no options then kotekan just starts a rest server, and waits for someone to send it a config in json
format on port `12048`
