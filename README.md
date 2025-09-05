
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

Cmake build options:

* `-DCMAKE_BUILD_TYPE=Debug` - Builds the project with asserts, debug logging and debug symbols.
* `-DCMAKE_BUILD_TYPE=Test` - Builds the project with asserts, debug logging, but without debug
  symbols.
* `-DUSE_OLD_DPDK=ON` - Include DPDK support for older (<19.11) versions (newer is included by default if installed).
  Optional `-DRTE_SDK=<build-location>` and `-DRTE_TARGET=x86_64-native-linuxapp-gcc`
  can be provided for non standard build locations.
* `-DUSE_OLD_ROCM=ON` - Build for ROCm versions 2.3 or older. Off by default.
  Direct toolchain calls are used by default.
* `-DUSE_OPENCL=ON` - Build with OpenCL support.
* `-DUSE_CUDA=ON` - Build with CUDA support, requires `nvcc`
* `-DUSE_HDF5=ON`, `-DUSE_ASDF=ON`, or `-DUSE_GDAL=ON`: HDF5, ASDF, and GDAL are auto‑enable if installed.
* `-DUSE_AIRSPY=ON` - Build the AirSpy producer. Requires libairspy.
* `-DUSE_FFTW=OFF` or `-DUSE_LAPACK_BLAZE=OFF` - FFTW and LAPACK/Blaze are auto‑enabled if installed. Optionally set `-DBLAZE_PATH=<blaze_path>` for Blaze headers.
* `-DCOMPILE_DOCS=ON` - Build kotekan documentation. Requires doxygen,
  sphinx (+ sphinx_rtd_theme), and breathe.
  Note that docs will only compile if explicitly told to,
  it is not part of the base compile, even when enabled.
* `-DUSE_OMP=ON` Build stages using OpenMP. This requires a compiler supporting OpenMP (>= 3.0)
* `-DOPENSSL_ROOT_DIR=<openssl_root_dir>` Only required for non-standard install locations of OpenSSL
* `-DWITH_TESTS=ON` Build and link kotekan's C++ testing helper library in `lib/testing` (used by some example/QA configs). This does not build Boost unit tests.
* `-DWITH_BOOST_TESTS=ON` Build the C++ unit tests under `tests/boost` (Boost.Test). Also disables DPDK to avoid linker issues in that configuration. `pytest-cpp` is required for pytest to discover and run them.
* `-DSUPERDEBUG=ON` Add extra debugging info and turn off all optimisation to improve coverage.
* `-DSANITIZE=ON` Turn on extra Clang sanitizers (currently the address sanitizer) for finding issues.

**Examples:**

To build with OpenCL and debug symbols and logging:

    cmake -DUSE_OPENCL=ON -DCMAKE_BUILD_TYPE=Debug ..

To build with CUDA:

    cmake -DUSE_CUDA=ON ..

At the end of configuration, CMake prints a colorized feature summary indicating which features were enabled (found) or disabled (missing/explicitly off). Each feature row shows its toggle flag, e.g. `CUDA: ON (found, toggle: -DUSE_CUDA=ON/OFF)`. Use `-D<OPTION>=ON|OFF` to include or exclude a feature present on your system.

To install kotekan (only works on CentOS at the moment):

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
