![](https://github.com/kotekan/kotekan/workflows/include-what-you-use/badge.svg?branch=develop)

# Documentation

Compiled docs are currently available at http://lwlab.dunlap.utoronto.ca/kotekan.
One day we'll move them to readthedocs...

# Build Instructions

Detailed instructions at http://lwlab.dunlap.utoronto.ca/kotekan/compiling/general.html.

The project is build using cmake, so you will need to install cmake
before starting a build.

To build just the base framework:

	cd build
	cmake <options> ..
	make

Cmake build options:

* `-DCMAKE_BUILD_TYPE=Debug` - Builds the project with asserts, debug logging and debug symbols.
* `-DCMAKE_BUILD_TYPE=Test` - Builds the project with asserts, debug logging, but without debug
symbols.
* `-DUSE_DPDK=ON` - Include DPDK support.  Optional `-DRTE_SDK=<build-location>` and
  `-DRTE_TARGET=x86_64-native-linuxapp-gcc` can be provided for non standard build locations.
* `-DUSE_HSA=ON` - Build with HSA support if available. On by default.
* `-DUSE_OLD_ROCM=ON` - Build for ROCm versions 2.3 or older. Off by default.
* `-DUSE_CLOC=ON` - For HSA, use cloc.sh to compile .hsaco binaries.
  Direct toolchain calls are used by default.
* `-DUSE_OPENCL=ON` - Build with OpenCL support.
* `-DUSE_CUDA=ON` - Build with CUDA support, requires `nvcc`
* `-DUSE_HDF5=ON` and `-DHIGHFIVE_PATH=<path>` - To enable the HDF5 writer
* `-DUSE_AIRSPY=ON` - Build the AirSpy producer. Requires libairspy.
* `-DUSE_FFTW=ON` - Build an FFTW-based F-engine. Requires FFTW3.
* `-DUSE_LAPACK=ON` - Build stages depending on LAPACK.
* `-DBLAZE_PATH=<blaze_path>` - Path to BLAZE headers. Required if `USE_LAPACK` is set.
* `-DCOMPILE_DOCS=ON` - Build kotekan documentation. Requires doxygen,
  sphinx (+ sphinx_rtd_theme), and breathe.
  Note that docs will only compile if explicitly told to,
  it is not part of the base compile, even when enabled.
* `-DUSE_OMP=ON` Build stages using OpenMP. This requires a compiler supporting OpenMP (>= 3.0)
* `-DOPENSSL_ROOT_DIR=<openssl_root_dir>` Only required for non-standard install locations of OpenSSL
* `-DWITH_TESTS=ON` Build kotekans test library and C++ unit tests using The Boost Test Framework.
  pytest-cpp needs to be installed for pytest to find them.
* `-DSUPERDEBUG=ON` Add extra debugging info and turn off all optimisation to improve coverage.
* `-DSANITIZE=ON` Turn on extra Clang sanitizers (currently the address sanitizer) for finding issues.

**Examples:**

To build with HSA, DPDK and debug symbols:

    cmake -DRTE_SDK=/opt/dpdk-stable-16.11.4/ -DRTE_TARGET=x86_64-native-linuxapp-gcc -DUSE_DPDK=ON -DUSE_HSA=ON -DCMAKE_BUILD_TYPE=Debug ..

To build with OpenCL and DPDK:

    cmake -DRTE_SDK=/opt/dpdk-stable-16.11.4/ -DRTE_TARGET=x86_64-native-linuxapp-gcc -DUSE_DPDK=ON -DUSE_OPENCL=ON ..

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

If running with no options then kotekan just starts a rest server, and waits for someone to send it a config in json format on port `12048`

# Building the Docker image

From the outermost directory of the repo, run:

	docker build -f tools/docker/Dockerfile.full -t kotekan:latest .

This will build the full image with kotekan and python-kotekan installed.

Tag for local registry and to dockerhub:
    docker tag kotekan:full chimefrb/kotekan:outrig
    docker tag kotekan:full aux:5000/kotekan:latest

Then `docker push` both.
    
