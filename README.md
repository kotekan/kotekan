# Documentation

Compiled docs are available at https://kotekan.readthedocs.io/latest/.

[![Documentation Status](https://img.shields.io/readthedocs/kotekan)](https://kotekan.readthedocs.io/latest/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# Build Instructions

[![kotekan-ci-tests](https://github.com/kotekan/kotekan/actions/workflows/main.yml/badge.svg)](https://github.com/kotekan/kotekan/actions/workflows/main.yaml)

Detailed instructions at https://kotekan.readthedocs.io/latest/.
Full list of CMake options: https://kotekan.readthedocs.io/latest/compiling/cmake_options.html

## Option A: Docker Build (Recommended)

The easiest way to build Kotekan is using our reproducible Docker container. This handles all dependencies (CUDA 13, GCC 14, Julia 1.10) automatically.

**Prerequisites:** You must have Docker and the NVIDIA Container Toolkit installed.
* See [`tools/docker/24.04/README.md`](tools/docker/24.04/README.md) for the host machine setup guide.

**1. Build the Environment**
```bash
cd tools/docker/24.04
docker build -t kotekan-build:latest .
cd ../../..
````

**2. Enter the Development Shell**
Mount the source code into the container to build it:

```bash
docker run --rm -it \
    --gpus all \
    --user $(id -u):$(id -g) \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    -v $(pwd):/code/kotekan \
    kotekan-build:latest
```

**3. Compile**
Once inside the container:

```bash
cmake /code/kotekan
make -j$(nproc)
```

-----

## Option B: Manual / Native Build

**⚠️ Warning:** Manual builds require installing a specific set of system libraries (CUDA 13, GCC 14, Boost, HDF5, DPDK, etc.). Please refer to [`tools/docker/24.04/Dockerfile`](https://www.google.com/search?q=tools/docker/24.04/Dockerfile) for the authoritative list of required `apt` packages and library versions.

To build just the base framework:

```bash
mkdir build
cd build
cmake <options> ..
make
```

### Building documentation

To build both the API (Doxygen) and user docs (Sphinx):

```bash
mkdir -p build-docs
cd build-docs
cmake -DCOMPILE_DOCS=ON ..
make docs
```

Individual doc targets are also available:

- `make doxygen` – generate API reference only
- `make sphinx` – build Sphinx HTML only

### CMake build options

Defaults are shown in **bold**. Most feature toggles accept `AUTO`, `ON`, or `OFF`.

| Option | Default | Description |
| :--- | :--- | :--- |
| `CMAKE_BUILD_TYPE` | **Test** | Choose preset: `Debug` (symbols), `Release` (optimized), or `Test` (asserts + logging). |
| `ARCH` | **native** | Helper to set `-march` or `-mtune` flags for the compiler. |
| `USE_CUDA` | **AUTO** | Enable CUDA support. |
| `USE_HIP` | **AUTO** | Enable HIP support (ROCm). |
| `USE_OPENCL` | **AUTO** | Enable OpenCL support. |
| `USE_DPDK` | **AUTO** | Enable DPDK support for high-speed networking. |
| `USE_NUMA` | **AUTO** | Enable NUMA awareness (requires `libnuma`). |
| `USE_AIRSPY` | **AUTO** | Enable Airspy SDR support. |
| `USE_FFTW` | **AUTO** | Enable FFTW support. |
| `USE_HDF5` | **AUTO** | Enable HDF5 file I/O. |
| `USE_GDAL` | **AUTO** | Enable GDAL support. |
| `USE_ASDF` | **AUTO** | Enable ASDF file I/O. |
| `USE_JULIA` | **AUTO** | Enable Julia language support. |
| `USE_LAPACK_BLAZE` | **AUTO** | Enable linear algebra via LAPACK/Blaze. |
| `WITH_TESTS` | **OFF** | Build the C++ unit tests under `lib/tests`. |
| `WITH_BOOST_TESTS` | **OFF** | Build the Boost.Test unit tests under `tests/boost` (requires `pytest-cpp`). Forces `USE_DPDK` to `OFF`. |

**Additional Helpers:**

* `-DOPENSSL_ROOT_DIR=<path>` - Point CMake at a non-default OpenSSL install when `USE_OPENSSL` is enabled.
* `-DBLAZE_PATH=<path>` - Provide a custom include path for Blaze headers if they are not in the standard search path.

**Examples:**

To build with OpenCL and debug symbols and logging:

```bash
cmake -DUSE_OPENCL=ON -DCMAKE_BUILD_TYPE=Debug ..
```

To build with CUDA:

```bash
cmake -DUSE_CUDA=ON ..
```

To install kotekan:

```bash
make install
```

# Running Kotekan

**Using systemd (if installed via package)**

```bash
sudo systemctl start kotekan
sudo systemctl stop kotekan
```

**Running manually (Development/Docker)**
From inside the build directory (`/code/build` in Docker):

```bash
# Run with a specific config file
./kotekan/kotekan -c <config_file>.yaml
```

For example:

```bash
./kotekan/kotekan -c ../config/examples/basic_pipeline.yaml
```

When installed, kotekan's default configuration file is at `/etc/kotekan/kotekan.yaml`.

---