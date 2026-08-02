#!/bin/bash
# Build one of the offline GNSS diagnostic tools against the already-built kotekan libraries.
# usage: ./build_tool.sh <name>          (compiles <name>.cpp -> <name> in this directory)
set -e
K=/home/kvand/gnss/kotekan
B=$K/build
S="$(cd "$(dirname "$0")" && pwd)"
N=$1

INC="-I$K/lib/stages/. -I$K/lib/stages/gnss -I/usr/include/gdal -I/usr/local/cuda/targets/x86_64-linux/include
 -I$K/lib/utils/. -I$K/lib/core/. -I$K/lib/version/. -I$K/external/MurmurHash3/. -I$K/external/modp_b64/.
 -I$K/lib/testing/. -I$K/lib/metadata/. -I$K/lib/cuda/. -I$K/lib/gpu/. -I$K/lib/dpdk/.
 -isystem $B/blaze_overrides -isystem /usr/include/libairspy -isystem /usr/include/hdf5/serial
 -isystem $K/external/fmt -isystem $K/external/gsl -isystem $K/external/json
 -isystem $K/external/ksgpu/include/. -isystem $K/external/ksgpu/. -isystem $K/external/n2k/include
 -isystem $K/external/n2k"
DEF="-DBLAZE_BLAS_IS_PARALLEL=0 -DBLAZE_BLAS_MODE=1 -DCL_TARGET_OPENCL_VERSION=220 -DGNSS_CUDA=1
 -DWITH_AIRSPY -DWITH_ASDF_CXX -DWITH_CUDA -DWITH_FFTW -DWITH_GDAL -DWITH_HDF5 -DWITH_OMP -DWITH_SSL"
FLAGS="-fopenmp -fcx-limited-range -O3 -DNDEBUG -std=c++17 -march=native -mtune=native -D_GNU_SOURCE"

/usr/bin/c++ $DEF $INC $FLAGS -MD -MF "$S/$N.o.d" -o "$S/$N.o" -c "$S/$N.cpp"

/usr/bin/c++ -fopenmp -O3 "$S/$N.o" -o "$S/$N" \
  -L/usr/local/cuda/targets/x86_64-linux/lib \
  -Wl,--start-group \
  $B/lib/stages/libkotekan_stages.a $B/lib/utils/libkotekan_utils.a $B/lib/core/libkotekan_core.a \
  $B/lib/cuda/libkotekan_cuda.a $B/lib/gpu/libkotekan_gpu.a $B/lib/metadata/libkotekan_metadata.a \
  $B/lib/testing/libkotekan_testing.a $B/lib/version/libkotekan_version.a $B/lib/dpdk/libkotekan_dpdk.a \
  $B/external/MurmurHash3/libMurmurHash3.a $B/external/modp_b64/libmodp_b64.a \
  $B/external/ksgpu/libksgpu.a \
  -Wl,--end-group \
  -lcudart -lcuda -lfftw3 -lfftw3f -lhdf5_serial -lyaml-cpp -lssl -lcrypto -lz -lnuma -lpthread -ldl -lm \
  -levent -levent_pthreads -levent_extra -lairspy -lasdf-cxx -lbz2 -lblosc -llz4 -lzstd -lgdal -lcurl \
  -Wl,-rpath,/usr/lib/x86_64-linux-gnu/hdf5/serial
echo "built $S/$N"
