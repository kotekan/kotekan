#!/bin/bash
# Build one of the offline GNSS diagnostic tools against the already-built kotekan libraries.
# usage: ./build_tool.sh <name>          (compiles <name>.cpp -> <name> in this directory)
#        KBUILD=build_nodpdk ./build_tool.sh <name>
#
# KBUILD picks the kotekan build tree to link against. It matters because /home/kvand is NFS
# shared: `build/` is a cx node's DPDK build and `build_nodpdk/` is cf06's, and linking a tool
# on cf06 against the node build fails on the DPDK symbols. Output goes to $S/$N.$HOSTNAME_SHORT
# with a plain $S/$N symlink, so two hosts building the same tool over NFS do not clobber each
# other's binary (they did, and the failure looks like "the tool suddenly segfaults").
set -e
K=/home/kvand/gnss/kotekan
B=$K/${KBUILD:-build}
S="$(cd "$(dirname "$0")" && pwd)"
N=$1
H=$(hostname -s)

INC="-I$K/lib/stages/. -I$K/lib/stages/gnss -I/usr/include/gdal -I/usr/local/cuda/targets/x86_64-linux/include
 -I$K/lib/utils/. -I$K/lib/core/. -I$K/lib/version/. -I$K/external/MurmurHash3/. -I$K/external/modp_b64/.
 -I$K/lib/testing/. -I$K/lib/metadata/. -I$K/lib/cuda/. -I$K/lib/gpu/. -I$K/lib/dpdk/.
 -isystem $B/blaze_overrides -isystem /usr/include/libairspy -isystem /usr/include/hdf5/serial
 -isystem $K/external/fmt -isystem $K/external/gsl -isystem $K/external/json
 -isystem $K/external/ksgpu/include/. -isystem $K/external/ksgpu/. -isystem $K/external/n2k/include
 -isystem $K/external/n2k -isystem $K/external/n2k_dual/include"
DEF="-DBLAZE_BLAS_IS_PARALLEL=0 -DBLAZE_BLAS_MODE=1 -DCL_TARGET_OPENCL_VERSION=220 -DGNSS_CUDA=1
 -DWITH_AIRSPY -DWITH_ASDF_CXX -DWITH_CUDA -DWITH_FFTW -DWITH_GDAL -DWITH_HDF5 -DWITH_OMP -DWITH_SSL"
FLAGS="-fopenmp -fcx-limited-range -O3 -DNDEBUG -std=c++17 -march=native -mtune=native -D_GNU_SOURCE"

/usr/bin/c++ $DEF $INC $FLAGS -MD -MF "$S/$N.$H.o.d" -o "$S/$N.$H.o" -c "$S/$N.cpp"

# Not every build tree has every library: cf06's build_nodpdk is configured without DPDK and
# without the testing lib. Link whatever this tree actually has rather than a fixed list.
LIBS=""
for l in lib/stages/libkotekan_stages.a lib/utils/libkotekan_utils.a lib/core/libkotekan_core.a \
         lib/cuda/libkotekan_cuda.a lib/gpu/libkotekan_gpu.a lib/metadata/libkotekan_metadata.a \
         lib/testing/libkotekan_testing.a lib/version/libkotekan_version.a \
         lib/dpdk/libkotekan_dpdk.a external/MurmurHash3/libMurmurHash3.a \
         external/modp_b64/libmodp_b64.a external/ksgpu/libksgpu.a; do
    [ -f "$B/$l" ] && LIBS="$LIBS $B/$l"
done

# The n2k correlator libs are SHARED; link + rpath them when the tree has them (tools that
# don't use them are unaffected -- lazy binding).
RPATHS=""
for l in external/n2k/libn2k.so external/n2k_dual/libn2k_dual.so; do
    [ -f "$B/$l" ] && LIBS="$LIBS $B/$l" && RPATHS="$RPATHS -Wl,-rpath,$B/$(dirname $l)"
done

/usr/bin/c++ -fopenmp -O3 "$S/$N.$H.o" -o "$S/$N.$H" \
  -L/usr/local/cuda/targets/x86_64-linux/lib $RPATHS \
  -Wl,--start-group \
  $LIBS \
  -Wl,--end-group \
  -lcudart -lcuda -lcufft -lfftw3 -lfftw3f -lhdf5_serial -lyaml-cpp -lssl -lcrypto -lz -lnuma \
  -lpthread -ldl -lm \
  -levent -levent_pthreads -levent_extra -lairspy -lasdf-cxx -lbz2 -lblosc -llz4 -lzstd -lgdal -lcurl \
  -Wl,-rpath,/usr/lib/x86_64-linux-gnu/hdf5/serial
ln -sf "$N.$H" "$S/$N"
echo "built $S/$N.$H (-> $S/$N)"
