# A script demonstrating building kotekan with tests and HDF5
# Designed for the kotekan Docker image

cd build
cmake \
        -DCMAKE_VERBOSE_MAKEFILE=ON\
        -DCMAKE_BUILD_TYPE=Debug \
        -DCMAKE_CXX_FLAGS_DEBUG="-g -O0"\
        -DCMAKE_C_FLAGS_DEBUG="-g -O0" \
	    -DNO_MEMLOCK=ON \
        -DSUPERDEBUG=ON \
        -DBLAZE_PATH=/code/build/blaze \
        -DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libopenblas.so.0 \
        -DLAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3.7.1 \
        -DCMAKE_BUILD_TYPE=Debug \
        -DUSE_HDF5=ON \
        -DUSE_LAPACK=ON \
        -DHIGHFIVE_PATH=/code/build/HighFive \
        -DWITH_TESTS=ON \
        ..
make
make install

