# A script to simplify building kotekan
# Designed for the kotekan Docker image

tests="OFF"
debug=0
# Get the options
while getopts ":t:d" option; do
   case $option in
      t) # Include tests
         tests='ON';;
      d) # Debug mode
	 debug=1;;
   esac
done


cd build

if [ $debug -eq 1 ]
then
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
        -DUSE_HDF5=ON \
        -DUSE_LAPACK=ON \
        -DHIGHFIVE_PATH=/code/build/HighFive \
        -DWITH_TESTS=$tests \
        ..
else
cmake \
	-DNO_MEMLOCK=ON \
        -DBLAZE_PATH=/code/build/blaze \
        -DBLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libopenblas.so.0 \
        -DLAPACK_LIBRARIES=/usr/lib/x86_64-linux-gnu/lapack/liblapack.so.3.7.1 \
        -DUSE_HDF5=ON \
        -DUSE_LAPACK=ON \
        -DHIGHFIVE_PATH=/code/build/HighFive \
        -DWITH_TESTS=$tests \
        ..
fi

make -j4
make install
