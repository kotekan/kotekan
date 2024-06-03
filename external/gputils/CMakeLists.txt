#
# From the makefile in the gputils code, the following should be done:
#
# Make sure to generate device_mma.hpp:
# include/gputils/device_mma.hpp: generate_device_mma_hpp.py
#     python3 $^ >$@
# 
# Then, compile cu files
# %.o: %.cu $(HFILES)
#     $(NVCC) -c -o $@ $<
# as a library.
#
cmake_minimum_required(VERSION 3.5)
project(gputils)

message(STATUS "GPUTILS Current source directory: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "GPUTILS Current binary directory: ${CMAKE_CURRENT_BINARY_DIR}")

# Special rule for generating 'device_mma.hpp'
add_custom_command(
    OUTPUT include/device_mma.hpp
    COMMAND mkdir -p include && python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate_device_mma_hpp.py > include/device_mma.hpp
    DEPENDS generate_device_mma_hpp.py
)

# Define source files and headers
set(HFILES
    include/gputils/Array.hpp
    include/gputils/Barrier.hpp
    include/gputils/CpuThreadPool.hpp
    include/gputils/CudaStreamPool.hpp
    include/gputils/ThreadSafeRingBuffer.hpp
    include/gputils/complex_type_traits.hpp
    include/gputils/constexpr_functions.hpp
    include/gputils/cuda_utils.hpp
    include/gputils/device_mma.hpp
    include/gputils/mem_utils.hpp
    include/gputils/memcpy_kernels.hpp
    include/gputils/rand_utils.hpp
    include/gputils/string_utils.hpp
    include/gputils/system_utils.hpp
    include/gputils/test_utils.hpp
    include/gputils/time_utils.hpp
    include/gputils.hpp
    include/device_mma.hpp
)

file(GLOB CUFILES "src_lib/*.cu")
message(STATUS "CUFILES: ${CUFILES}")

# Add library targets
add_library(gputils ${HFILES} ${CUFILES})

# Set architecture flags
set_target_properties(gputils PROPERTIES CUDA_ARCHITECTURES "80;86;89;90")

# Set language for CUDA files
set_target_properties(gputils PROPERTIES LANGUAGE CUDA)
set_target_properties(gputils PROPERTIES LINKER_LANGUAGE CUDA)

# Compiler options
set_property(TARGET gputils PROPERTY CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -m64 -O3 --compiler-options -Wall,-fPIC --forward-unknown-to-host-compiler")

# Include directories
target_include_directories(gputils INTERFACE include/.)
target_include_directories(gputils INTERFACE .)
