#
# From the makefile in the ksgpu code, the following should be done:
#
# Make sure to generate device_mma.hpp: include/ksgpu/device_mma.hpp: generate_device_mma_hpp.py
# python3 $^ >$@
#
# Then, compile cu files %.o: %.cu $(HFILES) $(NVCC) -c -o $@ $< as a library.
#
cmake_minimum_required(VERSION 3.5)
project(ksgpu)

message(STATUS "KSGPU Current source directory: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "KSGPU Current binary directory: ${CMAKE_CURRENT_BINARY_DIR}")

# Special rule for generating 'device_mma.hpp'
add_custom_command(
    OUTPUT include/device_mma.hpp
    COMMAND mkdir -p include && python3 ${CMAKE_CURRENT_SOURCE_DIR}/generate_device_mma_hpp.py >
            include/device_mma.hpp
    DEPENDS generate_device_mma_hpp.py
    COMMENT "generating device_mma.hpp")

# Define source files and headers
set(HFILES
    include/ksgpu/Array.hpp
    include/ksgpu/Barrier.hpp
    include/ksgpu/CpuThreadPool.hpp
    include/ksgpu/CudaStreamPool.hpp
    include/ksgpu/ThreadSafeRingBuffer.hpp
    include/ksgpu/complex_type_traits.hpp
    include/ksgpu/constexpr_functions.hpp
    include/ksgpu/cuda_utils.hpp
    include/ksgpu/device_mma.hpp
    include/ksgpu/mem_utils.hpp
    include/ksgpu/memcpy_kernels.hpp
    include/ksgpu/rand_utils.hpp
    include/ksgpu/string_utils.hpp
    include/ksgpu/test_utils.hpp
    include/ksgpu/time_utils.hpp
    include/ksgpu.hpp
    include/device_mma.hpp)

file(GLOB CUFILES "src_lib/*.cu")
message(STATUS "CUFILES: ${CUFILES}")

# Add library targets
add_library(ksgpu ${HFILES} ${CUFILES})

# Set architecture flags
set_target_properties(ksgpu PROPERTIES CUDA_ARCHITECTURES "86;89")

# Set language for CUDA files
set_target_properties(ksgpu PROPERTIES LANGUAGE CUDA)
set_target_properties(ksgpu PROPERTIES LINKER_LANGUAGE CUDA)

# Compiler options
set_property(TARGET ksgpu PROPERTY CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -m64 -O3 --compiler-options -Wall,-fPIC --forward-unknown-to-host-compiler"
)

# Include directories
target_include_directories(ksgpu PUBLIC include/.)
target_include_directories(ksgpu PUBLIC .)
