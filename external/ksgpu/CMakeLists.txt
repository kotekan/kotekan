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
file(GLOB HFILES "include/ksgpu/*.hpp" "include/*.hpp")

file(GLOB SRCFILES "src_lib/*.cu" "src_lib/*.cpp")
set_source_files_properties(${SRCFILES} PROPERTIES LANGUAGE CUDA)
message(STATUS "SRCFILES: ${SRCFILES}")

# Add library targets
add_library(ksgpu ${HFILES} ${SRCFILES})

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

# worarounds around issues with ksgpu
set_source_files_properties(src_lib/mem_utils.cpp PROPERTIES
    COMPILE_FLAGS -Wno-error=unused-parameter)
