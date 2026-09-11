cmake_minimum_required(VERSION 3.5)
project(pirate)

# Directories
message(STATUS "Pirate Current source directory: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "Pirate Current binary directory: ${CMAKE_CURRENT_BINARY_DIR}")

# Define source files and headers
set(HFILES include/pirate/ChimeBeamformer.hpp)

# Kernel filename syntax is kernel_{nstations}_{nfreq}.cu
set(CUFILES src_lib/ChimeBeamformer1.cu src_lib/ChimeBeamformer2.cu)

# Add library targets
add_library(pirate SHARED ${CUFILES} ${HFILES})
target_link_libraries(pirate ksgpu)

# Include directories
target_include_directories(pirate PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(pirate PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

# Install the shared library so installed kotekan can find libn2k at runtime.
install(TARGETS pirate LIBRARY DESTINATION lib)

# Set architecture flags
set_target_properties(pirate PROPERTIES CUDA_ARCHITECTURES "86;89")

# Set language for CUDA files
set_target_properties(pirate PROPERTIES LANGUAGE CUDA)
set_target_properties(pirate PROPERTIES LINKER_LANGUAGE CUDA)

# Compiler options
set_property(TARGET pirate PROPERTY CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -m64 -O3 --compiler-options -Wall,-fPIC --forward-unknown-to-host-compiler"
)

# Dependencies
get_target_property(cufftdx_SEPARATE_TWIDDLES_CUDA_ARCHITECTURES pirate CUDA_ARCHITECTURES)
find_package(cufftdx)
include_directories(${cufftdx_INCLUDE_DIR}/mathdx)
