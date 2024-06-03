cmake_minimum_required(VERSION 3.5)
project(n2k)

# Directories
message(STATUS "N2K Current source directory: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "N2K Current binary directory: ${CMAKE_CURRENT_BINARY_DIR}")

# Define source files and headers
set(HFILES
    include/n2k.hpp
    include/n2k_kernel.hpp
)

# Kernel files (to generate), filename syntax is kernel_{nstations}_{nfreq}.o
set(KERNELFILES
    kernel_128_1.cu
    kernel_128_2.cu
    kernel_128_4.cu
    kernel_128_8.cu
    kernel_128_16.cu
    kernel_128_32.cu
    kernel_128_64.cu
    kernel_128_128.cu
    kernel_128_256.cu
    kernel_128_384.cu
    kernel_128_512.cu
    kernel_128_1024.cu
    kernel_128_2048.cu
    kernel_1024_1.cu
    kernel_1024_2.cu
    kernel_1024_4.cu
    kernel_1024_8.cu
    kernel_1024_16.cu
    kernel_1024_32.cu
    kernel_1024_48.cu
    kernel_1024_64.cu
    kernel_1024_128.cu
    kernel_2048_16.cu
)

# Kernel filename syntax is kernel_{nstations}_{nfreq}.cu
set(CUFILES
    src/Correlator.cu
    src/kernel_table.cu
    src/precompute_offsets.cu
)

# Special rule for generating kernel_%d_%d.cu
foreach(FILE ${KERNELFILES})
    add_custom_command(
        OUTPUT template_instantiations/${FILE}
        COMMAND mkdir -p template_instantiations &&
            python3 ${CMAKE_CURRENT_SOURCE_DIR}/template_instantiations/make-instantiation.py ${FILE} &&
            mv ${FILE} template_instantiations/${FILE}
        DEPENDS template_instantiations/make-instantiation.py
    )
    list(APPEND CUFILES "${CMAKE_CURRENT_BINARY_DIR}/template_instantiations/${FILE}")
endforeach()


# Add library targets
add_library(n2k ${HFILES} ${CUFILES})
target_link_libraries(n2k gputils)

# Set architecture flags
set_target_properties(n2k PROPERTIES CUDA_ARCHITECTURES "80;86;89;90")

# Set language for CUDA files
set_target_properties(n2k PROPERTIES LANGUAGE CUDA)
set_target_properties(n2k PROPERTIES LINKER_LANGUAGE CUDA)

# Include directories
target_include_directories(n2k INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include/.")
target_include_directories(n2k INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/.")

# Compiler options
set_property(TARGET n2k PROPERTY CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -m64 -O3 --compiler-options -Wall,-fPIC --forward-unknown-to-host-compiler")

