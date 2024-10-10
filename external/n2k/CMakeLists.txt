cmake_minimum_required(VERSION 3.5)
project(n2k)

# Directories
message(STATUS "N2K Current source directory: ${CMAKE_CURRENT_SOURCE_DIR}")
message(STATUS "N2K Current binary directory: ${CMAKE_CURRENT_BINARY_DIR}")

# Define source files and headers
set(HFILES include/n2k/Correlator.hpp include/n2k/pl_kernels.hpp include/n2k/rfi_kernels.hpp
    include/n2k/internals/sk_globals.hpp include/n2k/internals/interpolation.hpp
    include/n2k/internals/internals.hpp include/n2k/internals/device_inlines.hpp
    include/n2k/internals/CorrelatorKernel.hpp include/n2k/internals/bad_feed_mask.hpp)

# Kernel files (to generate), filename syntax is kernel_{nstations}_{nfreq}.cu
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
    kernel_512_64.cu
    kernel_1024_1.cu
    kernel_1024_2.cu
    kernel_1024_4.cu
    kernel_1024_8.cu
    kernel_1024_16.cu
    kernel_1024_32.cu
    kernel_1024_48.cu
    kernel_1024_64.cu
    kernel_1024_128.cu
    kernel_2048_16.cu)

# Additional CUDA files
set(CUFILES src_lib/Correlator.cu src_lib/SkKernel.cu src_lib/internals.cu
    src_lib/kernel_table.cu src_lib/pl_1bit_correlator.cu src_lib/pl_mask_expander.cu
    src_lib/precompute_offsets.cu src_lib/s012_station_downsample.cu
    src_lib/s012_time_downsample.cu src_lib/s0_kernel.cu src_lib/s12_kernel.cu
    src_lib/sk_globals.cu)

# Special rule for generating kernel_%d_%d.cu
foreach(file ${KERNELFILES})
    add_custom_command(
        OUTPUT template_instantiations/${file}
        COMMAND
            mkdir -p template_instantiations && python3
            ${CMAKE_CURRENT_SOURCE_DIR}/template_instantiations/make-instantiation.py ${file} && mv
            ${file} template_instantiations/${file}
        DEPENDS template_instantiations/make-instantiation.py
        COMMENT "Generating template file ${file}")
    list(APPEND CUFILES "${CMAKE_CURRENT_BINARY_DIR}/template_instantiations/${file}")
endforeach()

# Add library targets
add_library(n2k SHARED ${CUFILES} ${HFILES})
target_link_libraries(n2k gputils)

# Include directories
target_include_directories(n2k PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(n2k PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

# Set architecture flags
set_target_properties(n2k PROPERTIES CUDA_ARCHITECTURES "80;86;89;90")

# Set language for CUDA files
set_target_properties(n2k PROPERTIES LANGUAGE CUDA)
set_target_properties(n2k PROPERTIES LINKER_LANGUAGE CUDA)

# Compiler options
set_property(TARGET n2k PROPERTY CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -m64 -O3 --compiler-options -Wall,-fPIC --forward-unknown-to-host-compiler"
)
