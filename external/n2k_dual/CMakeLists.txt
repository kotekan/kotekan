cmake_minimum_required(VERSION 3.5)
project(n2k_dual)

# Two-input clone of n2k -- see README.md and include/n2k_dual/DualCorrelator.hpp.
# n2k itself is NOT modified; this module reuses its headers (device_inlines.hpp).

# Kernel files (to generate), filename syntax is dual_kernel_{nstations_total}_{nfreq}.cu.
# The kernel template depends only on the TOTAL station count; one instantiation serves
# every 128-multiple A/B split of that total.
#   128_8  / 256_8   -- unit/regression tests (small NF for compile + test speed)
#   256_384          -- CHORD pathfinder deployment (128 antennas + 128 synthetic, 384 freqs)
set(KERNELFILES
    dual_kernel_128_8.cu
    dual_kernel_256_8.cu
    dual_kernel_256_384.cu)

set(CUFILES
    src_lib/DualCorrelator.cu
    src_lib/kernel_table.cu
    src_lib/precompute_offsets.cu)

set(HFILES
    include/n2k_dual/DualCorrelator.hpp
    include/n2k_dual/internals/DualCorrelatorKernel.hpp)

# Special rule for generating dual_kernel_%d_%d.cu
foreach(file ${KERNELFILES})
    add_custom_command(
        OUTPUT template_instantiations/${file}
        COMMAND ${CMAKE_COMMAND} -E make_directory template_instantiations
        COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/template_instantiations/make-instantiation.py
                ${file}
        COMMAND ${CMAKE_COMMAND} -E rename ${file} template_instantiations/${file}
        DEPENDS template_instantiations/make-instantiation.py
        COMMENT "Generating template file ${file}"
        VERBATIM)
    list(APPEND CUFILES "${CMAKE_CURRENT_BINARY_DIR}/template_instantiations/${file}")
endforeach()

add_library(n2k_dual SHARED ${CUFILES} ${HFILES})
target_link_libraries(n2k_dual ksgpu)

target_include_directories(n2k_dual PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(n2k_dual PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
# Header-only reuse of n2k internals (device_inlines.hpp); no link dependency.
target_include_directories(n2k_dual PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../n2k/include")

install(TARGETS n2k_dual LIBRARY DESTINATION lib)

# Set architecture flags (honor CMAKE_CUDA_ARCHITECTURES when provided, e.g. 121 for GB10)
if(CMAKE_CUDA_ARCHITECTURES)
    set_target_properties(n2k_dual PROPERTIES CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}")
else()
    set_target_properties(n2k_dual PROPERTIES CUDA_ARCHITECTURES "86;89")
endif()

set_target_properties(n2k_dual PROPERTIES LANGUAGE CUDA)
set_target_properties(n2k_dual PROPERTIES LINKER_LANGUAGE CUDA)

set_property(TARGET n2k_dual PROPERTY CUDA_STANDARD 17)
set(CMAKE_CUDA_FLAGS
    "${CMAKE_CUDA_FLAGS} -m64 -O3 --compiler-options -Wall,-fPIC --forward-unknown-to-host-compiler"
)
