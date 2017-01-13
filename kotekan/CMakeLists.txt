cmake_minimum_required(VERSION 2.8)

PROJECT (kotekan)

SET(SOURCE_EXE kotekan.cpp chimeShuffleMode.cpp packetCapMode.cpp kotekanMode.cpp
                            gpuTestMode.cpp)

include_directories (${CHIME_SOURCE_DIR}/lib)
include_directories (${CHIME_SOURCE_DIR}/lib/gpu_command_objects)

include_directories (${CHIME_SOURCE_DIR}/include)

link_directories (${CHIME_BINARY_DIR}/lib)

include_directories( ${OPENCL_INCLUDE_DIRS} )

include_directories ( $ENV{RTE_SDK}/$ENV{RTE_TARGET}/include )
link_directories ( $ENV{RTE_SDK}/$ENV{RTE_TARGET}/lib )

ADD_DEFINITIONS(-mssse3 -D__STDC_LIMIT_MACROS)
SET( DPDK_LIB  "-Wl,--whole-archive -lrte_mbuf  -lrte_eal -lrte_pmd_e1000 -lrte_mempool -lrte_pmd_i40e -lrte_ring -lethdev -lrte_pmd_ixgbe -Wl,--no-whole-archive" )

file(GLOB open_cl_kernels "${CHIME_SOURCE_DIR}/kernels/*.cl")

add_executable(kotekan ${SOURCE_EXE} ${open_cl_kernels})

target_link_libraries( kotekan ${DPDK_LIB} m rt dl pthread )
target_link_libraries( kotekan chime )
target_link_libraries( kotekan libinclude )

add_custom_target(kernel_copy
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CHIME_SOURCE_DIR}/kernels"
        "kernels")

add_dependencies( kotekan makeHSACOKernels )
add_dependencies( kotekan kernel_copy )

install(TARGETS kotekan DESTINATION /usr/sbin COMPONENT binaries)
install(FILES kotekan.conf DESTINATION /etc/kotekan/ COMPONENT config)
