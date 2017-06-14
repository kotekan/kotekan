cmake_minimum_required(VERSION 2.8)

PROJECT (kotekan)

SET(SOURCE_EXE kotekan.cpp kotekanMode.cpp)

#CUSTOM MODES:
SET(SOURCE_EXE ${SOURCE_EXE} singleDishMode.cpp packetCapMode.cpp)

if (${USE_HCC})
	#SET(SOURCE_EXE ${SOURCE_EXE} gpuTestMode.cpp)
endif ()
if (${USE_HSA})
	#SET(SOURCE_EXE ${SOURCE_EXE} gpuTestMode.cpp chimeShuffleMode.cpp)
endif ()
if (${USE_OPENCL})
	#SET(SOURCE_EXE ${SOURCE_EXE} )
endif ()


include_directories (${KOTEKAN_SOURCE_DIR}/lib)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/hsa)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/opencl)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/hcc)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/processes)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/testing)
include_directories (${KOTEKAN_SOURCE_DIR}/include)

link_directories (${KOTEKAN_BINARY_DIR}/lib)


if (${USE_DPDK})
	include_directories ( ${RTE_SDK}/${RTE_TARGET}/include )
    link_directories ( ${RTE_SDK}/${RTE_TARGET}/lib )
endif ()

ADD_DEFINITIONS(-mssse3 -D__STDC_LIMIT_MACROS)
SET( DPDK_LIB  "-Wl,--whole-archive -lrte_mbuf  -lrte_eal -lrte_pmd_e1000 -lrte_mempool -lrte_pmd_i40e -lrte_ring -lethdev -lrte_pmd_ixgbe -Wl,--no-whole-archive" )

file(GLOB open_cl_kernels "${KOTEKAN_SOURCE_DIR}/kernels/*.cl")

add_executable(kotekan ${SOURCE_EXE} ${open_cl_kernels})

target_link_libraries( kotekan ${DPDK_LIB} m rt dl pthread )
target_link_libraries( kotekan kotekan_libs )
target_link_libraries( kotekan libinclude )

add_dependencies( kotekan kotekan_libs )

install(TARGETS kotekan DESTINATION /usr/sbin COMPONENT binaries)
install(FILES kotekan.conf DESTINATION /etc/kotekan/ COMPONENT config)
