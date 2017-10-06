cmake_minimum_required(VERSION 2.8)

PROJECT (kotekan)

SET(SOURCE_EXE kotekan.cpp kotekanMode.cpp)

#CUSTOM MODES:
SET(SOURCE_EXE ${SOURCE_EXE} singleDishMode.cpp intensityReceiverMode.cpp gpuReplayMode.cpp)

if (${USE_DPDK})
	SET(SOURCE_EXE ${SOURCE_EXE} packetCapMode.cpp)
endif ()
if (${USE_HCC})
	#SET(SOURCE_EXE ${SOURCE_EXE} gpuTestMode.cpp)
endif ()
if (${USE_HSA})
	SET(SOURCE_EXE ${SOURCE_EXE} gpuTestMode.cpp chimeShuffleMode.cpp singleDishModeGpu.cpp frbMode.cpp)
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

ADD_DEFINITIONS(-mssse3 -D__STDC_LIMIT_MACROS)

file(GLOB open_cl_kernels "${KOTEKAN_SOURCE_DIR}/kernels/*.cl")

add_executable(kotekan ${SOURCE_EXE} ${open_cl_kernels})

target_link_libraries( kotekan m dl pthread )
target_link_libraries( kotekan kotekan_libs )
target_link_libraries( kotekan libinclude )

if (${USE_DPDK})
	include_directories ( ${RTE_SDK}/${RTE_TARGET}/include )
    link_directories ( ${RTE_SDK}/${RTE_TARGET}/lib )
    SET( DPDK_LIBS  "-Wl,--whole-archive -lrte_mbuf  -lrte_eal -lrte_pmd_e1000 -lrte_mempool -lrte_pmd_i40e -lrte_ring -lrte_ethdev -lrte_kvargs -lrte_pmd_ixgbe -ldl -Wl,--no-whole-archive" )
    target_link_libraries ( kotekan ${DPDK_LIBS} )
endif ()

add_dependencies( kotekan kotekan_libs )

install(TARGETS kotekan DESTINATION /usr/sbin COMPONENT binaries)
install(FILES kotekan.conf DESTINATION /etc/kotekan/ COMPONENT config)
