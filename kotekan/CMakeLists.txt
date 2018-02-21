cmake_minimum_required(VERSION 2.8)

PROJECT (kotekan)

SET(SOURCE_EXE kotekan.cpp kotekanMode.cpp)


include_directories (${KOTEKAN_SOURCE_DIR}/lib)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/hsa)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/opencl)
include_directories (${KOTEKAN_SOURCE_DIR}/kotekan/opencl/kernels)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/hcc)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/processes)
include_directories (${KOTEKAN_SOURCE_DIR}/lib/testing)
include_directories (${KOTEKAN_SOURCE_DIR}/include)

link_directories (${KOTEKAN_BINARY_DIR}/lib)

ADD_DEFINITIONS(-mssse3 -D__STDC_LIMIT_MACROS)

add_executable(kotekan ${SOURCE_EXE})

target_link_libraries( kotekan m dl pthread )
target_link_libraries( kotekan kotekan_libs )
target_link_libraries( kotekan libinclude )

if (${USE_DPDK})
	include_directories ( ${RTE_SDK}/${RTE_TARGET}/include )
    link_directories ( ${RTE_SDK}/${RTE_TARGET}/lib )
    SET( DPDK_LIBS  "-Wl,--whole-archive -lrte_mbuf  -lrte_eal -lrte_pmd_e1000 -lrte_mempool -lrte_pmd_i40e -lrte_ring -lrte_ethdev -lrte_kvargs -lrte_pmd_ixgbe -ldl -Wl,--no-whole-archive" )
    target_link_libraries ( kotekan ${DPDK_LIBS} )
endif ()

if (${USE_OPENCL})
    add_custom_target(kernel_opencl_copy
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${KOTEKAN_SOURCE_DIR}/lib/opencl/kernels"
        "opencl/kernels/")
    add_dependencies( kotekan kernel_opencl_copy )
    install(FILES kotekan_opencl_280x.yaml DESTINATION /etc/kotekan/ COMPONENT config)
    install(FILES kotekan_opencl_fury.yaml DESTINATION /etc/kotekan/ COMPONENT config)
    install(FILES kotekan_opencl_test.yaml DESTINATION /etc/kotekan/ COMPONENT config)
endif ()

add_dependencies( kotekan kotekan_libs )

install(TARGETS kotekan DESTINATION /usr/sbin COMPONENT binaries)
#install(FILES kotekan.conf DESTINATION /etc/kotekan/ COMPONENT config)
install(PROGRAMS ../scripts/yaml_to_json.py DESTINATION /usr/sbin COMPONENT scripts)

add_definitions(/D__STDC_FORMAT_MACROS)
