cmake_minimum_required (VERSION 2.6)
project (CHIME)

find_package(Threads REQUIRED)

include_directories ( $ENV{RTE_SDK}/$ENV{RTE_TARGET}/include )
link_directories ( $ENV{RTE_SDK}/$ENV{RTE_TARGET}/lib )

set (DPDK_LIST
        ethdev       rte_distributor  rte_kni     rte_mbuf      rte_pmd_bond   rte_pmd_ring
        rte_power    rte_timer        rte_ring    rte_sched     rte_table
        rte_acl      rte_eal          rte_kvargs  rte_mempool   rte_pmd_e1000  rte_pmd_virtio_uio
        rte_cfgfile  rte_hash         rte_lpm     rte_meter     rte_pmd_i40e   rte_pmd_vmxnet3_uio
        rte_cmdline  rte_ip_frag      rte_malloc  rte_pipeline  rte_pmd_ixgbe  rte_port
)

add_definitions(-mssse3 -D__STDC_LIMIT_MACROS)
set (DPDK_LIBS
        -Wl,--start-group ${DPDK_LIST} -Wl,--end-group )


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=gnu99 -O3 -D_GNU_SOURCE -Wall -march=native -mtune=native")

set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -ggdb ")

set(CMAKE_C_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE}")

set(CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/")

SET(CMAKE_INSTALL_PREFIX "/root/kotekan/")

install(DIRECTORY kernels DESTINATION . )

add_subdirectory(lib)
add_subdirectory(kotekan)
add_subdirectory(tools)
add_subdirectory(scripts)
add_subdirectory(vdif_dump)
