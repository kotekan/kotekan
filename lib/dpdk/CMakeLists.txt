project(kotekan_libs)

include ( dpdk.cmake )

add_library (kotekan_dpdk ${KOTEKAN_DPDK_SOURCES})

# DPDK Includes
# TODO This should use find_library with location hints, etc.
include_directories ( ${RTE_SDK}/${RTE_TARGET}/include )
link_directories ( ${RTE_SDK}/${RTE_TARGET}/lib )

set (DPDK_LIST
        rte_ethdev   rte_distributor  rte_kni     rte_mbuf      rte_pmd_bond   rte_pmd_ring
        rte_power    rte_timer        rte_ring    rte_sched     rte_table
        rte_acl      rte_eal          rte_kvargs  rte_mempool   rte_pmd_e1000
        rte_cfgfile  rte_hash         rte_lpm     rte_meter     rte_pmd_i40e   rte_pmd_vmxnet3_uio
        rte_cmdline  rte_ip_frag      rte_pipeline  rte_pmd_ixgbe  rte_port
)
set (DPDK_LIBS
    -Wl,--start-group ${DPDK_LIST} -Wl,--end-group -ldl -L${RTE_SDK}/${RTE_TARGET}/lib )

target_link_libraries ( kotekan_dpdk ${DPDK_LIBS} )

add_definitions(-DWITH_DPDK)

