# Tries to find DPDK on local system
# If found it defines DPDK_FOUND, DPDK_INCLUDE_DIR, and DPDK_LIBRARIES

find_path(DPDK_INCLUDE_DIR rte_config.h
          PATH_SUFFIXES dpdk)
find_library(rte_hash_LIBRARY rte_hash)
find_library(rte_kvargs_LIBRARY rte_kvargs)
find_library(rte_mbuf_LIBRARY rte_mbuf)
find_library(rte_ethdev_LIBRARY rte_ethdev)
find_library(rte_mempool_LIBRARY rte_mempool)
find_library(rte_ring_LIBRARY rte_ring)
find_library(rte_eal_LIBRARY rte_eal)
find_library(rte_cmdline_LIBRARY rte_cmdline)
find_library(rte_pmd_bond_LIBRARY rte_pmd_bond)
find_library(rte_pmd_ixgbe_LIBRARY rte_pmd_ixgbe)
find_library(rte_pmd_i40e_LIBRARY rte_pmd_i40e)
find_library(rte_pmd_ring_LIBRARY rte_pmd_ring)
find_library(rte_pmd_af_packet_LIBRARY rte_pmd_af_packet)

set(dpdk_list_LIBRARIES
    ${rte_hash_LIBRARY}
    ${rte_kvargs_LIBRARY}
    ${rte_mbuf_LIBRARY}
    ${rte_ethdev_LIBRARY}
    ${rte_mempool_LIBRARY}
    ${rte_ring_LIBRARY}
    ${rte_eal_LIBRARY}
    ${rte_cmdline_LIBRARY}
    ${rte_pmd_bond_LIBRARY}
    ${rte_pmd_vmxnet3_uio_LIBRARY}
    ${rte_pmd_ixgbe_LIBRARY}
    ${rte_pmd_i40e_LIBRARY}
    ${rte_pmd_ring_LIBRARY}
    ${rte_pmd_af_packet_LIBRARY})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(dpdk DEFAULT_MSG
                                  DPDK_INCLUDE_DIR
                                  dpdk_list_LIBRARIES)

if(DPDK_FOUND)
    set(DPDK_LIBRARIES
            -Wl,--whole-archive ${dpdk_list_LIBRARIES} -lpthread -Wl,--no-whole-archive)
endif(DPDK_FOUND)