# Tries to find DPDK on local system
# If found it defines DPDK_FOUND, DPDK_INCLUDE_DIR, and DPDK_LIBRARIES
# If DPDK is installed from source then this requires
# ${RTE_SDK} and ${RTE_TARGET}

include(FindPackageHandleStandardArgs)

# Default install locations
set(DPDK_SEARCH_PATHS /usr/include
        /usr/local/include)

# Note the ${RTE_SDK}/${RTE_TARGET} path is for backwards compatibility
# with non-standard DPDK install locations
if (RTE_SDK)
    set(DPDK_LINK_HINT "${RTE_SDK}/${RTE_TARGET}/lib")
    message("DPDK Path set to: ${DPDK_LINK_HINT}")
else()
    set(DPDK_LINK_HINT "")
endif()

find_path(DPDK_INCLUDE_DIR rte_common.h
        PATHS ${DPDK_SEARCH_PATHS} ${RTE_SDK}/${RTE_TARGET}/include
        PATH_SUFFIXES dpdk)

find_library(rte_hash_LIBRARY NAMES librte_hash.a PATHS ${DPDK_LINK_HINT})
find_library(rte_kvargs_LIBRARY NAMES librte_kvargs.a PATHS ${DPDK_LINK_HINT})
find_library(rte_mbuf_LIBRARY NAMES librte_mbuf.a PATHS ${DPDK_LINK_HINT})
find_library(rte_ethdev_LIBRARY NAMES librte_ethdev.a PATHS ${DPDK_LINK_HINT})
find_library(rte_mempool_LIBRARY NAMES librte_mempool.a PATHS ${DPDK_LINK_HINT})
find_library(rte_ring_LIBRARY NAMES librte_ring.a PATHS ${DPDK_LINK_HINT})
find_library(rte_eal_LIBRARY NAMES librte_eal.a PATHS ${DPDK_LINK_HINT})
find_library(rte_cmdline_LIBRARY NAMES librte_cmdline.a PATHS ${DPDK_LINK_HINT})
find_library(rte_pci_LIBRARY NAMES librte_pci.a PATHS ${DPDK_LINK_HINT})
find_library(rte_bus_pci_LIBRARY NAMES librte_bus_pci.a PATHS ${DPDK_LINK_HINT})
find_library(rte_bus_vdev_LIBRARY NAMES librte_bus_vdev.a PATHS ${DPDK_LINK_HINT})
find_library(rte_pmd_bond_LIBRARY NAMES librte_pmd_bond.a PATHS ${DPDK_LINK_HINT})
find_library(rte_pmd_ixgbe_LIBRARY NAMES librte_pmd_ixgbe.a PATHS ${DPDK_LINK_HINT})
find_library(rte_pmd_i40e_LIBRARY NAMES librte_pmd_i40e.a PATHS ${DPDK_LINK_HINT})
find_library(rte_pmd_ring_LIBRARY NAMES librte_pmd_ring.a PATHS ${DPDK_LINK_HINT})
find_library(rte_pmd_af_packet_LIBRARY NAMES librte_pmd_af_packet.a PATHS ${DPDK_LINK_HINT})

set(DPDK_LIST_LIBRARIES
        ${rte_hash_LIBRARY}
        ${rte_kvargs_LIBRARY}
        ${rte_mbuf_LIBRARY}
        ${rte_ethdev_LIBRARY}
        ${rte_mempool_LIBRARY}
        ${rte_ring_LIBRARY}
        ${rte_eal_LIBRARY}
        ${rte_cmdline_LIBRARY}
        ${rte_pci_LIBRARY}
        ${rte_bus_pci_LIBRARY}
        ${rte_bus_vdev_LIBRARY}
        ${rte_pmd_bond_LIBRARY}
        ${rte_pmd_vmxnet3_uio_LIBRARY}
        ${rte_pmd_ixgbe_LIBRARY}
        ${rte_pmd_i40e_LIBRARY}
        ${rte_pmd_ring_LIBRARY}
        ${rte_pmd_af_packet_LIBRARY})

find_package_handle_standard_args(dpdk DEFAULT_MSG
        DPDK_LIST_LIBRARIES
        DPDK_INCLUDE_DIR)

message("DPDK List: ${DPDK_LIST_LIBRARIES}")

if(DPDK_FOUND)
    set(DPDK_LIBRARIES
            -Wl,--start-group ${DPDK_LIST_LIBRARIES} -Wl,--end-group -ldl)
endif(DPDK_FOUND)

mark_as_advanced(DPDK_INCLUDE_DIR
        DPDK_LIBRARIES
        DPDK_LIST_LIBRARIES
        rte_hash_LIBRARY
        rte_kvargs_LIBRARY
        rte_mbuf_LIBRARY
        rte_ethdev_LIBRARY
        rte_mempool_LIBRARY
        rte_ring_LIBRARY
        rte_eal_LIBRARY
        rte_cmdline_LIBRARY
        rte_pci_LIBRARY
        rte_bus_pci_LIBRARY
        rte_bus_vdev_LIBRARY
        rte_pmd_bond_LIBRARY
        rte_pmd_ixgbe_LIBRARY
        rte_pmd_i40e_LIBRARY
        rte_pmd_ring_LIBRARY
        rte_pmd_af_packet_LIBRARY)
