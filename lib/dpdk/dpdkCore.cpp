#include "dpdkCore.hpp"

#include "fmt.hpp"
#include "json.hpp"

#ifdef WITH_NUMA
#include <numa.h>
#endif
#include <signal.h>
#include <stdexcept>
#include <unistd.h>
#include <vector>

using nlohmann::json;
using std::string;
using std::to_string;
using std::vector;

#include "captureHandler.hpp"
#include "iceBoardShuffle.hpp"
#include "iceBoardStandard.hpp"
#include "iceBoardVDIF.hpp"

/// TODO move this to an inline static once we go to C++17
stream_id_t iceBoardShuffle::all_stream_ids[iceBoardShuffle::shuffle_size];

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(dpdkCore);

static bool __eal_initalized = false;

dpdkCore::dpdkCore(Config& config, const string& unique_name, bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&dpdkCore::main_thread, this)) {

    uint32_t num_mbufs = config.get_default<uint32_t>(unique_name, "num_mbufs", 1024);
    const uint32_t mbuf_cache_size =
        config.get_default<uint32_t>(unique_name, "mbuf_cache_size", 250);
    burst_size = config.get_default<uint32_t>(unique_name, "burst_size", 32);
    rx_ring_size = config.get_default<uint32_t>(unique_name, "rx_ring_size", 512);
    tx_ring_size = config.get_default<uint32_t>(unique_name, "tx_ring_size", 512);

    num_mem_channels = config.get_default<uint32_t>(unique_name, "num_mem_channels", 4);
    init_mem_alloc = config.get_default<uint32_t>(unique_name, "init_mem_alloc", 256);

    // Setup the lcore mappings
    // Basically this is mapping the DPDK EAL framework way of assigning threads
    // into the kotekan framework.
    vector<int> lcore_cpu_map = config.get<std::vector<int>>(unique_name, "lcore_cpu_map");
    uint32_t master_lcore_cpu = config.get<uint32_t>(unique_name, "master_lcore_cpu");

    num_lcores = lcore_cpu_map.size();

    dpdk_init(lcore_cpu_map, master_lcore_cpu);

    num_system_ports = rte_eth_dev_count();

    // This default works well for ICE boards,
    // but we might change this to something more genertic
    memset((void*)&port_conf, 0, sizeof(struct rte_eth_conf));
    port_conf.rxmode.max_rx_pkt_len =
        config.get_default<uint32_t>(unique_name, "max_rx_pkt_len", 5000);
    port_conf.rxmode.jumbo_frame =
        (uint16_t)config.get_default<bool>(unique_name, "jumbo_frame", true);
    port_conf.rxmode.hw_strip_crc = 0;
    port_conf.rxmode.header_split = 0;
    port_conf.rxmode.hw_ip_checksum = 1;

    // TODO reference why this needs to be 2048
    const uint32_t max_data_size = 2048;
    const uint32_t mbuf_size = max_data_size + sizeof(struct rte_mbuf) + RTE_PKTMBUF_HEADROOM;

    // Convert the lcore to port map into a simple c style struct
    // This is basically done to remove overhead in the critial packet processing loop.
    // TODO check there are no ports assigned twice
    lcore_port_list = (struct portList*)malloc(num_lcores * sizeof(struct portList));
    CHECK_MEM(lcore_port_list);
    int lcore_id = 0;
    num_ports = 0;
    json lcore_port_map = config.get_value(unique_name, "lcore_port_map");
    for (vector<int> ports : lcore_port_map) {
        lcore_port_list[lcore_id].ports = (uint32_t*)malloc(ports.size() * sizeof(uint32_t));
        CHECK_MEM(lcore_port_list[lcore_id].ports);
        int port_id = 0;
        for (uint32_t port : ports) {
            lcore_port_list[lcore_id].ports[port_id++] = port;
        }
        lcore_port_list[lcore_id].num_ports = ports.size();
        num_ports += ports.size();
        lcore_id++;
    }

    create_handlers(buffer_container);

    if (num_ports > num_system_ports) {
        throw std::runtime_error(
            fmt::format(fmt("Trying to create more ports: {:d}, than DPDK found: {:d}"), num_ports,
                        num_system_ports));
    }

    // The plus one is for the master lcore.
    if (rte_lcore_count() != num_lcores + 1) {
        ERROR("Mismatch in the number of lcores");
        throw std::runtime_error(fmt::format(
            fmt("Num lcores set to: {:d} in the config, but the DPDK run time has: {:d} lcores."),
            num_lcores, rte_lcore_count()));
    }

    // Get the number of ports on each numa node and create an mbuf pool for that node.
    for (int node_id = 0; node_id < numa_num_configured_nodes(); ++node_id) {
        int num_ports_on_node = 0;
        for (size_t j = 0; j < lcore_cpu_map.size(); ++j) {
            auto lcore_id = lcore_cpu_map.at(j);
            if (numa_node_of_cpu(lcore_id) == -1) {
                throw std::runtime_error(
                    "lcore_id '" + to_string(lcore_id)
                    + "' failed to map to numa node, is this a valid CPU core id?");
            }
            if (numa_node_of_cpu(lcore_id) == node_id) {
                num_ports_on_node += lcore_port_list[j].num_ports;
            }
        }
        DEBUG("Number of ports on numa node {:d}: {:d}", node_id, num_ports_on_node);
        struct rte_mempool* pool = rte_mempool_create(
            ("MBUF_POOL_" + to_string(node_id)).c_str(), num_mbufs * num_ports_on_node, mbuf_size,
            mbuf_cache_size, sizeof(struct rte_pktmbuf_pool_private), rte_pktmbuf_pool_init, NULL,
            rte_pktmbuf_init, NULL, node_id, 0);
        if (pool == NULL) {
            throw std::runtime_error("Cannot create DPDK mbuf pool.");
        }
        mbuf_pools.push_back(pool);
    }

    // Init ports referenced in the lcore port mapping
    int i = 0; // Index into lcore_cpu_map
    for (vector<int> ports : lcore_port_map) {
        for (uint32_t port : ports) {
            // TODO This will fail in a strange way if a port is listed more than once in the
            // config. We should have a check that each port assignment is unique.
            if (port_init(port, lcore_cpu_map.at(i)) != 0) {
                throw std::runtime_error(fmt::format(fmt("DPDK Cannot init port: {:d}"), port));
            }
        }
        i++;
    }
}

void dpdkCore::create_handlers(bufferContainer& buffer_container) {
    // Create the handlers
    // TODO This could likely be refactored out of this system.
    // The one problem is that we are using header only builds for efficency,
    // so the normal factory model doesn't work here.
    vector<json> handlers_block = config.get<std::vector<json>>(unique_name, "handlers");
    uint32_t port = 0;
    if (handlers_block.size() != num_system_ports) {
        throw std::runtime_error(fmt::format(fmt("The number of DPDK handlers ({:d}) must be equal "
                                                 "to the number of system ports ({:d})"),
                                             handlers_block.size(), num_system_ports));
    }
    handlers = (dpdkRXhandler**)malloc(num_system_ports * sizeof(dpdkRXhandler*));
    CHECK_MEM(handlers);
    for (json& handler : handlers_block) {

        string handler_name = handler["dpdk_handler"];
        string handler_unique_name = fmt::format(fmt("{:s}/handlers/{:d}"), unique_name, port);

        if (handler_name == "iceBoardShuffle") {
            handlers[port] =
                new iceBoardShuffle(config, handler_unique_name, buffer_container, port);
        } else if (handler_name == "iceBoardStandard") {
            handlers[port] =
                new iceBoardStandard(config, handler_unique_name, buffer_container, port);
        } else if (handler_name == "iceBoardVDIF") {
            handlers[port] = new iceBoardVDIF(config, handler_unique_name, buffer_container, port);
        } else if (handler_name == "captureHandler") {
            handlers[port] =
                new captureHandler(config, handler_unique_name, buffer_container, port);
        } else if (handler_name == "none") {
            handlers[port] = nullptr;
        } else {
            throw std::runtime_error(
                fmt::format(fmt("The dpdk handler type '{:s}' does not exist."), handler_name));
        }

        port++;
    }
}

void dpdkCore::dpdk_init(vector<int> lcore_cpu_map, uint32_t master_lcore_cpu) {

    string dpdk_lcore_map = fmt::format(fmt("0@{:d},"), master_lcore_cpu);
    int i = 1;
    for (int& core_id : lcore_cpu_map) {
        dpdk_lcore_map += fmt::format(fmt("{:d}@{:d},"), i++, core_id);
    }
    dpdk_lcore_map.pop_back(); // Remove the last ","

    DEBUG("Using DPDK lcore map: {:s}", dpdk_lcore_map);

    // App name, this can be anything.
    char arg0[] = "kotekan";
    // Number of memory channels
    char arg1[] = "-n";
    char* arg2 = (char*)malloc(std::to_string(num_mem_channels).length() + 1);
    strncpy(arg2, std::to_string(num_mem_channels).c_str(),
            std::to_string(num_mem_channels).length() + 1);
    // Lcore map
    char arg3[] = "--lcores";
    char* arg4 = (char*)malloc(dpdk_lcore_map.length() + 1);
    strncpy(arg4, dpdk_lcore_map.c_str(), dpdk_lcore_map.length() + 1);
    // Initial memory allocation
    char arg5[] = "-m";
    char* arg6 = (char*)malloc(std::to_string(init_mem_alloc).length() + 1);
    strncpy(arg6, std::to_string(init_mem_alloc).c_str(),
            std::to_string(init_mem_alloc).length() + 1);
    // Generate final options string for EAL initialization
    char* argv2[] = {&arg0[0], &arg1[0], &arg2[0], &arg3[0], &arg4[0], &arg5[0], &arg6[0], NULL};
    int argc2 = (int)(sizeof(argv2) / sizeof(argv2[0])) - 1;

    // Initialize the Environment Abstraction Layer (EAL).
    // Currently closing DPDKs EAL isn't offically supported,
    // so we only do it once.
    if (!__eal_initalized) {
        int ret = rte_eal_init(argc2, argv2);
        if (ret < 0)
            throw std::runtime_error(
                fmt::format(fmt("Failed to init DPDK EAL with error code: {:d}"), ret));
        __eal_initalized = true;
    }
}

void dpdkCore::main_thread() {

    // Start the packet receiving lcores (basically pthreads)
    rte_eal_mp_remote_launch(dpdkCore::lcore_rx, (void*)this, SKIP_MASTER);

    while (!stop_thread) {
        sleep(1);

        // Get the handlers to update their stats about packets
        // Some of these stats might be moved to this class, but
        // it seemed like it was worth leaving it upto the handler
        // to say which stats we actually care about recording.
        for (uint32_t i = 0; i < num_system_ports; ++i) {
            if (handlers[i] != nullptr)
                handlers[i]->update_stats();
        }

        // Check port status
    }

    // Wait for the lcores to join
    rte_eal_mp_wait_lcore();
}

dpdkCore::~dpdkCore() {
    // TODO Make sure DPDK is stopped
    // Requires an experimental feature not yet the version of DPDK used by kotekan

    for (auto& pool : mbuf_pools) {
        rte_mempool_free(pool);
    }

    // Free the handlers
    for (uint32_t i = 0; i < num_system_ports; ++i) {
        if (handlers[i] != nullptr)
            delete handlers[i];
    }
    free(handlers);
}

int32_t dpdkCore::port_init(uint8_t port, uint32_t lcore_id) {
    const uint16_t rx_rings = 1, tx_rings = 1;
    int retval;
    uint16_t q;

    if (port >= num_system_ports)
        return -1;

    // Configure the Ethernet device.
    retval = rte_eth_dev_configure(port, rx_rings, tx_rings, &port_conf);
    if (retval != 0) {
        ERROR("Failed to configure device, port {:d}, error: {:d}", port, retval);
        return retval;
    }

    // Allocate and set up 1 RX queue per Ethernet port.
    for (q = 0; q < rx_rings; q++) {
        retval = rte_eth_rx_queue_setup(port, q, rx_ring_size, rte_eth_dev_socket_id(port), NULL,
                                        mbuf_pools.at(numa_node_of_cpu(lcore_id)));
        if (retval < 0) {
            ERROR("Failed to setupt RX queue for port {:d}, error: {:d}", port, retval);
            return retval;
        }
    }

    // Allocate and set up 1 TX queue per Ethernet port.
    // TODO Do we need this?
    for (q = 0; q < tx_rings; q++) {
        retval = rte_eth_tx_queue_setup(port, q, tx_ring_size, rte_eth_dev_socket_id(port), NULL);
        if (retval < 0) {
            ERROR("Failed to setupt TX queue for port {:d}, error: {:d}", port, retval);
            return retval;
        }
    }

    // Start the Ethernet port.
    retval = rte_eth_dev_start(port);
    if (retval < 0) {
        ERROR("Failed to start port: {:d}", port);
        return retval;
    }

    // Report the port MAC address.
    // TODO record the MAC address for export to JSON
    struct ether_addr addr;
    rte_eth_macaddr_get(port, &addr);
    INFO("Port {:d} MAC: {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} "
         "memory assigned to numa_node {:d}",
         (unsigned)port, addr.addr_bytes[0], addr.addr_bytes[1], addr.addr_bytes[2],
         addr.addr_bytes[3], addr.addr_bytes[4], addr.addr_bytes[5], numa_node_of_cpu(lcore_id));

    // Enable promiscuous mode.
    rte_eth_promiscuous_enable(port);

    return 0;
}

int dpdkCore::lcore_rx(void* args) {
    dpdkCore* core = (dpdkCore*)args;

    struct rte_mbuf* mbufs[core->burst_size];

    // non-master cores start at 1, but it's easier to 0 base here
    uint32_t lcore = rte_lcore_id() - 1;
    fprintf(stderr, "lcore ID: %u\n", lcore);
    if (lcore > core->num_lcores) {
        throw std::runtime_error("lcore mapping error");
    }

    // The list of ports this thread processes
    struct dpdkCore::portList port_list = core->lcore_port_list[lcore];
    const uint32_t num_local_ports = port_list.num_ports;
    const uint32_t* ports = port_list.ports;
    const uint32_t burst_size = core->burst_size;

    for (uint32_t i = 0; i < num_local_ports; ++i) {
        uint32_t port = ports[i];
        if (core->handlers[port] == nullptr) {
            WARN_NON_OO("No valid handler provided for port {:d}", port);
            return 0;
        }
    }

    while (!core->stop_thread) {
        for (uint32_t i = 0; i < num_local_ports; ++i) {
            uint32_t port = ports[i];

            const uint16_t num_rx = rte_eth_rx_burst(port, 0, mbufs, burst_size);

            for (uint16_t j = 0; j < num_rx; ++j) {

                // Process the packet with the required handler
                // NOTE: Ideally this wouldn't be a call to a virtual function,
                // but it's an overhead that's hard to avoid here.
                if (unlikely(core->handlers[port]->handle_packet(mbufs[j]) != 0)) {
                    goto exit_lcore;
                }

                rte_pktmbuf_free(mbufs[j]);
            }
        }
    }
exit_lcore:
    return 0;
}
