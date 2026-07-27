#include "dpdkCore.hpp"

#include "Config.hpp"                  // for Config
#include "PipelineGraph.hpp"           // for PipelineGraph, GraphNode
#include "StageFactory.hpp"            // for REGISTER_KOTEKAN_STAGE
#include "captureHandler.hpp"          // for captureHandler
#include "crs16BoardCaptureWorker.hpp" // for crs16BoardCaptureWorker
#include "crs16BoardDistributor.hpp"   // for crs16BoardDistributor
#include "crs1BoardCaptureWorker.hpp"  // for crs1BoardCaptureWorker
#include "crs1BoardDistributor.hpp"    // for crs1BoardDistributor
#include "iceBoardShuffle.hpp"         // for iceBoardShuffle
#include "iceBoardStandard.hpp"        // for iceBoardStandard
#include "iceBoardVDIF.hpp"            // for iceBoardVDIF

#include "fmt.hpp"  // for format, compile_string_to_view, fmt, format_string
#include "json.hpp" // for basic_json, json, iter_impl

#include <functional>              // for bind, function
#include <numa.h>                  // for numa_node_of_cpu, numa_num_configured_nodes
#include <rte_branch_prediction.h> // for unlikely
#include <rte_config.h>            // for RTE_PKTMBUF_HEADROOM
#include <rte_eal.h>               // for rte_eal_init
#include <rte_errno.h>             // for rte_strerror, per_lcore__rte_errno, rte_errno
#include <rte_ether.h>             // for rte_ether_addr
#include <rte_launch.h>            // for rte_eal_mp_remote_launch, rte_eal_mp_wait_lcore
#include <rte_lcore.h>             // for rte_lcore_count, rte_lcore_id
#include <rte_mbuf.h>              // for rte_pktmbuf_free, rte_pktmbuf_init, rte_pktmbuf_p...
#include <rte_mbuf_core.h>         // for rte_mbuf
#include <rte_mempool.h>           // for rte_mempool_create, rte_mempool_free
#include <rte_memzone.h>           // for rte_memzone_max_set
#include <rte_ring.h>              // for rte_ring_create, rte_ring_dequeue_burst
#include <set>                     // for set
#include <stdexcept>               // for runtime_error
#include <stdio.h>                 // for fprintf, NULL, size_t, stderr
#include <stdlib.h>                // for free, malloc
#include <string.h>                // for memset
#include <sys/types.h>             // for uint
#include <unistd.h>                // for sleep
#include <vector>                  // for vector

using nlohmann::json;
using std::string;
using std::to_string;
using std::vector;

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(dpdkCore);

static bool __eal_initalized = false;

dpdkCore::dpdkCore(Config& config, const string& unique_name, bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&dpdkCore::main_thread, this)),
    nic_rx_packets_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_nic_rx_packets_total", unique_name, {"port"})),
    nic_rx_missed_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_nic_rx_missed_total", unique_name, {"port"})),
    nic_rx_errors_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_nic_rx_errors_total", unique_name, {"port"})),
    nic_rx_nombuf_total_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_nic_rx_nombuf_total", unique_name, {"port"})),
    nic_link_up_metric(kotekan::prometheus::Metrics::instance().add_gauge(
        "kotekan_dpdk_nic_link_up", unique_name, {"port"})) {

    uint32_t num_mbufs = config.get_default<uint32_t>(unique_name, "num_mbufs", 1024);
    const uint32_t mbuf_cache_size =
        config.get_default<uint32_t>(unique_name, "mbuf_cache_size", 250);
    burst_size = config.get_default<uint32_t>(unique_name, "burst_size", 32);
    rx_ring_size = config.get_default<uint32_t>(unique_name, "rx_ring_size", 512);
    tx_ring_size = config.get_default<uint32_t>(unique_name, "tx_ring_size", 512);

    num_mem_channels = config.get_default<uint32_t>(unique_name, "num_mem_channels", 4);
    init_mem_alloc = config.get_default<std::string>(unique_name, "init_mem_alloc", "256");
    lcore_start_delay = config.get_default<uint32_t>(unique_name, "lcore_start_delay", 40);

    // Setup the lcore mappings
    // Basically this is mapping the DPDK EAL framework way of assigning threads
    // into the kotekan framework.
    vector<int> lcore_cpu_map = config.get<std::vector<int>>(unique_name, "lcore_cpu_map");
    uint32_t main_lcore_cpu = config.get<uint32_t>(unique_name, "main_lcore_cpu");

    num_lcores = lcore_cpu_map.size();

    dpdk_init(lcore_cpu_map, main_lcore_cpu);

    num_system_ports = rte_eth_dev_count_avail();

    // Storage for the per-port NIC hardware stats and link state polling.
    last_eth_stats.resize(num_system_ports);
    link_is_up.resize(num_system_ports, false);

    // This default works well for ICE boards,
    // but we might change this to something more genertic
    memset((void*)&port_conf, 0, sizeof(struct rte_eth_conf));
    uint32_t max_rx_pkt_len = config.get_default<uint32_t>(unique_name, "max_rx_pkt_len", 9000);
    port_conf.rxmode.max_lro_pkt_size = max_rx_pkt_len;
    port_conf.rxmode.mtu = max_rx_pkt_len;
    port_conf.rxmode.offloads =
        RTE_ETH_RX_OFFLOAD_KEEP_CRC | RTE_ETH_RX_OFFLOAD_IPV4_CKSUM | RTE_ETH_RX_OFFLOAD_UDP_CKSUM;


    // Change hardcoded 2048 to support jumbo frames if configured
    // We add RTE_PKTMBUF_HEADROOM to max_rx_pkt_len to ensure the payload fits comfortably
    // or simply use a fixed large size like 9600 if max_rx_pkt_len is high.
    uint32_t configured_mbuf_size =
        config.get_default<uint32_t>(unique_name, "mbuf_data_size", 2048);

    // If the user hasn't explicitly set mbuf_data_size, but has set a large max_rx_pkt_len,
    // we should probably default to the larger size to avoid scatter-gather.
    if (configured_mbuf_size == 2048 && max_rx_pkt_len > 2048) {
        configured_mbuf_size = max_rx_pkt_len + RTE_PKTMBUF_HEADROOM;
        // Align to 1024 for sanity, though not strictly required
        configured_mbuf_size = ((configured_mbuf_size + 1023) / 1024) * 1024;
    }

    const uint32_t max_data_size = configured_mbuf_size;
    const uint32_t mbuf_size = max_data_size + sizeof(struct rte_mbuf) + RTE_PKTMBUF_HEADROOM;

    int lcore_id = 0;
    json lcore_port_map = config.get_value(unique_name, "lcore_port_map");
    std::set<uint32_t> unique_ports;
    for (vector<int> ports : lcore_port_map) {
        lcore_port_list.push_back(vector<uint32_t>());
        for (uint32_t port : ports) {
            lcore_port_list[lcore_id].push_back(port);
            unique_ports.insert(port);
        }
        lcore_id++;
    }
    num_ports = unique_ports.size();

    // Allocate worker rings
    if (config.exists(unique_name, "worker_ring_map")) {
        json worker_ring_map = config.get_value(unique_name, "worker_ring_map");
        for (const auto& ring : worker_ring_map) {
            string ring_name = ring["name"];
            uint32_t num_slots = ring["num_slots"];
            int socket_id = ring["socket_id"];

            DEBUG("Creating worker ring '{:s}' with {:d} slots on socket {:d}", ring_name,
                  num_slots, socket_id);

            // TODO Possibly make the flags configurable?
            // We are assuming single producer single consumer for now.
            struct rte_ring* r = rte_ring_create(ring_name.c_str(), num_slots, socket_id,
                                                 RING_F_SP_ENQ | RING_F_SC_DEQ);
            if (r == nullptr) {
                throw std::runtime_error("Cannot create worker ring: "
                                         + std::string(rte_strerror(rte_errno)));
            }
            worker_rings.push_back(r);
        }
    }

    create_handlers(buffer_container);
    create_workers(buffer_container);

    if (num_ports > num_system_ports) {
        throw std::runtime_error(
            fmt::format(fmt("Trying to create more ports: {:d}, than DPDK found: {:d}"), num_ports,
                        num_system_ports));
    }

    // The plus one is for the main lcore.
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
            if (numa_node_of_cpu(lcore_id) == node_id && j < lcore_port_list.size()) {
                num_ports_on_node += lcore_port_list.at(j).size();
            }
        }
        DEBUG("Number of ports on numa node {:d}: {:d}", node_id, num_ports_on_node);
        struct rte_mempool* pool = nullptr;
        if (num_ports_on_node > 0) {
            pool = rte_mempool_create(
                ("MBUF_POOL_" + to_string(node_id)).c_str(), num_mbufs * num_ports_on_node,
                mbuf_size, mbuf_cache_size, sizeof(struct rte_pktmbuf_pool_private),
                rte_pktmbuf_pool_init, nullptr, rte_pktmbuf_init, nullptr, node_id, 0);
            if (pool == nullptr) {
                throw std::runtime_error("Cannot create DPDK mbuf pool: "
                                         + std::string(rte_strerror(rte_errno)));
            }
            INFO("Created mbuf pool on node {:d} with {:d} mbufs of size {:d} bytes", node_id,
                 num_mbufs * num_ports_on_node, mbuf_size);
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
    // The one problem is that we are using header only builds for efficiency,
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
        } else if (handler_name == "crs1BoardDistributor") {
            handlers[port] = new crs1BoardDistributor(config, handler_unique_name, buffer_container,
                                                      port, worker_rings);
        } else if (handler_name == "crs16BoardDistributor") {
            handlers[port] = new crs16BoardDistributor(config, handler_unique_name,
                                                       buffer_container, port, worker_rings);
        } else if (handler_name == "none") {
            handlers[port] = nullptr;
        } else {
            throw std::runtime_error(
                fmt::format(fmt("The dpdk handler type '{:s}' does not exist."), handler_name));
        }

        port++;
    }
}

void dpdkCore::create_workers(bufferContainer& buffer_container) {
    // Create the workers
    // Does not use the factory model because this is header only.
    if (!config.exists(unique_name, "workers"))
        return;
    vector<json> workers_block = config.get<std::vector<json>>(unique_name, "workers");
    workers.resize(workers_block.size());
    uint32_t worker_id = 0;
    for (json& worker : workers_block) {

        string worker_name = worker["dpdk_handler"];
        string worker_unique_name = fmt::format(fmt("{:s}/workers/{:d}"), unique_name, worker_id);

        if (worker_name == "crs1BoardCaptureWorker") {
            workers[worker_id] = new crs1BoardCaptureWorker(
                config, worker_unique_name, buffer_container, worker_id, worker_rings);
        } else if (worker_name == "crs16BoardCaptureWorker") {
            workers[worker_id] = new crs16BoardCaptureWorker(
                config, worker_unique_name, buffer_container, worker_id, worker_rings);
        } else if (worker_name == "none") {
            workers[worker_id] = nullptr;
        } else {
            throw std::runtime_error(
                fmt::format(fmt("The dpdk worker type '{:s}' does not exist."), worker_name));
        }
        if (workers[worker_id] != nullptr)
            INFO("Created DPDK worker '{:s}' with ID {:d}", worker_name, worker_id);
        else
            FATAL_ERROR("DPDK worker '{:s}' with ID {:d} is nullptr", worker_name, worker_id);

        worker_id++;
    }
    active_workers = worker_id;
}

void dpdkCore::dpdk_init(vector<int> lcore_cpu_map, uint32_t main_lcore_cpu) {

    string dpdk_lcore_map = fmt::format(fmt("0@{:d},"), main_lcore_cpu);
    int i = 1;
    for (int& core_id : lcore_cpu_map) {
        dpdk_lcore_map += fmt::format(fmt("{:d}@{:d},"), i++, core_id);
    }
    dpdk_lcore_map.pop_back(); // Remove the last ","

    DEBUG("Using DPDK lcore map: {:s}", dpdk_lcore_map);

    std::vector<std::string> str_args;
    std::vector<char*> argv2_vec;

    str_args.push_back("kotekan");
    str_args.push_back("-n");
    str_args.push_back(std::to_string(num_mem_channels));
    str_args.push_back("--lcores");
    str_args.push_back(dpdk_lcore_map);
    str_args.push_back("--socket-mem");
    str_args.push_back(init_mem_alloc);

    // Get PCIe block list from config
    // Used to block NICs that DPDK detects, but that shouldn't be used by DPDK
    std::vector<std::string> pcie_block_list =
        config.get_default<std::vector<std::string>>(unique_name, "pcie_block_list", {});
    for (const auto& pcie : pcie_block_list) {
        str_args.push_back("-b");
        str_args.push_back(pcie);
    }

    for (auto& s : str_args) {
        argv2_vec.push_back(const_cast<char*>(s.c_str()));
    }
    argv2_vec.push_back(nullptr);
    int argc2 = argv2_vec.size() - 1;


    // Suppress deprecated warning for rte_memzone_max_set
    // This is purely because it's an experimental API, but we have to use it
    // since there number of ports we are using is too high for the default.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif
    rte_memzone_max_set(8192);
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif

    // Initialize the Environment Abstraction Layer (EAL).
    // Currently closing DPDKs EAL isn't offically supported,
    // so we only do it once.
    INFO("Initializing DPDK EAL with arguments:");
    for (const auto& arg : str_args) {
        INFO("  {:s}", arg);
    }
    if (!__eal_initalized) {
        int ret = rte_eal_init(argc2, argv2_vec.data());
        if (ret < 0)
            throw std::runtime_error(
                fmt::format(fmt("Failed to init DPDK EAL with error code: {:d}"), ret));
        __eal_initalized = true;
    }
}

void dpdkCore::main_thread() {

    // Start the packet receiving lcores (basically pthreads)
    rte_eal_mp_remote_launch(dpdkCore::lcore_rx, (void*)this, SKIP_MAIN);

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

        // Poll the NIC hardware counters and link state for each port.
        update_port_stats();
    }

    // Wait for the lcores to join
    rte_eal_mp_wait_lcore();
}

void dpdkCore::update_port_stats() {

    for (uint32_t port = 0; port < num_system_ports; ++port) {
        if (handlers[port] == nullptr)
            continue;

        // Check the link state (without blocking on the PHY).
        struct rte_eth_link link;
        if (rte_eth_link_get_nowait(port, &link) == 0) {
            const bool up = (link.link_status == RTE_ETH_LINK_UP);
            if (up != link_is_up[port]) {
                if (up) {
                    INFO("Port {:d}: link is UP at {:d} Mbps", port, link.link_speed);
                } else {
                    ERROR("Port {:d}: link went DOWN! The NIC, cable, or switch port may have "
                          "reset.",
                          port);
                }
                link_is_up[port] = up;
            }
        }

        // Read the NIC hardware counters.
        struct rte_eth_stats stats;
        if (rte_eth_stats_get(port, &stats) != 0)
            continue;

        // imissed counts packets that arrived at the NIC but were dropped because
        // the rx ring was full, i.e. the lcore didn't service this port fast enough.
        const uint64_t missed_delta = stats.imissed - last_eth_stats[port].imissed;
        if (missed_delta > 0) {
            WARN("Port {:d}: NIC dropped {:d} packets since the last poll (~1s) because the rx "
                 "ring was full (lcore stalled or too slow); total missed: {:d}",
                 port, missed_delta, stats.imissed);
        }

        const uint64_t error_delta = stats.ierrors - last_eth_stats[port].ierrors;
        if (error_delta > 0) {
            WARN("Port {:d}: NIC reported {:d} rx errors since the last poll (~1s); total: {:d}",
                 port, error_delta, stats.ierrors);
        }

        const uint64_t nombuf_delta = stats.rx_nombuf - last_eth_stats[port].rx_nombuf;
        if (nombuf_delta > 0) {
            WARN("Port {:d}: {:d} rx mbuf allocation failures since the last poll (~1s); "
                 "increase num_mbufs; total: {:d}",
                 port, nombuf_delta, stats.rx_nombuf);
        }

        std::vector<std::string> port_label = {to_string(port)};
        nic_rx_packets_total_metric.labels(port_label).set(stats.ipackets);
        nic_rx_missed_total_metric.labels(port_label).set(stats.imissed);
        nic_rx_errors_total_metric.labels(port_label).set(stats.ierrors);
        nic_rx_nombuf_total_metric.labels(port_label).set(stats.rx_nombuf);
        nic_link_up_metric.labels(port_label).set(link_is_up[port] ? 1 : 0);

        last_eth_stats[port] = stats;
    }
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
        retval = rte_eth_rx_queue_setup(port, q, rx_ring_size, rte_eth_dev_socket_id(port), nullptr,
                                        mbuf_pools.at(numa_node_of_cpu(lcore_id)));
        if (retval < 0) {
            ERROR("Failed to setupt RX queue for port {:d}, error: {:d}", port, retval);
            return retval;
        }
    }

    // Allocate and set up 1 TX queue per Ethernet port.
    // TODO Do we need this?
    for (q = 0; q < tx_rings; q++) {
        retval =
            rte_eth_tx_queue_setup(port, q, tx_ring_size, rte_eth_dev_socket_id(port), nullptr);
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
    rte_ether_addr addr;

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

    // non-main cores start at 1, but it's easier to 0 base here
    uint32_t lcore = rte_lcore_id() - 1;
    fprintf(stderr, "lcore ID: %u\n", lcore);
    if (lcore > core->num_lcores) {
        throw std::runtime_error("lcore mapping error");
    }

    // If we have no map for this lcore, it must be a worker lcore.
    if (lcore >= core->lcore_port_list.size()) {
        uint32_t worker_id = lcore - core->lcore_port_list.size();
        INFO_NON_OO("Starting worker: worker_id {:d}, lcore {:d}", worker_id, lcore);

        // Cache the worker pointer and ring pointer locally; neither changes at runtime.
        dpdkRXhandler* local_worker = core->workers.at(worker_id);
        rte_ring* local_ring = core->worker_rings.at(worker_id);

        while (!core->stop_thread) {
            uint16_t num_rx =
                rte_ring_dequeue_burst(local_ring, (void**)mbufs, core->burst_size, NULL);

            if (num_rx == 0)
                continue;

            for (uint16_t j = 0; j < num_rx; ++j) {
                if (unlikely(local_worker->handle_packet(mbufs[j]) != 0)) {
                    rte_pktmbuf_free(mbufs[j]);
                    goto exit_loop;
                }
                rte_pktmbuf_free(mbufs[j]);
            }
        }
    exit_loop:

        INFO_NON_OO("Exiting worker: worker_id {:d}, lcore {:d}", worker_id, lcore);
        core->active_workers--;
        if (core->active_workers == 0) {
            core->stop_thread = true;
        }
        return 0;
    }

    // The list of ports this thread processes
    const std::vector<uint32_t>& ports = core->lcore_port_list.at(lcore);
    const uint32_t burst_size = core->burst_size;

    for (uint32_t i = 0; i < ports.size(); ++i) {
        uint32_t port = ports[i];
        if (core->handlers[port] == nullptr) {
            WARN_NON_OO("No valid handler provided for port {:d}", port);
            return 0;
        }
    }

    // TODO Figure out why this sleep is needed.  It seems like when starting the E810
    // ports, there is a delay between when they are started in DPDK and when they become
    // ready.  There might be a function to check their readiness state we could query?
    INFO_NON_OO("DPDK Sleeping for {} seconds before starting network handlers",
                core->lcore_start_delay);
    sleep(core->lcore_start_delay);
    const uint32_t num_local_ports = ports.size();

    // Enforce that all handlers on this lcore are either all distributors or all
    // non-distributors.
    const bool is_distributor_lcore = core->handlers[ports[0]]->is_distributor();
    for (uint32_t i = 1; i < num_local_ports; ++i) {
        if (core->handlers[ports[i]]->is_distributor() != is_distributor_lcore) {
            throw std::runtime_error(
                "All handlers on a single lcore must be either all distributors or all "
                "non-distributors. Mixed lcores are not supported.");
        }
    }

    // Cache handler pointers locally. The handlers array does not change at runtime.
    std::vector<dpdkRXhandler*> local_handlers(num_local_ports);
    for (uint32_t i = 0; i < num_local_ports; ++i) {
        local_handlers[i] = core->handlers[ports[i]];
    }

    if (is_distributor_lcore) {
        // Distributor handlers take ownership of the mbuf; do not free here.
        // Instead the free happens in the worker (above)
        while (!core->stop_thread) {
            for (uint32_t i = 0; i < num_local_ports; ++i) {
                uint32_t port = ports[i];
                const uint16_t num_rx = rte_eth_rx_burst(port, 0, mbufs, burst_size);
                for (uint16_t j = 0; j < num_rx; ++j) {
                    if (unlikely(local_handlers[i]->handle_packet(mbufs[j]) != 0)) {
                        goto exit_lcore;
                    }
                }
            }
        }
    } else {
        while (!core->stop_thread) {
            for (uint32_t i = 0; i < num_local_ports; ++i) {
                uint32_t port = ports[i];
                const uint16_t num_rx = rte_eth_rx_burst(port, 0, mbufs, burst_size);
                for (uint16_t j = 0; j < num_rx; ++j) {
                    if (unlikely(local_handlers[i]->handle_packet(mbufs[j]) != 0)) {
                        goto exit_lcore;
                    }
                    rte_pktmbuf_free(mbufs[j]);
                }
            }
        }
    }
exit_lcore:
    return 0;
}

void dpdkCore::add_graph_details(kotekan::PipelineGraph& graph) const {
    const std::string name = get_unique_name();

    auto& stage_node = graph.add_node(name);
    auto& cluster = graph.add_cluster(name);
    cluster.label = fmt::format(fmt("{:s} ({:d} ports)"), kotekan::leaf_name(name), num_ports);
    cluster.set_attr("style", "rounded,filled")
        .set_attr("fillcolor", kotekan::graph_device_fill)
        .set_attr("color", kotekan::graph_device_line);
    // Keep whatever grouping the pipeline put this stage in as the region's parent.
    cluster.parent = stage_node.cluster;

    // The stage node itself belongs in the region, so that the buffer edges added
    // centrally land inside the box rather than on an empty node beside it.
    stage_node.cluster = cluster.id;

    for (uint i = 0; i < num_ports; ++i) {
        auto& handler = graph.add_node(handlers[i]->unique_name);
        handler.add_line(kotekan::leaf_name(handlers[i]->unique_name));
        handler.cluster = cluster.id;
        handler.set_category(kotekan::GraphCategory::Io);

        // Port node ids carry the stage, so several dpdkCore stages each with a
        // port 0 stay distinct.
        const std::string port_id = fmt::format("{:s}/port_{:d}", name, i);
        graph.add_node(port_id)
            .add_line(fmt::format(fmt("port {:d}"), i))
            .set_category(kotekan::GraphCategory::Endpoint);
        graph.add_edge(port_id, handlers[i]->unique_name);
    }
}
