#include "dpdkCore.hpp"

#include "Config.hpp"           // for Config
#include "ICETelescope.hpp"     // for ice_stream_id_t
#include "StageFactory.hpp"     // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "captureHandler.hpp"   // for captureHandler
#include "iceBoardShuffle.hpp"  // for iceBoardShuffle, iceBoardShuffle::shuffle_size
#include "iceBoardStandard.hpp" // for iceBoardStandard
#include "iceBoardVDIF.hpp"     // for iceBoardVDIF
#include "rfsocHandler.hpp"

#include "fmt.hpp"   // for format, fmt
#include "json.hpp"  // for json, basic_json<>::object_t, basic_json, basic_json<...

#include <algorithm> // for copy, max
#include <atomic>    // for atomic_bool
// cinttypes needed by some CentOS systems.
#include <cinttypes>               // IWYU pragma: keep
#include <functional>              // for _Bind_helper<>::type, bind, function
#include <numa.h>                  // for numa_node_of_cpu, numa_num_configured_nodes
#include <regex>                   // for match_results<>::_Base_type
#include <rte_branch_prediction.h> // for unlikely
#include <rte_config.h>            // for RTE_PKTMBUF_HEADROOM
#include <rte_eal.h>               // for rte_eal_init
#include <rte_errno.h>             // for rte_strerror, per_lcore__rte_errno, rte_errno
#include <rte_ethdev.h>
#include <rte_ether.h>             // for ether_addr
#include <rte_launch.h>            // for rte_eal_mp_remote_launch, rte_eal_mp_wait_lcore, SKIP...
#include <rte_lcore.h>             // for rte_lcore_count, rte_lcore_id
#include <rte_mbuf.h>              // for rte_mbuf, rte_pktmbuf_free, rte_pktmbuf_init, rte_pkt...
#include <rte_mempool.h>           // for rte_mempool, rte_mempool_create, rte_mempool_free
#include <rte_ring.h>
#include <stdexcept>               // for runtime_error
#include <stdio.h>                 // for fprintf, size_t, stderr
#include <stdlib.h>                // for malloc, free
#include <string.h>                // for strncpy, memset
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

/// TODO move this to an inline static once we go to C++17
ice_stream_id_t iceBoardShuffle::all_stream_ids[iceBoardShuffle::shuffle_size];

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
    init_mem_alloc = config.get_default<std::string>(unique_name, "init_mem_alloc", "256");
    num_workers = config.get_default<uint32_t>(unique_name, "num_workers", 0);
    round_robbin_length = config.get_default<uint32_t>(unique_name, "round_robbin_length", 64);

    // Setup the lcore mappings
    // Basically this is mapping the DPDK EAL framework way of assigning threads
    // into the kotekan framework.
    lcore_cpu_map = config.get<std::vector<int>>(unique_name, "lcore_cpu_map");
    uint32_t main_lcore_cpu = config.get<uint32_t>(unique_name, "main_lcore_cpu");

    num_lcores = lcore_cpu_map.size();

    dpdk_init(lcore_cpu_map, main_lcore_cpu);

#ifndef OLD_DPDK
    num_system_ports = rte_eth_dev_count_avail();
#else
    num_system_ports = rte_eth_dev_count();
#endif

    // This default works well for ICE boards,
    // but we might change this to something more genertic
    memset((void*)&port_conf, 0, sizeof(struct rte_eth_conf));
    uint32_t max_rx_pkt_len = config.get_default<uint32_t>(unique_name, "max_rx_pkt_len", 5000);
#ifndef OLD_DPDK
    port_conf.rxmode.max_lro_pkt_size = max_rx_pkt_len;
    port_conf.rxmode.mtu = max_rx_pkt_len;
    port_conf.rxmode.offloads =
        RTE_ETH_RX_OFFLOAD_KEEP_CRC | RTE_ETH_RX_OFFLOAD_IPV4_CKSUM | RTE_ETH_RX_OFFLOAD_UDP_CKSUM;
    // port_conf.rxmode.split_hdr_size = 0;
    port_conf.link_speeds = RTE_ETH_LINK_SPEED_100G | RTE_ETH_LINK_SPEED_FIXED;
#else
    port_conf.rxmode.max_rx_pkt_len = max_rx_pkt_len;
    port_conf.rxmode.jumbo_frame =
        (uint16_t)config.get_default<bool>(unique_name, "jumbo_frame", true);
    port_conf.rxmode.hw_strip_crc = 0;
    port_conf.rxmode.header_split = 0;
    port_conf.rxmode.hw_ip_checksum = 1;
#endif

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
            if (num_workers == 0) {
                if (numa_node_of_cpu(lcore_id) == node_id) {
                    num_ports_on_node += lcore_port_list[j].num_ports;
                }
            } else {
                if (node_id == 0)
                    num_ports_on_node = 1;
                if (node_id == 1)
                    num_ports_on_node = 8;
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
        }
        mbuf_pools.push_back(pool);
    }

    for (uint32_t worker_id = 0; worker_id < num_workers; ++worker_id) {
        rte_ring* rx_worker_ring =
            rte_ring_create(("Worker_" + std::to_string(worker_id)).c_str(), 512, 0, 0);
        if (rx_worker_ring == nullptr)
            throw std::runtime_error("Cannot create worker ring");
        worker_rings.push_back(rx_worker_ring);
    }

    // Init ports referenced in the lcore port mapping
    int i = 0; // Index into lcore_cpu_map
    for (vector<int> ports : lcore_port_map) {
        for (uint32_t port : ports) {
            // TODO This will fail in a strange way if a port is listed more than once in the
            // config. We should have a check that each port assignment is unique.
            if (i == 0) {
                if (port_init(port, lcore_cpu_map.at(i)) != 0) {
                    throw std::runtime_error(fmt::format(fmt("DPDK Cannot init port: {:d}"), port));
                }
            } else {
                if (port_init(port, lcore_cpu_map.at(i + 2)) != 0) {
                    throw std::runtime_error(fmt::format(fmt("DPDK Cannot init port: {:d}"), port));
                }
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
    // if (handlers_block.size() != num_system_ports) {
    //     throw std::runtime_error(fmt::format(fmt("The number of DPDK handlers ({:d}) must be
    //     equal "
    //                                              "to the number of system ports ({:d})"),
    //                                          handlers_block.size(), num_system_ports));
    // }
    handlers = (dpdkRXhandler**)malloc(20 * sizeof(dpdkRXhandler*));
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
        } else if (handler_name == "rfsocHandler") {
            handlers[port] = new rfsocHandler(config, handler_unique_name, buffer_container, port);
        } else if (handler_name == "none") {
            handlers[port] = nullptr;
        } else {
            throw std::runtime_error(
                fmt::format(fmt("The dpdk handler type '{:s}' does not exist."), handler_name));
        }

        port++;
    }
}

void dpdkCore::dpdk_init(vector<int> lcore_cpu_map, uint32_t main_lcore_cpu) {

    string dpdk_lcore_map = fmt::format(fmt("0@{:d},"), main_lcore_cpu);
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
    char arg5[] = "--socket-mem";
    char* arg6 = (char*)malloc(init_mem_alloc.length() + 1);
    strncpy(arg6, init_mem_alloc.c_str(), init_mem_alloc.length() + 1);
    // Generate final options string for EAL initialization
    char* argv2[] = {&arg0[0], &arg1[0], &arg2[0], &arg3[0], &arg4[0], &arg5[0], &arg6[0], nullptr};
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

    if (num_workers == 0) {
// Start the packet receiving lcores (basically pthreads)
#ifndef OLD_DPDK
        rte_eal_mp_remote_launch(dpdkCore::lcore_rx, (void*)this, SKIP_MAIN);
#else
        // Support old DPDK versions
        rte_eal_mp_remote_launch(dpdkCore::lcore_rx, (void*)this, SKIP_MASTER);
#endif
    } else {
        INFO("Starting distributor lcore on {:d}, with {:d} workers", 1, num_workers);
        if (rte_eal_remote_launch(dpdkCore::lcore_rx_distributor, (void*)this, 1) != 0) {
            ERROR("Could not start distributor lcore");
        }
        for (uint32_t i = 0; i < num_workers; ++i) {
            INFO("Starting worker on lcore {:d}", i + 2);
            if (rte_eal_remote_launch(dpdkCore::lcore_worker, (void*)this, i + 2) != 0) {
                ERROR("Could not start worker lcore");
            }
        }
        for (uint32_t i = 0; i < 8; ++i) {
            INFO("Starting 10G capture handler on lcore {:d}", i + num_workers + 2);
            if (rte_eal_remote_launch(dpdkCore::lcore_rx, (void*)this, i + num_workers + 2) != 0) {
                ERROR("Could not start worker lcore");
            }
        }
    }

    while (!stop_thread) {
        sleep(1);

        // Get the handlers to update their stats about packets
        // Some of these stats might be moved to this class, but
        // it seemed like it was worth leaving it upto the handler
        // to say which stats we actually care about recording.
        for (uint32_t i = 0; i < num_system_ports; ++i) {
            // if (handlers[i] != nullptr)
            //     handlers[i]->update_stats();
        }

        // TODO Check port status
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

    INFO("Called port_init port {:d} lcore_id {:d}", port, lcore_id);

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
        assert(mbuf_pools.at(numa_node_of_cpu(lcore_id)) != nullptr);
        retval = rte_eth_rx_queue_setup(port, q, 1024, rte_eth_dev_socket_id(port), nullptr,
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

    /*
        uint32_t link_fec_capa;
        retval = rte_eth_fec_get(port, &link_fec_capa);
        if (retval != 0) {
            WARN("Failed to get FEC status, port {:d}, error: {:d}", port, retval);
            if (retval == -ENOTSUP) {
                WARN("Card does not support FEC");
            } else if (retval == -EIO) {
                WARN("Device is removed?");
            } else if (retval == -ENODEV) {
                WARN("Invalid port");
            }
        } else {
            INFO("Port {:d} FEC Status: {:b}", port, link_fec_capa);
        }

        rte_eth_fec_capa link_fec_capa_array[50];
        retval = rte_eth_fec_get_capability(port, link_fec_capa_array, 50);
        if (retval < 0) {
            WARN("Failed to get FEC status, port {:d}, error: {:d}", port, retval);
            if (retval == -ENOTSUP) {
                WARN("Card does not support FEC");
            } else if (retval == -EIO) {
                WARN("Device is removed?");
            } else if (retval == -ENODEV) {
                WARN("Invalid port");
            }
        } else {
            INFO("Port {:d} Retval num_capa: {:d}", port, retval);
        }
    */

    // Start the Ethernet port.
    retval = rte_eth_dev_start(port);
    if (retval < 0) {
        ERROR("Failed to start port: {:d}", port);
        return retval;
    }


    // Report the port MAC address.
    // TODO record the MAC address for export to JSON
#ifndef OLD_DPDK
    rte_ether_addr addr;
#else
    struct ether_addr addr;
#endif
    rte_eth_macaddr_get(port, &addr);
    INFO("Port {:d} MAC: {:02x} {:02x} {:02x} {:02x} {:02x} {:02x} "
         "memory assigned to numa_node {:d}",
         (unsigned)port, addr.addr_bytes[0], addr.addr_bytes[1], addr.addr_bytes[2],
         addr.addr_bytes[3], addr.addr_bytes[4], addr.addr_bytes[5], numa_node_of_cpu(lcore_id));

    // Enable promiscuous mode.
    rte_eth_promiscuous_enable(port);

    return 0;
}

int dpdkCore::lcore_worker(void* args) {
    dpdkCore* core = (dpdkCore*)args;

    const uint32_t burst_size = core->burst_size;
    rte_mbuf* mbufs[burst_size];

    // TODO make this more general
    int port = 0;

    int lcore_id = rte_lcore_id();
    int worker_id = lcore_id - 2;
    INFO_NON_OO("Worker: lcore {:d}, worker_id: {:d}", lcore_id, worker_id);

    uint64_t total_packets = 0;
    struct timeval last_time;
    gettimeofday(&last_time, nullptr);

    while (!core->stop_thread) {
        uint16_t num_rx =
            rte_ring_dequeue_burst(core->worker_rings[worker_id], (void**)mbufs, burst_size, NULL);

        if (num_rx == 0)
            continue;
        /*
                for (uint16_t j = 0; j < num_rx; ++j) {
                    total_packets += 1;
                    if (unlikely(total_packets % (1250000 * 1) == 0)) {
                        struct timeval now;
                        gettimeofday(&now, nullptr);
                        double elapsed_time = tv_to_double(now) - tv_to_double(last_time);
                        INFO_NON_OO("Packet rate: {:.0f} pps, data rate: {:.4f}Gb/s",
                                    total_packets / elapsed_time,
                                    (double)total_packets * 8224 * 8 / 1e9 / elapsed_time);
                        last_time = now;
                        total_packets = 0;
                    }
                    // rte_pktmbuf_free(mbufs[j]);
                }
        */
        for (uint16_t j = 0; j < num_rx; ++j) {
            if (unlikely(core->handlers[worker_id]->handle_packet(mbufs[j]) != 0))
                break;
        }
        rte_pktmbuf_free_bulk(mbufs, num_rx);
    }
}

int dpdkCore::lcore_rx_distributor(void* args) {
    dpdkCore* core = (dpdkCore*)args;

    const uint32_t burst_size = core->burst_size;
    const uint32_t round_robbin_length = core->round_robbin_length;

    INFO_NON_OO("Started lcore_rx_distributor on lcore: {:d}", rte_lcore_id());

    const uint32_t port = 0;
    const uint32_t num_workers = core->num_workers;

    rte_mbuf* mbufs[burst_size];

    uint16_t distributor_idx = 0;
    uint32_t packets_sent_to_worker = 0;
    uint32_t worker_id = 0;
    uint64_t total_packets = 0;
    uint64_t last_seq = 0;
    uint64_t lost_packets = 0;
    struct timeval last_time;

    while (!core->stop_thread) {
        uint16_t num_rx = rte_eth_rx_burst(port, 0, mbufs, burst_size);
        if (unlikely(num_rx == 0))
            continue;

        // Process some of packet header information, copy happens in an other thread
        for (uint16_t j = 0; j < num_rx; ++j) {
            total_packets += 1;
            uint64_t seq_num = *rte_pktmbuf_mtod_offset(mbufs[j], uint64_t*, 50);

            if (unlikely(last_seq) == 0) {
                last_seq = seq_num;
                gettimeofday(&last_time, nullptr);
            } else if (unlikely(seq_num > last_seq + 16)) {
                lost_packets += (seq_num - last_seq - 16) / 16;
            }

            last_seq = seq_num;

            if (unlikely(total_packets % (1450000 * 5) == 0)) {
                struct timeval now;
                gettimeofday(&now, nullptr);
                double elapsed_time = tv_to_double(now) - tv_to_double(last_time);
                INFO_NON_OO("Port: {:d} Packet rate: {:.0f} pps, data rate: {:.4f}Gb/s, "
                            "lost_packets rate: {:.4f}%",
                            port, total_packets / elapsed_time,
                            (double)total_packets * 8224 * 8 / 1e9 / elapsed_time,
                            (double)lost_packets / (double)total_packets * 100.0);
                last_time = now;
                total_packets = 0;
                lost_packets = 0;
            }
        }


        // INFO_NON_OO("Sending packets to worker {:d}, total packets: {:d}", worker_id,
        //             total_packets);
        //   Transferring the packets directly into the next ring might not be the
        //   most optimal way to do things since there is more queue overhead
        //   However this might be offset by not having to store mbufs


        uint16_t num_enqueued =
            rte_ring_enqueue_burst(core->worker_rings[worker_id], (void**)mbufs, num_rx, NULL);
        packets_sent_to_worker += num_enqueued;

        if (unlikely(num_enqueued < num_rx)) {
            // TODO we could try another working lcore in this case,
            // but that has balancing considerations
            WARN_NON_OO("Packet loss due to full worker lcore ring");
            while (num_enqueued < num_rx)
                rte_pktmbuf_free(mbufs[num_enqueued++]);
        }

        // Note this isn't a perfect load_balancing system.  However, it's cheaper to implement
        // it this way, and the different should average out over time.
        // TODO check if the above is true.
        if (packets_sent_to_worker > round_robbin_length) {
            // Switch to next worker
            worker_id = (worker_id + 1) % num_workers;
            packets_sent_to_worker = 0;
        }
    }
exit_lcore_rx_distributor:
    return 0;
}

int dpdkCore::lcore_rx(void* args) {
    dpdkCore* core = (dpdkCore*)args;

    // return 0;
    struct rte_mbuf* mbufs[core->burst_size];

    // non-main cores start at 1, but it's easier to 0 base here
    uint32_t lcore = rte_lcore_id();
    const uint32_t port = lcore - 3;
    fprintf(stderr, "starting lcore_rx on lcore ID: %u using port %u\n", lcore, port);
    if (lcore > core->num_lcores) {
        throw std::runtime_error("lcore mapping error");
    }

    const uint32_t burst_size = core->burst_size;

    uint16_t distributor_idx = 0;
    uint32_t packets_sent_to_worker = 0;
    uint32_t worker_id = 0;
    uint64_t total_packets = 0;
    uint32_t last_seq = 0;
    uint64_t lost_packets = 0;
    struct timeval last_time;

    while (!core->stop_thread) {

        const uint16_t num_rx = rte_eth_rx_burst(port, 0, mbufs, burst_size);

        for (uint16_t j = 0; j < num_rx; ++j) {

            total_packets += 1;
            uint32_t seq_num = *rte_pktmbuf_mtod_offset(mbufs[j], uint32_t*, 42);

            if (unlikely(last_seq) == 0) {
                last_seq = seq_num;
                gettimeofday(&last_time, nullptr);
            } else if (unlikely(seq_num > last_seq + 1)) {
                lost_packets += seq_num - last_seq - 1;
            }

            last_seq = seq_num;

            if (unlikely(total_packets % (304875 * 5) == 0)) {
                struct timeval now;
                gettimeofday(&now, nullptr);
                double elapsed_time = tv_to_double(now) - tv_to_double(last_time);
                INFO_NON_OO("Port: {:d} Packet rate: {:.0f} pps, effective data rate: {:.4f}Gb/s, "
                            "lost_packets rate: {:.4f}%",
                            port, total_packets / elapsed_time,
                            (double)total_packets * 4076 * 8 / 1e9 / elapsed_time,
                            (double)lost_packets / (double)total_packets * 100.0);
                last_time = now;
                total_packets = 0;
                lost_packets = 0;
            }

            // Process the packet with the required handler
            // NOTE: Ideally this wouldn't be a call to a virtual function,
            // but it's an overhead that's hard to avoid here.
            if (unlikely(core->handlers[port + 1]->handle_packet(mbufs[j]) != 0)) {
                goto exit_lcore;
            }

            rte_pktmbuf_free(mbufs[j]);
        }
    }
exit_lcore:
    return 0;
}

std::string dpdkCore::dot_string(const std::string& prefix) const {
    std::string dot = fmt::format("{:s}subgraph \"cluster_{:s}\" {{\n", prefix, get_unique_name());

    dot += fmt::format("{:s}{:s}style=filled;\n", prefix, prefix);
    dot += fmt::format("{:s}{:s}color=lightgrey;\n", prefix, prefix);
    dot += fmt::format("{:s}{:s}node [style=filled,color=white];\n", prefix, prefix);
    dot += fmt::format("{:s}{:s}label = \"{:s}\";\n", prefix, prefix, get_unique_name());

    for (uint i = 0; i < num_ports; ++i) {
        dot += fmt::format("{:s}{:s} \"{:s}\" [shape=box];\n", prefix, prefix,
                           handlers[i]->unique_name);
    }

    dot += fmt::format("{:s}}}\n", prefix);

    for (uint i = 0; i < num_ports; ++i) {
        dot += fmt::format("{:s}port_{:d} [shape=doubleoctagon style=filled,color=lightblue];\n",
                           prefix, i);
        dot += fmt::format("{:s}port_{:d} -> \"{:s}\";\n", prefix, i, handlers[i]->unique_name);
    }

    return dot;
}