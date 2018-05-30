#ifndef DPDK_BASE_HPP
#define DPDK_BASE_HPP

// DPDK!
extern "C" {
#include <rte_config.h>
#include <rte_common.h>
#include <rte_log.h>
#include <rte_memory.h>
#include <rte_memcpy.h>
#include <rte_memzone.h>
#include <rte_eal.h>
#include <rte_per_lcore.h>
#include <rte_launch.h>
#include <rte_atomic.h>
#include <rte_cycles.h>
#include <rte_prefetch.h>
#include <rte_lcore.h>
#include <rte_per_lcore.h>
#include <rte_branch_prediction.h>
#include <rte_interrupts.h>
#include <rte_pci.h>
#include <rte_random.h>
#include <rte_debug.h>
#include <rte_ether.h>
#include <rte_ethdev.h>
#include <rte_ring.h>
#include <rte_mempool.h>
#include <rte_mbuf.h>
}

#include <emmintrin.h>
#include <string>

#include "kotekanLogging.hpp"
#include "KotekanProcess.hpp"

// There is one handler per port
class dpdkRXhandler : public kotekanLogging {
public:
    dpdkRXhandler(Config &config, const std::string &unique_name,
                  bufferContainer &buffer_container, int port) :
                  config(config), unique_name(unique_name),
                  buffer_container(buffer_container), port(port) {};

    virtual ~dpdkRXhandler() {};

    virtual int handle_packet(struct rte_mbuf *mbuf) = 0;

    virtual void update_stats() = 0;

protected:

    Config &config;
    std::string unique_name;
    bufferContainer &buffer_container;
    uint32_t port;
};

class dpdkCore : public KotekanProcess {
public:

    dpdkCore(Config& config, const string& unique_name,
             bufferContainer &buffer_container);
    ~dpdkCore();

    void main_thread();
    void apply_config(uint64_t fpga_seq) {};

private:

    static int lcore_rx(void *args);

    void dpdk_init(vector<int> lcore_cpu_map, uint32_t master_lcore_cpu);

    void check_port_socket_assignment();

    int32_t port_init(uint8_t port);

    void create_handlers(bufferContainer &buffer_container);

    struct rte_mempool *mbuf_pool;
    struct rte_eth_conf port_conf;

    uint32_t num_lcores;
    uint32_t num_ports;
    uint32_t burst_size;
    uint32_t rx_ring_size;
    uint32_t tx_ring_size;

    // Set to 1 if we should stop the lcores, and zero if running.
    uint32_t stop_lcores;

    struct portList {
        uint32_t * ports;
        uint32_t num_ports;
    };

    // One of these port list structs exists per lcore
    struct portList * lcore_port_list;

    // One of these exists per port
    dpdkRXhandler ** handlers;
};


#endif /* DPDK_BASE_HPP */