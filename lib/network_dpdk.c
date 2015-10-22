#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include <sys/time.h>
#include <assert.h>

#define __STDC_FORMAT_MACROS
#include <inttypes.h>

// DPDK!
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
#include <emmintrin.h>

#include "packet_copy.h"
#include "network_dpdk.h"
#include "config.h"

#define RX_RING_SIZE 64
#define TX_RING_SIZE 512

#define DATA_MAX_SIZE 2048
#define NUM_MBUFS 256
#define MBUF_SIZE (DATA_MAX_SIZE + sizeof(struct rte_mbuf) + RTE_PKTMBUF_HEADROOM)
#define MBUF_CACHE_SIZE 250
#define BURST_SIZE 32

// Count max
#define COUNTER_BITS 32
#define COUNTER_MAX (1ll << COUNTER_BITS) - 1ll

static inline double e_time(void) {
    static struct timeval now;
    gettimeofday(&now, NULL);
    return (double)(now.tv_sec  + now.tv_usec/1000000.0);
}

static const struct rte_eth_conf port_conf_default = {
    .rxmode = { .max_rx_pkt_len = 9000,
        .jumbo_frame = 1,
        .hw_strip_crc = 0,
        .header_split = 0,
        .hw_ip_checksum = 1
    }
};

/* basicfwd.c: Basic DPDK skeleton forwarding example. */

/*
 * Initializes a given port using global settings and with the RX buffers
 * coming from the mbuf_pool passed as a parameter.
 */
static inline int port_init(uint8_t port, struct rte_mempool *mbuf_pool)
{
    struct rte_eth_conf port_conf = port_conf_default;
    const uint16_t rx_rings = 1, tx_rings = 1;
    int retval;
    uint16_t q;

    if (port >= rte_eth_dev_count())
        return -1;

    /* Configure the Ethernet device. */
    retval = rte_eth_dev_configure(port, rx_rings, tx_rings, &port_conf);
    if (retval != 0)
        return retval;

    /* Allocate and set up 1 RX queue per Ethernet port. */
    for (q = 0; q < rx_rings; q++) {
        retval = rte_eth_rx_queue_setup(port, q, RX_RING_SIZE,
                                        rte_eth_dev_socket_id(port), NULL, mbuf_pool);
        if (retval < 0)
            return retval;
    }

    /* Allocate and set up 1 TX queue per Ethernet port. */
    for (q = 0; q < tx_rings; q++) {
        retval = rte_eth_tx_queue_setup(port, q, TX_RING_SIZE,
                                        rte_eth_dev_socket_id(port), NULL);
        if (retval < 0)
            return retval;
    }

    /* Start the Ethernet port. */
    retval = rte_eth_dev_start(port);
    if (retval < 0)
        return retval;

    /* Display the port MAC address. */
    struct ether_addr addr;
    rte_eth_macaddr_get(port, &addr);
    printf("Port %u MAC: %02" PRIx8 " %02" PRIx8 " %02" PRIx8
    " %02" PRIx8 " %02" PRIx8 " %02" PRIx8 "\n",
    (unsigned)port,
           addr.addr_bytes[0], addr.addr_bytes[1],
           addr.addr_bytes[2], addr.addr_bytes[3],
           addr.addr_bytes[4], addr.addr_bytes[5]);

    /* Enable RX in promiscuous mode for the Ethernet device. */
    rte_eth_promiscuous_enable(port);

    return 0;
}

static void check_port_socket_assignment() {

    const uint8_t nb_ports = rte_eth_dev_count();
    assert(nb_ports == NUM_LINKS);

    /*
    * Check that the port is on the same NUMA node as the polling thread
    * for best performance.
    */
    for (int port = 0; port < nb_ports; port++)
        if (rte_eth_dev_socket_id(port) > 0 &&
            rte_eth_dev_socket_id(port) !=
            (int)rte_socket_id())
            WARN("WARNING, port %u is on remote NUMA node to "
            "polling thread.\n\tPerformance will "
            "not be optimal.\n", port);

        INFO("network_dpdk: Core %u forwarding packets. [Ctrl+C to quit]\n",
            rte_lcore_id());
}


static inline void print_eth_stats(const uint32_t port) {

    static struct rte_eth_stats rx_stats[4];

    rte_eth_stats_get(port, &rx_stats[port]);
    INFO("rx_stats[%d].ibadcrc = %" PRIu64 "; imissed = %" PRIu64"; ibadlen = %" PRIu64"; ierrors = %" PRIu64"; rx_nombuf = %" PRIu64"; q_errors = %" PRIu64"; ipackets = %" PRIu64 "",
         port,
         rx_stats[port].ibadcrc,
         rx_stats[port].imissed,
         rx_stats[port].ibadlen,
         rx_stats[port].ierrors,
         rx_stats[port].rx_nombuf,
         rx_stats[port].q_errors[0],
         rx_stats[port].ipackets );
    // Reset the counter.
    rte_eth_stats_reset(port);
}

static void init_network_object(struct NetworkDPDK * net_dpdk) {

    for (int i = 0; i < NUM_LINKS; ++i) {

        net_dpdk->link_data[i].buffer_id = 0;
        net_dpdk->link_data[i].seq = -1;
        net_dpdk->link_data[i].lost_packets = 0;
        net_dpdk->link_data[i].last_seq = -1;
        net_dpdk->link_data[i].cur_seq64_edge = 0;
        net_dpdk->link_data[i].num_packets = 0;
        net_dpdk->link_data[i].seq64 = 0;
        net_dpdk->link_data[i].num_packets = 0;
        net_dpdk->link_data[i].finished_buffer = 0;
    }

    net_dpdk->num_unused_cycles = 0;
    net_dpdk->start_time = e_time();
    net_dpdk->end_time = e_time();
}

static void check_and_advance_frame(struct NetworkDPDK * dpdk_net,
                                    const int port,
                                    int * data_id) {

    // TODO it is really bad to have a blocking call here(!)

    mark_buffer_full(&dpdk_net->args->buf[port], dpdk_net->link_data[port].buffer_id);

    dpdk_net->link_data[port].buffer_id =
    (dpdk_net->link_data[port].buffer_id + 1) % dpdk_net->args->buf[port].num_buffers;

    static struct timeval now;
    gettimeofday(&now, NULL);

    wait_for_empty_buffer(&dpdk_net->args->buf[port], dpdk_net->link_data[port].buffer_id);
    set_data_ID(&dpdk_net->args->buf[port], dpdk_net->link_data[port].buffer_id, (*data_id)++);
    set_stream_ID(&dpdk_net->args->buf[port], dpdk_net->link_data[port].buffer_id, *data_id);
    set_fpga_seq_num(&dpdk_net->args->buf[port], dpdk_net->link_data[port].buffer_id, (*data_id/4)*262144/2);
    set_first_packet_recv_time(&dpdk_net->args->buf[port], dpdk_net->link_data[port].buffer_id, now);

}



static inline void copy_data_with_shuffle(struct NetworkDPDK * dpdk_net,
                                          struct rte_mbuf * cur_mbuf,
                                          int port) {
    int offset = 58;

    for (int frame = 0;
         frame < dpdk_net->args->config->fpga_network.timesamples_per_packet;
         ++frame) {

        // TODO this should only mark the buffers as full once we have
        // a full set of data from all links.  At the moment we are
        if (unlikely(dpdk_net->link_data[port].seq64 >=
                        dpdk_net->link_data[port].cur_seq64_edge)) {
            for (int i = 0; i < 4; ++i) {

            }
        }

        // TODO this 4 shouldn't be hard coded
        for (int freq = 0; freq < 4; ++freq) {
            copy_block(&cur_mbuf,
                       (uint8_t *) &dpdk_net->args->buf[freq].data[dpdk_net->link_data[freq].buffer_id][512*port],
                       512,
                       &offset);

        }

    }
}

static inline void copy_data_no_shuffle() {


}

/*
 * The lcore main. This is the main thread that does the work, reading from
 * an input port and writing to an output port.
 */
static __attribute__((noreturn)) void
lcore_main(void *args)
{
    struct NetworkDPDK dpdk_net;
    uint8_t port;

    dpdk_net.args = (struct networkDPDKArg *)args;

    struct rte_mbuf *bufs[BURST_SIZE];

    int data_id = 0;

    check_port_socket_assignment();

    static struct timeval now;
    gettimeofday(&now, NULL);

    for (port = 0; port < NUM_LINKS; ++port) {
        wait_for_empty_buffer(&dpdk_net.args->buf[port], dpdk_net.link_data[port].buffer_id);
        set_data_ID(&dpdk_net.args->buf[port], dpdk_net.link_data[port].buffer_id, data_id++);
        set_stream_ID(&dpdk_net.args->buf[port], dpdk_net.link_data[port].buffer_id, port);
        set_fpga_seq_num(&dpdk_net.args->buf[port], dpdk_net.link_data[port].buffer_id, 0);
        set_first_packet_recv_time(&dpdk_net.args->buf[port], dpdk_net.link_data[port].buffer_id, now);
    }


    /* Run until the application is quit or killed. */
    for (;;) {

        // For each port.
        for (port = 0; port < dpdk_net.args->num_links; port++) {

            const int32_t nb_rx = rte_eth_rx_burst(port, 0,
                                                    bufs, BURST_SIZE);

            if (likely(nb_rx == 0)) {
                dpdk_net.num_unused_cycles++;
                continue;
            }
            dpdk_net.link_data[port].num_packets += nb_rx;

            //printf("Got packet!\n");
            if (unlikely (dpdk_net.link_data[port].num_packets > 16*390625/2)) {

                INFO("network_dpdk; link %d lost_packets %.6f%%", port,
                    (double)dpdk_net.link_data[port].lost_packets/
                    (double)dpdk_net.link_data[port].num_packets);

                dpdk_net.link_data[port].num_packets = 0;
                dpdk_net.link_data[port].lost_packets = 0;
            }

            // For each packet on that port.
            for (int i = 0; i < nb_rx; ++i) {

                if ((bufs[i]->ol_flags | PKT_RX_IP_CKSUM_BAD) == 1) {
                    ERROR("network_dpdk: Got bad packet!");
                    continue;
                }

                dpdk_net.link_data[port].seq = *(uint32_t *)(rte_pktmbuf_mtod(bufs[i], char *) + 54);

                if (unlikely(dpdk_net.link_data[port].last_seq == -1)) {
                    dpdk_net.link_data[port].last_seq = dpdk_net.link_data[port].seq;
                    dpdk_net.link_data[port].stream_ID = *(uint16_t *)(rte_pktmbuf_mtod(bufs[i], char *) + 44);
                }

                int64_t diff = dpdk_net.link_data[port].seq - dpdk_net.link_data[port].last_seq;
                if (unlikely(diff < 0)) {
                    diff += COUNTER_MAX + (uint64_t)dpdk_net.args->config->fpga_network.timesamples_per_packet;
                }

                dpdk_net.link_data[port].lost_packets = diff - (uint64_t)dpdk_net.args->config->fpga_network.timesamples_per_packet;

                // TODO Mark lost packets here to be zeroed from GPU frame.

                // Copy the packet to the GPU staging buffer.
                copy_data_with_shuffle(&dpdk_net, bufs[i], port);


                dpdk_net.link_data[port].last_seq = dpdk_net.link_data[port].seq;

                rte_pktmbuf_free(bufs[i]);
            }
        }
    }
}

/*
 * The main function, which does initialization and calls the per-lcore
 * functions.
 */
void
network_dpdk_thread(void * args)
{
    struct rte_mempool *mbuf_pool;
    unsigned nb_ports;
    uint8_t portid;

    /* Check that there is an even number of ports to send/receive on. */
    nb_ports = rte_eth_dev_count();
    fprintf(stdout, "Number of ports: %d\n", nb_ports);

    /* Creates a new mempool in memory to hold the mbufs. */
    mbuf_pool = rte_mempool_create("MBUF_POOL",
                                   NUM_MBUFS * nb_ports,
                                   MBUF_SIZE,
                                   MBUF_CACHE_SIZE,
                                   sizeof(struct rte_pktmbuf_pool_private),
                                   rte_pktmbuf_pool_init, NULL,
                                   rte_pktmbuf_init,      NULL,
                                   rte_socket_id(),
                                   0);

    if (mbuf_pool == NULL)
        rte_exit(EXIT_FAILURE, "Cannot create mbuf pool\n");

    /* Initialize all ports. */
    for (portid = 0; portid < nb_ports; portid++){
        if (port_init(portid, mbuf_pool) != 0) {
            rte_exit(EXIT_FAILURE, "Cannot init port %"PRIu8 "\n", portid);
        }
    }

    if (rte_lcore_count() > 1)
        printf("\nWARNING: Too many lcores enabled. Only 1 used.\n");

    /* Call lcore_main on the master core only. */
    lcore_main(args);

    return;
}