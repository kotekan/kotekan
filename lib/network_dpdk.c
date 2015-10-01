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
static inline int
port_init(uint8_t port, struct rte_mempool *mbuf_pool)
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

static inline void
print_eth_stats(const uint32_t port) {

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

/*
 * The lcore main. This is the main thread that does the work, reading from
 * an input port and writing to an output port.
 */
static __attribute__((noreturn)) void
lcore_main(void *args)
{
    const uint8_t nb_ports = rte_eth_dev_count();
    uint8_t port;
    uint32_t num_packets = 0;
    uint32_t num_unused_cycles = 0;
    double start_time = e_time();
    double end_time = e_time();
    struct rte_mbuf *bufs[BURST_SIZE];
    int64_t seq[4];
    int64_t last_seq[4];
    uint16_t stream_ID[4];
    uint64_t lost_packets[4];
    uint32_t num_packets_per_link[4];

    struct networkDPDKArg * arg = (struct networkDPDKArg *)args;

    int buffer_ids[arg->num_links];
    int num_frames = 0;

    static struct timeval now;
    gettimeofday(&now, NULL);
    int data_id = 0;

    for (int i = 0; i < 4; ++i) {
        buffer_ids[i] = 0;
        seq[i] = -1;
        lost_packets[i] = 0;
        last_seq[i] = -1;
        wait_for_empty_buffer(&arg->buf[i], buffer_ids[i]);
        set_data_ID(&arg->buf[i], buffer_ids[i], data_id++);
        set_stream_ID(&arg->buf[i], buffer_ids[i], i);
        set_fpga_seq_num(&arg->buf[i], buffer_ids[i], 0*262144);
        set_first_packet_recv_time(&arg->buf[i], buffer_ids[i], now);
    }

    int out_i[4] = {0, 0, 0, 0};

    /*
     * Check that the port is on the same NUMA node as the polling thread
     * for best performance.
     */
    for (port = 0; port < nb_ports; port++)
        if (rte_eth_dev_socket_id(port) > 0 &&
            rte_eth_dev_socket_id(port) !=
            (int)rte_socket_id())
            printf("WARNING, port %u is on remote NUMA node to "
            "polling thread.\n\tPerformance will "
            "not be optimal.\n", port);

        printf("\nCore %u forwarding packets. [Ctrl+C to quit]\n",
               rte_lcore_id());

        /* Run until the application is quit or killed. */
        for (;;) {
            /*
             * Receive packets
             */
            for (port = 0; port < nb_ports; port++) {

                const int32_t nb_rx = rte_eth_rx_burst(port, 0,
                                                        bufs, BURST_SIZE);

                if (likely(nb_rx == 0)) {
                    num_unused_cycles++;
                    continue;
                }
                num_packets += nb_rx;
                num_packets_per_link[port] += nb_rx;

                //printf("Got packet!\n");
                if (unlikely (num_packets > 16*390625/2)) {
                    //struct rte_mbuf * pkt = bufs[0];
                    //end_time = e_time();
                    /*INFO("Packet rate: %f PPS\n Unused cycles: %d mbuf data len %d mbuf pkt len %d seq_num %u\n",
                           (double)num_packets / (end_time - start_time),
                           num_unused_cycles,
                           pkt->data_len,
                           pkt->pkt_len,
                           *(uint32_t *)(rte_pktmbuf_mtod(pkt, char *) + 54) );*/
                    /*printf("Frame rate: %f Frames/second\n",
                           (double)num_frames / 4.0 / (end_time - start_time));*/

                    num_frames = 0;
                    num_packets = 0;
                    num_unused_cycles = 0;
                    INFO("network_dpdk; lost_packets %.6f%%, %.6f%%, %.6f%%, %.6f%%",
                         (double)lost_packets[0]/(double)num_packets_per_link[0],
                         (double)lost_packets[1]/(double)num_packets_per_link[1],
                         (double)lost_packets[2]/(double)num_packets_per_link[2],
                         (double)lost_packets[3]/(double)num_packets_per_link[3] );
                    for (int i = 0; i < 4; i++) {
                        lost_packets[i] = 0;
                        num_packets_per_link[i] = 0;
                    }
                    //start_time = end_time;
                }

                for (int i = 0; i < nb_rx; ++i) {
                    //fprintf(stderr, "output + out_i: %p\n", output + out_i);

                    if ((bufs[i]->ol_flags | PKT_RX_IP_CKSUM_BAD) == 1) {
                        fprintf(stderr, "Got bad packet!");
                    }

                    struct rte_mbuf * cur_mbuf = bufs[i];
                    int offset = 58;

                    seq[port] = *(uint32_t *)(rte_pktmbuf_mtod(cur_mbuf, char *) + 54);

                    if (unlikely(last_seq[port] == -1)) {
                        last_seq[port] = seq[port];
                        stream_ID[port] = *(uint16_t *)(rte_pktmbuf_mtod(cur_mbuf, char *) + 44) >> 1;
                    }

                    int64_t diff = seq[port] - last_seq[port];
                    if (unlikely(diff < 0)) {
                        diff += COUNTER_MAX + (uint64_t)num_frames_per_packet;
                    }

                    int64_t lost = (diff - (uint64_t)num_frames_per_packet);
                    lost_packets[port] += lost;

                    //INFO("seq %d; last_seq %d; diff %d", seq[port], last_seq[port], diff);
                    last_seq[port] = seq[port];

                    for (int frame = 0; frame < 2; ++frame) {
                        for (int freq = 0; freq < 4; ++freq) {
                            //INFO("cur_buf %p; port %d; out_i[freq] %d; freq %d; offset %d; buf_size %d\n",
                            //     cur_mbuf, port, out_i[freq],
                            //     freq, offset, arg->buf[freq].buffer_size);
                            copy_block(&cur_mbuf,
                                    (uint8_t *) &arg->buf[freq].data[buffer_ids[freq]][out_i[port] + 512*port],
                                    512,
                                    &offset);

                        }
                        out_i[port] += 2048;
                        // TODO this should only mark the buffers as full once we have
                        // a full set of data from all links.  At the moment we are
                        if (unlikely(out_i[port] >= arg->buf[0].buffer_size)) {
                            for (int i = 0; i < 4; ++i) {
                                // TODO it is really bad to have a blocking call here(!)
                                // INFO("Getting new buffer for buf % 4 = %d", i);
                                mark_buffer_full(&arg->buf[i], buffer_ids[i]);

                                buffer_ids[i] = (buffer_ids[i] + 1) % arg->buf[i].num_buffers;

                                static struct timeval now;
                                gettimeofday(&now, NULL);

                                num_frames++;

                                wait_for_empty_buffer(&arg->buf[i], buffer_ids[i]);
                                set_data_ID(&arg->buf[i], buffer_ids[i], data_id++);
                                set_stream_ID(&arg->buf[i], buffer_ids[i], data_id);
                                set_fpga_seq_num(&arg->buf[i], buffer_ids[i], (data_id/4)*262144/2);
                                set_first_packet_recv_time(&arg->buf[i], buffer_ids[i], now);

                                out_i[i] = 0;
                            }
                        }
                    }

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