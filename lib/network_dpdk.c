#include <stdint.h>
#include <stdio.h>
#include <inttypes.h>
#include <sys/time.h>
#include <assert.h>
#include <math.h>

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
#include "nt_memset.h"
#include "fpga_header_functions.h"
#include "vdif_functions.h"

// 256
#define RX_RING_SIZE 512
#define TX_RING_SIZE 512

#define DATA_MAX_SIZE 2048
// 512
#define NUM_MBUFS 1024
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
    .rxmode = { .max_rx_pkt_len = 5000,
        .jumbo_frame = 1,
        .hw_strip_crc = 0,
        .header_split = 0,
        .hw_ip_checksum = 1
    }
};

// Annoyingly DPDK doesn't seem to have a stop function
// So we make sure we only start it once...
int __dpdk_internal_started = 0;

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
        for (int j = 0; j < NUM_FREQ; ++j) {
            net_dpdk->link_data[i][j].buffer_id = net_dpdk->args->link_id[i];
            net_dpdk->link_data[i][j].seq = 0;
            net_dpdk->link_data[i][j].last_seq = 0;
            net_dpdk->link_data[i][j].first_packet = 1;
            net_dpdk->link_data[i][j].finished_buffer = 0;
            net_dpdk->link_data[i][j].vdif_buffer_id = 0;
            net_dpdk->link_data[i][j].dump_location = 0;
        }
    }

    net_dpdk->num_unused_cycles = 0;
    net_dpdk->start_time = e_time();
    net_dpdk->end_time = e_time();
}

static void advance_frame(struct NetworkDPDK * dpdk_net,
                          const int port,
                          const int freq,
                          uint64_t new_seq) {


    //
    INFO ("DPDK: advance_frame: port %d; freq %d; buffer %p; buffer_id &d; new_seq %d",
            port, freq, dpdk_net->args->buf[port][freq], dpdk_net->link_data[port][freq].buffer_id, new_seq);
    mark_buffer_full(dpdk_net->args->buf[port][freq], dpdk_net->link_data[port][freq].buffer_id);

    dpdk_net->link_data[port][freq].buffer_id =
        (dpdk_net->link_data[port][freq].buffer_id + dpdk_net->args->num_links_in_group[port]) % dpdk_net->args->buf[port][freq]->num_buffers;

    // TODO this should be based on packet arrival time - or the seq->time mapping.
    static struct timeval now;
    gettimeofday(&now, NULL);

    // TODO it is really bad to have a blocking call here(!)
    wait_for_empty_buffer(dpdk_net->args->buf[port][freq], dpdk_net->link_data[port][freq].buffer_id);
    set_data_ID(dpdk_net->args->buf[port][freq],
                dpdk_net->link_data[port][freq].buffer_id,
                dpdk_net->link_data[port][freq].data_id++);

    set_first_packet_recv_time(dpdk_net->args->buf[port][freq],
                               dpdk_net->link_data[port][freq].buffer_id,
                               now);

    set_stream_ID(dpdk_net->args->buf[port][freq],
                  dpdk_net->link_data[port][freq].buffer_id,
                  dpdk_net->link_data[port][freq].stream_ID);

    set_fpga_seq_num(dpdk_net->args->buf[port][freq],
                     dpdk_net->link_data[port][freq].buffer_id,
                     new_seq);

}

//  TODO this is very close to the one above, merge?
static void advance_vdif_frame(struct NetworkDPDK * dpdk_net,
                          const int port,
                          uint64_t new_seq) {

    // TODO it is really bad to have a blocking call here(!)
    //DEBUG("Port %d, marking buffer %d as full\n", port, dpdk_net->link_data[port][0].vdif_buffer_id);
    mark_buffer_full(dpdk_net->args->vdif_buf, dpdk_net->link_data[port][0].vdif_buffer_id);

    dpdk_net->link_data[port][0].vdif_buffer_id =
        (dpdk_net->link_data[port][0].vdif_buffer_id + 1) % dpdk_net->args->vdif_buf->num_buffers;

    // TODO this should be based on packet arrival time - or the seq->time mapping.
    static struct timeval now;
    gettimeofday(&now, NULL);

    wait_for_empty_buffer(dpdk_net->args->vdif_buf, dpdk_net->link_data[port][0].vdif_buffer_id);
    set_data_ID(dpdk_net->args->vdif_buf,
                dpdk_net->link_data[port][0].vdif_buffer_id,
                dpdk_net->link_data[port][0].data_id++);

    set_first_packet_recv_time(dpdk_net->args->vdif_buf,
                               dpdk_net->link_data[port][0].vdif_buffer_id,
                               now);

    set_stream_ID(dpdk_net->args->vdif_buf,
                  dpdk_net->link_data[port][0].vdif_buffer_id,
                  dpdk_net->link_data[port][0].stream_ID);

    set_fpga_seq_num(dpdk_net->args->vdif_buf,
                     dpdk_net->link_data[port][0].vdif_buffer_id,
                     new_seq);

}

static inline void copy_data_with_shuffle(struct NetworkDPDK * dpdk_net,
                                          struct rte_mbuf * cur_mbuf,
                                          int port) {
    const int header_offset = 58;  //FPGA/UDP/IP/Ethernet headers
    int pkt_offset = header_offset;

    // TODO Don't hard code.
    const int frame_size = 2048;
    // == 512, number of elements in an fpga crate pair
    const int sub_frame_size = 2048 / NUM_FREQ;

    int buffer_id[NUM_FREQ];
    int64_t frame_location[NUM_FREQ];

    // Get buffers IDs and advance if needed.
    for (int freq = 0; freq < NUM_FREQ; ++freq) {
        buffer_id[freq] = dpdk_net->link_data[port][freq].buffer_id;

        frame_location[freq] = dpdk_net->link_data[port][freq].seq -
                                get_fpga_seq_num(dpdk_net->args->buf[port][freq],
                                                   dpdk_net->link_data[port][freq].buffer_id);

        if (unlikely(frame_location[freq] * frame_size == dpdk_net->args->buf[port][freq]->buffer_size)) {
            advance_frame(dpdk_net, port, freq, dpdk_net->link_data[port][freq].seq);
            frame_location[freq] = 0;
            buffer_id[freq] = dpdk_net->link_data[port][freq].buffer_id;
        }
    }

    // Copy the packet in packet memory order.
    for (int frame = 0;
             frame < dpdk_net->args->timesamples_per_packet;
             ++frame) {

        for (int freq = 0; freq < NUM_FREQ; ++freq) {
            uint64_t copy_location = frame_location[freq] * frame_size + frame * frame_size + freq * sub_frame_size;
            //DEBUG("Port %d; frame %d, freq %d; cur_mbuf %p; offset %d, copy location: %d", port, frame, freq, cur_mbuf, pkt_offset, copy_location);
            copy_block(&cur_mbuf,
                       (uint8_t *) &dpdk_net->args->buf[port][freq]->data[buffer_id[freq]][copy_location],
                       sub_frame_size,
                       &pkt_offset);
        }
    }
}

static inline uint64_t get_mbuf_seq_num(struct rte_mbuf * cur_mbuf) {
    return (uint64_t)(*(uint32_t *)(rte_pktmbuf_mtod(cur_mbuf, char *) + 54)) +
            (((uint64_t) (0xFFFF & (*(uint32_t *)(rte_pktmbuf_mtod(cur_mbuf, char *) + 50)))) << 32);
}

static inline uint16_t get_mbuf_stream_id(struct rte_mbuf * cur_mbuf) {
    return *(uint16_t *)(rte_pktmbuf_mtod(cur_mbuf, char *) + 44);
}

static void check_data_zero(struct NetworkDPDK * dpdk_net, int port, uint8_t * frame_start, int len);

static inline void set_vdif_header_options(struct NetworkDPDK * dpdk_net,
                                            int vdif_frame_location,
                                            int num_elements, int vdif_packet_len,
                                            int invalid, uint64_t seq, int port) {

    int buffer_id = dpdk_net->link_data[port][0].vdif_buffer_id;
    for(int time_step = 0;
            time_step < dpdk_net->args->timesamples_per_packet; ++time_step) {
        for (int elem = 0; elem < num_elements; ++elem) {
            int header_idx = vdif_frame_location +
                vdif_packet_len * num_elements * time_step +
                vdif_packet_len * elem;

            assert(header_idx < dpdk_net->args->vdif_buf->buffer_size);

            struct VDIFHeader * vdif_header =
                (struct VDIFHeader *)&dpdk_net->args->vdif_buf->data[buffer_id][header_idx];

            vdif_header->legacy = 0;
            vdif_header->vdif_version = 1;
            vdif_header->data_type = 1;
            vdif_header->unused = 0;
            vdif_header->ref_epoch = 0;
            vdif_header->frame_len = 132;
            vdif_header->log_num_chan = 10;
            vdif_header->bits_depth = 3;
            vdif_header->edv = 0;
            vdif_header->eud1 = 0;
            vdif_header->eud2 = 0;
            vdif_header->eud3 = 0;
            vdif_header->eud4 = 0;
            // TODO make this dynamic
            vdif_header->station_id = 0x4151; // 0x5141 AQ check which order is correct
            vdif_header->thread_id = elem;
            // Only set the invalid bit if it is invalid, since it is preset to valid
            if (invalid == 1)
                vdif_header->invalid = invalid;
            if (likely(dpdk_net->vdif_time_set)) {
                vdif_header->seconds =
                    (seq + time_step - dpdk_net->vdif_offset) / 390625 +
                    dpdk_net->vdif_base_time;
                vdif_header->data_frame =
                    (seq + time_step - dpdk_net->vdif_offset) % 390625;
            } else {
                vdif_header->invalid = 1;
                vdif_header->seconds = 0;
                vdif_header->data_frame = 0;
            }

        }
    }
}

static inline void copy_data_to_vdif(struct NetworkDPDK * dpdk_net,
                                        struct rte_mbuf * cur_mbuf,
                                        int port) {
    // TODO make some of these more dynamic.
    const int64_t fpga_header_len = 58;
    const int64_t vdif_header_len = 32;
    const int64_t vdif_packet_len = vdif_header_len + 1024;
    const int64_t total_num_elements = 16;
    const int64_t num_elements = 2; // This is also the number of threads.
    const int64_t frame_size = vdif_packet_len * num_elements;
    const int64_t offset = 0;

    int buffer_id = dpdk_net->link_data[port][0].vdif_buffer_id;

    int64_t vdif_frame_location = dpdk_net->link_data[port][0].seq -
                                  get_fpga_seq_num(dpdk_net->args->vdif_buf,
                                                   dpdk_net->link_data[port][0].vdif_buffer_id);

    if (unlikely(vdif_frame_location * frame_size == dpdk_net->args->vdif_buf->buffer_size)) {
        advance_vdif_frame(dpdk_net, port, dpdk_net->link_data[port][0].seq);
        vdif_frame_location = 0;
        buffer_id = dpdk_net->link_data[port][0].vdif_buffer_id;
    }

    stream_id_t stream_id = dpdk_net->link_data[port][0].s_stream_ID;

    //if (port == 0) DEBUG("vdif_frame_location * frame_size = %lld; buffer size = %lld; frame_size = %lld; seq = %lld ",
    //        vdif_frame_location * frame_size, dpdk_net->args->vdif_buf->buffer_size,
    //        frame_size, dpdk_net->link_data[port][0].seq);
    assert(((vdif_frame_location + dpdk_net->args->timesamples_per_packet)* frame_size) <= dpdk_net->args->vdif_buf->buffer_size);

    // Setup the VDIF headers
    if (port == 0) {
        set_vdif_header_options(dpdk_net, vdif_frame_location * frame_size,
                                num_elements, vdif_packet_len, 0,
                                dpdk_net->link_data[port][0].seq, port);
    }

    // Create the parts of the VDIF frame that are in this packet.

    int from_idx = fpga_header_len + offset;
    int mbuf_len = cur_mbuf->data_len;
    for (int time_step = 0;
            time_step < dpdk_net->args->timesamples_per_packet; ++time_step ){
        for (int freq = 0; freq < 128; ++freq) {
            for (int elem = 0; elem < num_elements; ++elem) {

                // Advance to the next mbuf in the chain.
                if (unlikely(from_idx >= mbuf_len)) {
                    cur_mbuf = cur_mbuf->next;
                    assert(cur_mbuf);
                    from_idx -= mbuf_len;  // Subtract the last mbuf_len from the current idx.
                    mbuf_len = cur_mbuf->data_len;
                }

                int output_idx = vdif_frame_location * frame_size + // Frame location in output buffer.
                    vdif_packet_len * num_elements * time_step + // Time step in output frame.
                    vdif_packet_len * elem + // VDIF pack for the correct element (ThreadID).
                    vdif_header_len + // Offset for the vdif header.
                    bin_number_16_elem(&stream_id, freq); // Location in the VDIF packet is just frequency.
                    // TODO on the last point, make sure that works with VDIF endianness.

                // After all that indexing copy one byte :)
                dpdk_net->args->vdif_buf->data[buffer_id][output_idx] =
                        *(rte_pktmbuf_mtod(cur_mbuf, char *) + from_idx);

                from_idx += 1;
            }
            // If only take 2 elements, then we have to skip 14
            from_idx += total_num_elements - num_elements;
        }
    }
}

static inline void copy_data_no_shuffle(struct NetworkDPDK * dpdk_net,
                                        struct rte_mbuf * cur_mbuf,
                                        int port) {
    int offset = 58;  //FPGA/UDP/IP/Ethernet headers

    // TODO Don't hard code.
    const int frame_size = 2048;
    const int packet_data_size = frame_size * dpdk_net->args->timesamples_per_packet;

    int buffer_id = dpdk_net->link_data[port][0].buffer_id;

    int64_t frame_location = dpdk_net->link_data[port][0].seq -
                            get_fpga_seq_num(dpdk_net->args->buf[port][0],
                                               dpdk_net->link_data[port][0].buffer_id);

    if (unlikely(frame_location * frame_size == dpdk_net->args->buf[port][0]->buffer_size)) {
        advance_frame(dpdk_net, port, 0, dpdk_net->link_data[port][0].seq);
        frame_location = 0;
        buffer_id = dpdk_net->link_data[port][0].buffer_id;
    }

    copy_block(&cur_mbuf,
               (uint8_t*)&dpdk_net->args->buf[port][0]->data[buffer_id][frame_location * frame_size],
               packet_data_size,
               &offset);
}

static void check_data_zero(struct NetworkDPDK * dpdk_net, int port, uint8_t * frame_start, int len)  {
    uint64_t * packet_array = (uint64_t *)  frame_start;
    for (int i = 0; i < len/8; ++i) {
        if (packet_array[i] != 0x8888888888888888) {
            fprintf(stderr, "location %d", i);
            assert(0==1);
	    }
    }
}

static void setup_for_first_packet(struct NetworkDPDK * dpdk_net, int port) {

    // Since this is first packet we can expect this to be an instant call
    if (dpdk_net->args->buf != NULL) {
        for (int freq = 0; freq < NUM_FREQ; ++freq) {
            wait_for_empty_buffer(dpdk_net->args->buf[port][freq], dpdk_net->link_data[port][freq].buffer_id);
            set_data_ID(dpdk_net->args->buf[port][freq], dpdk_net->link_data[port][0].buffer_id, dpdk_net->link_data[port][freq].data_id++);
        }
    }
    if (dpdk_net->args->vdif_buf != NULL) {
        wait_for_empty_buffer(dpdk_net->args->vdif_buf, 0);
        set_data_ID(dpdk_net->args->vdif_buf, dpdk_net->link_data[port][0].vdif_buffer_id, dpdk_net->link_data[port][0].data_id++);
    }

}

static inline int align_first_packet(struct NetworkDPDK * dpdk_net,
                                      struct rte_mbuf * cur_mbuf,
                                      int port) {
    uint64_t seq = get_mbuf_seq_num(cur_mbuf);
    uint16_t stream_id = get_mbuf_stream_id(cur_mbuf);
    uint64_t integration_period = dpdk_net->args->samples_per_data_set *
                                    dpdk_net->args->num_data_sets *
                                    dpdk_net->args->num_gpu_frames;

    if ( ((seq % integration_period) <= 100) && ((seq % integration_period) >= 0 )) {

        static struct timeval now;
        gettimeofday(&now, NULL);

        stream_id_t s_stream_id;

        for (int freq = 0; freq < NUM_FREQ; ++freq) {
            s_stream_id = extract_stream_id(stream_id);
            INFO("dpdk: port %d; Got StreamID: crate: %d, slot: %d, link: %d, unused: %d\n",
                    port, s_stream_id.crate_id, s_stream_id.slot_id, s_stream_id.link_id, s_stream_id.unused);
            if (dpdk_net->args->fake_stream_ids == 1) {
                s_stream_id.unused = freq;
                // HACK for 2048 with one pair of crates
                s_stream_id.crate_id = port * 2;
                s_stream_id.link_id = 0;
                INFO("dpdk: Faked StreamID: crate: %d, slot: %d, link: %d, unused: %d\n",
                        s_stream_id.crate_id, s_stream_id.slot_id, s_stream_id.link_id, s_stream_id.unused);
            }
            stream_id = encode_stream_id(s_stream_id);

            dpdk_net->link_data[port][freq].stream_ID = stream_id;
            dpdk_net->link_data[port][freq].s_stream_ID = s_stream_id;

            dpdk_net->link_data[port][freq].last_seq = seq - seq % integration_period;
            dpdk_net->link_data[port][freq].seq = seq;


            if (dpdk_net->args->buf != NULL) {
                set_fpga_seq_num(dpdk_net->args->buf[port][freq],
                                 dpdk_net->link_data[port][freq].buffer_id,
                                 seq - seq % integration_period);
                set_first_packet_recv_time(dpdk_net->args->buf[port][freq],
                                        dpdk_net->link_data[port][freq].buffer_id,
                                        now);
                set_stream_ID(dpdk_net->args->buf[port][freq],
                              dpdk_net->link_data[port][freq].buffer_id,
                              stream_id);
            }
        }

        if (dpdk_net->args->vdif_buf != NULL) {
            set_fpga_seq_num(dpdk_net->args->vdif_buf,
                             dpdk_net->link_data[port][0].vdif_buffer_id,
                             seq - seq % integration_period);
            set_first_packet_recv_time(dpdk_net->args->vdif_buf,
                                    dpdk_net->link_data[port][0].vdif_buffer_id,
                                    now);
            set_stream_ID(dpdk_net->args->vdif_buf,
                          dpdk_net->link_data[port][0].vdif_buffer_id,
                          stream_id);

            if (port == 0) {
                // We solve for a solution to the congruence
                // (seq - usec * 5^8) === offset mod 5^8
                dpdk_net->vdif_offset = ((uint64_t)floor(seq - 390625.0*((double)now.tv_usec/1000000.0))) % 390625;
                // This allows us to take a future (seq - offset) mod 5^8 to get the VDIF data frame.

                // Now we get the VDIF second
                // TODO check that this offset it correct and accounts for the leap seconds correctly
                // 946684800 => 2000-01-01T12:00:00+00:00
                dpdk_net->vdif_base_time = now.tv_sec - 946684800 - ((seq - dpdk_net->vdif_offset) / 390625);
                // To get the current time stamp with the 2000-01-01 epoch:
                // seconds = (seq - offset) / 5^8 + vdif_base_time

                // Indicate that we have this data set for other threads
                dpdk_net->vdif_time_set = 1;

                // Debug test to make sure this works.
                DEBUG("Set VDIF time offsets: base_time: %f; VDIF seconds %" PRIu64 ", data frame %d, vdif_time: %f",
                        (double)now.tv_sec+(double)now.tv_usec/1000000.0,
                        dpdk_net->vdif_base_time + ((seq - dpdk_net->vdif_offset) / 390625),
                        (seq - dpdk_net->vdif_offset) % 390625,
                        (double)(((seq - dpdk_net->vdif_offset) / 390625) + 946684800) +
                        (double)dpdk_net->vdif_base_time +
                        (double)((seq - dpdk_net->vdif_offset) % 390625)/390625.0 );
            }
        }
        INFO("Got first packet: port: %d, seq: %" PRId64 "\n",
                port, dpdk_net->link_data[port][0].seq);

        return 1;
    }
    return 0;
}

static void handle_lost_packets(struct NetworkDPDK * dpdk_net, int port) {

    for (int freq = 0; freq < NUM_FREQ; ++freq) {
        // TODO Consider extracting this to another thread since it is non-deterministic.
        int lost_frames = dpdk_net->link_data[port][freq].seq - dpdk_net->link_data[port][freq].last_seq;
        const int64_t timesamples_per_packet = dpdk_net->args->timesamples_per_packet;
        const int64_t frame_size = 2048;
        const int64_t sub_frame_size = frame_size / NUM_FREQ;

        int64_t frame_location = dpdk_net->link_data[port][freq].last_seq +
            timesamples_per_packet -
            get_fpga_seq_num(dpdk_net->args->buf[port][freq],
                               dpdk_net->link_data[port][freq].buffer_id);
        int64_t cur_seq_num = dpdk_net->link_data[port][freq].last_seq + timesamples_per_packet;

        int buffer_id = dpdk_net->link_data[port][freq].buffer_id;

        struct ErrorMatrix * error_matrix = get_error_matrix(dpdk_net->args->buf[port][freq], buffer_id);
        add_bad_timesamples(error_matrix, lost_frames);

        //fprintf(stderr, "Number of lost frames: %d\n", lost_frames);
        while (lost_frames > 0) {
            if (unlikely(frame_location * frame_size == dpdk_net->args->buf[port][freq]->buffer_size)) {
                advance_frame(dpdk_net, port, freq, cur_seq_num);
                frame_location = 0;
            }
            for (int i = 0; i < timesamples_per_packet; ++ i) {
                nt_memset((void *)&dpdk_net->args->buf[port][freq]->data[buffer_id][frame_location * frame_size + i*frame_size + freq*sub_frame_size],
                      0x88,
                      sub_frame_size);
            }
            cur_seq_num += timesamples_per_packet;
            frame_location += timesamples_per_packet;
            lost_frames -= timesamples_per_packet;
        }
    }
}

static void handle_lost_raw_packets(struct NetworkDPDK * dpdk_net, int port) {

    // TODO Consider extracting this to another thread since it is non-deterministic.
    int lost_frames = dpdk_net->link_data[port][0].seq - dpdk_net->link_data[port][0].last_seq;
    const int64_t timesamples_per_packet = dpdk_net->args->timesamples_per_packet;
    const int vdif_header_len = 32;
    const int vdif_packet_len = vdif_header_len + 1024;
    const int num_elements = 2; // This is also the number of threads.
    const int frame_size = vdif_packet_len * num_elements;

    int64_t frame_location = dpdk_net->link_data[port][0].last_seq +
        timesamples_per_packet -
        get_fpga_seq_num(dpdk_net->args->vdif_buf,
                           dpdk_net->link_data[port][0].vdif_buffer_id);
    int64_t cur_seq_num = dpdk_net->link_data[port][0].last_seq + timesamples_per_packet;

    int buffer_id = dpdk_net->link_data[port][0].vdif_buffer_id;

    struct ErrorMatrix * error_matrix = get_error_matrix(dpdk_net->args->vdif_buf, buffer_id);
    add_bad_timesamples(error_matrix, lost_frames);

    //fprintf(stderr, "Number of lost frames: %d\n", lost_frames);
    while (lost_frames > 0) {
        if (unlikely(frame_location * frame_size == dpdk_net->args->vdif_buf->buffer_size)) {
            advance_vdif_frame(dpdk_net, port, cur_seq_num);
            frame_location = 0;
        }
        set_vdif_header_options(dpdk_net, frame_location * frame_size,
                                num_elements, vdif_packet_len, 1, cur_seq_num, port);
        // TODO zero the corresponding values?
        cur_seq_num += timesamples_per_packet;
        frame_location += timesamples_per_packet;
        lost_frames -= timesamples_per_packet;
    }
}

int lcore_recv_pkt_dump(void *args) {

    INFO("Reached lcore_recv_pkt_dump");

    struct rte_mbuf *mbufs[BURST_SIZE];

    struct NetworkDPDK * dpdk_net = (struct NetworkDPDK *)args;

    uint8_t port;
    unsigned int lcore;
    int64_t lost_frames[dpdk_net->args->num_links];
    for (int i = 0; i < dpdk_net->args->num_links; ++i){
        lost_frames[i] = 0;
    }

    lcore = rte_lcore_id();
    INFO("lcore ID: %d", lcore);
    // Because of the way that we launch the network thread we run the master cores out
    // side of the DPDK RTE framework, so the lcore id of the master core becomes -1.
    if (lcore == -1)
        lcore = 0;

    const int port_offset = dpdk_net->args->port_offset[lcore];
    for (port = port_offset;
         port < dpdk_net->args->num_links_per_lcore + port_offset;
         ++port) {
        setup_for_first_packet(dpdk_net, port);
        INFO("port reached %d", port);
    }

    uint64_t integration_period = dpdk_net->args->samples_per_data_set *
                                    dpdk_net->args->num_data_sets *
                                    dpdk_net->args->num_gpu_frames;

    for (;;) { // Main DPDK loop.

        // For each port.
        for (port = port_offset;
             port < dpdk_net->args->num_links_per_lcore + port_offset;
             ++port) {

            const int32_t nb_rx = rte_eth_rx_burst(port, 0, mbufs, BURST_SIZE);

            if (unlikely(dpdk_net->args->stop_capture == 1)) {
                for (int i = 0; i < nb_rx; ++i) {
                    rte_pktmbuf_free(mbufs[i]);
                }
                goto exit_dpdk_lcore_loop;
            }

            // For each packet on that port.
            for (int i = 0; i < nb_rx; ++i) {

                if (unlikely(dpdk_net->link_data[port][0].finished_buffer == 1)) {
                    goto release_frame;
                }

                if (unlikely((mbufs[i]->ol_flags | PKT_RX_IP_CKSUM_BAD) == 1)) {
                    ERROR("network_dpdk: Got bad packet checksum!");
                    goto release_frame;
                }

                if (unlikely( mbufs[i]->pkt_len
                        != dpdk_net->args->udp_packet_size)) {
                    WARN("Got packet with incorrect length: %d; expected: %d",
                            mbufs[i]->pkt_len,
                            dpdk_net->args->udp_packet_size);
                    goto release_frame;
                }

                //INFO("Got packet on port %d, size %d", port, mbufs[i]->pkt_len);
                uint64_t seq = get_mbuf_seq_num(mbufs[i]);

                // Align first frame
                if (unlikely(dpdk_net->link_data[port][0].first_packet == 1)) {
                    if ( ((seq % integration_period) <= 100) && ((seq % integration_period) >= 0 )) {
                        dpdk_net->link_data[port][0].first_packet = 0;
                        INFO("Got first packet on port %d, with seq%" PRIu64 " ", port, seq);
                        dpdk_net->link_data[port][0].last_seq = seq - dpdk_net->args->timesamples_per_packet;
                    } else {
                        goto release_frame;
                    }
                }

                // Assumes that packet numbers only go up...
                // TODO add case to handle FPGA reset
                int64_t diff = seq - dpdk_net->link_data[port][0].last_seq;
                lost_frames[port] += diff - dpdk_net->args->timesamples_per_packet;
                dpdk_net->link_data[port][0].last_seq = seq;
                if (unlikely((seq % 65536) == 0)) {
                    INFO("Lost frames: Port %d; lost_frames %" PRIi64 ",  %lf%% of total", port,
			lost_frames[port], 100.0*((double)lost_frames[port]/(double)65536) );
                    lost_frames[port] = 0;
                }

                int buffer_id = dpdk_net->link_data[port][0].buffer_id;
                int dump_location = dpdk_net->link_data[port][0].dump_location;
                int offset = 0;

                //assert((dump_location + dpdk_net->args->udp_packet_size)
                //        <= dpdk_net->args->buf[port][0]->buffer_size);

                struct rte_mbuf * pkt = mbufs[i];
                copy_block(&pkt,
                           (uint8_t *)&dpdk_net->args->buf[port][0]->data[buffer_id][dump_location],
                           dpdk_net->args->udp_packet_size,
                           (int *)&offset);

                dpdk_net->link_data[port][0].dump_location += dpdk_net->args->udp_packet_size;

                //INFO("Port %d dump location %d, buffer size %d", port, dpdk_net->link_data[port][0].dump_location, dpdk_net->args->buf[port][0]->buffer_size);

                if (dpdk_net->link_data[port][0].dump_location ==
                        dpdk_net->args->buf[port][0]->buffer_size) {

                    mark_buffer_full(dpdk_net->args->buf[port][0], buffer_id);

                    buffer_id = dpdk_net->link_data[port][0].buffer_id =
                            (dpdk_net->link_data[port][0].buffer_id + 1) %
                            dpdk_net->args->buf[port][0]->num_buffers;

                    if (is_buffer_empty(dpdk_net->args->buf[port][0], buffer_id) == 0) {
                        INFO("dpdk: finishing capture on port %d", port);
                        mark_producer_done(dpdk_net->args->buf[port][0], 0);
                        dpdk_net->link_data[port][0].finished_buffer = 1;
                        goto release_frame;
                    }

                    wait_for_empty_buffer(dpdk_net->args->buf[port][0], buffer_id);
                    dpdk_net->link_data[port][0].dump_location = 0;
                    set_data_ID(dpdk_net->args->buf[port][0], buffer_id, dpdk_net->data_id++);
                }

                release_frame:
                rte_pktmbuf_free(mbufs[i]);
            }
        }
    } // Main DPDK lcore loop
    exit_dpdk_lcore_loop:
    INFO("Existing DPDK on lcore: %d", lcore);
    for (port = port_offset;
             port < dpdk_net->args->num_links_per_lcore + port_offset;
             ++port) {
        mark_producer_done(dpdk_net->args->buf[port][0], 0);
    }
    return 0;
}

/*
 * The lcore main. This is the main thread that does the work, reading from
 * an input port and writing to an output port.
 */
int lcore_recv_pkt(void *args)
{
    struct rte_mbuf *mbufs[BURST_SIZE];

    struct NetworkDPDK * dpdk_net = (struct NetworkDPDK *)args;

    uint8_t port;
    unsigned int lcore;

    lcore = rte_lcore_id();
    INFO("lcore ID: %d", lcore);
    // Because of the way that we launch the network thread we run the master cores out
    // side of the DPDK RTE framework, so the lcore id of the master core becomes -1.
    if (lcore == -1)
        lcore = 0;

    const int port_offset = dpdk_net->args->port_offset[lcore];
    for (port = port_offset;
         port < dpdk_net->args->num_links_per_lcore + port_offset;
         ++port) {
        setup_for_first_packet(dpdk_net, port);
        INFO("port reached %d", port);
    }

    // Main DPDK lcore loop
    for (;;) {

        // For each port.
        for (port = port_offset;
             port < dpdk_net->args->num_links_per_lcore + port_offset;
             ++port) {

            const int32_t nb_rx = rte_eth_rx_burst(port, 0, mbufs, BURST_SIZE);

            if (unlikely(dpdk_net->args->stop_capture == 1)) {
                for (int i = 0; i < nb_rx; ++i) {
                    rte_pktmbuf_free(mbufs[i]);
                }
                goto exit_dpdk_lcore_loop;
            }

            // For each packet on that port.
            for (int i = 0; i < nb_rx; ++i) {

                if (unlikely((mbufs[i]->ol_flags | PKT_RX_IP_CKSUM_BAD) == 1)) {
                    ERROR("network_dpdk: Got bad packet checksum!");
                    goto release_frame;
                }

                if (unlikely( mbufs[i]->pkt_len
                        != dpdk_net->args->udp_packet_size)) {
                    WARN("Got packet with incorrect length: %d; expected: %d",
                            mbufs[i]->pkt_len,
                            dpdk_net->args->udp_packet_size);
                    goto release_frame;
                }

                //INFO("Got packet on port %d, size %d", port, mbufs[i]->pkt_len);

                if (unlikely(dpdk_net->link_data[port][0].first_packet == 1)) {
                    if (likely((align_first_packet(dpdk_net, mbufs[i], port) == 0))) {
                        goto release_frame;
                    }
                    for (int freq = 0; freq < NUM_FREQ; ++freq)
                        dpdk_net->link_data[port][freq].first_packet = 0;
                }

                for (int freq = 0; freq < NUM_FREQ; ++freq)
                    dpdk_net->link_data[port][freq].seq = get_mbuf_seq_num(mbufs[i]);

                // There is only possible diff for all freqs.  TODO: Idealy this value would be a per port only.
                int64_t diff = (int64_t)dpdk_net->link_data[port][0].seq - (int64_t)dpdk_net->link_data[port][0].last_seq;
                if (unlikely(diff < 0)) {
                    DEBUG("Port: %d; Diff %" PRId64 " less than zero, duplicate, bad, or out-of-order packet; last %" PRIu64 "; cur: %" PRIu64 "",
                            port, diff, dpdk_net->link_data[port][0].last_seq, dpdk_net->link_data[port][0].seq);
                    goto release_frame;
                }

                // This allows us to not do the normal GPU buffer operations.
                if (dpdk_net->args->buf != NULL) {
                    if (unlikely(diff > (int64_t)dpdk_net->args->timesamples_per_packet)) {
                        handle_lost_packets(dpdk_net, port);
                    }

                    // Copy the packet to the GPU staging buffer.
                    if (dpdk_net->args->enable_shuffle == 1) {
                        copy_data_with_shuffle(dpdk_net, mbufs[i], port);
                    } else {
                        copy_data_no_shuffle(dpdk_net, mbufs[i], port);
                    }
                }
                if (dpdk_net->args->vdif_buf != NULL) {
                    if (unlikely(diff > (int64_t)dpdk_net->args->timesamples_per_packet)) {
                        handle_lost_raw_packets(dpdk_net, port);
                    }
                    copy_data_to_vdif(dpdk_net, mbufs[i], port);
                }

                for (int freq = 0; freq < NUM_FREQ; ++freq)
                    dpdk_net->link_data[port][freq].last_seq = dpdk_net->link_data[port][freq].seq;

                release_frame:
                rte_pktmbuf_free(mbufs[i]);
            }
        }
    } // Main DPDK lcore loop
    exit_dpdk_lcore_loop:
    INFO("Existing DPDK on lcore: %d", lcore);
    for (port = port_offset;
             port < dpdk_net->args->num_links_per_lcore + port_offset;
             ++port) {
        for (int freq = 0; freq < NUM_FREQ; ++freq)
            mark_producer_done(dpdk_net->args->buf[port][freq], freq);
    }
    return 0;
}

/*
 * The main function, which does initialization and calls the per-lcore
 * functions.
 */
void*
network_dpdk_thread(void * args)
{

    INFO("reached dpdk thread");

    struct NetworkDPDK dpdk_net;

    dpdk_net.args = (struct networkDPDKArg *)args;

    // Shared between all threads.
    dpdk_net.vdif_time_set = 0;

    init_network_object(&dpdk_net);

    if (__dpdk_internal_started == 0) {

        check_port_socket_assignment();

        struct rte_mempool *mbuf_pool;
        unsigned nb_ports;
        uint8_t portid;

        /* Check that there is an even number of ports to send/receive on. */
        nb_ports = rte_eth_dev_count();
        INFO("Number of ports: %d", nb_ports);

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

        if (rte_lcore_count() != dpdk_net.args->num_lcores) {
            INFO("WARNING: The number of lcores %d doesn't match the expected value %d", rte_lcore_count(), dpdk_net.args->num_lcores);
        }
    }
    __dpdk_internal_started = 1;

    INFO("DPDK: Starting lcores!")

    // Start the packet receiving lcores (basically pthreads)
    if (dpdk_net.args->dump_full_packets == 1) {
        rte_eal_mp_remote_launch(lcore_recv_pkt_dump, (void *) &dpdk_net, CALL_MASTER);
    } else {
        rte_eal_mp_remote_launch(lcore_recv_pkt, (void *) &dpdk_net, CALL_MASTER);
    }

    rte_eal_mp_wait_lcore();

    INFO("DPDK stoped lcores!");

    return NULL;
}
