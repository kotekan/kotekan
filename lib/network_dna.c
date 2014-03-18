#include "network_dna.h"
#include "buffers.h"
#include "pfring.h"
#include "errors.h"
#include "test_data_generation.h"
#include "error_correction.h"

#include <sys/socket.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <memory.h>
#include <unistd.h>
#include <assert.h>
#include <pthread.h>
#include <inttypes.h>

//#define UDP_PACKETSIZE 8256
#define UDP_PACKETSIZE 9296
#define UDP_PAYLOADSIZE 8192
#define NUM_TIMESAMPLES_PER_PACKET 4

// Count max
#define COUNTER_BITS 30
#define COUNTER_MAX (1ll << COUNTER_BITS) - 1ll

#define SEQ_NUM_EDGE 100000

double e_time(void) {
    static struct timeval now;
    gettimeofday(&now, NULL);
    return (double)(now.tv_sec  + now.tv_usec/1000000.0);
}

void check_if_done(int * total_buffers_filled, struct networkThreadArg * args,
                    long long int total_packets, int32_t total_lost, int32_t total_out_of_order, 
                   int32_t total_duplicate, double start_time) {

    (*total_buffers_filled)++;

    // If data_limit = 0 => unlimited
    if (args->data_limit == 0) {
        return;
    }

    if ( (*total_buffers_filled) * (args->buf->buffer_size / (1024*1024)) >= args->data_limit * 1024) {
        double end_time = e_time();
        printf("Stopping packet capture, ran for ~ %f seconds.\n", end_time - start_time);
        printf("\nStats:\nTotal Packets Captured: %lld\nPackets lost: %d\nOut of order packets: %d\nDuplicate Packets: %d\n", 
                total_packets, total_lost, total_out_of_order, total_duplicate);
        mark_producer_done(args->buf, args->link_id);
        int ret = 0;
        pthread_exit((void *) &ret);
    }
}


void network_thread(void * arg) {

    struct networkThreadArg * args;
    args = (struct networkThreadArg *) arg;

    // Setup the PF_RING.
    pfring *pd;
    pd = pfring_open(args->ip_address, UDP_PACKETSIZE, PF_RING_PROMISC );

    if(pd == NULL) {
        ERROR("pfring_open error [%s] (pf_ring not loaded or quick mode is enabled you and have already a socket bound to %s?)\n",
            strerror(errno), args->ip_address);
        exit(EXIT_FAILURE);
    }

    pfring_set_application_name(pd, "gpu_correlator");

    if (pd->dna.dna_mapped_device == 0) {
        ERROR("The device is not in DNA mode.?");
    }

    pfring_set_poll_duration(pd, 1000);
    pfring_set_poll_watermark(pd, 10000);

    if (pfring_enable_ring(pd) != 0) {
        ERROR("Cannot enable the PF_RING.");
        exit(EXIT_FAILURE);
    }

    uint64_t count = 0;
    double last_time = e_time();
    double current_time = e_time();
    int64_t seq = 0;
    int64_t last_seq = -1;
    int64_t diff = 0;
    int64_t total_lost = 0;
    int64_t grand_total_lost = 0;
    int64_t lost = 0;
    int64_t total_out_of_order = 0;
    int64_t total_duplicate = 0;
    long long int total_packets = 0;

    int buffer_location = 0;
    int buffer_id = args->link_id;
    int data_id = 0;
    int total_buffers_filled = 0;

    int64_t out_of_order_event = 0;

    struct ErrorMatrix * error_matrix = NULL;

    // Make sure the first buffer is ready to go. (this check is not really needed)
    wait_for_empty_buffer(args->buf, buffer_id);

    set_data_ID(args->buf, buffer_id, data_id++);
    error_matrix = get_error_matrix(args->buf, buffer_id);

    double start_time = -1;

    for (;;) {

        assert (buffer_location <= args->buf->buffer_size);

        // See if we need to get a new buffer
        if (buffer_location == args->buf->buffer_size) {

            // Notify the we have filled a buffer.
            mark_buffer_full(args->buf, buffer_id);
            pthread_yield();

            // Check if we should stop collecting data.
            check_if_done(&total_buffers_filled, args, total_packets, 
                        grand_total_lost, total_out_of_order, total_duplicate, start_time);

            buffer_id = (buffer_id + args->num_links) % (args->buf->num_buffers);

            // This call will block if the buffer has not been written out to disk yet.
            wait_for_empty_buffer(args->buf, buffer_id);

            buffer_location = 0;

            set_data_ID(args->buf, buffer_id, data_id++);
            error_matrix = get_error_matrix(args->buf, buffer_id);
            if (last_seq != -1) {
                // If not the first packet we need to set BufferInfo data.
                set_fpga_seq_num(args->buf, buffer_id, last_seq + 1);
                // TODO This is close, but not perfect timing - but this shouldn't really matter.
                static struct timeval now;
                gettimeofday(&now, NULL);
                set_first_packet_recv_time(args->buf, buffer_id, now);
            }
        }

        struct pfring_pkthdr pf_header;
        u_char *pkt_buf;
        int rc = pfring_recv(pd, &pkt_buf, 0, &pf_header, 1);
        if (rc <= 0) {
            // No packets available.
            if (rc < 0) { fprintf(stderr,"Error in pfring_recv! %d\n", rc); }
            pthread_yield();
            continue;
        }

        if (pf_header.len != UDP_PACKETSIZE) {
            fprintf(stderr,"Got wrong sized packet with len: %d", pf_header.len);
            continue;
        }

        // Do seq number related stuff (location will change.)
        seq = ((((uint32_t *) &pkt_buf[54])[0]) + 0 ) >> 2;
        //INFO("seq: %u", seq);

        // First packet alignment code.
        if (last_seq == -1) {

            if ( !( (seq % SEQ_NUM_EDGE) <= 10 && (seq % SEQ_NUM_EDGE) >= 0 ) ) {
                continue;
            }

            INFO("Got first packet %" PRId64, seq);
            // Set the time we got the first packet.
            static struct timeval now;
            gettimeofday(&now, NULL);
            set_first_packet_recv_time(args->buf, buffer_id, now);
            set_fpga_seq_num(args->buf, buffer_id, seq - seq % SEQ_NUM_EDGE);

            // Time for internal counters.
            start_time = e_time();

            // TODO This is only correct with high probability,
            // this should be made deterministic. 
            if (seq % SEQ_NUM_EDGE ==  0) {
                last_seq = seq;
                memcpy(&args->buf->data[buffer_id][buffer_location], pkt_buf + 58, UDP_PAYLOADSIZE);
                count++;
                buffer_location += UDP_PAYLOADSIZE;
            } else {
                // If we have lost the packet on the edge,
                // we set the last_seq to the edge so that the buffer will still be aligned.
                // We also ignore the current packet, and just allow it to be lost for simplicity.
                last_seq = seq - seq % SEQ_NUM_EDGE;
            }
            continue;
        }


        memcpy(&args->buf->data[buffer_id][buffer_location], pkt_buf + 58, UDP_PAYLOADSIZE);

        //INFO("seq_num: %d", seq);

        // If this is the first packet, we don't need to check the later cases,
        // just move the buffer location, and continue.
        if (last_seq == -1) {
            last_seq = seq;
            count++;
            buffer_location += UDP_PAYLOADSIZE;
            continue;
        }

        if (seq == last_seq) {
            total_duplicate++;
            // Do nothing in this case, because if the buffer_location doesn't change, 
            // we over write this duplicate with the next packet.
            // We continue since we don't count this as a reciveved packet.
            continue;
        }

        if ( (seq < last_seq && last_seq - seq > 1ll << (COUNTER_BITS - 1ll) ) 
                    || (seq > last_seq && seq - last_seq < 1ll << (COUNTER_BITS - 1ll) ) ) {
            // See RFC 1982 for above statement details. 
            // Result: seq follows last_seq if above is true.
            // This is the most common case.

            // Compute the true distance between seq numbers, and packet loss (if any).
            diff = seq - last_seq;
            if (diff < 0) {
                diff += COUNTER_MAX + 1ll;
            }

            lost = diff - 1ll;
            total_lost += lost;

            // We have packet loss, we have two cases.
            // Case 1:  There is room in the buffer to move the data we put in the
            // wrong place.  So we move our data, and zero out the missing values.
            // Case 2:  We ended up in a new buffer...  So we need to zero the value
            // we recorded, and zero the values in the next buffer(s), upto the point we want to
            // start writing new data.  Note in this case we zero out even the last packet that we
            // read.  So losses over a buffer edge result in one extra "lost" packet. 
            if ( lost > 0 ) {

                if (lost > 1000000) {
                    ERROR("Packet loss is very high! lost packets: %lld\n", (long long int)lost);
                }

                // The location the packet should have been in.
                int realPacketLocation = buffer_location + lost * UDP_PAYLOADSIZE;
                if ( realPacketLocation < args->buf->buffer_size ) { // Case 1:
                    // Copy the memory in the right location.
                    assert(buffer_id < args->buf->num_buffers);
                    assert(realPacketLocation <= args->buf->buffer_size - UDP_PAYLOADSIZE);
                    assert(buffer_location <= args->buf->buffer_size - UDP_PAYLOADSIZE);
                    assert(buffer_id >= 0);
                    assert(buffer_location >= 0);
                    assert(realPacketLocation >= 0);
                    memcpy((void *) &args->buf->data[buffer_id][realPacketLocation], 
                            (void *) &args->buf->data[buffer_id][buffer_location], UDP_PAYLOADSIZE);

                    // Zero out the lost part of the buffer.
                    for (int i = 0; i < lost; ++i) {
                        memset(&args->buf->data[buffer_id][buffer_location + i*UDP_PAYLOADSIZE], 0x88, UDP_PAYLOADSIZE);
                    }
                    add_bad_timesamples(error_matrix, lost * NUM_TIMESAMPLES_PER_PACKET);

                    buffer_location = realPacketLocation + UDP_PAYLOADSIZE;

                    //ERROR("Case 1 packet loss event on data_id: %d", data_id);
                } else { // Case 2 (the hard one):

                    // zero out the rest of the current buffer and mark it as full.
                    int i, j;
                    for (i = 0; (buffer_location + i*UDP_PAYLOADSIZE) < args->buf->buffer_size; ++i) {
                        memset(&args->buf->data[buffer_id][buffer_location + i*UDP_PAYLOADSIZE], 0x88, UDP_PAYLOADSIZE);
                        add_bad_timesamples(error_matrix, NUM_TIMESAMPLES_PER_PACKET);
                    }

                    // Keep track of the last edge seq number in case we need it later.
                    uint32_t last_edge = get_fpga_seq_num(args->buf, buffer_id);

                    // Notify the we have filled a buffer.
                    mark_buffer_full(args->buf, buffer_id);

                    // Check if we should stop collecting data.
                    check_if_done(&total_buffers_filled, args, total_packets, 
                                grand_total_lost, total_out_of_order, total_duplicate, start_time);

                    // Get the number of lost packets in the new buffer(s).
                    int num_lost_packets_new_buf = (lost+1) - (args->buf->buffer_size - buffer_location)/UDP_PAYLOADSIZE;

                    assert(num_lost_packets_new_buf > 0);

                    i = 0;
                    j = 1;

                    // We may have lost more packets than will fit in a buffer.
                    do {

                        // Get a new buffer.
                        buffer_id = (buffer_id + args->num_links) % (args->buf->num_buffers);

                        // This call will block if the buffer has not been written out to disk yet
                        // shouldn't be an issue if everything runs correctly.
                        wait_for_empty_buffer(args->buf, buffer_id);

                        set_data_ID(args->buf, buffer_id, data_id++);
                        error_matrix = get_error_matrix(args->buf, buffer_id);

                        uint32_t fpga_seq_number = last_edge + j * (args->buf->buffer_size/UDP_PAYLOADSIZE); // == number of iterations FIXME.

                        set_fpga_seq_num(args->buf, buffer_id, fpga_seq_number);

                        // This really isn't the correct time, but this is the best we can do here.
                        struct timeval now;
                        gettimeofday(&now, NULL);
                        set_first_packet_recv_time(args->buf, buffer_id, now);

                        for (i = 0; num_lost_packets_new_buf > 0 && (i*UDP_PAYLOADSIZE) < args->buf->buffer_size; ++i) {
                            memset(&args->buf->data[buffer_id][i*UDP_PAYLOADSIZE], 0x88, UDP_PAYLOADSIZE);
                            add_bad_timesamples(error_matrix, NUM_TIMESAMPLES_PER_PACKET);
                            num_lost_packets_new_buf--;
                        }

                        // Check if we need to run another iteration of the loop.
                        // i.e. get another buffer.
                        if (num_lost_packets_new_buf > 0) {

                            // Notify the we have filled a buffer.
                            mark_buffer_full(args->buf, buffer_id);
                            pthread_yield();

                            // Check if we should stop collecting data.
                            check_if_done(&total_buffers_filled, args, total_packets, 
                                        grand_total_lost, total_out_of_order, total_duplicate, start_time);

                        }
                        j++;

                    } while (num_lost_packets_new_buf > 0);

                    // Update the new buffer location.
                    buffer_location = i*UDP_PAYLOADSIZE;

                    // We need to increase the total number of lost packets since we 
                    // just tossed away the packet we read.
                    total_lost++;
                    printf("Case 2 packet loss event on data_id: %d\n", data_id);

                }
            } else {
                // This is the normal case; a valid packet. 
                buffer_location += UDP_PAYLOADSIZE;
            }

        } else {
            // seq is before last_seq.  We have an out of order packet.

            total_out_of_order++;

            if (out_of_order_event == 0 || out_of_order_event != last_seq) {
                ERROR("Out of order event in data_id: %d\n", data_id);
                out_of_order_event = last_seq;
            }

            // In this case, we could write the out of order packet into the right location,
            // upto buffer edge issues. 
            // However at the moment, we are just ignoring it, since that location will already
            // have been zeroed out as a lost packet, or written if this is a late duplicate.
            // We don't advance the buffer location, so that this location is overwritten.

            // Continue so we don't update last_seq, or count.
            continue;
        }

        last_seq = seq;

        // Compute speed at packet loss every X packets
        count++;
        total_packets++;
        static const int X = 10*39062;
        if (count % (X+1) == 0) {
            current_time = e_time();
            INFO("Receive Speed: %1.3f Gbps %.0f pps\n", (((double)X*UDP_PACKETSIZE*8) / (current_time - last_time)) / (1024*1024*1024), X / (current_time - last_time) );
            last_time = current_time;
            if (total_lost != 0) {
                INFO("Packet loss: %.6f%%\n", ((double)total_lost/(double)X)*100); 
            } else {
                INFO("Packet loss: %.6f%%\n", (double)0.0);
            }
            grand_total_lost += total_lost;
            total_lost = 0;

            INFO("Data received: %.2f GB -- ", ((double)total_buffers_filled * ((double)args->buf->buffer_size / (1024.0*1024.0)))/1024.0);
            INFO("Number of full buffers: %d\n", get_num_full_buffers(args->buf));
        }
    }
}

void test_network_thread(void * arg) {
    struct networkThreadArg * args;
    args = (struct networkThreadArg *) arg;
    int buffer_id = args->link_id;
    int data_id = 0;

    // Make sure the first buffer is ready to go. (this check is not really needed)
    wait_for_empty_buffer(args->buf, buffer_id);

    set_data_ID(args->buf, buffer_id, data_id++);

    generate_char_data_set(GENERATE_DATASET_CONSTANT,
                           GEN_DEFAULT_SEED,
                           GEN_DEFAULT_RE,
                           GEN_DEFAULT_IM,
                           GEN_INITIAL_RE,
                           GEN_INITIAL_IM,
                           GEN_FREQ,
                           args->num_timesamples,
                           args->actual_num_freq, 
                           args->actual_num_elements, 
                           (unsigned char *) args->buf->data[buffer_id]);

    for (int i = 0; i < 100; i++) {
        INFO("Thread ID=%d; buf[%d]=0x%X", args->link_id, i, *(unsigned int *)(&args->buf->data[buffer_id][i*4]) );
    }

    mark_buffer_full(args->buf, buffer_id);

    mark_producer_done(args->buf, args->link_id);
    int ret = 0;
    pthread_exit((void *) &ret);

}
