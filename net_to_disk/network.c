
#include "network.h"
#include "buffers.h"
#include "pfring.h"

#include <sys/epoll.h>
#include <sys/socket.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <memory.h>
#include <unistd.h>
#include <assert.h>

#define UDP_PACKETSIZE 9296
#define SEQ_NUM_EDGE 100000

// Count max
#define COUNTER_BITS 30
#define COUNTER_MAX (1ll << COUNTER_BITS) - 1ll

double e_time(void){
    static struct timeval now;
    gettimeofday(&now, NULL);
    return (double)(now.tv_sec  + now.tv_usec/1000000.0);
}

void check_if_done(int * total_buffers_filled, struct network_thread_arg * args,
                    long long int total_packets, int32_t total_lost, int32_t total_out_of_order,
                   int32_t total_duplicate, double start_time) {
        (*total_buffers_filled)++;
    if ( (*total_buffers_filled) * (args->buf->buffer_size / (1024*1024)) >= args->data_limit * 1024) {
        double end_time = e_time();
        printf("Stopping packet capture, ran for ~ %f seconds.\n", end_time - start_time);
        printf("\nStats:\nTotal Packets Captured: %lld\nPackets lost: %d\nOut of order packets: %d\nDuplicate Packets: %d\n",
                total_packets, total_lost, total_out_of_order, total_duplicate);
        printf("Writing remaining buffers to disk...\n");
        markProducerDone(args->buf);
        int ret;
        pthread_exit((void *) &ret);
    }
}
/*
// num_freq is global number of freq, default 1024
void copyFrame(char * output_buf, char * packet_buf, int link_id,
                int freq_offset, int num_freq ) {

    const int packet_offset = 58;

    if (link_id == 0) {
        // Copy header
        memcpy(output_buf, packet_buf, packet_offset);
    }

    const int frame_size = 2 * num_freq;

    for (int frame = 0; frame < 4; ++frame) {
        for (int i = freq_offset; i < (num_freq / 8); ++i) {
            output_buf[packet_offset + frame*frame_size + i*16 + link_id*2]  = packet_buf[packet_offset + frame*2048 + frame*16];
            output_buf[packet_offset + frame*frame_size + i*16 + link_id*2 + 1]  = packet_buf[packet_offset + frame*2048 + frame*16 + 1];
        }
    }
}*/

void copyFrame(char * output_buf, char * packet_buf, int link_id, struct timeval * time,
                int num_frames, int num_inputs, int num_freq, int offset) {

    const int header_offset = 58;

    if (link_id == 0) {
        // Copy header
        memcpy(output_buf, packet_buf, header_offset);

        *(uint32_t*)(output_buf + 30) = num_frames; // num_frames = 4
        *(uint32_t*)(output_buf + 34) = num_inputs; // num_inputs = 2
        *(uint32_t*)(output_buf + 38) = num_freq; // num_freq = 1024
        *(uint32_t*)(output_buf + 42) = offset; // offset
        *(uint32_t*)(output_buf + 46) = (uint32_t)time->tv_sec; // seconds
        *(uint32_t*)(output_buf + 50) = (uint32_t)time->tv_usec; // microseconds
    }

    int frame_size = num_inputs * num_freq;

    for (int frame = 0; frame < num_frames; ++frame) {
        int output_row = 0;
        for (int row = offset/8; row < (offset/8 + num_freq/8); ++row ) {
            for (int i = 0; i < num_inputs; ++i) {
                output_buf[header_offset + frame*frame_size + output_row*num_inputs*8 + link_id*num_inputs + i] =
                    packet_buf[header_offset + frame*2048 + row*16 + i];
            }
            output_row++;
        }
    }
/*
    for (int i = 0; i < 512; ++i) {
        output_buf[header_offset + i*16 + link_id*2]  = packet_buf[header_offset + i*16];
        output_buf[header_offset + i*16 + link_id*2 + 1]  = packet_buf[header_offset + i*16 + 1];
    }*/
}

void network_thread(void * arg) {

    struct network_thread_arg * args;
    args = (struct network_thread_arg *) arg;

    // Setup the PF_RING.
    pfring *pd;
    pd = pfring_open(args->interface, UDP_PACKETSIZE, PF_RING_PROMISC );

    if(pd == NULL) {
        printf("pfring_open error [%s] (pf_ring not loaded or quick mode is enabled you and have already a socket bound to %s?)\n",
              strerror(errno), args->interface);
        exit(EXIT_FAILURE);
    }

    pfring_set_application_name(pd, "gpu_correlator");

    if (pd->dna.dna_mapped_device == 0) {
        printf("The device is not in DNA mode.?");
    }

    pfring_set_poll_duration(pd, 1000);
    pfring_set_poll_watermark(pd, 10000);

    if (pfring_enable_ring(pd) != 0) {
        printf("Cannot enable the PF_RING.");
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
    struct timeval first_packet_time;

    int buffer_location = 0;
    int buffer_id = 0;
    int data_id = 0;
    int total_buffers_filled = 0;

    int out_of_order_event = 0;

    int output_packet_size = args->num_frames * args->num_inputs * args->num_freq + 58;

    // Make sure the first buffer is ready to go. (this check is not really needed)
    waitForEmptyBuffer(args->buf, buffer_id);

    setDataID(args->buf, buffer_id, data_id++);

    double start_time = -1;

    for (;;) {

        assert (buffer_location <= args->buf->buffer_size);

        // See if we need to get a new buffer
        if (buffer_location == args->buf->buffer_size) {

            // Notify the we have filled a buffer.
            markBufferFull_nProducers(args->buf, buffer_id);

            // Check if we should stop collecting data.
            check_if_done(&total_buffers_filled, args, total_packets,
                            grand_total_lost, total_out_of_order, total_duplicate, start_time);

            buffer_id = (buffer_id + 1) % (args->buf->num_buffers);

            // This call will block if the buffer has not been written out to disk yet.
            waitForEmptyBuffer(args->buf, buffer_id);

            gettimeofday(&first_packet_time, NULL);

            buffer_location = 0;

            setDataID(args->buf, buffer_id, data_id++);
        }

        struct pfring_pkthdr pf_header;
        u_char *pkt_buf;
        int rc = pfring_recv(pd, &pkt_buf, 0, &pf_header, 1);
        if (rc <= 0) {
            // No packets available.
            if (rc < 0) { fprintf(stderr,"Error in pfring_recv! %d\n", rc); }
            continue;
        }

        if (pf_header.len != UDP_PACKETSIZE) {
            fprintf(stderr,"Got wrong sized packet with len: %d", pf_header.len);
            continue;
        }

        // Do seq number related stuff (location will change.)
        seq = ((((uint32_t *) &pkt_buf[54])[0]) + 0 ) >> 2;
        //INFO("seq: %u", seq);

        // Set the time we got the first packet.
        if (start_time == -1) {
            start_time = e_time();
        }

        // If this is the first packet, we don't need to check the later cases,
        // just move the buffer location, and continue.
        if ( last_seq == -1 ) {
            if ( !( (seq % SEQ_NUM_EDGE) <= 10 && (seq % SEQ_NUM_EDGE) >= 0 ) ) {
                continue;
            }

            printf("Got first packet %d", seq);
            // Set the time we got the first packet.
            gettimeofday(&first_packet_time, NULL);
            //set_first_packet_recv_time(args->buf, buffer_id, now);
            //set_fpga_seq_num(args->buf, buffer_id, seq - seq % SEQ_NUM_EDGE);

            // Time for internal counters.
            start_time = e_time();

            // TODO This is only correct with high probability,
            // this should be made deterministic.
            if (seq % SEQ_NUM_EDGE == 0) {
                last_seq = seq;
                copyFrame(&args->buf->data[buffer_id][buffer_location], pkt_buf, args->link_id, &first_packet_time,
                        args->num_frames, args->num_inputs, args->num_freq, args->offset);
                count++;
                buffer_location += output_packet_size;
            } else {
                // If we have lost the packet on the edge,
                // we set the last_seq to the edge so that the buffer will still be aligned.
                // We also ignore the current packet, and just allow it to be lost for simplicity.
                last_seq = seq - seq % SEQ_NUM_EDGE;
            }
            continue;
        }

        copyFrame(&args->buf->data[buffer_id][buffer_location], pkt_buf, args->link_id, &first_packet_time,
                  args->num_frames, args->num_inputs, args->num_freq, args->offset);

        if (seq == last_seq) {
            total_duplicate++;
            // Do nothing in this case, because if the buffer_location doesn't change,
            // we over write this duplicate with the next packet.
            // We continue since we don't count this as a reciveved packet.
            continue;
        }

        if ( (seq < last_seq && last_seq - seq > 1 << (COUNTER_BITS - 1) )
                    || (seq > last_seq && seq - last_seq < 1 << (COUNTER_BITS - 1) ) ) {
            // See RFC 1982 for above statement details.
            // Result: seq follows last_seq if above is true.
            // This is the most common case.


            // Compute the true distance between seq numbers, and packet loss (if any).
            diff = seq - last_seq;
            if (diff < 0) {
                diff += COUNTER_MAX + 1;
            }

            lost = diff - 1;
            total_lost += lost;

            // We have packet loss, we have two cases.
            // Case 1:  There is room in the buffer to move the data we put in the
            // wrong place.  So we move our data, and zero out the missing values.
            // Case 2:  We ended up in a new buffer...  So we need to zero the value
            // we recorded, and zero the values in the next buffer(s), upto the point we want to
            // start writing new data.  Note in this case we zero out even the last packet that we
            // read.  So losses over a buffer edge result in one extra "lost" packet.
            if ( lost > 0 ) {

                // The location the packet should have been in.
                int realPacketLocation = buffer_location + lost * output_packet_size;
                if ( realPacketLocation < args->buf->buffer_size ) { // Case 1:
                    // Copy the memory in the right location.
                    copyFrame((void *) &args->buf->data[buffer_id][realPacketLocation],
                            (void *) &args->buf->data[buffer_id][buffer_location], args->link_id,
                              &first_packet_time,
                              args->num_frames, args->num_inputs, args->num_freq, args->offset);

                    // Zero out the lost part of the buffer.
                    for (int i = 0; i < lost; ++i) {
                        memset(&args->buf->data[buffer_id][buffer_location + i*output_packet_size], 0, output_packet_size);
                    }

                    buffer_location = realPacketLocation + output_packet_size;

                    //printf("Case 1 packet loss event on data_id: %d", data_id);
                } else { // Case 2 (the hard one):

                    // zero out the rest of the current buffer and mark it as full.
                    for (int i = 0; (buffer_location + i*output_packet_size) < args->buf->buffer_size; ++i) {
                        memset(&args->buf->data[buffer_id][buffer_location + i*output_packet_size], 0, output_packet_size);
                    }

                    // Notify the we have filled a buffer.
                    markBufferFull_nProducers(args->buf, buffer_id);

                    // Check if we should stop collecting data.
                    check_if_done(&total_buffers_filled, args, total_packets,
                                grand_total_lost, total_out_of_order, total_duplicate, start_time);

                    // Get the number of lost packets in the new buffer(s).
                    int num_lost_packets_new_buf = lost - (args->buf->buffer_size - buffer_location)/output_packet_size;

                    assert(num_lost_packets_new_buf > 0);

                    int i = 0;

                    // We may have lost more packets than will fit in a buffer.
                    do {

                        // Get a new buffer.
                        buffer_id = (buffer_id + 1) % (args->buf->num_buffers);

                        // This call will block if the buffer has not been written out to disk yet
                        // shouldn't be an issue if everything runs correctly.
                        waitForEmptyBuffer(args->buf, buffer_id);
                        gettimeofday(&first_packet_time, NULL);

                        setDataID(args->buf, buffer_id, data_id++);

                        for (i = 0; num_lost_packets_new_buf >= 0 && (i*output_packet_size) < args->buf->buffer_size; ++i) {
                            memset(&args->buf->data[buffer_id][i*output_packet_size], 0, output_packet_size);
                            num_lost_packets_new_buf--;
                        }

                        // Check if we need to run another iteration of the loop.
                        // i.e. get another buffer.
                        if (num_lost_packets_new_buf > 0) {

                            // Notify the we have filled a buffer.
                            markBufferFull_nProducers(args->buf, buffer_id);

                            // Check if we should stop collecting data.
                            check_if_done(&total_buffers_filled, args, total_packets,
                                        grand_total_lost, total_out_of_order, total_duplicate, start_time);

                        }

                    } while (num_lost_packets_new_buf > 0);

                    // Update the new buffer location.
                    buffer_location = i*output_packet_size;

                    // We need to increase the total number of lost packets since we
                    // just tossed away the packet we read.
                    total_lost++;
                    printf("Case 2 packet loss event on data_id: %d", data_id);

                }
            } else {
                // This is the normal case; a valid packet.
                buffer_location += output_packet_size;
            }

        } else {
            // seq is before last_seq.  We have an out of order packet.

            total_out_of_order++;

            if (out_of_order_event == 0 || out_of_order_event != last_seq) {
                printf("Out of order event in data_id: %d", data_id);
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
        static const int X = 4*64*1024;
        if (count % (X+1) == 0) {
            current_time = e_time();
            printf("Receive Speed: %.0f Mbps %.0f pps\n", (((double)X*UDP_PACKETSIZE*8) / (current_time - last_time)) / (1024*1024), X / (current_time - last_time) );
            last_time = current_time;
            if (total_lost != 0) {
                printf("Packet loss: %.6f%%\n", ((double)total_lost/(double)X)*100);
            } else {
                printf("Packet loss: %.6f%%\n", (double)0.0);
            }
            grand_total_lost += total_lost;
            total_lost = 0;

            printf("Data recieved: %.2f GB -- ", ((double)total_buffers_filled * ((double)args->buf->buffer_size / (1024.0*1024.0)))/1024.0);
            printf("Number of full buffers: %d\n", getNumFullBuffers(args->buf));
        }
    }
}