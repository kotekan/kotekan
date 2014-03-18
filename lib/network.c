
#include "network.h"
#include "buffers.h"
#include "errors.h"

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

#define MAX_EVENTS 100

#define UDP_PACKETSIZE 8192

// Count max
#define COUNTER_BITS 32
#define COUNTER_MAX (1ll << COUNTER_BITS) - 1ll

double e_time(void) {
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
        INFO("Stopping packet capture, ran for ~ %f seconds.\n", end_time - start_time);
        INFO("\nStats:\nTotal Packets Captured: %lld\nPackets lost: %d\nOut of order packets: %d\nDuplicate Packets: %d\n", 
                total_packets, total_lost, total_out_of_order, total_duplicate);
        mark_producer_done(args->buf, args->link_id);
        int ret = 0;
        pthread_exit((void *) &ret);
    }
}

void network_thread(void * arg) {

    struct networkThreadArg * args;
    args = (struct networkThreadArg *) arg;

    struct epoll_event ev;
    struct epoll_event events[MAX_EVENTS];
    int epoll_fd;
    int num_events;

    // Create the EPOLL file handle
    epoll_fd = epoll_create(10);
    if (epoll_fd == -1) {
        ERROR("epoll_create; errno %d", errno);
        exit(EXIT_FAILURE);
    }

    // Setup the UDP server.

    int udp_socket;
    struct sockaddr_in server_address;

    udp_socket = socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK , IPPROTO_UDP);

    memset(&server_address, 0, sizeof(server_address));
    server_address.sin_family = AF_INET;
    inet_pton(AF_INET, args->ip_address, &(server_address.sin_addr));
    server_address.sin_port=htons(args->port_number);
    if ( bind(udp_socket,(struct sockaddr *)&server_address,sizeof(server_address)) == -1) {
        ERROR("Socket bind; errno %d", errno);
    }

    int n = 256 * 1024 * 1024;
    if (setsockopt(udp_socket, SOL_SOCKET, SO_RCVBUF, (void *) &n, sizeof(n)) == -1) {
        ERROR("Error in set socket rcvbuf size; errno %d", errno);
    }

    ev.events = EPOLLIN;
    ev.data.fd = udp_socket;
    if (epoll_ctl(epoll_fd, EPOLL_CTL_ADD, udp_socket, &ev) == -1) {
        ERROR("epoll_ctl: listen_sock; errno %d", errno);
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

    // Make sure the first buffer is ready to go. (this check is not really needed)
    wait_for_empty_buffer(args->buf, buffer_id);

    set_data_ID(args->buf, buffer_id, data_id++);

    double start_time = -1;

    for (;;) {

        num_events = epoll_wait(epoll_fd, events, MAX_EVENTS, -1);
        if (num_events == -1) {
            ERROR("epoll_pwait; errno %d", errno);
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < num_events; i++) {

            if (events[i].data.fd != udp_socket) {
                continue;
            }

            assert (buffer_location <= args->buf->buffer_size);

            // See if we need to get a new buffer
            if (buffer_location == args->buf->buffer_size) {

                // Notify the we have filled a buffer.
                assert(args->buf->info[buffer_id] != NULL);
                mark_buffer_full(args->buf, buffer_id);

                // Check if we should stop collecting data.
                check_if_done(&total_buffers_filled, args, total_packets, 
                              grand_total_lost, total_out_of_order, total_duplicate, start_time);

                buffer_id = (buffer_id + args->num_links) % (args->buf->num_buffers);

                // This call will block if the buffer has not been written out to disk yet.
                wait_for_empty_buffer(args->buf, buffer_id);

                buffer_location = 0;

                set_data_ID(args->buf, buffer_id, data_id++);
            }

            // TODO This will need to be changed once we start getting packets with header information.
            // This changes to readv for reading the header into a seperate memory space. 
            ssize_t bytes_read = read(udp_socket, (void*) &args->buf->data[buffer_id][buffer_location], UDP_PACKETSIZE);

            if (bytes_read == -1) {
                ERROR("Failed to read data; bytes_read: %d; Buffer Id: %d; Buffer location: %d; Errno: %d", (int)bytes_read, buffer_id, buffer_location, errno);
                exit(errno);
            }

            if (bytes_read != UDP_PACKETSIZE) {
                ERROR("Bad packet. Panic, Abort, etc. This should never happen.\n");
                exit(EXIT_FAILURE);
            }

            // Set the time we got the first packet.
            if (start_time == -1) {
                start_time = e_time();
            }

            // Do seq number related stuff (location will change.)
            seq = ntohl(((uint32_t *) &args->buf->data[buffer_id][buffer_location] )[0]);

            // If this is the first packet, we don't need to check the later cases,
            // just move the buffer location, and continue.
            if (last_seq == -1) {
                last_seq = seq;
                count++;
                buffer_location += UDP_PACKETSIZE;
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
                        ERROR("Packet loss is very high! lost packets: %lld", (long long int)lost);
                    }

                    // The location the packet should have been in.
                    int realPacketLocation = buffer_location + lost * UDP_PACKETSIZE;
                    if ( realPacketLocation < args->buf->buffer_size ) { // Case 1:
                        // Copy the memory in the right location.
                        assert(buffer_id < args->buf->num_buffers);
                        assert(realPacketLocation <= args->buf->buffer_size - UDP_PACKETSIZE);
                        assert(buffer_location <= args->buf->buffer_size - UDP_PACKETSIZE );
                        assert(buffer_id >= 0);
                        assert(buffer_location >= 0);
                        assert(realPacketLocation >= 0);
                        memcpy((void *) &args->buf->data[buffer_id][realPacketLocation], 
                                (void *) &args->buf->data[buffer_id][buffer_location], UDP_PACKETSIZE);

                        // Zero out the lost part of the buffer.
                        for (int i = 0; i < lost; ++i) {
                            memset(&args->buf->data[buffer_id][buffer_location + i*UDP_PACKETSIZE], 0, UDP_PACKETSIZE);
                        }

                        buffer_location = realPacketLocation + UDP_PACKETSIZE;

                    } else { // Case 2 (the hard one):

                        // zero out the rest of the current buffer and mark it as full.
                        for (int i = 0; (buffer_location + i*UDP_PACKETSIZE) < args->buf->buffer_size; ++i) {
                            memset(&args->buf->data[buffer_id][buffer_location + i*UDP_PACKETSIZE], 0, UDP_PACKETSIZE);
                        }

                        // Notify the we have filled a buffer.
                        assert(args->buf->info[buffer_id] != NULL);
                        mark_buffer_full(args->buf, buffer_id);

                        // Check if we should stop collecting data.
                        check_if_done(&total_buffers_filled, args, total_packets, 
                                    grand_total_lost, total_out_of_order, total_duplicate, start_time);

                        // Get the number of lost packets in the new buffer(s).
                        int num_lost_packets_new_buf = (lost+1) - (args->buf->buffer_size - buffer_location)/UDP_PACKETSIZE;

                        assert(num_lost_packets_new_buf > 0);

                        int i = 0;

                        // We may have lost more packets than will fit in a buffer.
                        do {

                            // Get a new buffer.
                            buffer_id = (buffer_id + args->num_links) % (args->buf->num_buffers);

                            // This call will block if the buffer has not been written out to disk yet
                            // shouldn't be an issue if everything runs correctly.
                            wait_for_empty_buffer(args->buf, buffer_id);

                            set_data_ID(args->buf, buffer_id, data_id++);

                            for (i = 0; num_lost_packets_new_buf > 0 && (i*UDP_PACKETSIZE) < args->buf->buffer_size; ++i) {
                                memset(&args->buf->data[buffer_id][i*UDP_PACKETSIZE], 0, UDP_PACKETSIZE);
                                num_lost_packets_new_buf--;
                            }

                            // Check if we need to run another iteration of the loop.
                            // i.e. get another buffer.
                            if (num_lost_packets_new_buf > 0) {

                                // Notify the we have filled a buffer.
                                assert(args->buf->info[buffer_id] != NULL);
                                mark_buffer_full(args->buf, buffer_id);

                                // Check if we should stop collecting data.
                                check_if_done(&total_buffers_filled, args, total_packets, 
                                            grand_total_lost, total_out_of_order, total_duplicate, start_time);

                            }

                        } while (num_lost_packets_new_buf > 0);

                        // Update the new buffer location.
                        buffer_location = i*UDP_PACKETSIZE;

                        // We need to increase the total number of lost packets since we 
                        // just tossed away the packet we read.
                        total_lost++;
                        DEBUG("Case 2 packet loss event on data_id: %d", data_id);

                    }
                } else {
                    // This is the normal case; a valid packet. 
                    buffer_location += UDP_PACKETSIZE;
                }

            } else {
                // seq is before last_seq.  We have an out of order packet.

                total_out_of_order++;

                if (out_of_order_event == 0 || out_of_order_event != last_seq) {
                    DEBUG("Out of order event in data_id: %d\n", data_id);
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
            static const int X = 4*128*1024;
            if (count % (X+1) == 0) {
                current_time = e_time();
                INFO("Receive Speed: %.0f Mbps %.0f pps\n", (((double)X*UDP_PACKETSIZE*8) / (current_time - last_time)) / (1024*1024), X / (current_time - last_time) );
                last_time = current_time;
                if (total_lost != 0) {
                    INFO("Packet loss: %.6f%%\n", ((double)total_lost/(double)X)*100); 
                } else {
                    INFO("Packet loss: %.6f%%\n", (double)0.0);
                }
                grand_total_lost += total_lost;
                total_lost = 0;

                INFO("Data received: %.2f GB -- ", ((double)total_buffers_filled * ((double)args->buf->buffer_size / (1024.0*1024.0)))/1024.0);
                INFO("Number of full buffers: %d\n", getNumFullBuffers(args->buf));
            }
        }
    }
}