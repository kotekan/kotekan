#include "network_dna.h"
#include "buffers.h"
#include "pfring.h"
#include "errors.h"
#include "test_data_generation.h"
#include "error_correction.h"
#include "nt_memcpy.h"
#include "config.h"

#include <dirent.h>
#include <sys/socket.h>
#include <stdio.h>
#include <errno.h>
#include <stdlib.h>
#include <arpa/inet.h>
#include <sys/time.h>
#include <sys/stat.h>
#include <memory.h>
#include <unistd.h>
#include <assert.h>
#include <pthread.h>
#include <inttypes.h>

// Count max
#define COUNTER_BITS 30
#define COUNTER_MAX (1ll << COUNTER_BITS) - 1ll

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
    if (args->config->processing.data_limit == 0) {
        return;
    }

    if ( (*total_buffers_filled) * (args->buf->buffer_size / (1024*1024)) >= args->config->processing.data_limit * 1024) {
        double end_time = e_time();
        INFO("Stopping packet capture, ran for ~ %f seconds.\n", end_time - start_time);
        INFO("\nStats:\nTotal Packets Captured: %lld\nPackets lost: %d\nOut of order packets: %d\nDuplicate Packets: %d\n",
                total_packets, total_lost, total_out_of_order, total_duplicate);
        mark_producer_done(args->buf, args->link_id);
        int ret = 0;
        pthread_exit((void *) &ret);
    }
}

FILE * open_next_file(const char * file_base_dir, struct dirent * file_info) {
    char full_name[256];
    sprintf(full_name, "%s/%s", file_base_dir, file_info->d_name);

    FILE * data_file = fopen(full_name, "rp");
    if (!data_file) {
        ERROR("Cannot open file %s, errno: %d", full_name, errno);
        return NULL;
    }
    INFO("Opened file %s", full_name);
    return data_file;
}

int file_select(const struct dirent *entry) {
    return strstr(entry->d_name, ".dat") || strstr(entry->d_name, ".pkt");
}

void network_thread(void * arg) {

    struct networkThreadArg * args;
    args = (struct networkThreadArg *) arg;

    struct Config * config = args->config;
    const int udp_payload_size = config->fpga_network.udp_frame_size *
        config->fpga_network.timesamples_per_packet;

    uint64_t count = 0;
    double last_time = e_time();
    double current_time = e_time();
    int64_t seq = 0;
    int64_t last_seq = -1;
    uint32_t stream_ID = 0xffff;
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

    // NOTE: This is a temporary solution to aligning frames. Since it requires
    // an integration period which is a power of 2.
    const uint32_t integration_edge =
        (config->processing.samples_per_data_set * config->processing.num_gpu_frames *
        config->processing.num_data_sets) >> 2;

    // Testing variables.
    FILE * data_file = NULL;
    u_char file_buf[config->fpga_network.udp_packet_size];
    struct dirent ** file_list;
    int num_files = 0;
    int cur_file_id = 0;

    // Setup the PF_RING.
    pfring *pd = NULL;

    if (args->read_from_file == 0) {
        pd = pfring_open(args->ip_address, config->fpga_network.udp_packet_size, PF_RING_PROMISC );

        if(pd == NULL) {
            ERROR("pfring_open error [%s] (pf_ring not loaded or quick mode is enabled you and have already a socket bound to %s?)\n",
                  strerror(errno), args->ip_address);
            exit(EXIT_FAILURE);
        }

        pfring_set_application_name(pd, "gpu_correlator");

        if (pd->dna.dna_mapped_device == 0) {
            ERROR("The device is not in DNA mode.?");
        }

        pfring_set_poll_duration(pd, 100);
        pfring_set_poll_watermark(pd, 1000);

        if (pfring_enable_ring(pd) != 0) {
            ERROR("Cannot enable the PF_RING.");
            exit(EXIT_FAILURE);
        }
    }

    // Make sure the first buffer is ready to go. (this check is not really needed)
    wait_for_empty_buffer(args->buf, buffer_id);

    set_data_ID(args->buf, buffer_id, data_id++);
    error_matrix = get_error_matrix(args->buf, buffer_id);

    double start_time = -1;

    // Code for reading data from files.
    if (args->read_from_file == 1) {
        struct stat file_info;

        if (stat(args->file_name, &file_info) != 0) {
            ERROR("File or directory does not exist");
            return;
        }

        if (file_info.st_mode & S_IFREG) {
            file_list = malloc(sizeof(void*));
            file_list[0] = malloc(sizeof(struct dirent));
            strcpy(file_list[0]->d_name, args->file_name);
            strcpy(args->file_name, "");
        } else if (file_info.st_mode & S_IFDIR) {
            INFO("Reading directory %s", args->file_name);

            num_files = scandir(args->file_name, &file_list, file_select, versionsort);

            if (num_files <= 0) {
                ERROR("No files in directory");
                return;
            }
        } else {
            ERROR("Not a file or directory. mode: %xxd", file_info.st_mode);
            return;
        }

        data_file = open_next_file(args->file_name, file_list[cur_file_id]);
    }

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

            buffer_id = (buffer_id + args->num_links_in_group) % (args->buf->num_buffers);

            // Add some delay so that ch_master can keep up with the files.
            if (args->read_from_file == 1) {
                usleep(10000);
            }
            // This call will block if the buffer has not been written out to disk yet.
            wait_for_empty_buffer(args->buf, buffer_id);

            buffer_location = 0;

            set_data_ID(args->buf, buffer_id, data_id++);
            error_matrix = get_error_matrix(args->buf, buffer_id);
            if (last_seq != -1) {
                // If not the first packet we need to set BufferInfo data.
                set_fpga_seq_num(args->buf, buffer_id, last_seq + 1);
                set_stream_ID(args->buf, buffer_id, stream_ID);
                // TODO This is close, but not perfect timing - but this shouldn't really matter.
                static struct timeval now;
                gettimeofday(&now, NULL);
                set_first_packet_recv_time(args->buf, buffer_id, now);
            }
        }

        u_char *pkt_buf;

        if (likely(args->read_from_file == 0)) {
            struct pfring_pkthdr pf_header;
            int rc = pfring_recv(pd, &pkt_buf, 0, &pf_header, 1);
            if (rc <= 0) {
                // No packets available.
                if (rc < 0) {
                    ERROR("Error in pfring_recv! %d", rc);
                }
                pthread_yield();
                continue;
            }

            if (pf_header.len != config->fpga_network.udp_packet_size) {
                INFO("Link id: %d; Received packet with incorrect length: %d",
                     args->dev_id,
                     pf_header.len);
                continue;
            }
        } else {
            // File read code.
            if (fread(file_buf, config->fpga_network.udp_packet_size, 1, data_file) != 1) {
                if (!feof(data_file)) {
                    ERROR("Error reading file %s", args->file_name);
                    break;
                } else {
                    INFO("Reached EOF");
                    INFO("Current data_id = %d", get_buffer_data_ID(args->buf, buffer_id));
                    cur_file_id++;
                    if (cur_file_id < num_files) {
                        fclose(data_file);
                        data_file = open_next_file(args->file_name, file_list[cur_file_id]);
                        continue;
                    } else {
                        INFO("Reached end of files");
                        mark_producer_done(args->buf, args->link_id);
                        int ret = 0;
                        pthread_exit((void *) &ret);
                    }
                }
            }
            pkt_buf = file_buf;

            if (((uint32_t *)&pkt_buf[44])[0] == 0) {
                INFO("Skiping empty packet");
                continue;
            }
        }

        // Do seq number related stuff (location will change.)
        seq = ((((uint32_t *) &pkt_buf[54])[0]) + 0 ) >> 2;
        stream_ID = ((uint16_t *) &pkt_buf[44])[0];
        //INFO("Network thread: %d, seq: %u", args->dev_id, seq);

        // First packet alignment code.
        if (unlikely(last_seq == -1)) {

            if ( !( (seq % integration_edge) <= 10 && (seq % integration_edge) >= 0 ) && args->read_from_file == 0) {
                continue;
            }

            INFO("Network Thread: %d, Got first packet %" PRId64, args->dev_id, seq << 2);
            uint16_t link_ID = stream_ID & 0x000F;
            uint16_t slot_ID = (stream_ID & 0x00F0) >> 4;
            uint16_t crate_ID = (stream_ID & 0x0F00) >> 8;
            INFO("Network Thread: %d, Link ID: %u; Slot ID: %u; Crate ID: %u",
                 args->dev_id, link_ID, slot_ID, crate_ID);
            if (link_ID != args->dev_id) {
                // This shouldn't really be necessary, since the system should work with any cable configuration
                // However for now we will enforce it, since the cables are supposed to be connected in this way.
                ERROR("Cable connected incorrectly on link %d", args->dev_id);
            }

            // Set the time we got the first packet.
            static struct timeval now;
            gettimeofday(&now, NULL);
            set_first_packet_recv_time(args->buf, buffer_id, now);
            if (args->read_from_file == 0) {
                set_fpga_seq_num(args->buf, buffer_id, seq - seq % integration_edge);
                set_stream_ID(args->buf, buffer_id, stream_ID);
            } else {
                set_fpga_seq_num(args->buf, buffer_id, seq);
                set_stream_ID(args->buf, buffer_id, stream_ID);
            }

            // Time for internal counters.
            start_time = e_time();

            // TODO This is only correct with high probability,
            // this should be made deterministic. 
            if (seq % integration_edge == 0 || args->read_from_file == 1) {
                last_seq = seq;
                nt_memcpy(&args->buf->data[buffer_id][buffer_location], pkt_buf + 58, udp_payload_size);
                count++;
                buffer_location += udp_payload_size;
            } else {
                // If we have lost the packet on the edge,
                // we set the last_seq to the edge so that the buffer will still be aligned.
                // We also ignore the current packet, and just allow it to be lost for simplicity.
                last_seq = seq - seq % integration_edge;
            }
            continue;
        }


        nt_memcpy(&args->buf->data[buffer_id][buffer_location], pkt_buf + 58, udp_payload_size);

        //INFO("seq_num: %d", seq);

        if (unlikely(seq == last_seq)) {
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
            if (unlikely(diff < 0)) {
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
            if ( unlikely(lost > 0) ) {

                if (lost > 1000000) {
                    ERROR("Packet loss is very high! lost packets: %lld\n", (long long int)lost);
                }

                // The location the packet should have been in.
                int realPacketLocation = buffer_location + lost * udp_payload_size;
                if ( realPacketLocation < args->buf->buffer_size ) { // Case 1:
                    // Copy the memory in the right location.
                    assert(buffer_id < args->buf->num_buffers);
                    assert(realPacketLocation <= args->buf->buffer_size - udp_payload_size);
                    assert(buffer_location <= args->buf->buffer_size - udp_payload_size);
                    assert(buffer_id >= 0);
                    assert(buffer_location >= 0);
                    assert(realPacketLocation >= 0);
                    nt_memcpy((void *) &args->buf->data[buffer_id][realPacketLocation],
                              (void *) &args->buf->data[buffer_id][buffer_location], udp_payload_size);

                    // Zero out the lost part of the buffer.
                    for (int i = 0; i < lost; ++i) {
                        memset(&args->buf->data[buffer_id][buffer_location + i*udp_payload_size], 0x88, udp_payload_size);
                    }
                    add_bad_timesamples(error_matrix, lost * config->fpga_network.timesamples_per_packet);

                    buffer_location = realPacketLocation + udp_payload_size;

                    //ERROR("Case 1 packet loss event on data_id: %d", data_id);
                } else { // Case 2 (the hard one):

                    // zero out the rest of the current buffer and mark it as full.
                    int i, j;
                    for (i = 0; (buffer_location + i*udp_payload_size) < args->buf->buffer_size; ++i) {
                        memset(&args->buf->data[buffer_id][buffer_location + i*udp_payload_size], 0x88, udp_payload_size);
                        add_bad_timesamples(error_matrix, config->fpga_network.timesamples_per_packet);
                    }

                    WARN("In Case 2 packet loss on link: %d ; data_id: %d ; buffer_id: %d", args->link_id, data_id, buffer_id);
                    // Keep track of the last edge seq number in case we need it later.
                    uint32_t last_edge = get_fpga_seq_num(args->buf, buffer_id);

                    // Notify the we have filled a buffer.
                    mark_buffer_full(args->buf, buffer_id);

                    // Check if we should stop collecting data.
                    check_if_done(&total_buffers_filled, args, total_packets, 
                                grand_total_lost, total_out_of_order, total_duplicate, start_time);

                    // Get the number of lost packets in the new buffer(s).
                    int num_lost_packets_new_buf = (lost+1) - (args->buf->buffer_size - buffer_location)/udp_payload_size;

                    assert(num_lost_packets_new_buf > 0);

                    i = 0;
                    j = 1;

                    // We may have lost more packets than will fit in a buffer.
                    do {

                        // Get a new buffer.
                        buffer_id = (buffer_id + args->num_links_in_group) % (args->buf->num_buffers);

                        // This call will block if the buffer has not been written out to disk yet
                        // shouldn't be an issue if everything runs correctly.
                        wait_for_empty_buffer(args->buf, buffer_id);

                        set_data_ID(args->buf, buffer_id, data_id++);
                        error_matrix = get_error_matrix(args->buf, buffer_id);

                        uint32_t fpga_seq_number = last_edge + j * (args->buf->buffer_size/udp_payload_size); // == number of iterations FIXME.

                        set_fpga_seq_num(args->buf, buffer_id, fpga_seq_number);
                        set_stream_ID(args->buf, buffer_id, stream_ID);

                        // This really isn't the correct time, but this is the best we can do here.
                        struct timeval now;
                        gettimeofday(&now, NULL);
                        set_first_packet_recv_time(args->buf, buffer_id, now);

                        for (i = 0; num_lost_packets_new_buf > 0 && (i*udp_payload_size) < args->buf->buffer_size; ++i) {
                            memset(&args->buf->data[buffer_id][i*udp_payload_size], 0x88, udp_payload_size);
                            add_bad_timesamples(error_matrix, config->fpga_network.timesamples_per_packet);
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
                    buffer_location = i*udp_payload_size;

                    // We need to increase the total number of lost packets since we 
                    // just tossed away the packet we read.
                    total_lost++;
                    WARN("Case 2 packet loss event on link: %d ; data_id: %d\n", args->link_id, data_id);

                }
            } else {
                // This is the normal case; a valid packet. 
                buffer_location += udp_payload_size;
            }

        } else {
            // seq is before last_seq.  We have an out of order packet.

            total_out_of_order++;

            if (out_of_order_event == 0 || out_of_order_event != last_seq) {
                WARN("Out of order event in data_id: %d\n", data_id);
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

        // Compute speed and packet loss for every recorded frame.
        count++;
        total_packets++;
        int output_period = config->processing.num_gpu_frames *
            config->processing.samples_per_data_set / config->fpga_network.timesamples_per_packet;

        if (count % (output_period+1) == 0) {
            current_time = e_time();
            DEBUG("Link id: %d; Receive Speed: %1.3f Gbps %.0f pps\n", args->dev_id,
                  (((double)output_period*config->fpga_network.udp_packet_size*8) /
                  (current_time - last_time)) / (1024*1024*1024), output_period / (current_time - last_time) );
            last_time = current_time;
            if (total_lost != 0) {
                INFO("Link id: %d; Packet loss on %.6f%%\n",
                     args->dev_id, ((double)total_lost/(double)output_period)*100);
            } else {
                INFO("Link id: %d; Packet loss on %.6f%%\n",
                     args->dev_id, (double)0.0);
            }
            grand_total_lost += total_lost;
            total_lost = 0;

            INFO("Link id: %d; Number of full buffers: %d/%d; Total data received: %.2f GB\n",
                 args->dev_id, get_num_full_buffers(args->buf),
                 args->buf->num_buffers,
                 ((double)total_buffers_filled *
                 ((double)args->buf->buffer_size / (1024.0*1024.0)))/1024.0);
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
                           args->config->processing.samples_per_data_set,
                           args->config->processing.num_local_freq,
                           args->config->processing.num_elements,
                           (unsigned char *) args->buf->data[buffer_id]);

    for (int i = 0; i < 100; i++) {
        INFO("Thread ID=%d; buf[%d]=0x%X", args->link_id, i, *(unsigned int *)(&args->buf->data[buffer_id][i*4]) );
    }

    mark_buffer_full(args->buf, buffer_id);

    mark_producer_done(args->buf, args->link_id);
    int ret = 0;
    pthread_exit((void *) &ret);

}
