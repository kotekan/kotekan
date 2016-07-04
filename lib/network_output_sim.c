#include "network_output_sim.h"
#include "buffers.h"
#include "errors.h"
#include "test_data_generation.h"
#include "error_correction.h"
#include "nt_memcpy.h"
#include "config.h"
#include "util.h"
#include "time_tracking.h"

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


void* network_output_sim(void * arg) {
    struct networkOutputSim * args;
    args = (struct networkOutputSim *) arg;
    int buffer_id = args->link_id;
    int data_id = 0;
    uint64_t fpga_seq_num = 0;

    for (EVER) {
        wait_for_empty_buffer(args->buf, buffer_id);

        set_data_ID(args->buf, buffer_id, data_id++);
        set_stream_ID(args->buf, buffer_id, args->stream_id);
        set_fpga_seq_num(args->buf, buffer_id, fpga_seq_num);
        struct timeval now;
        gettimeofday(&now, NULL);
        set_first_packet_recv_time(args->buf, buffer_id, now);

        if (args->pattern == SIM_CONSTANT) {
            //INFO("Generating a constant data set all (1,1).");
            generate_const_data_set(9, 9,
                                args->config->processing.samples_per_data_set,
                                args->config->processing.num_local_freq,
                                args->config->processing.num_elements,
                                args->buf->data[buffer_id]);
        } else if (args->pattern == SIM_FULL_RANGE) {
            //INFO("Generating a full range of all possible values.");
            generate_full_range_data_set(0,
                                args->config->processing.samples_per_data_set,
                                args->config->processing.num_local_freq,
                                args->config->processing.num_elements,
                                args->buf->data[buffer_id]);
        } else if (args->pattern == SIM_SINE) {
            stream_id_t stream_id;
            stream_id.link_id = args->stream_id;
            //INFO("Generating data with a complex sine in frequency.");
            generate_complex_sine_data_set(stream_id,
                                args->config->processing.samples_per_data_set,
                                args->config->processing.num_local_freq,
                                args->config->processing.num_elements,
                                args->buf->data[buffer_id]);
        } else {
            ERROR("Invalid Pattern");
            exit(-1);
        }

        mark_buffer_full(args->buf, buffer_id);

        buffer_id = (buffer_id + args->num_links_in_group) % (args->buf->num_buffers);

        fpga_seq_num += args->config->processing.samples_per_data_set;
    }

    mark_producer_done(args->buf, args->link_id);
    int ret = 0;
    pthread_exit((void *) &ret);

}
