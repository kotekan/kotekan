#include "networkOutputSim.hpp"
#include "buffer.h"
#include "chimeMetadata.h"
#include "errors.h"
#include "test_data_generation.h"
#include "nt_memcpy.h"
#include "Config.hpp"
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
#include <functional>

networkOutputSim::networkOutputSim(Config &config_,
                     const string& unique_name,
                     bufferContainer &buffer_container) :
                  KotekanProcess(config_, unique_name, buffer_container, std::bind(&networkOutputSim::main_thread, this))
{

    buf = get_buffer("network_out_buf");
    register_producer(buf, unique_name.c_str());
    num_links_in_group = config.get_int(unique_name, "num_links_in_group");
    link_id = config.get_int(unique_name,"link_id");
    pattern = config.get_int(unique_name,"pattern");
    stream_id = config.get_int(unique_name,"stream_id");
}

networkOutputSim::~networkOutputSim() {
}

void networkOutputSim::apply_config(uint64_t fpga_seq) {
    if (!config.update_needed(fpga_seq))
        return;

    _samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    _num_local_freq = config.get_int(unique_name, "num_local_freq");
    _num_elem = config.get_int(unique_name, "num_elements");
}

void networkOutputSim::main_thread() {

    apply_config(0);

    int frame_id = link_id;
    unsigned char * frame = NULL;
    uint64_t fpga_seq_num = 0;
    int constant = 9;

    for (EVER) {
        frame = (unsigned char *)wait_for_empty_frame(buf, unique_name.c_str(), frame_id);

        if ((fpga_seq_num / _samples_per_data_set) % 2 == 0) {
            constant = 10;
        } else {
            constant = 9;
        }

        set_stream_id(buf, frame_id, stream_id);
        set_fpga_seq_num(buf, frame_id, fpga_seq_num);
        struct timeval now;
        gettimeofday(&now, NULL);
        set_first_packet_recv_time(buf, frame_id, now);

        // TODO perfect place for lambdas here.
        if (pattern == SIM_CONSTANT) {
            //INFO("Generating a constant data set all (1,1).");
            generate_const_data_set(constant, constant,
                                _samples_per_data_set,
                                _num_local_freq,
                                _num_elem,
                                frame);
        } else if (pattern == SIM_FULL_RANGE) {
            //INFO("Generating a full range of all possible values.");
            generate_full_range_data_set(0,
                                _samples_per_data_set,
                                _num_local_freq,
                                _num_elem,
                                frame);
        } else if (pattern == SIM_SINE) {
            stream_id_t stream_id;
            stream_id.link_id = link_id;
            //INFO("Generating data with a complex sine in frequency.");
            generate_complex_sine_data_set(stream_id,
                                _samples_per_data_set,
                                _num_local_freq,
                                _num_elem,
                                frame);
        } else {
            ERROR("Invalid Pattern");
            exit(-1);
        }

        mark_frame_full(buf, unique_name.c_str(), frame_id);

        frame_id = (frame_id + num_links_in_group) % (buf->num_frames);

        fpga_seq_num += _samples_per_data_set;

    }

    int ret = 0;
    pthread_exit((void *) &ret);

}
