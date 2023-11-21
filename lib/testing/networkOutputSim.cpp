#include "networkOutputSim.hpp"

#include "Config.hpp" // for Config
#include "ICETelescope.hpp"
#include "StageFactory.hpp" // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"
#include "buffer.hpp"          // for mark_frame_full, register_producer, wait_for_empty_frame
#include "bufferContainer.hpp" // for bufferContainer
#include "chimeMetadata.hpp"   // for set_first_packet_recv_time, set_fpga_seq_num, set_str...
#include "kotekanLogging.hpp"  // for ERROR

#include <atomic>     // for atomic_bool
#include <cstdint>    // for int32_t
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <pthread.h>  // for pthread_exit
#include <regex>      // for match_results<>::_Base_type
#include <stdlib.h>   // for exit
#include <sys/time.h> // for gettimeofday, timeval
#include <vector>     // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;


// Test data patterns
void generate_full_range_data_set(int offset, int num_time_steps, int num_freq, int num_elem,
                                  unsigned char* out_data) {
    unsigned char real = 0;
    unsigned char imag = offset;
    int idx = 0;

    for (int time_step = 0; time_step < num_time_steps; ++time_step) {
        for (int freq = 0; freq < num_freq; ++freq) {
            for (int elem = 0; elem < num_elem; ++elem) {
                idx = time_step * num_elem * num_freq + freq * num_elem + elem;
                out_data[idx] = ((real << 4) & 0xF0) + (imag & 0x0F);

                // Note this is the same as doing [0,255] in the char
                // but this is more clear as to what the components are doing.
                if (imag == 15) {
                    real = (real + 1) % 16;
                }
                imag = (imag + 1) % 16;
            }
        }
    }
}

void generate_const_data_set(unsigned char real, unsigned char imag, int num_time_steps,
                             int num_freq, int num_elem, unsigned char* out_data) {
    int idx = 0;

    for (int time_step = 0; time_step < num_time_steps; ++time_step) {
        for (int freq = 0; freq < num_freq; ++freq) {
            for (int elem = 0; elem < num_elem; ++elem) {
                idx = time_step * num_elem * num_freq + freq * num_elem + elem;
                out_data[idx] = ((real << 4) & 0xF0) + (imag & 0x0F);
            }
        }
    }
}

void generate_complex_sine_data_set(stream_t stream_id, int num_time_steps, int num_freq,
                                    int num_elem, unsigned char* out_data) {

    int idx = 0;
    int imag = 0;
    int real = 0;

    auto& tel = Telescope::instance();

    for (int time_step = 0; time_step < num_time_steps; ++time_step) {
        for (int freq = 0; freq < num_freq; ++freq) {
            for (int elem = 0; elem < num_elem; ++elem) {
                idx = time_step * num_elem * num_freq + freq * num_elem + elem;
                imag = tel.to_freq_id(stream_id, freq) % 16;
                real = 9;
                out_data[idx] = ((real << 4) & 0xF0) + (imag & 0x0F);
            }
        }
    }
}

REGISTER_KOTEKAN_STAGE(networkOutputSim);

networkOutputSim::networkOutputSim(Config& config_, const std::string& unique_name,
                                   bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container, std::bind(&networkOutputSim::main_thread, this)) {

    buf = get_buffer("network_out_buf");
    buf->register_producer(unique_name);
    num_links_in_group = config.get<int>(unique_name, "num_links_in_group");
    link_id = config.get<int>(unique_name, "link_id");
    pattern = config.get<int>(unique_name, "pattern");
    stream_id.id = config.get<uint64_t>(unique_name, "stream_id");
}

networkOutputSim::~networkOutputSim() {}

void networkOutputSim::main_thread() {

    // Apply config.
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");
    _num_local_freq = config.get<int32_t>(unique_name, "num_local_freq");
    _num_elem = config.get<int32_t>(unique_name, "num_elements");

    int frame_id = link_id;
    unsigned char* frame = nullptr;
    uint64_t fpga_seq_num = 0;
    int constant = 9;

    while (!stop_thread) {
        frame = (unsigned char*)buf->wait_for_empty_frame(unique_name, frame_id);
        if (frame == nullptr)
            break;

        if ((fpga_seq_num / _samples_per_data_set) % 2 == 0) {
            constant = 10;
        } else {
            constant = 9;
        }

        set_stream_id(buf, frame_id, stream_id);
        set_fpga_seq_num(buf, frame_id, fpga_seq_num);
        struct timeval now;
        gettimeofday(&now, nullptr);
        set_first_packet_recv_time(buf, frame_id, now);

        // TODO perfect place for lambdas here.
        if (pattern == SIM_CONSTANT) {
            // INFO("Generating a constant data set all (1,1).");
            generate_const_data_set(constant, constant, _samples_per_data_set, _num_local_freq,
                                    _num_elem, frame);
        } else if (pattern == SIM_FULL_RANGE) {
            // INFO("Generating a full range of all possible values.");
            generate_full_range_data_set(0, _samples_per_data_set, _num_local_freq, _num_elem,
                                         frame);
        } else if (pattern == SIM_SINE) {
            ice_stream_id_t stream_id;
            stream_id.link_id = link_id;
            // INFO("Generating data with a complex sine in frequency.");
            generate_complex_sine_data_set(ice_encode_stream_id(stream_id), _samples_per_data_set,
                                           _num_local_freq, _num_elem, frame);
        } else {
            ERROR("Invalid Pattern");
            exit(-1);
        }

        buf->mark_frame_full(unique_name, frame_id);

        frame_id = (frame_id + num_links_in_group) % (buf->num_frames);

        fpga_seq_num += _samples_per_data_set;
    }

    int ret = 0;
    pthread_exit((void*)&ret);
}
