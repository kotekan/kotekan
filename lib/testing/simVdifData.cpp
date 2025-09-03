#include "simVdifData.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"          // for mark_frame_full, register_producer, wait_for_empty_frame
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for INFO
#include "util.h"              // for e_time
#include "vdif_functions.h"    // for VDIFHeader

#include <atomic>     // for atomic_bool
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <random>     // for mt19937, random_device, uniform_int_distribution
#include <regex>      // for match_results<>::_Base_type
#include <string.h>   // for memcpy, memset
#include <unistd.h>   // for usleep
#include <vector>     // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(simVdifData);

simVdifData::simVdifData(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&simVdifData::main_thread, this)) {
    buf = get_buffer("network_out_buf");
    register_producer(buf, unique_name.c_str());
}

simVdifData::~simVdifData() {}

void simVdifData::main_thread() {
    int times = config.get<int>(unique_name, "samples_per_data_set");
    int elements = config.get<int>(unique_name, "num_elements");
    int freqs = config.get<int>(unique_name, "num_local_freq");

    double time_available = 2.56e-6 * times; // microseconds
    int frame_id = 0;

    struct VDIFHeader header = {
        0, // uint32_t seconds : 30;
        0, // uint32_t legacy : 1;
        0, // uint32_t invalid : 1;
        0, // uint32_t data_frame : 24;
        0, // uint32_t ref_epoch : 6;
        0, // uint32_t unused : 2;
        0, // uint32_t frame_len : 24;
        0, // uint32_t log_num_chan : 5;
        0, // uint32_t vdif_version : 3;
        0, // uint32_t station_id : 16;
        0, // uint32_t thread_id : 10;
        0, // uint32_t bits_depth : 5;
        0, // uint32_t data_type : 1;
        0, // uint32_t eud1 : 24;
        0, // uint32_t edv : 8;
        0, // uint32_t eud2 : 32;
        0, // uint32_t eud3 : 32;
        0  // uint32_t eud4 : 32;
    };

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    start_time = e_time();
    //    for (int ct=0; ct<100; ct++) {
    while (!stop_thread) {
        unsigned char* buf_ptr =
            (unsigned char*)wait_for_empty_frame(buf, unique_name.c_str(), frame_id);
        if (buf_ptr == nullptr)
            break;
        stop_time = e_time();
        double dt = stop_time - start_time;
        if (dt < time_available) {
            usleep((time_available - dt) * 1e6);
            stop_time = e_time();
        }

        for (int t = 0; t < times; t++) {
            for (int e = 0; e < elements; e++) {
                memcpy(buf_ptr, (void*)&header, sizeof(header));
                buf_ptr += sizeof(header);
                memset(buf_ptr, (unsigned char)dis(gen), freqs * sizeof(char));
                //                for (int f = 0; f < freqs; f++) buf_ptr[f] = (f/4 % 16);
                //                for (int f = 0; f < freqs; f++) buf_ptr[f] = (unsigned
                //                char)dis(gen);
                buf_ptr += freqs;
            }
        }
        //        INFO("Generated a test data set in {:s}[{:d}]", buf.buffer_name, frame_id);

        mark_frame_full(buf, unique_name.c_str(), frame_id);
        frame_id = (frame_id + 1) % buf->num_frames;
        header.data_frame++;

        INFO("{:4.1f}% of {:6.4f}s available.\n", 100 * dt / time_available, time_available);
        start_time = stop_time;
    }
}
