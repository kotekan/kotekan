#include "testDataGenFloat.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for allocate_new_metadata_object, mark_frame_full, register_p...
#include "bufferContainer.hpp" // for bufferContainer
#include "chimeMetadata.hpp"   // for set_first_packet_recv_time, set_fpga_seq_num, set_stream_id
#include "kotekanLogging.hpp"  // for DEBUG

#include <assert.h>    // for assert
#include <atomic>      // for atomic_bool
#include <cmath>       // for fmod
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, bind, function
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error
#include <stdint.h>    // for uint64_t
#include <stdlib.h>    // for rand, srand
#include <sys/time.h>  // for gettimeofday, timeval
#include <sys/types.h> // for uint
#include <unistd.h>    // for usleep
#include <vector>      // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(testDataGenFloat);

testDataGenFloat::testDataGenFloat(Config& config, const std::string& unique_name,
                                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&testDataGenFloat::main_thread, this)) {

    buf = get_buffer("network_out_buf");
    register_producer(buf, unique_name.c_str());
    type = config.get<std::string>(unique_name, "type");
    assert(type == "const" || type == "random" || type == "ramp");
    if (type == "const" || type == "random" || type == "ramp")
        value = config.get<int>(unique_name, "value");
    _pathfinder_test_mode = config.get_default<bool>(unique_name, "pathfinder_test_mode", false);
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    _first_frame_index = config.get_default<uint32_t>(unique_name, "first_frame_index", 0);
    _gen_all_const_data = config.get_default<bool>(unique_name, "gen_all_const_data", false);
}

testDataGenFloat::~testDataGenFloat() {}

void testDataGenFloat::main_thread() {

    int frame_id = 0;
    float* frame = nullptr;
    uint64_t seq_num = _samples_per_data_set * _first_frame_index;
    bool finished_seeding_consant = false;
    static struct timeval now;

    int link_id = 0;

    while (!stop_thread) {
        frame = (float*)wait_for_empty_frame(buf, unique_name.c_str(), frame_id);
        if (frame == nullptr)
            break;

        allocate_new_metadata_object(buf, frame_id);
        set_fpga_seq_num(buf, frame_id, seq_num);
        // TODO This should be dynamic/config controlled.
        set_stream_id(buf, frame_id, {0});

        gettimeofday(&now, nullptr);
        set_first_packet_recv_time(buf, frame_id, now);

        // std::random_device rd;
        // std::mt19937 gen(rd());
        // std::uniform_int_distribution<> dis(0, 255);
        if (type == "random")
            srand(value);
        unsigned char temp_output;
        for (uint j = 0; j < buf->frame_size / sizeof(float); ++j) {
            if (type == "const") {
                if (finished_seeding_consant)
                    break;
                frame[j] = value;
            } else if (type == "ramp") {
                frame[j] = fmod(j * value, 256 * value);
            } else if (type == "random") {
                unsigned char new_real;
                unsigned char new_imaginary;
                new_real = rand() % 16;
                new_imaginary = rand() % 16;
                temp_output = ((new_real << 4) & 0xF0) + (new_imaginary & 0x0F);
                frame[j] = temp_output;
            }
        }
        usleep(83000);
        DEBUG("Generated a {:s} test data set in {:s}[{:d}]", type, buf->buffer_name, frame_id);

        mark_frame_full(buf, unique_name.c_str(), frame_id);

        frame_id = (frame_id + 1) % buf->num_frames;

        if (_pathfinder_test_mode == true) {
            // Test PF seq_num increment.
            if (link_id == 7) {
                link_id = 0;
                seq_num += _samples_per_data_set;
            } else {
                link_id++;
            }
        } else {
            seq_num += _samples_per_data_set;
        }
        if (frame_id == 0 && !_gen_all_const_data)
            finished_seeding_consant = true;
    }
}
