#include "testDataGen.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"       // for Telescope
#include "buffer.h"            // for Buffer, allocate_new_metadata_object, mark_frame_full
#include "bufferContainer.hpp" // for bufferContainer
#include "chimeMetadata.hpp"   // for set_first_packet_recv_time, set_fpga_seq_num, set_stream_id
#include "errors.h"            // for exit_kotekan, CLEAN_EXIT, ReturnCode
#include "kotekanLogging.hpp"  // for DEBUG, INFO
#include "kotekanTrackers.hpp" // for KotekanTrackers
#include "restServer.hpp"      // for restServer, connectionInstance, HTTP_RESPONSE, HTTP_RESPO...
#include "visUtil.hpp"         // for current_time, ts_to_double

#include <assert.h>    // for assert
#include <atomic>      // for atomic_bool
#include <cmath>       // for fmod
#include <cstdint>     // for uint64_t
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, _Placeholder, bind, _1, _2, function
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error
#include <stdint.h>    // for uint64_t, uint32_t, uint8_t
#include <stdlib.h>    // for rand, srand
#include <sys/time.h>  // for gettimeofday, timeval
#include <sys/types.h> // for uint
#include <unistd.h>    // for usleep
#include <vector>      // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

using kotekan::connectionInstance;
using kotekan::HTTP_RESPONSE;
using kotekan::restServer;

REGISTER_KOTEKAN_STAGE(testDataGen);


testDataGen::testDataGen(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&testDataGen::main_thread, this)) {

    buf = get_buffer("out_buf");
    register_producer(buf, unique_name.c_str());
    type = config.get<std::string>(unique_name, "type");
    assert(type == "const" || type == "random" || type == "ramp" || type == "tpluse"
           || type == "tpluseplusf" || type == "tpluseplusfprime" || type == "square");
    if (type == "const" || type == "random" || type == "ramp")
        value = config.get<int>(unique_name, "value");
    _pathfinder_test_mode = config.get_default<bool>(unique_name, "pathfinder_test_mode", false);

    samples_per_data_set = config.get_default<int>(unique_name, "samples_per_data_set", 32768);
    stream_id.id = config.get_default<uint64_t>(unique_name, "stream_id", 0);
    num_frames = config.get_default<int>(unique_name, "num_frames", -1);
    // TODO: rename this parameter to `num_freq_per_stream` in the config
    _num_freq_in_frame = config.get_default<size_t>(unique_name, "num_local_freq", 1);
    // Try to generate data based on `samples_per_dataset` cadence or else just generate it as
    // fast as possible.
    wait = config.get_default<bool>(unique_name, "wait", true);
    // Whether to wait for is rest signal to start or generate next frame. Useful for testing
    // stages that must interact rest commands. Valid modes are "start", "step", and "none".
    rest_mode = config.get_default<std::string>(unique_name, "rest_mode", "none");
    assert(rest_mode == "none" || rest_mode == "start" || rest_mode == "step");
    step_to_frame = 0;
    _first_frame_index = config.get_default<uint32_t>(unique_name, "first_frame_index", 0);

    endpoint = unique_name + "/generate_test_data";
    using namespace std::placeholders;
    restServer::instance().register_post_callback(
        endpoint, std::bind(&testDataGen::rest_callback, this, _1, _2));

    // Create stat tracker
    kotekan::KotekanTrackers& KT = kotekan::KotekanTrackers::instance();
    timer = KT.add_tracker(unique_name, "frame_fill_time", "sec");
}


testDataGen::~testDataGen() {
    restServer::instance().remove_json_callback(endpoint);
}

bool testDataGen::can_i_go(int frame_id_abs) {
    if (rest_mode == "none")
        return true;
    if (step_to_frame > 0 && rest_mode == "start")
        return true;
    // Yes, this is a race condition, but it is fine since don't need perfect synchorization.
    if (frame_id_abs < step_to_frame)
        return true;
    return false;
}


void testDataGen::rest_callback(connectionInstance& conn, nlohmann::json& request) {
    int num_frames;
    try {
        num_frames = request["num_frames"];
    } catch (...) {
        conn.send_error("Could not parse number of frames.", HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    conn.send_empty_reply(HTTP_RESPONSE::OK);
    step_to_frame += num_frames;
}


void testDataGen::main_thread() {

    int frame_id = 0;
    int frame_id_abs = 0;
    uint8_t* frame = nullptr;
    uint64_t seq_num = samples_per_data_set * _first_frame_index;
    bool finished_seeding_consant = false;
    static struct timeval now;

    int link_id = 0;

    double frame_length = samples_per_data_set * ts_to_double(Telescope::instance().seq_length());

    while (!stop_thread) {
        double start_time = current_time();

        if (!can_i_go(frame_id_abs)) {
            usleep(1e5);
            continue;
        }

        frame = (uint8_t*)wait_for_empty_frame(buf, unique_name.c_str(), frame_id);
        if (frame == nullptr)
            break;

        allocate_new_metadata_object(buf, frame_id);
        set_fpga_seq_num(buf, frame_id, seq_num);
        set_stream_id(buf, frame_id, stream_id);

        gettimeofday(&now, nullptr);
        set_first_packet_recv_time(buf, frame_id, now);

        // std::random_device rd;
        // std::mt19937 gen(rd());
        // std::uniform_int_distribution<> dis(0, 255);
        if (type == "random")
            srand(value);
        unsigned char temp_output;
        int num_elements = buf->frame_size / samples_per_data_set / _num_freq_in_frame;
        for (uint j = 0; j < buf->frame_size / sizeof(uint8_t); ++j) {
            if (type == "const") {
                if (finished_seeding_consant)
                    break;
                frame[j] = value;
            } else if (type == "ramp") {
                frame[j] = fmod(j * value, 256 * value);
                //                frame[j] = j*value;
            } else if (type == "random") {
                unsigned char new_real;
                unsigned char new_imaginary;
                new_real = rand() % 16;
                new_imaginary = rand() % 16;
                temp_output = ((new_real << 4) & 0xF0) + (new_imaginary & 0x0F);
                frame[j] = temp_output;
            } else if (type == "tpluse") {
                int time_idx = j / num_elements;
                int elem_idx = j % num_elements;
                frame[j] = seq_num + time_idx + elem_idx;
            } else if (type == "tpluseplusf") {
                int time_idx = j / (_num_freq_in_frame * num_elements);
                int stream_freq_idx = Telescope::instance().to_freq_id(
                    stream_id,
                    j % (_num_freq_in_frame * num_elements)
                        / num_elements); // translate local freq_idx (0..._num_freq_in_frame
                                         // - 1) to global frequency index (0...1023)
                int elem_idx = j % num_elements;
                frame[j] = seq_num + time_idx + stream_freq_idx + elem_idx;
            } else if (type == "tpluseplusfprime") {
                int time_idx = j / (_num_freq_in_frame * num_elements);
                int stream_freq_idx = Telescope::instance().to_freq_id(
                    stream_id,
                    j % (_num_freq_in_frame * num_elements)
                        / num_elements); // translate local freq_idx (0..._num_freq_in_frame
                                         // - 1) to global frequency index (0...1023)
                int elem_idx = j % num_elements;
                frame[j] = 2 * (seq_num + time_idx) + 3 * stream_freq_idx + 5 * elem_idx;
            } else if (type == "square") {
                unsigned char new_real;
                unsigned char new_imaginary;
                if ((j / num_elements) % 8 < 4) {
                    new_real = 0;
                    new_imaginary = 0;
                } else {
                    new_real = 4;
                    new_imaginary = 0;
                }
                temp_output = ((new_real << 4) & 0xF0) + (new_imaginary & 0x0F);
                frame[j] = temp_output;
            }
        }
        DEBUG("Generated a {:s} test data set in {:s}[{:d}]", type, buf->buffer_name, frame_id);

        mark_frame_full(buf, unique_name.c_str(), frame_id);

        frame_id_abs += 1;
        if (num_frames >= 0 && frame_id_abs >= num_frames) {
            INFO("Frame ID greater than the no. of frames");
            exit_kotekan(ReturnCode::CLEAN_EXIT);
            break;
        };
        frame_id = frame_id_abs % buf->num_frames;

        if (_pathfinder_test_mode == true) {
            // Test PF seq_num increment.
            if (link_id == 7) {
                link_id = 0;
                seq_num += samples_per_data_set;
            } else {
                link_id++;
            }
        } else {
            seq_num += samples_per_data_set;
        }
        if (frame_id == 0)
            finished_seeding_consant = true;

        if (wait) {
            double time = current_time();
            double frame_end_time = start_time + frame_length;
            timer->add_sample(time - start_time);
            if (time < frame_end_time)
                usleep((int)(1e6 * (frame_end_time - time)));
        }
    }
}
