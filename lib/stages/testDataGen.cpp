#include "testDataGen.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"       // for Telescope, stream_t
#include "buffer.h"            // for Buffer, allocate_new_metadata_object, mark_frame_full
#include "bufferContainer.hpp" // for bufferContainer
#include "chimeMetadata.hpp"   // for set_first_packet_recv_time, set_fpga_seq_num, set_stream_id
#include "kotekanLogging.hpp"  // for INFO, DEBUG
#include "kotekanTrackers.hpp" // for KotekanTrackers
#include "oneHotMetadata.hpp"  // for metadata_is_onehot, set_onehot_frame_counter, set_onehot_...
#include "restServer.hpp"      // for restServer, connectionInstance, HTTP_RESPONSE, HTTP_RESPO...
#include "visUtil.hpp"         // for current_time, ts_to_double, StatTracker

#include <algorithm>   // for copy, max
#include <assert.h>    // for assert
#include <atomic>      // for atomic_bool
#include <cmath>       // for fmod
#include <cstdint>     // for uint64_t
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, _Placeholder, bind, _1, _2, function
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error, invalid_argument
#include <stdint.h>    // for uint64_t, uint32_t, uint8_t, int32_t
#include <stdlib.h>    // for rand, srand
#include <strings.h>   // for bzero
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
    assert(type == "const" || type == "const32" || type == "random" || type == "random_signed"
           || type == "ramp" || type == "tpluse" || type == "tpluseplusf"
           || type == "tpluseplusfprime" || type == "square" || type == "onehot");
    if (type == "const" || type == "const32" || type == "random" || type == "random_signed"
        || type == "ramp" || type == "onehot") {
        value = config.get_default<int>(unique_name, "value", -1999);
        _value_array =
            config.get_default<std::vector<int>>(unique_name, "values", std::vector<int>());
    }
    _seed = config.get_default<int>(unique_name, "seed", 0);
    _pathfinder_test_mode = config.get_default<bool>(unique_name, "pathfinder_test_mode", false);
    _array_shape =
        config.get_default<std::vector<int>>(unique_name, "array_shape", std::vector<int>());
    if (_array_shape.size()) {
        size_t sz = 1;
        for (int s : _array_shape)
            sz *= s;
        if (sz != buf->frame_size)
            // clang-format off
            throw std::invalid_argument("testDataGen: product of 'array_shape' config setting must equal the buffer frame size");
        // clang-format on
    }
    samples_per_data_set = config.get_default<int>(unique_name, "samples_per_data_set", 32768);
    stream_id.id = config.get_default<uint64_t>(unique_name, "stream_id", 0);
    num_frames = config.get_default<int>(unique_name, "num_frames", -1);
    num_links = config.get_default<uint32_t>(unique_name, "num_links", 1);
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
    int32_t* frame32 = nullptr;
    uint64_t seq_num = samples_per_data_set * _first_frame_index;
    bool finished_seeding_consant = false;
    static struct timeval now;
#if KOTEKAN_FLOAT16
    float16_t* framef16 = nullptr;
#endif

    int link_id = 0;

    double frame_length =
        samples_per_data_set * ts_to_double(Telescope::instance().seq_length()) / num_links;

    if (((type == "random") || (type == "random_signed") || (type == "onehot")) && _seed)
        srand(_seed);

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

        unsigned char temp_output;
        int num_elements = buf->frame_size / samples_per_data_set / _num_freq_in_frame;
        uint n_to_set = buf->frame_size / sizeof(uint8_t);
        if (type == "const32") {
            n_to_set /= sizeof(int32_t);
            frame32 = (int32_t*)frame;
#if KOTEKAN_FLOAT16
        } else if (type == "constf16") {
            n_to_set /= sizeof(float16_t);
            framef16 = (float16_t*)frame;
#endif
        }
        if (type == "onehot") {
            int val = value;
            if (_value_array.size())
                val = _value_array[frame_id_abs % _value_array.size()];
            bzero(frame, n_to_set);
            if (_array_shape.size()) {
                std::string istring = "";
                size_t j = 0;
                std::vector<int> indices;
                for (size_t i = 0; i < _array_shape.size(); i++) {
                    int n = _array_shape[i];
                    int k = rand() % n;
                    j = j * n + k;
                    if (i)
                        istring += ", ";
                    istring += std::to_string(k);
                    indices.push_back(k);
                }
                frame[j] = val;
                INFO("Set {:s}[{:d}] index [{:s}] (flat: {:d} = 0x{:x}) to 0x{:x} ({:d})",
                     buf->buffer_name, frame_id, istring, j, j, val, val);
                if (metadata_is_onehot(buf, frame_id)) {
                    DEBUG("One-hot metadata; setting indices");
                    set_onehot_indices(buf, frame_id, indices);
                    set_onehot_frame_counter(buf, frame_id, frame_id_abs);
                    INFO("Set {:s}[{:d}] frame counter {:d}", buf->buffer_name, frame_id,
                         frame_id_abs);
                }
                INFO("PY onehot[{:d}] = (({:s}), 0x{:x})", frame_id_abs, istring, val);
            } else {
                int j = rand() % n_to_set;
                INFO("Set {:s}[{:d}] flat index {:d} = 0x{:x} to 0x{:x} ({:d})", buf->buffer_name,
                     frame_id, j, j, val, val);
                frame[j] = val;
            }
            n_to_set = 0;
        }
        for (uint j = 0; j < n_to_set; ++j) {
            if (type == "const") {
                if (finished_seeding_consant)
                    break;
                frame[j] = value;
            } else if (type == "const32") {
                if (finished_seeding_consant)
                    break;
                frame32[j] = value;
#if KOTEKAN_FLOAT16
            } else if (type == "constf16") {
                if (finished_seeding_consant)
                    break;
                framef16[j] = fvalue;
#endif
            } else if (type == "ramp") {
                frame[j] = fmod(j * value, 256 * value);
                //                frame[j] = j*value;
            } else if (type == "random") {
                char new_real;
                char new_imaginary;
                new_real = (rand() % 15) + 1;      // Limit to [-7, 7]
                new_imaginary = (rand() % 15) + 1; // Limit to [-7, 7]
                temp_output = ((new_real << 4) & 0xF0) + (new_imaginary & 0x0F);
                frame[j] = temp_output;
            } else if (type == "random_signed") {
                char new_real;
                char new_imaginary;
                new_real = (rand() % 15) + 1;      // Limit to [-7, 7]
                new_imaginary = (rand() % 15) + 1; // Limit to [-7, 7]
                temp_output = ((new_real << 4) & 0xF0) + (new_imaginary & 0x0F);
                frame[j] = temp_output ^ 0x88;
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
            INFO("Generated the requested number of frames ({:d}) - exiting", num_frames);
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
