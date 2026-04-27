#include "testDataGen.hpp"

#include "CHORDTelescope.hpp"  // for CHORDTelescope
#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for DataType, KOTEKAN_FLOAT16, float16_t
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "Symbol.hpp"          // for Symbol
#include "Telescope.hpp"       // for Telescope, stream_t
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, get_chord_metadata, CHORD_META_MAX_FREQ
#include "kotekanLogging.hpp"  // for INFO, DEBUG, ERROR
#include "kotekanTrackers.hpp" // for KotekanTrackers
#include "oneHotMetadata.hpp"  // for metadata_is_onehot, set_onehot_frame_counter, set_onehot_...
#include "restServer.hpp"      // for HTTP_RESPONSE, restServer, connectionInstance
#include "visUtil.hpp"         // for current_time, ts_to_double, StatTracker

#include "fmt.hpp" // for compile_string_to_view

#include <algorithm>   // for max
#include <assert.h>    // for assert
#include <cmath>       // for fmod
#include <functional>  // for bind, function, _1, _2
#include <random>      // for mt19937
#include <signal.h>    // for raise, SIGINT
#include <stdexcept>   // for invalid_argument
#include <stdint.h>    // for int8_t, uint32_t, uint8_t, int16_t, int32_t, uint64_t
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
    buf->register_producer(unique_name);
    type = config.get<std::string>(unique_name, "type");
    assert(type == "const" || type == "const_offset" || type == "const8" || type == "const1x8"
           || type == "const16" || type == "const32" || type == "constf16" || type == "random"
           || type == "random_signed" || type == "random_signed_offset" || type == "random1x8"
           || type == "random32" || type == "const64" || type == "random64"
           || type == "constu64" || type == "randomu64" || type == "random8" || type == "ramp"
           || type == "tpluse" || type == "tpluseplusf" || type == "tpluseplusfprime"
           || type == "square" || type == "onehot");
    assert(!((type == "constf16") && (KOTEKAN_FLOAT16 == 0)));
    int type_size = 1; // default
    if (type == "const")
        type_size = 1;
    if (type == "const_offset")
        type_size = 1;
    if (type == "const8" || type == "const1x8")
        type_size = 1;
    if (type == "const16")
        type_size = 2;
    if (type == "const32" || type == "random32")
        type_size = 4;
    if (type == "constf16")
        type_size = 2;
    if (type == "constu64" || type == "randomu64" || type == "const64" || type == "random64")
        type_size = 8;
    if (type == "const" || type == "const_offset" || type == "const8" || type == "const1x8"
        || type == "const16" || type == "const32" || type == "random" || type == "random32" || type == "random_signed"
        || type == "random_signed_offset" || type == "random1x8" || type == "random8"
        || type == "ramp" || type == "onehot") {
        value = config.get_default<int>(unique_name, "value", -1999);
        _value_array =
            config.get_default<std::vector<int>>(unique_name, "values", std::vector<int>());
    } else if (type == "constf16") {
        fvalue = config.get_default<float>(unique_name, "value", -1.0);
        _fvalue_array =
            config.get_default<std::vector<float>>(unique_name, "values", std::vector<float>());
    } else if (type == "constu64" || type == "randomu64") {
        ulvalue = config.get_default<uint64_t>(unique_name, "value", 0);
        _ulvalue_array = config.get_default<std::vector<uint64_t>>(unique_name, "values",
                                                                  std::vector<uint64_t>());
    } else if (type == "const64" || type == "random64") {
        lvalue = config.get_default<uint64_t>(unique_name, "value", 0);
        _lvalue_array = config.get_default<std::vector<uint64_t>>(unique_name, "values",
                                                                  std::vector<uint64_t>());
    }
    _reuse_random = config.get_default<bool>(unique_name, "reuse_random", false);
    _seed = config.get_default<int>(unique_name, "seed", 0);
    _pathfinder_test_mode = config.get_default<bool>(unique_name, "pathfinder_test_mode", false);
    _name = config.get_default<std::string>(unique_name, "name", "E");
    _array_shape = config.get_default<std::vector<int>>(
        unique_name, "array_shape", std::vector<int>({int(buf->frame_size) / type_size}));
    {
        size_t sz = type_size;
        for (int s : _array_shape)
            sz *= s;
        if (sz != buf->frame_size)
            // clang-format off
            throw std::invalid_argument("testDataGen: product of 'array_shape' config setting must equal the buffer frame size");
        // clang-format on
    }
    _dim_name = config.get_default<std::vector<std::string>>(unique_name, "dim_name",
                                                             std::vector<std::string>({"D"}));
    if (_array_shape.size() != _dim_name.size()) {
        throw std::invalid_argument("testDataGen: 'array_shape' and 'dim_name' config "
                                    "settings must be the same length!");
    }

    samples_per_data_set = config.get_default<int>(unique_name, "samples_per_data_set", 32768);
    stream_id.id = config.get_default<uint64_t>(unique_name, "stream_id", 0);
    num_frames = config.get_default<int>(unique_name, "num_frames", -1);
    num_links = config.get_default<uint32_t>(unique_name, "num_links", 1);
    set_dish_index = config.get_default<bool>(unique_name, "set_dish_index", false);
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
    _meta_time_downsample_factor =
        config.get_default<int>(unique_name, "meta_time_downsample_factor", 1);

    _manual_freq_ids = config.get_default<std::vector<uint32_t>>(unique_name, "manual_freq_ids",
                                                                 std::vector<uint32_t>());

    // Get simulate_fpga_restart_at_frame option
    _simulate_fpga_restart_at_frame =
        config.get_default<int64_t>(unique_name, "simulate_fpga_restart_at_frame", -1);

    // Get end_interrupt option
    _end_interrupt = config.get_default<bool>(unique_name, "end_interrupt", false);

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
    uint32_t frame_id_abs = _first_frame_index;
    uint8_t* frame = nullptr;
    int8_t* frame8 = nullptr;
    int16_t* frame16 = nullptr;
    int32_t* frame32 = nullptr;
    int64_t* frame64 = nullptr;
    uint64_t* frameu64 = nullptr;
    uint64_t seq_num = samples_per_data_set * _first_frame_index;
    bool finished_seeding_constant = false;
    static struct timeval now;
#if KOTEKAN_FLOAT16
    float16_t* framef16 = nullptr;
#endif

    int link_id = 0;

    auto& telescope = Telescope::instance();
    std::string telescope_type = telescope.get_name();

    double frame_length = samples_per_data_set * ts_to_double(telescope.seq_length()) / num_links;

    std::mt19937 rng(_seed);

    while (!stop_thread) {
        double start_time = current_time();

        if (!can_i_go(frame_id_abs)) {
            usleep(1e5);
            continue;
        }

        frame = (uint8_t*)buf->wait_for_empty_frame(unique_name, frame_id);
        if (frame == nullptr)
            break;

        buf->allocate_new_metadata_object(frame_id);
        std::shared_ptr<chordMetadata> chordmeta = get_chord_metadata(buf, frame_id);

        chordmeta->set_fpga_seq_num(seq_num);
        chordmeta->set_time_downsampling_fpga(_meta_time_downsample_factor);

        // TODO: Fix this, cannot change from frame to frame (and should not be "now")
        gettimeofday(&now, nullptr);
        chordmeta->set_first_packet_recv_time(now);

        chordmeta->set_name(_name);
        chordmeta->dims = (int)_array_shape.size();
        for (int d = 0; d < chordmeta->dims; ++d)
            chordmeta->set_array_dimension(d, _array_shape[d], _dim_name[d]);
        chordmeta->set_strides_simple();
        // frame_desc is set only after "type" has been decoded below

        // Set dish information
        // (This is the outdated way; the modern way uses the telescope object)
        if (set_dish_index && !chordmeta->dish_index) {
            const auto& chord_telescope = Telescope::instance().cast<CHORDTelescope>();
            const auto& dish_grid = chord_telescope.get_dish_grid();
            const int num_dish_locations_ew = dish_grid.get_num_dishes_x();
            const int num_dish_locations_ns = dish_grid.get_num_dishes_y();
            const int num_dish_locations = num_dish_locations_ew * num_dish_locations_ns;
            std::vector<int> dish_index(num_dish_locations, -1);
            int num_dishes = 0;
            for (int dish_loc_ns = 0; dish_loc_ns < num_dish_locations_ns; ++dish_loc_ns) {
                for (int dish_loc_ew = 0; dish_loc_ew < num_dish_locations_ew; ++dish_loc_ew) {
                    const int dish_ind = dish_grid.dish_index(dish_loc_ew, dish_loc_ns);
                    if (dish_ind >= 0) {
                        ++num_dishes;
                        assert(dish_index.at(dish_loc_ew + num_dish_locations_ew * dish_loc_ns)
                               == -1);
                        dish_index.at(dish_loc_ew + num_dish_locations_ew * dish_loc_ns) = dish_ind;
                    }
                }
            }
            chordmeta->ndishes = num_dishes;
            chordmeta->n_dish_locations_ew = num_dish_locations_ew;
            chordmeta->n_dish_locations_ns = num_dish_locations_ns;
            chordmeta->dish_index =
                new dish_index_t[chordmeta->n_dish_locations_ns * chordmeta->n_dish_locations_ew];
            std::copy(dish_index.begin(), dish_index.end(), chordmeta->dish_index);
        }

        // Set frequency channel metadata

        assert(_num_freq_in_frame <= CHORD_META_MAX_FREQ);
        std::vector<int> coarse_freq(_num_freq_in_frame);
        std::vector<int> freq_upchan_factor(coarse_freq.size());
        std::vector<int> freq_upchan_index(coarse_freq.size());
        for (int f = 0; f < static_cast<int>(coarse_freq.size()); f++) {
            if (_manual_freq_ids.size() > 0)
                coarse_freq[f] = _manual_freq_ids[f % _manual_freq_ids.size()];
            else if (telescope_type == "CHORDTelescope")
                coarse_freq[f] = telescope.cast<CHORDTelescope>().min_science_freq_id() + f;
            else
                coarse_freq[f] = f;
            freq_upchan_factor[f] = 1;
            freq_upchan_index[f] = 0;
        }

        chordmeta->set_coarse_freq(coarse_freq);
        chordmeta->set_freq_upchan_factor(freq_upchan_factor);
        chordmeta->set_freq_upchan_index(freq_upchan_index);

        chordmeta->set_frame_counter(frame_id_abs);

        unsigned char temp_output;
        int num_elements = buf->frame_size / samples_per_data_set / _num_freq_in_frame;
        uint n_to_set = buf->frame_size / sizeof(uint8_t);

        if (type == "const") {
            n_to_set /= sizeof(int8_t);
            frame8 = (int8_t*)frame;
            if (chordmeta)
                chordmeta->type = kotekan::int4x2;
        } else if (type == "const_offset") {
            n_to_set /= sizeof(int8_t);
            frame8 = (int8_t*)frame;
            if (chordmeta)
                chordmeta->type = kotekan::int4x2_swapped_withoffset;
        } else if (type == "const8" || type == "random8") {
            n_to_set /= sizeof(int8_t);
            frame8 = (int8_t*)frame;
            if (chordmeta)
                chordmeta->type = kotekan::int8;
        } else if (type == "const1x8") {
            n_to_set /= sizeof(int8_t);
            frame8 = (int8_t*)frame;
            if (chordmeta)
                chordmeta->type = kotekan::uint1x8;
        } else if (type == "const16") {
            n_to_set /= sizeof(int16_t);
            frame16 = (int16_t*)frame;
            if (chordmeta)
                chordmeta->type = kotekan::int16;
        } else if (type == "const32" || type == "random32") {
            n_to_set /= sizeof(int32_t);
            frame32 = (int32_t*)frame;
            if (chordmeta)
                chordmeta->type = kotekan::int32;
#if KOTEKAN_FLOAT16
        } else if (type == "constf16") {
            n_to_set /= sizeof(float16_t);
            framef16 = (float16_t*)frame;
            if (chordmeta)
                chordmeta->type = kotekan::float16;
#endif
        } else if (type == "constu64" || type == "randomu64") {
            n_to_set /= sizeof(uint64_t);
            frameu64 = (uint64_t*)frame;
            if (chordmeta)
                chordmeta->type = kotekan::uint64;
        } else if (type == "const64" || type == "random64") {
            n_to_set /= sizeof(int64_t);
            frame64 = (int64_t*)frame;
            if (chordmeta)
                chordmeta->type = kotekan::int64;
        } else if (type == "random") {
            if (chordmeta)
                chordmeta->type = kotekan::uint4x2;
        } else if (type == "random_signed") {
            if (chordmeta)
                chordmeta->type = kotekan::int4x2;
        } else if (type == "random_signed_offset") {
            if (chordmeta)
                chordmeta->type = kotekan::int4x2_swapped_withoffset;
        } else if (type == "random1x8") {
            if (chordmeta)
                chordmeta->type = kotekan::uint1x8;
        } else if (type == "ramp") {
            if (chordmeta)
                chordmeta->type = kotekan::uint8;
        } else if (type == "tpluse") {
            if (chordmeta)
                chordmeta->type = kotekan::uint1x8;
        } else if (type == "tpluseplusf") {
            if (chordmeta)
                chordmeta->type = kotekan::uint1x8;
        } else if (type == "tpluseplusfprime") {
            if (chordmeta)
                chordmeta->type = kotekan::uint1x8;
        } else if (type == "square") {
            if (chordmeta)
                chordmeta->type = kotekan::int4x2;
        } else if (type == "onehot") {
            if (chordmeta)
                chordmeta->type = kotekan::uint8;
        } else {
            ERROR("unexpected type: {:s}", type);
            throw std::runtime_error("unexpected type: " + type);
        }

        // this needs the decoded type
        // could be moved into constructor, but need the bit of code above
        /* new style array description */
        const std::vector<ptrdiff_t> extents(_array_shape.begin(), _array_shape.end());
        const std::vector<kotekan::Symbol> dimnames(_dim_name.begin(), _dim_name.end());

        buf->allocate_ndarray_frame_desc(chordmeta->type, _name, extents, dimnames);
        /* test that things are consistent */
        chordmeta->check_frame_desc(buf->get_ndarray_frame_desc());

        if (type == "onehot") {
            int val = value;
            if (_value_array.size())
                val = _value_array[frame_id_abs % _value_array.size()];
            bzero(frame, n_to_set);
            std::string istring = "";
            size_t j = 0;
            std::vector<int> indices;
            for (size_t i = 0; i < _array_shape.size(); i++) {
                int n = _array_shape[i];
                int k = rng() % n;
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
                INFO("Set {:s}[{:d}] frame counter {:d}", buf->buffer_name, frame_id, frame_id_abs);
            } else {
                ERROR("Metadata type is not one-hot, not recording one-hot indices anywhere!");
            }
            DEBUG("PY onehot[{:d}] = (({:s}), 0x{:x})", frame_id_abs, istring, val);
            n_to_set = 0;
        }

        if (_value_array.size()
            && ((type == "const") || (type == "const_offset") || (type == "const8")
                || (type == "const1x8") || (type == "const16") || (type == "const32")))
            // Cycle through "values" array, if given
            value = _value_array[frame_id_abs % _value_array.size()];
        if (_lvalue_array.size() && type == "const64")
            lvalue = _lvalue_array[frame_id_abs % _lvalue_array.size()];
        if (_lvalue_array.size() && type == "constu64")
            ulvalue = _ulvalue_array[frame_id_abs % _ulvalue_array.size()];
        for (uint j = 0; j < n_to_set; ++j) {
            if (type == "const") {
                if (finished_seeding_constant)
                    break;
                frame[j] = value;
            } else if (type == "const_offset") {
                if (finished_seeding_constant)
                    break;
                frame[j] = value;
            } else if (type == "const8" || type == "const1x8") {
                if (finished_seeding_constant)
                    break;
                frame8[j] = value;
            } else if (type == "const16") {
                if (finished_seeding_constant)
                    break;
                frame16[j] = value;
            } else if (type == "const32") {
                if (finished_seeding_constant)
                    break;
                frame32[j] = value;
#if KOTEKAN_FLOAT16
            } else if (type == "constf16") {
                if (finished_seeding_constant)
                    break;
                framef16[j] = (float16_t)fvalue;
#endif
            } else if (type == "const64") {
                if (finished_seeding_constant)
                    break;
                frame64[j] = lvalue;
            } else if (type == "constu64") {
                if (finished_seeding_constant)
                    break;
                frameu64[j] = ulvalue;
            } else if (type == "ramp") {
                frame[j] = fmod(j * value, 256 * value);
                //                frame[j] = j*value;
            } else if (type == "random") {
                char new_real;
                char new_imaginary;
                if (_reuse_random && finished_seeding_constant)
                    break;
                new_real = (rng() % 15) + 1;      // Limit to [-7, 7]
                new_imaginary = (rng() % 15) + 1; // Limit to [-7, 7]
                temp_output = ((new_real << 4) & 0xF0) + (new_imaginary & 0x0F);
                frame[j] = temp_output;
            } else if (type == "random_signed") {
                char new_real;
                char new_imaginary;
                if (_reuse_random && finished_seeding_constant)
                    break;
                uint32_t r = rng();
                new_real = (r % 15) + 1; // Limit to [-7, 7]
                r >>= 4;
                new_imaginary = (r % 15) + 1; // Limit to [-7, 7]
                temp_output = ((new_real << 4) & 0xF0) + (new_imaginary & 0x0F);
                frame[j] = temp_output ^ 0x88;
            } else if (type == "random_signed_offset") {
                char new_real;
                char new_imaginary;
                if (_reuse_random && finished_seeding_constant)
                    break;
                uint32_t r = rng();
                new_real = (r % 15) + 1; // Limit to [-7, 7]
                r >>= 4;
                new_imaginary = (r % 15) + 1; // Limit to [-7, 7]
                frame[j] = ((new_real << 4) & 0xF0) + (new_imaginary & 0x0F);
            } else if (type == "random1x8") {
                if (_reuse_random && finished_seeding_constant)
                    break;
                uint8_t rand_val = rng() & 0xFFu;
                frame[j] = rand_val;
            } else if (type == "random32") {
                if (_reuse_random && finished_seeding_constant)
                    break;
                frame32[j] = static_cast<int32_t>(rng());
            } else if (type == "random64") {
                if (_reuse_random && finished_seeding_constant)
                    break;
                uint64_t lo = static_cast<uint64_t>(rng());
                uint64_t hi = static_cast<uint64_t>(rng());
                frame64[j] = static_cast<int64_t>((hi << 32) | lo);
            } else if (type == "randomu64") {
                if (_reuse_random && finished_seeding_constant)
                    break;
                uint64_t lo = static_cast<uint64_t>(rng());
                uint64_t hi = static_cast<uint64_t>(rng());
                frameu64[j] = (hi << 32) | lo;
            } else if (type == "random8") {
                if (_reuse_random && finished_seeding_constant)
                    break;
                uint32_t rand_val = rng() % 256;                           // rand in [0, 255]
                int32_t rand_i_val = static_cast<int32_t>(rand_val) - 128; // rand in [-128, 127]
                frame8[j] = static_cast<int8_t>(rand_i_val);
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
            } else {
                ERROR("unexpected type: {:s}", type);
                throw std::runtime_error("unexpected type: " + type);
            }
        }
        DEBUG("Generated a {:s} test data set in {:s}[{:d}] at seq {:d}", type, buf->buffer_name,
              frame_id, seq_num);

        buf->mark_frame_full(unique_name, frame_id);

        frame_id_abs += 1;
        if (num_frames >= 0 && frame_id_abs >= num_frames + _first_frame_index) {
            INFO("Generated the requested number of frames ({:d}) - exiting", num_frames);
            if (_end_interrupt) {
                raise(SIGINT);
            }
            break;
        };
        frame_id = (frame_id_abs - _first_frame_index) % buf->num_frames;

        if (_pathfinder_test_mode) {
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

        // Simulate FPGA restart if requested
        if (_simulate_fpga_restart_at_frame >= 0
            && frame_id_abs == _simulate_fpga_restart_at_frame) {
            INFO("Simulating FPGA restart at frame {:d}: resetting seq_num from {:d} to 0",
                 frame_id_abs, seq_num);
            seq_num = 0;
        }
        if (frame_id == 0) {
            if (_value_array.size() && (_value_array.size() != (size_t)buf->num_frames)) {
                // this "finished_seeding" business does not work
            } else
                finished_seeding_constant = true;
        }

        if (wait) {
            double time = current_time();
            double frame_end_time = start_time + frame_length;
            timer->add_sample(time - start_time);
            if (time < frame_end_time)
                usleep((int)(1e6 * (frame_end_time - time)));
        }
    }
}
