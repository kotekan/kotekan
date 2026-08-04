#include "testDataGenFloat.hpp"

#include "CHORDTelescope.hpp"  // for CHORDTelescope
#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for get_chord_metadata, chordMetadata
#include "kotekanLogging.hpp"  // for DEBUG

#include "fmt.hpp" // for compile_string_to_view

#include <assert.h>    // for assert
#include <cmath>       // for fmod
#include <functional>  // for bind, function
#include <memory>      // for __shared_ptr_access, shared_ptr
#include <random>      // for uniform_real_distribution, mt19937
#include <stdint.h>    // for uint32_t, uint64_t
#include <stdlib.h>    // for srand
#include <sys/time.h>  // for gettimeofday, timeval
#include <sys/types.h> // for uint
#include <unistd.h>    // for usleep


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(testDataGenFloat);

testDataGenFloat::testDataGenFloat(Config& config, const std::string& unique_name,
                                   bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&testDataGenFloat::main_thread, this)) {

    buf = get_buffer("network_out_buf");
    buf->register_producer(unique_name);
    type = config.get<std::string>(unique_name, "type");
    assert(type == "const" || type == "random" || type == "ramp");
    if (type == "const" || type == "ramp")
        value = config.get<float>(unique_name, "value");
    if (type == "random") {
        seed = config.get<int>(unique_name, "seed");
        _rand_min = config.get_default<float>(unique_name, "rand_min", 0);
        _rand_max = config.get_default<float>(unique_name, "rand_max", 1);
    }
    _pathfinder_test_mode = config.get_default<bool>(unique_name, "pathfinder_test_mode", false);
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    _first_frame_index = config.get_default<uint32_t>(unique_name, "first_frame_index", 0);
    _gen_all_const_data = config.get_default<bool>(unique_name, "gen_all_const_data", false);

    _name = config.get_default<std::string>(unique_name, "name", "E");

    _value_type = kotekan::string_to_type(
        config.get_default<std::string>(unique_name, "value_type", "float32"));
    if (!(_value_type == kotekan::float16 || _value_type == kotekan::float32
          || _value_type == kotekan::float64)) {
        FATAL_ERROR(
            "value_type must be on of {:s}, {:s}, or {:s}, but got {:s}",
            kotekan::type_to_string(kotekan::float16), kotekan::type_to_string(kotekan::float32),
            kotekan::type_to_string(kotekan::float64), kotekan::type_to_string(_value_type));
    }

    const int type_size = kotekan::type_total_bytes(_value_type);
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

    // TODO: rename this parameter to `num_freq_per_stream` in the config
    _num_freq_in_frame = config.get_default<size_t>(unique_name, "num_local_freq", 1);
    _manual_freq_ids = config.get_default<std::vector<uint32_t>>(unique_name, "manual_freq_ids",
                                                                 std::vector<uint32_t>());

    _meta_time_downsample_factor =
        config.get_default<int>(unique_name, "meta_time_downsample_factor", 1);
}

testDataGenFloat::~testDataGenFloat() {}

void testDataGenFloat::main_thread() {

    int frame_id = 0;
    int frame_id_abs = 0;
    void* frame = nullptr;
    uint64_t seq_num = _samples_per_data_set * _first_frame_index;
    bool finished_seeding_consant = false;
    static struct timeval now;

    int link_id = 0;

    auto& telescope = Telescope::instance();
    std::string telescope_type = telescope.get_name();

    while (!stop_thread) {
        frame = (void*)buf->wait_for_empty_frame(unique_name, frame_id);
        if (frame == nullptr)
            break;

        buf->allocate_new_metadata_object(frame_id);
        std::shared_ptr<chordMetadata> chordmeta = get_chord_metadata(buf, frame_id);
        chordmeta->set_fpga_seq_num(seq_num);

        // TODO: Fix this, cannot change from frame to frame (and should not be "now")
        gettimeofday(&now, nullptr);
        chordmeta->set_first_packet_recv_time(now);

        chordmeta->set_name(_name);
        chordmeta->dims = (int)_array_shape.size();
        for (int d = 0; d < chordmeta->dims; ++d)
            chordmeta->set_array_dimension(d, _array_shape[d], _dim_name[d], _dim_scaling[d]);
        chordmeta->set_strides_simple();

        chordmeta->type = _value_type;

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

        // this needs the decoded type
        // could be moved into constructor, but need the bit of code above
        /* new style array description */
        const std::vector<ptrdiff_t> extents(_array_shape.begin(), _array_shape.end());
        const std::vector<kotekan::Symbol> dimnames(_dim_name.begin(), _dim_name.end());
        const std::vector<ptrdiff_t> dimscalings(_dim_scaling.begin(), _dim_scaling.end());

        buf->ensure_frame_desc(kotekan::GenericNDArray::describe(chordmeta->type, _name, extents,
                                                                 dimnames, dimscalings));
        /* test that things are consistent */
        chordmeta->check_frame_desc(buf->get_frame_desc<kotekan::GenericNDArray>());

        // std::random_device rd;
        // std::mt19937 gen(rd());
        // std::uniform_int_distribution<> dis(0, 255);
        if (type == "random")
            srand(seed);
        const uint type_size = kotekan::type_total_bytes(_value_type);
        for (uint j = 0; j < buf->frame_size / type_size; ++j) {
            float fvalue;
            if (type == "const") {
                if (finished_seeding_consant)
                    break;
                fvalue = value;
            } else if (type == "ramp") {
                fvalue = fmod(j * value, 256 * value);
            } else if (type == "random") {
                // Generate a random float between _rand_min and _rand_max
                static std::mt19937 eng(seed);
                static std::uniform_real_distribution<float> dis(_rand_min, _rand_max);
                fvalue = dis(eng);
            } else {
                throw std::runtime_error("Unexpected type " + type);
            }
            switch (_value_type) {
                case kotekan::float16:
                    static_cast<float16_t*>(frame)[j] = static_cast<float16_t>(fvalue);
                    break;
                case kotekan::float32:
                    static_cast<float*>(frame)[j] = static_cast<float>(fvalue);
                    break;
                case kotekan::float64:
                    static_cast<double*>(frame)[j] = static_cast<double>(fvalue);
                    break;
                default:
                    throw std::runtime_error("Unexpected kotekan type: "
                                             + kotekan::type_to_string(_value_type));
            }
        }
        usleep(83000);
        DEBUG("Generated a {:s} test data set in {:s}[{:d}]", type, buf->buffer_name, frame_id);

        buf->mark_frame_full(unique_name, frame_id);

        frame_id_abs += 1;
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
