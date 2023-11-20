#include "GenHFBFrames.hpp"

#include "Config.hpp"         // for Config
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"      // for stream_t
#include "buffer.hpp"         // for allocate_new_metadata_object, mark_frame_full, register_pr...
#include "chimeMetadata.hpp"  // for atomic_add_lost_timesamples, set_fpga_seq_num, set_stream_id
#include "datasetManager.hpp" // for dset_id_t
#include "kotekanLogging.hpp" // for DEBUG, INFO
#include "visUtil.hpp"        // for frameID, modulo

#include <atomic>     // for atomic_bool
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <math.h>     // for sqrt
#include <random>     // for default_random_engine, normal_distribution
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GenHFBFrames);

GenHFBFrames::GenHFBFrames(Config& config, const std::string& unique_name,
                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&GenHFBFrames::main_thread, this)),
    _samples_per_data_set(config.get<uint32_t>(unique_name, "samples_per_data_set")),
    _first_frame_index(config.get_default<uint32_t>(unique_name, "first_frame_index", 0)),
    _rng_mean(config.get<float>(unique_name, "rng_mean")),
    _rng_stddev(config.get_default<float>(unique_name, "rng_stddev", sqrt(_rng_mean))),
    _pattern(config.get<std::string>(unique_name, "pattern")) {

    uint32_t _downsample_time = config.get<uint32_t>(unique_name, "downsample_time");
    uint32_t _factor_upchan = config.get<uint32_t>(unique_name, "factor_upchan");
    _num_samples = _samples_per_data_set / _factor_upchan / _downsample_time;

    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    cls_out_buf = get_buffer("cls_out_buf");
    cls_out_buf->register_producer(unique_name);

    dataset_id = config.get_default<dset_id_t>(
        unique_name, "dataset_id", dset_id_t::from_string("f65bec4949ca616fbeea62660351edcb"));
}

void GenHFBFrames::main_thread() {
    frameID out_frame_id(out_buf), cls_frame_id(cls_out_buf);
    uint64_t seq_num = _samples_per_data_set * _first_frame_index;
    stream_t stream_id;
    stream_id.id = 0;

    std::default_random_engine gen;
    std::normal_distribution<float> gaussian(_rng_mean, _rng_stddev);

    uint32_t num_lost_samples = 0.01 * _num_samples;
    float lost_frac = (float)(_num_samples - num_lost_samples) / (float)_num_samples;
    INFO("num_lost_samples: {}, lost_frac: {:e}", num_lost_samples, lost_frac);

    while (!stop_thread) {
        uint8_t* frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_id);
        if (frame == nullptr)
            break;

        uint8_t* cls_frame = wait_for_empty_frame(cls_out_buf, unique_name.c_str(), cls_frame_id);
        if (cls_frame == nullptr)
            break;

        float* data = (float*)frame;
        uint32_t* cls_data = (uint32_t*)cls_frame;
        uint32_t total_lost_samples = 0;

        for (uint32_t i = 0; i < cls_out_buf->frame_size / sizeof(uint32_t); i++) {
            cls_data[i] = 0;
        }

        // Only drop samples on odd frames if test pattern set
        if (out_frame_id % 2 != 0 && _pattern == "drop") {

            for (uint32_t i = 0; i < out_buf->frame_size / sizeof(float); i++) {

                data[i] = gaussian(gen) * lost_frac;
            }

            // Populate dropped sample mask
            for (uint32_t i = 0; i < num_lost_samples; i++) {
                cls_data[i] = 1;
            }

            total_lost_samples += (int)((float)_samples_per_data_set * (1.f - lost_frac));
        } else {
            for (uint32_t i = 0; i < out_buf->frame_size / sizeof(float); i++) {
                data[i] = gaussian(gen);
            }
        }

        DEBUG("data: [{:f} ... {:f} ... {:f}], lost_samples: {}", data[0], data[131072 / 2],
              data[131072 - 1], total_lost_samples);

        // Create metadata
        allocate_new_metadata_object(out_buf, out_frame_id);
        set_fpga_seq_num(out_buf, out_frame_id, seq_num);
        set_dataset_id(out_buf, out_frame_id, dataset_id);
        set_stream_id(out_buf, out_frame_id, stream_id);

        allocate_new_metadata_object(cls_out_buf, cls_frame_id);
        set_fpga_seq_num(cls_out_buf, cls_frame_id, seq_num);
        zero_lost_samples(cls_out_buf, cls_frame_id);
        atomic_add_lost_timesamples(cls_out_buf, cls_frame_id, total_lost_samples);

        seq_num += _samples_per_data_set;
        mark_frame_full(out_buf, unique_name.c_str(), out_frame_id++);
        mark_frame_full(cls_out_buf, unique_name.c_str(), cls_frame_id++);

    } // end stop thread
}
