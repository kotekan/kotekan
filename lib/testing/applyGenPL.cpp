#include "applyGenPL.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for FATAL_ERROR, INFO

#include "fmt.hpp" // for compile_string_to_view

#include <cstdint>    // for int32_t, uint64_t
#include <cstdlib>    // for abort, size_t
#include <functional> // for bind, function

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(applyGenPL);

applyGenPL::applyGenPL(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&applyGenPL::main_thread, this)) {

    // Apply config.
    _num_elements = config.get<int32_t>(unique_name, "num_elements"); // = "2*D"
    _num_local_freq = config.get<int32_t>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");

    // Check parameter compatibility
    if (_num_elements <= 0) {
        FATAL_ERROR("num_elements ({:d}) is not positive.", _num_elements);
        std::abort();
    }
    if (_num_local_freq <= 0) {
        FATAL_ERROR("num_local_freq ({:d}) is not positive.", _num_local_freq);
        std::abort();
    }
    if (_samples_per_data_set <= 0) {
        FATAL_ERROR("samples_per_data_set ({:d}) is not positive.", _samples_per_data_set);
        std::abort();
    }
    if (_samples_per_data_set % 2 != 0) {
        FATAL_ERROR("samples_per_data_set ({:d}) is not even.", _samples_per_data_set);
        std::abort();
    }
    if (_num_local_freq % 4 != 0) {
        FATAL_ERROR("num_local_freq ({:d}) is not a multiple of 4.", _num_local_freq);
        std::abort();
    }
    if (_num_elements % 8 != 0) {
        FATAL_ERROR("num_elements ({:d}) is not a multiple of 8.", _num_elements);
        std::abort();
    }

    input_buf = get_buffer("voltage_in_buf");
    input_buf->register_consumer(unique_name);
    plmask_buf = get_buffer("plmask_in_buf");
    plmask_buf->register_consumer(unique_name);
    output_buf = get_buffer("voltage_out_buf");
    output_buf->register_producer(unique_name);

    size_t voltage_frame_size =
        _samples_per_data_set * _num_local_freq * _num_elements * sizeof(char);
    size_t pl_frame_size =
        (_samples_per_data_set / 2) * (_num_local_freq / 4) * (_num_elements / 8) / 8;

    if (input_buf->frame_size != voltage_frame_size) {
        FATAL_ERROR("applyGenPL voltage_in_buf ({:s}) has frame size {:d}, expected {:d}.",
                    input_buf->buffer_name, input_buf->frame_size, voltage_frame_size);
        std::abort();
    }
    if (plmask_buf->frame_size != pl_frame_size) {
        FATAL_ERROR("applyGenPL plmask_in_buf ({:s}) has frame size {:d}, expected {:d}.",
                    plmask_buf->buffer_name, plmask_buf->frame_size, voltage_frame_size);
        std::abort();
    }
    if (output_buf->frame_size != voltage_frame_size) {
        FATAL_ERROR("applyGenPL voltage_out_buf ({:s}) has frame size {:d}, expected {:d}.",
                    output_buf->buffer_name, output_buf->frame_size, voltage_frame_size);
        std::abort();
    }
    
    output_buf->allocate_ndarray_frame_desc(kotekan::int4x2_swapped_withoffset,
        "E", {_samples_per_data_set, _num_local_freq, 2, _num_elements / 2},
        {"T", "F", "P", "D"});
}

applyGenPL::~applyGenPL() {}

void applyGenPL::main_thread() {

    int input_frame_id = 0;
    int plmask_frame_id = 0;
    int output_frame_id = 0;

    while (!stop_thread) {
        char* input = (char*)input_buf->wait_for_full_frame(unique_name, input_frame_id);
        if (input == nullptr)
            break;
        uint64_t* plmask = (uint64_t*)plmask_buf->wait_for_full_frame(unique_name, plmask_frame_id);
        if (plmask == nullptr)
            break;
        char* output = (char*)output_buf->wait_for_empty_frame(unique_name, output_frame_id);
        if (output == nullptr)
            break;

        INFO("Applying PL for {:s}[{:d}] and {:s}[{:d}] putting result in {:s}[{:d}]",
             input_buf->buffer_name, input_frame_id, plmask_buf->buffer_name, plmask_frame_id,
             output_buf->buffer_name, output_frame_id);


        int fstride = _num_elements;
        int tstride = _num_local_freq * fstride;
        int fstride_pl = _num_elements / 8;
        int tstride_pl = (_num_local_freq / 4) * fstride_pl;

        for (int t = 0; t < _samples_per_data_set; t++) {
            for (int f = 0; f < _num_local_freq; f++) {
                for (int e = 0; e < _num_elements; e++) {
                    int pl_t = t / 2;           // downsample time by 2
                    int pl_t_bit = pl_t & 0x3F; // last 6 bits to index into a
                                                // uint64_t
                    int pl_t_slow = pl_t >> 6;  // the rest are for the array
                    int pl_f = f / 4;           // downsample freq by 4
                    int pl_e = e / 8;           // downsample element by 8

                    int pl_idx = tstride_pl * pl_t_slow + fstride_pl * pl_f + pl_e;
                    int pl = (plmask[pl_idx] >> pl_t_bit) & 0x1;

                    int v_idx = t * tstride + f * fstride + e;

                    if (pl)
                        output[v_idx] = input[v_idx];
                    else
                        output[v_idx] = 16*8 + 8; // = 0 in offset-encoding
                } // e
            } // f
        } // tout

        input_buf->pass_metadata(input_frame_id, output_buf, output_frame_id);

        input_buf->mark_frame_empty(unique_name, input_frame_id);
        plmask_buf->mark_frame_empty(unique_name, plmask_frame_id);

        output_buf->mark_frame_full(unique_name, output_frame_id);

        INFO("Apply PL done for {:s}[{:d}] and {:s}[{:d}] result is in {:s}[{:d}]",
             input_buf->buffer_name, input_frame_id, plmask_buf->buffer_name, plmask_frame_id,
             output_buf->buffer_name, output_frame_id);

        input_frame_id = (input_frame_id + 1) % input_buf->num_frames;
        plmask_frame_id = (plmask_frame_id + 1) % plmask_buf->num_frames;
        output_frame_id = (output_frame_id + 1) % output_buf->num_frames;
    }
}
