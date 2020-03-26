#include "integrateHFBData.hpp"

#include "StageFactory.hpp"        // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"                // for mark_frame_empty, Buffer, register_consumer, wait_for...
#include "chimeMetadata.h"
#include "datasetManager.hpp"      // for state_id_t, datasetManager, dset_id_t
#include "fpga_header_functions.h" // for bin_number_chime, extract_stream_id, stream_id_t
#include "gpsTime.h"
#include "hfbMetadata.hpp"
#include "kotekanLogging.hpp"      // for DEBUG, DEBUG2
#include "version.h"               // for get_git_commit_hash
#include "visUtil.hpp"             // for freq_ctype

#include <atomic>      // for atomic_bool
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, bind, function
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error
#include <string.h>    // for memcpy
#include <string>      // for string
#include <sys/types.h> // for uint
#include <vector>      // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(integrateHFBData);

integrateHFBData::integrateHFBData(Config& config_, const std::string& unique_name,
                                   bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container, std::bind(&integrateHFBData::main_thread, this)) {

    // Apply config.
    _num_frb_total_beams = config.get<uint32_t>(unique_name, "num_frb_total_beams");
    _num_frames_to_integrate =
        config.get_default<uint32_t>(unique_name, "num_frames_to_integrate", 80);
    _factor_upchan = config.get<uint32_t>(unique_name, "factor_upchan");
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    _good_samples_threshold = config.get<float>(unique_name, "good_samples_threshold");

    in_buf = get_buffer("hfb_input_buf");
    register_consumer(in_buf, unique_name.c_str());

    compressed_lost_samples_buf = get_buffer("compressed_lost_samples_buf");
    register_consumer(compressed_lost_samples_buf, unique_name.c_str());

    out_buf = get_buffer("hfb_output_buf");
    register_producer(out_buf, unique_name.c_str());

    // weight calculation is hardcoded, so is the weight type name
    const std::string weight_type = "hfb_weight_type";
    const std::string git_tag = get_git_commit_hash();
    const std::string instrument_name =
        config.get_default<std::string>(unique_name, "instrument_name", "chime");

    std::vector<uint32_t> freq_ids;

    // Get the frequency IDs that are on this stream, check the config or just
    // assume all CHIME channels
    // TODO: CHIME specific
    if (config.exists(unique_name, "freq_ids")) {
        freq_ids = config.get<std::vector<uint32_t>>(unique_name, "freq_ids");
    } else {
        freq_ids.resize(1024);
        std::iota(std::begin(freq_ids), std::end(freq_ids), 0);
    }

    // Create the frequency specification
    // TODO: CHIME specific
    std::vector<std::pair<uint32_t, freq_ctype>> freqs;
    std::transform(std::begin(freq_ids), std::end(freq_ids), std::back_inserter(freqs),
                   [](uint32_t id) -> std::pair<uint32_t, freq_ctype> {
                       return {id, {800.0 - 400.0 / 1024 * id, 400.0 / 1024}};
                   });

    // Create the beam indices
    std::vector<uint32_t> beams;
    beams.resize(_num_frb_total_beams);
    std::iota(std::begin(beams), std::end(beams), 0);
    
    // Create the sub-frequencies specification
    std::vector<std::pair<uint32_t, freq_ctype>> sub_freqs;
    sub_freqs.resize(1024 * _factor_upchan);

    uint32_t index = 0;
    double freq_width = 400.0 / 1024;
    double sub_freq_width = freq_width / _factor_upchan;
    for(auto const &f : freqs) {
        for(uint32_t sub_freq_index = 0; sub_freq_index<_factor_upchan; sub_freq_index++) {
            double sub_freq_centre = (f.second.centre - 0.5 * freq_width) + (sub_freq_index * sub_freq_width) + (0.5 * sub_freq_width);
            sub_freqs.push_back({index++, {sub_freq_centre, sub_freq_width}});
        }
    }

    // create all the states
    datasetManager& dm = datasetManager::instance();
    std::vector<state_id_t> base_states;
    base_states.push_back(dm.create_state<freqState>(freqs).first);
    base_states.push_back(dm.create_state<beamState>(beams).first);
    base_states.push_back(dm.create_state<subfreqState>(sub_freqs).first);
    base_states.push_back(
        dm.create_state<metadataState>(weight_type, instrument_name, git_tag).first);

    // register root dataset
    ds_id = dm.add_dataset(base_states);
}

integrateHFBData::~integrateHFBData() {}

void integrateHFBData::initFirstFrame(float* input_data, float* sum_data,
                                      const uint32_t in_buffer_ID) {

    int64_t fpga_seq_num_start =
        fpga_seq_num_end - (_num_frames_to_integrate - 1) * _samples_per_data_set;
    memcpy(sum_data, input_data, _num_frb_total_beams * _factor_upchan * sizeof(float));
    total_lost_timesamples += get_fpga_seq_num(in_buf, in_buffer_ID) - fpga_seq_num_start;
    // Get the first FPGA sequence no. to check for missing frames
    fpga_seq_num = get_fpga_seq_num(in_buf, in_buffer_ID);
    frame++;

    DEBUG("\nInit frame. fpga_seq_num: {:d}. sum_data[0]: {:f}",
          get_fpga_seq_num(in_buf, in_buffer_ID), sum_data[0]);
}

void integrateHFBData::integrateFrame(float* input_data, float* sum_data,
                                      const uint32_t in_buffer_ID) {
    frame++;
    fpga_seq_num += _samples_per_data_set;
    total_lost_timesamples += get_fpga_seq_num(in_buf, in_buffer_ID) - fpga_seq_num;
    fpga_seq_num = get_fpga_seq_num(in_buf, in_buffer_ID);

    // Integrates data from the input buffer to the output buffer.
    for (uint beam = 0; beam < _num_frb_total_beams; beam++) {
        for (uint freq = 0; freq < _factor_upchan; freq++) {
            sum_data[beam * _factor_upchan + freq] += input_data[beam * _factor_upchan + freq];
        }
    }

    DEBUG("\nIntegrate frame {:d}, total_lost_timesamples: {:d}, sum_data[0]: {:f}\n", frame,
          total_lost_timesamples, sum_data[0]);
}

float integrateHFBData::normaliseFrame(float* sum_data, const uint32_t in_buffer_ID) {

    const float normalise_frac =
        (float)total_timesamples / (total_timesamples - total_lost_timesamples);

    for (uint beam = 0; beam < _num_frb_total_beams; beam++) {
        for (uint freq = 0; freq < _factor_upchan; freq++) {
            sum_data[beam * _factor_upchan + freq] *= normalise_frac;
        }
    }

    DEBUG("Integration completed with {:d} lost samples", total_lost_timesamples);

    fpga_seq_num = get_fpga_seq_num(in_buf, in_buffer_ID);

    return normalise_frac;
}

void integrateHFBData::main_thread() {

    uint in_buffer_ID = 0,
         compress_buffer_ID = 0; // Process only 1 GPU buffer, cycle through buffer depth
    uint8_t* in_frame;
    uint8_t* compressed_lost_samples_frame;
    int out_buffer_ID = 0, first = 1;
    int64_t fpga_seq_num_end_old;
    total_timesamples = _samples_per_data_set * _num_frames_to_integrate;
    total_lost_timesamples = 0;
    fpga_seq_num = 0;
    fpga_seq_num_end = (_num_frames_to_integrate - 1) * _samples_per_data_set;
    frame = 0;

    // Get the first output buffer which will always be id = 0 to start.
    uint8_t* out_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_buffer_ID);
    if (out_frame == nullptr)
        return;

    while (!stop_thread) {
        // Get an input buffer, This call is blocking!
        in_frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_buffer_ID);
        if (in_frame == nullptr)
            return;
        compressed_lost_samples_frame = wait_for_full_frame(
            compressed_lost_samples_buf, unique_name.c_str(), compress_buffer_ID);
        if (compressed_lost_samples_frame == nullptr)
            return;

        // Try and synchronize up the frames. Even though they arrive at
        // different rates, this should eventually sync them up.
        auto hfb_seq_num = get_fpga_seq_num(in_buf, in_buffer_ID);
        auto cls_seq_num = get_fpga_seq_num(compressed_lost_samples_buf, compress_buffer_ID);

        if (hfb_seq_num < cls_seq_num) {
            DEBUG("Dropping incoming HFB frame to sync up. HFB frame: {}; Compressed Lost Samples "
                  "frame: {}, diff {}",
                  hfb_seq_num, cls_seq_num, hfb_seq_num - cls_seq_num);
            mark_frame_empty(in_buf, unique_name.c_str(), in_buffer_ID);
            in_buffer_ID = (in_buffer_ID + 1) % in_buf->num_frames;
            continue;
        }
        if (cls_seq_num < hfb_seq_num) {
            DEBUG("Dropping incoming Compressed Lost Samples frame to sync up. HFB frame: {}; "
                  "Compressed Lost Samples frame: {}, diff {}",
                  hfb_seq_num, cls_seq_num, hfb_seq_num - cls_seq_num);
            mark_frame_empty(compressed_lost_samples_buf, unique_name.c_str(), compress_buffer_ID);
            compress_buffer_ID = (compress_buffer_ID + 1) % compressed_lost_samples_buf->num_frames;
            continue;
        }
        DEBUG2("Frames are synced. HFB frame: {}; Compressed Lost Samples frame: {}, diff {}",
               hfb_seq_num, cls_seq_num, hfb_seq_num - cls_seq_num);

        float* sum_data = (float*)out_buf->frames[out_buffer_ID];
        float* input_data = (float*)in_buf->frames[in_buffer_ID];

        // Find where the end of the integration is
        fpga_seq_num_end = get_fpga_seq_num(in_buf, in_buffer_ID)
                           + ((_num_frames_to_integrate * _samples_per_data_set
                               - (get_fpga_seq_num(in_buf, in_buffer_ID)
                                  % (_num_frames_to_integrate * _samples_per_data_set)))
                              - _samples_per_data_set);
        if (first) {
            fpga_seq_num_end_old = fpga_seq_num_end;
            first = 0;
        }

        DEBUG("fpga_seq_num: {:d}, fpga_seq_num_end: {:d}, num_frames * num_samples: {:d}, fpga % "
              "(align): {:d}",
              get_fpga_seq_num(in_buf, in_buffer_ID), fpga_seq_num_end,
              _num_frames_to_integrate * _samples_per_data_set,
              get_fpga_seq_num(in_buf, in_buffer_ID)
                  % (_num_frames_to_integrate * _samples_per_data_set));

        // Get the no. of lost samples in this frame
        total_lost_timesamples +=
            get_lost_timesamples(compressed_lost_samples_buf, compress_buffer_ID);

        // When all frames have been integrated output the result
        if (get_fpga_seq_num(in_buf, in_buffer_ID)
            >= fpga_seq_num_end_old + _samples_per_data_set) {

            DEBUG("fpga_seq_num_end_old: {:d}, fpga_seq_num: {:d}", fpga_seq_num_end_old,
                  fpga_seq_num);
            // Increment the no. of lost frames if there are missing frames
            total_lost_timesamples += fpga_seq_num_end_old - fpga_seq_num;

            const float good_samples_frac =
                (float)(total_timesamples - total_lost_timesamples) / total_timesamples;

            // Normalise data
            const float norm_frac = normaliseFrame(sum_data, in_buffer_ID);

            // Only output integration if there are enough good samples
            if (good_samples_frac >= _good_samples_threshold) {

                // Create new metadata
                allocate_new_metadata_object(out_buf, out_buffer_ID);

                // Populate metadata
                int64_t fpga_seq =
                    fpga_seq_num_end_old - ((_num_frames_to_integrate - 1) * _samples_per_data_set);
                set_fpga_seq_num_hfb(out_buf, out_buffer_ID, fpga_seq);

                // Check if GPS time is set
                if (!is_gps_global_time_set())
                    set_gps_time_flag(out_buf, out_buffer_ID, 0);
                else {
                    set_gps_time_flag(out_buf, out_buffer_ID, 1);
                    set_gps_time_hfb(out_buf, out_buffer_ID, compute_gps_time(fpga_seq));
                }

                set_norm_frac(out_buf, out_buffer_ID, norm_frac);
                set_num_samples_integrated(out_buf, out_buffer_ID,
                                           total_timesamples - total_lost_timesamples);
                set_num_samples_expected(out_buf, out_buffer_ID, total_timesamples);

                const stream_id_t stream_id =
                    extract_stream_id(get_stream_id(in_buf, in_buffer_ID));
                uint32_t freq_bin_num = bin_number_chime(&stream_id);
                set_freq_bin_num(out_buf, out_buffer_ID, freq_bin_num);

                set_dataset_id(out_buf, out_buffer_ID, ds_id);
                set_num_beams(out_buf, out_buffer_ID, _num_frb_total_beams);
                set_num_subfreq(out_buf, out_buffer_ID, _factor_upchan);

                DEBUG("Dataset ID: {}, freq_bin: {:d}", ds_id, freq_bin_num);

                mark_frame_full(out_buf, unique_name.c_str(), out_buffer_ID);

                // Get a new output buffer
                out_buffer_ID = (out_buffer_ID + 1) % out_buf->num_frames;
                out_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_buffer_ID);
                if (out_frame == nullptr)
                    return;

                sum_data = (float*)out_buf->frames[out_buffer_ID];
            } else {
                DEBUG("Integration discarded. Too many lost samples.");
            }

            // Reset the no. of lost samples and frame counter
            total_lost_timesamples = 0;
            frame = 0;

            // Already started next integration
            if (fpga_seq_num > fpga_seq_num_end_old)
                initFirstFrame(input_data, sum_data, in_buffer_ID);

        } else {
            // If we are on the first frame copy it directly into the
            // output buffer frame so that we don't need to zero the frame
            if (frame == 0)
                initFirstFrame(input_data, sum_data, in_buffer_ID);
            else
                integrateFrame(input_data, sum_data, in_buffer_ID);
        }

        fpga_seq_num_end_old = fpga_seq_num_end;

        // Release the input buffers
        mark_frame_empty(in_buf, unique_name.c_str(), in_buffer_ID);
        mark_frame_empty(compressed_lost_samples_buf, unique_name.c_str(), compress_buffer_ID);
        in_buffer_ID = (in_buffer_ID + 1) % in_buf->num_frames;
        compress_buffer_ID = (compress_buffer_ID + 1) % compressed_lost_samples_buf->num_frames;

    } // end stop thread
}
