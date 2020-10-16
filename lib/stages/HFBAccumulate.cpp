#include "HFBAccumulate.hpp"

#include "HFBFrameView.hpp"   // for HFBFrameView
#include "HFBMetadata.hpp"    // for get_fpga_seq_start_hfb, set_ctime_hfb, set_dataset_id, set...
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"      // for Telescope, freq_id_t
#include "buffer.h"           // for mark_frame_empty, Buffer, register_consumer, wait_for_empt...
#include "chimeMetadata.h"    // for get_lost_timesamples
#include "datasetManager.hpp" // for state_id_t, datasetManager
#include "datasetState.hpp"   // for beamState, freqState, metadataState, subfreqState
#include "kotekanLogging.hpp" // for DEBUG, DEBUG2
#include "version.h"          // for get_git_commit_hash
#include "visUtil.hpp"        // for freq_ctype

#include "gsl-lite.hpp" // for span

#include <algorithm>   // for max, transform, copy
#include <atomic>      // for atomic_bool
#include <cstdint>     // for uint32_t
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, bind, function
#include <iterator>    // for back_insert_iterator, begin, end, back_inserter
#include <numeric>     // for iota
#include <regex>       // for match_results<>::_Base_type
#include <stdexcept>   // for runtime_error
#include <string.h>    // for memcpy
#include <string>      // for string
#include <sys/types.h> // for uint
#include <utility>     // for pair
#include <vector>      // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(HFBAccumulate);

HFBAccumulate::HFBAccumulate(Config& config_, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container, std::bind(&HFBAccumulate::main_thread, this)) {

    auto& tel = Telescope::instance();

    // Apply config.
    _num_frb_total_beams = config.get<uint32_t>(unique_name, "num_frb_total_beams");
    _num_frames_to_integrate =
        config.get_default<uint32_t>(unique_name, "num_frames_to_integrate", 80);
    _factor_upchan = config.get<uint32_t>(unique_name, "factor_upchan");
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    _good_samples_threshold = config.get<float>(unique_name, "good_samples_threshold");

    in_buf = get_buffer("hfb_input_buf");
    register_consumer(in_buf, unique_name.c_str());

    cls_buf = get_buffer("compressed_lost_samples_buf");
    register_consumer(cls_buf, unique_name.c_str());

    out_buf = get_buffer("hfb_output_buf");
    register_producer(out_buf, unique_name.c_str());

    // weight calculation is hardcoded, so is the weight type name
    const std::string weight_type = "inverse_var";
    const std::string git_tag = get_git_commit_hash();
    const std::string instrument_name =
        config.get_default<std::string>(unique_name, "instrument_name", "chime");

    std::vector<uint32_t> freq_ids;

    // Get the frequency IDs that are on this stream, check the config or just
    // assume all CHIME channels
    if (config.exists(unique_name, "freq_ids")) {
        freq_ids = config.get<std::vector<uint32_t>>(unique_name, "freq_ids");
    } else {
        freq_ids.resize(tel.num_freq());
        std::iota(std::begin(freq_ids), std::end(freq_ids), 0);
    }

    // Create the frequency specification
    std::vector<std::pair<uint32_t, freq_ctype>> freqs;
    std::transform(std::begin(freq_ids), std::end(freq_ids), std::back_inserter(freqs),
                   [&tel](uint32_t id) -> std::pair<uint32_t, freq_ctype> {
                       return {id, {tel.to_freq(id), tel.freq_width(id)}};
                   });

    // create all the states
    datasetManager& dm = datasetManager::instance();
    std::vector<state_id_t> base_states;
    base_states.push_back(dm.create_state<freqState>(freqs).first);
    base_states.push_back(dm.create_state<beamState>(_num_frb_total_beams).first);
    base_states.push_back(dm.create_state<subfreqState>(_factor_upchan).first);
    base_states.push_back(
        dm.create_state<metadataState>(weight_type, instrument_name, git_tag).first);

    // register root dataset
    ds_id = dm.add_dataset(base_states);
}

HFBAccumulate::~HFBAccumulate() {}

void HFBAccumulate::init_first_frame(HFBFrameView& in_frame, float* sum_data) {

    int64_t fpga_seq_num_start =
        fpga_seq_num_end - (_num_frames_to_integrate - 1) * _samples_per_data_set;
    memcpy(sum_data, in_frame.hfb.data(), _num_frb_total_beams * _factor_upchan * sizeof(float));
    total_lost_timesamples += in_frame.fpga_seq_start - fpga_seq_num_start;
    // Get the first FPGA sequence no. to check for missing frames
    fpga_seq_num = in_frame.fpga_seq_start;
    frame++;

    DEBUG("\nInit frame. fpga_seq_start: {:d}. sum_data[0]: {:f}",
          in_frame.fpga_seq_start, sum_data[0]);
}

void HFBAccumulate::integrate_frame(HFBFrameView& in_frame, float* sum_data) {
    frame++;
    fpga_seq_num += _samples_per_data_set;
    total_lost_timesamples += in_frame.fpga_seq_start - fpga_seq_num;
    fpga_seq_num = in_frame.fpga_seq_start;

    // Integrates data from the input buffer to the output buffer.
    for (uint32_t i = 0; i < _num_frb_total_beams * _factor_upchan; i++) {
        sum_data[i] += in_frame.hfb.data()[i];
    }

    DEBUG("\nIntegrate frame {:d}, total_lost_timesamples: {:d}, sum_data[0]: {:f}\n", frame,
          total_lost_timesamples, sum_data[0]);
}

void HFBAccumulate::normalise_frame(HFBFrameView& in_frame, float* sum_data) {

    const float normalise_frac =
        (float)total_timesamples / (total_timesamples - total_lost_timesamples);

    for (uint32_t i = 0; i < _num_frb_total_beams * _factor_upchan; i++) {
        sum_data[i] *= normalise_frac;
    }

    DEBUG("Integration completed with {:d} lost samples", total_lost_timesamples);

    fpga_seq_num = in_frame.fpga_seq_start;
}

void HFBAccumulate::main_thread() {

    frameID in_frame_id(in_buf), out_frame_id(out_buf), cls_frame_id(cls_buf);
    int first = 1;
    int64_t fpga_seq_num_end_old;

    auto& tel = Telescope::instance();

    total_timesamples = _samples_per_data_set * _num_frames_to_integrate;
    total_lost_timesamples = 0;
    fpga_seq_num = 0;
    fpga_seq_num_end = (_num_frames_to_integrate - 1) * _samples_per_data_set;
    frame = 0;

    if(wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_id) == nullptr)
        return;
    
    while (!stop_thread) {
        // Get an input buffer. This call is blocking!
        if (wait_for_full_frame(in_buf, unique_name.c_str(), in_frame_id) == nullptr)
            return;
        if (wait_for_full_frame(cls_buf, unique_name.c_str(), cls_frame_id) == nullptr)
            return;

        // Try and synchronize up the frames. Even though they arrive at
        // different rates, this should eventually sync them up.
        auto hfb_seq_num = get_fpga_seq_start_hfb(in_buf, in_frame_id);
        auto cls_seq_num = get_fpga_seq_start_hfb(cls_buf, cls_frame_id);

        if (hfb_seq_num < cls_seq_num) {
            DEBUG("Dropping incoming HFB frame to sync up. HFB frame: {}; Compressed Lost Samples "
                  "frame: {}, diff {}",
                  hfb_seq_num, cls_seq_num, hfb_seq_num - cls_seq_num);
            mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id++);
            continue;
        }
        if (cls_seq_num < hfb_seq_num) {
            DEBUG("Dropping incoming Compressed Lost Samples frame to sync up. HFB frame: {}; "
                  "Compressed Lost Samples frame: {}, diff {}",
                  hfb_seq_num, cls_seq_num, hfb_seq_num - cls_seq_num);
            mark_frame_empty(cls_buf, unique_name.c_str(), cls_frame_id++);
            continue;
        }
        DEBUG2("Frames are synced. HFB frame: {}; Compressed Lost Samples frame: {}, diff {}",
               hfb_seq_num, cls_seq_num, hfb_seq_num - cls_seq_num);

        auto in_frame = HFBFrameView(in_buf, in_frame_id);
        float* sum_data = (float*)out_buf->frames[out_frame_id];

        // Find where the end of the integration is
        fpga_seq_num_end = in_frame.fpga_seq_start
                           + ((_num_frames_to_integrate * _samples_per_data_set
                               - (in_frame.fpga_seq_start
                                  % (_num_frames_to_integrate * _samples_per_data_set)))
                              - _samples_per_data_set);
        if (first) {
            fpga_seq_num_end_old = fpga_seq_num_end;
            first = 0;
        }

        DEBUG(
            "fpga_seq_start: {:d}, fpga_seq_num_end: {:d}, num_frames * num_samples: {:d}, fpga % "
            "(align): {:d}",
            in_frame.fpga_seq_start, fpga_seq_num_end,
            _num_frames_to_integrate * _samples_per_data_set,
            in_frame.fpga_seq_start
                % (_num_frames_to_integrate * _samples_per_data_set));

        // Get the no. of lost samples in this frame
        total_lost_timesamples +=
            get_lost_timesamples(cls_buf, cls_frame_id);

        // When all frames have been integrated output the result
        if (in_frame.fpga_seq_start
            >= fpga_seq_num_end_old + _samples_per_data_set) {

            DEBUG("fpga_seq_num_end_old: {:d}, fpga_seq_start: {:d}", fpga_seq_num_end_old,
                  fpga_seq_num);
            // Increment the no. of lost frames if there are missing frames
            total_lost_timesamples += fpga_seq_num_end_old - fpga_seq_num;

            const float good_samples_frac =
                (float)(total_timesamples - total_lost_timesamples) / total_timesamples;

            // Normalise data
            normalise_frame(in_frame, sum_data);

            // Only output integration if there are enough good samples
            if (good_samples_frac >= _good_samples_threshold) {

                // Create new metadata
                allocate_new_metadata_object(out_buf, out_frame_id);

                // Populate metadata
                int64_t fpga_seq =
                    fpga_seq_num_end_old - ((_num_frames_to_integrate - 1) * _samples_per_data_set);
                set_fpga_seq_start_hfb(out_buf, out_frame_id, fpga_seq);

                // Set GPS time
                set_ctime_hfb(out_buf, out_frame_id, tel.to_time(fpga_seq));

                set_fpga_seq_total(out_buf, out_frame_id,
                                   total_timesamples - total_lost_timesamples);
                set_fpga_seq_length(out_buf, out_frame_id, total_timesamples);

                freq_id_t freq_id = tel.to_freq_id(in_buf, in_frame_id);
                set_freq_id(out_buf, out_frame_id, freq_id);

                set_dataset_id(out_buf, out_frame_id, ds_id);
                set_num_beams(out_buf, out_frame_id, _num_frb_total_beams);
                set_num_subfreq(out_buf, out_frame_id, _factor_upchan);

                DEBUG("Dataset ID: {}, freq ID: {:d}", ds_id, freq_id);

                // Set weights to zero for now
                auto frame = HFBFrameView(out_buf, out_frame_id);
                for (uint32_t i = 0; i < _num_frb_total_beams * _factor_upchan; i++) {
                    frame.weight[i] = 0.0;
                }

                mark_frame_full(out_buf, unique_name.c_str(), out_frame_id++);

                // Get a new output buffer
                if(wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_id) == nullptr)
                  return;
    
                sum_data = (float*)out_buf->frames[out_frame_id];
            } else {
                DEBUG("Integration discarded. Too many lost samples.");
            }

            // Reset the no. of lost samples and frame counter
            total_lost_timesamples = 0;
            frame = 0;

            // Already started next integration
            if (fpga_seq_num > fpga_seq_num_end_old)
                init_first_frame(in_frame, sum_data);

        } else {
            // If we are on the first frame copy it directly into the
            // output buffer frame so that we don't need to zero the frame
            if (frame == 0)
                init_first_frame(in_frame, sum_data);
            else
                integrate_frame(in_frame, sum_data);
        }

        fpga_seq_num_end_old = fpga_seq_num_end;

        // Release the input buffers
        mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id++);
        mark_frame_empty(cls_buf, unique_name.c_str(), cls_frame_id++);

    } // end stop thread
}
