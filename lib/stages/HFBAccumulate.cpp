#include "HFBAccumulate.hpp"

#include "HFBFrameView.hpp"   // for HFBFrameView
#include "HFBMetadata.hpp"    // for get_fpga_seq_start_hfb
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"      // for Telescope, freq_id_t
#include "buffer.h"           // for mark_frame_empty, Buffer, register_consumer, wait_for_empt...
#include "chimeMetadata.hpp"  // for get_lost_timesamples
#include "datasetManager.hpp" // for state_id_t, datasetManager
#include "datasetState.hpp"   // for beamState, freqState, metadataState, subfreqState
#include "kotekanLogging.hpp" // for DEBUG, DEBUG2
#include "version.h"          // for get_git_commit_hash
#include "visUtil.hpp"        // for frameID, modulo, freq_ctype

#include "gsl-lite.hpp" // for span

#include <algorithm>  // for max, fill, transform, copy
#include <atomic>     // for atomic_bool
#include <cstdint>    // for uint32_t, int32_t
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <iterator>   // for back_insert_iterator, begin, end, back_inserter
#include <math.h>     // for pow
#include <numeric>    // for iota
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <string.h>   // for memcpy, size_t
#include <string>     // for string
#include <time.h>     // for timespec
#include <utility>    // for pair
#include <vector>     // for vector, vector<>::iterator

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

void HFBAccumulate::init_first_frame(float* input_data, float* sum_data,
                                     const uint32_t in_frame_id) {

    int64_t fpga_seq_num_start =
        fpga_seq_num_end - (_num_frames_to_integrate - 1) * _samples_per_data_set;
    memcpy(sum_data, input_data, _num_frb_total_beams * _factor_upchan * sizeof(float));
    total_lost_timesamples += get_fpga_seq_start_hfb(in_buf, in_frame_id) - fpga_seq_num_start;
    // Get the first FPGA sequence no. to check for missing frames
    fpga_seq_num = get_fpga_seq_start_hfb(in_buf, in_frame_id);
    frame++;

    DEBUG("\nInit frame. fpga_seq_start: {:d}. sum_data[0]: {:f}",
          get_fpga_seq_start_hfb(in_buf, in_frame_id), sum_data[0]);
}

void HFBAccumulate::integrate_frame(float* input_data, float* sum_data,
                                    const uint32_t in_frame_id) {
    frame++;
    fpga_seq_num += _samples_per_data_set;
    total_lost_timesamples += get_fpga_seq_start_hfb(in_buf, in_frame_id) - fpga_seq_num;
    fpga_seq_num = get_fpga_seq_start_hfb(in_buf, in_frame_id);

    // Integrates data from the input buffer to the output buffer.
    for (uint32_t i = 0; i < _num_frb_total_beams * _factor_upchan; i++) {
        sum_data[i] += input_data[i];
    }

    DEBUG2("\nIntegrate frame {:d}, total_lost_timesamples: {:d}, sum_data[0]: {:f}\n", frame,
           total_lost_timesamples, sum_data[0]);
}

void HFBAccumulate::normalise_frame(float* sum_data, const uint32_t in_frame_id) {

    const float normalise_frac = (float)1.f / (total_timesamples - total_lost_timesamples);

    for (uint32_t i = 0; i < _num_frb_total_beams * _factor_upchan; i++) {
        sum_data[i] *= normalise_frac;
    }

    DEBUG("Integration completed with {:d} lost samples (~{}%). normalise_frac: {}",
          total_lost_timesamples, 100.f * (((float)total_lost_timesamples) / total_timesamples),
          normalise_frac);

    fpga_seq_num = get_fpga_seq_start_hfb(in_buf, in_frame_id);
}

void HFBAccumulate::main_thread() {

    frameID in_frame_id(in_buf), out_frame_id(out_buf), cls_frame_id(cls_buf);
    int first = 1;
    int64_t fpga_seq_num_end_old = 0;

    // Temporary arrays for storing intermediates
    std::vector<float> hfb_even(_num_frb_total_beams * _factor_upchan);
    int32_t samples_even = 0;
    internalState dset = internalState(_num_frb_total_beams, _factor_upchan);

    auto& tel = Telescope::instance();

    total_timesamples = _samples_per_data_set * _num_frames_to_integrate;
    total_lost_timesamples = 0;
    fpga_seq_num = 0;
    fpga_seq_num_end = (_num_frames_to_integrate - 1) * _samples_per_data_set;
    frame = 0;

    if (wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_id) == nullptr)
        return;

    while (!stop_thread) {
        // Get an input buffer. This call is blocking!
        uint8_t* in_frame_ptr = wait_for_full_frame(in_buf, unique_name.c_str(), in_frame_id);
        if (in_frame_ptr == nullptr)
            return;
        if (wait_for_full_frame(cls_buf, unique_name.c_str(), cls_frame_id) == nullptr)
            return;

        float* input = (float*)in_frame_ptr;
        uint64_t frame_count = (get_fpga_seq_num(in_buf, in_frame_id) / _samples_per_data_set);

        // Try and synchronize up the frames. Even though they arrive at
        // different rates, this should eventually sync them up.
        auto hfb_seq_num = get_fpga_seq_start_hfb(in_buf, in_frame_id);
        auto cls_seq_num = get_fpga_seq_num(cls_buf, cls_frame_id);

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

        // Create new metadata for output frame
        allocate_new_metadata_object(out_buf, out_frame_id);
        HFBFrameView::set_metadata(out_buf, out_frame_id, _num_frb_total_beams,
                                   _factor_upchan);

        auto out_frame = HFBFrameView(out_buf, out_frame_id);

        float* sum_data = (float*)out_frame.hfb.data();
        float* input_data = (float*)in_buf->frames[in_frame_id];

        // Find where the end of the integration is
        fpga_seq_num_end = get_fpga_seq_start_hfb(in_buf, in_frame_id)
                           + ((_num_frames_to_integrate * _samples_per_data_set
                               - (get_fpga_seq_start_hfb(in_buf, in_frame_id)
                                  % (_num_frames_to_integrate * _samples_per_data_set)))
                              - _samples_per_data_set);
        if (first) {
            fpga_seq_num_end_old = fpga_seq_num_end;
            first = 0;
        }

        DEBUG2(
            "fpga_seq_start: {:d}, fpga_seq_num_end: {:d}, num_frames * num_samples: {:d}, fpga % "
            "(align): {:d}",
            get_fpga_seq_start_hfb(in_buf, in_frame_id), fpga_seq_num_end,
            _num_frames_to_integrate * _samples_per_data_set,
            get_fpga_seq_start_hfb(in_buf, in_frame_id)
                % (_num_frames_to_integrate * _samples_per_data_set));

        // Get the no. of lost samples in this frame
        int32_t lost_in_frame = get_lost_timesamples(cls_buf, cls_frame_id);
        int32_t samples_in_frame = _samples_per_data_set - lost_in_frame;

        total_lost_timesamples += lost_in_frame;

        // TODO: implement generalised non uniform weighting, I'm primarily
        // not doing this because I don't want to burn cycles doing the
        // multiplications
        // Perform primary accumulation (assume that the weight is one)
        for (size_t i = 0; i < _num_frb_total_beams * _factor_upchan; i++) {
            dset.hfb1[i] += input[i];
        }

        // We are calculating the weights by differencing even and odd samples.
        // Every even sample we save the HFB data...
        if (frame_count % 2 == 0) {
            std::memcpy(hfb_even.data(), input,
                        _num_frb_total_beams * _factor_upchan * sizeof(float));
            samples_even = samples_in_frame;
        }
        // ... every odd sample we accumulate the squared difference into the weight dataset
        // NOTE: this incrementally calculates the variance, but eventually
        // output_frame.weight will hold the *inverse* variance
        // TODO: we might need to account for packet loss in here too, but it
        // would require some awkward rescalings
        else {
            for (size_t i = 0; i < _num_frb_total_beams * _factor_upchan; i++) {
                float d = input[i] - hfb_even[i];
                dset.hfb2[i] += d * d;
            }
            DEBUG2("hfb2[{}]: {}, input[0]: {}, hfb_even[0]: {}", 0, dset.hfb2[0], input[0],
                   hfb_even[0]);

            // Accumulate the squared samples difference which we need for
            // debiasing the variance estimate
            float samples_diff = samples_in_frame - samples_even;
            dset.weight_diff_sum += samples_diff * samples_diff;
        }

        // When all frames have been integrated output the result
        if (get_fpga_seq_start_hfb(in_buf, in_frame_id)
            >= fpga_seq_num_end_old + _samples_per_data_set) {

            DEBUG("fpga_seq_num_end_old: {:d}, fpga_seq_start: {:d}", fpga_seq_num_end_old,
                  fpga_seq_num);
            // Increment the no. of lost frames if there are missing frames
            total_lost_timesamples += fpga_seq_num_end_old - fpga_seq_num;

            const float good_samples_frac =
                (float)(total_timesamples - total_lost_timesamples) / total_timesamples;

            // Normalise data
            normalise_frame(sum_data, in_frame_id);

            // Only output integration if there are enough good samples
            if (good_samples_frac >= _good_samples_threshold) {

                // Populate metadata using HFBFrameView
                int64_t fpga_seq =
                    fpga_seq_num_end_old - ((_num_frames_to_integrate - 1) * _samples_per_data_set);

                out_frame.fpga_seq_start = fpga_seq;
                out_frame.time = tel.to_time(fpga_seq);
                out_frame.fpga_seq_total = total_timesamples - total_lost_timesamples;
                out_frame.fpga_seq_length = total_timesamples;
                out_frame.freq_id = tel.to_freq_id(in_buf, in_frame_id);
                out_frame.dataset_id = ds_id;

                // Set the weights
                float sample_weight_total = total_timesamples - total_lost_timesamples;

                // Debias the weights estimate, by subtracting out the bias estimation
                float w_debias = dset.weight_diff_sum / pow(sample_weight_total, 2);
                for (size_t i = 0; i < _num_frb_total_beams * _factor_upchan; i++) {
                    float d = dset.hfb1[i];
                    dset.hfb2[i] -= w_debias * (d * d);
                }

                // Determine the weighting factors (if weight is zero we should just
                // multiply the HFB data by zero so as not to generate Infs)
                float w = sample_weight_total;

                // Unpack and invert the weights
                for (uint32_t i = 0; i < _num_frb_total_beams * _factor_upchan; i++) {
                    float t = dset.hfb2[i];
                    out_frame.weight[i] = w * w / t;
                }

                DEBUG("Dataset ID: {}, freq ID: {:d}, data: [{:f} ... {:f} ... {:f}], weight: "
                      "[{:f} ... {:f} ... {:f}]",
                      out_frame.dataset_id, out_frame.freq_id, out_frame.hfb[0],
                      out_frame.hfb[_num_frb_total_beams * _factor_upchan / 2],
                      out_frame.hfb[_num_frb_total_beams * _factor_upchan - 1], out_frame.weight[0],
                      out_frame.weight[_num_frb_total_beams * _factor_upchan / 2],
                      out_frame.weight[_num_frb_total_beams * _factor_upchan - 1]);

                mark_frame_full(out_buf, unique_name.c_str(), out_frame_id++);

                // Get a new output buffer
                if (wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_id) == nullptr)
                    return;

                sum_data = (float*)out_buf->frames[out_frame_id];
            } else {
                DEBUG("Integration discarded. Too many lost samples.");
            }

            // Reset the no. of lost samples and frame counter
            total_lost_timesamples = 0;
            frame = 0;

            // Already started next integration
            if (fpga_seq_num > fpga_seq_num_end_old) {
                init_first_frame(input_data, sum_data, in_frame_id);
                reset_state(dset);
            }
        } else {
            // If we are on the first frame copy it directly into the
            // output buffer frame so that we don't need to zero the frame
            if (frame == 0) {
                init_first_frame(input_data, sum_data, in_frame_id);
                reset_state(dset);
            } else
                integrate_frame(input_data, sum_data, in_frame_id);
        }

        fpga_seq_num_end_old = fpga_seq_num_end;

        // Release the input buffers
        mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id++);
        mark_frame_empty(cls_buf, unique_name.c_str(), cls_frame_id++);

    } // end stop thread
}

bool HFBAccumulate::reset_state(HFBAccumulate::internalState& state) {
    // Reset the internal counters
    state.weight_diff_sum = 0;

    // Zero out accumulation arrays
    std::fill(state.hfb1.begin(), state.hfb1.end(), 0.0);
    std::fill(state.hfb2.begin(), state.hfb2.end(), 0.0);

    return true;
}

HFBAccumulate::internalState::internalState(size_t num_beams, size_t num_sub_freqs) :
    hfb1(num_beams * num_sub_freqs),
    hfb2(num_beams * num_sub_freqs) {}
