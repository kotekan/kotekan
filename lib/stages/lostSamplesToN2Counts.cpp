#include "lostSamplesToN2Counts.hpp"

#include "Config.hpp" // for Config
#include "DataType.hpp"
#include "N2Util.hpp"          // for frameID
#include "NDArray.hpp"         // for GenericNDArray, NDArray
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for get_chord_metadata, chordMetadata

#include "json.hpp" // for basic_json, json, iter_impl

#include <memory>
#include <string>
#include <sys/types.h>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;
using nlohmann::json;


REGISTER_KOTEKAN_STAGE(lostSamplesToN2Counts);

// Random helpers for clarity
#define BITS_PER_BYTE 8
// Counts ALWAYS have tile size of 8
#define COUNTS_BLOCK_SIZE 8
#define COUNTS_ELEMENT_DOWNSAMPLE 8


lostSamplesToN2Counts::lostSamplesToN2Counts(Config& config, const std::string& unique_name,
                                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&lostSamplesToN2Counts::main_thread, this)),
    num_elements(config.get<size_t>(unique_name, "num_elements")),
    num_n2k_freq(config.get<size_t>(unique_name, "num_n2k_freq")),
    samples_per_data_set(config.get<size_t>(unique_name, "samples_per_data_set")),
    sub_integration_ntime(config.get<size_t>(unique_name, "sub_integration_ntime")) {

    // Set up for n2 counts buffer. One buffer containing all
    // frequencies on this node
    n2k_counts_buf = get_buffer("n2k_counts_buf");
    n2k_counts_buf->register_producer(unique_name);

    // Register as a consumer for the rfi mask buffer, which should
    // also be a single buffer for all frequencies
    rfi_mask_buf = get_buffer("rfi_mask_buf");
    rfi_mask_buf->register_consumer(unique_name);

    // Register as a consumer for all the lost samples buffers
    json ls_in_bufs = config.get_value(unique_name, "lost_samples_buffers");

    for (json::iterator it = ls_in_bufs.begin(); it != ls_in_bufs.end(); ++it) {
        Buffer* buf = buffer_container.get_buffer(it.value());

        lost_samples_buffers.push_back(buf);
        buf->register_consumer(unique_name);

        if (buf->frame_size != lost_samples_buffers.at(0)->frame_size)
            FATAL_ERROR("Input lost_samples buffers have different frame sizes: {:d} != {:d}",
                        buf->frame_size, lost_samples_buffers.at(0)->frame_size);

        if (buf->num_frames != lost_samples_buffers.at(0)->num_frames)
            FATAL_ERROR("Input lost_samples buffers have different number of frames: {:d} != {:d}",
                        buf->num_frames, lost_samples_buffers.at(0)->num_frames);
    }

    // Check counts size against both inputs
    // lost_samples->frame_size == samples_per_data_set
    // rfi_mask->frame_size == nfreq * samples_per_data_set / 8
    // counts->frame_size == nfreq * subintegrations * _counts_stride * sizeof(int32_t)
    if (samples_per_data_set != lost_samples_buffers.at(0)->frame_size)
        FATAL_ERROR("lost_samples_buf does not have the expected number of samples. Expected {:d}, "
                    "got {:d}",
                    samples_per_data_set, lost_samples_buffers.at(0)->frame_size);

    size_t _nbufs = lost_samples_buffers.size();
    // Can't have a fractional number of frequencies
    // rfi_mask is a bitmap, so frame size is reduced by a factor of 8
    if ((BITS_PER_BYTE * rfi_mask_buf->frame_size) % samples_per_data_set != 0)
        FATAL_ERROR("rfi_mask_buf and lost_samples_buf frame sizes are incompatable. rfi_mask: "
                    "{:d}; lost_samples: {:d}",
                    rfi_mask_buf->frame_size, samples_per_data_set);

    if (num_n2k_freq % _nbufs != 0)
        FATAL_ERROR(
            "Number of lost_samples buffers ({:d}) does not evenly divide total frequencies ({:d})",
            _nbufs, num_n2k_freq);

    // Check rfi make frame size against expectation
    size_t _expected_rfi_frame_size = num_n2k_freq * samples_per_data_set / BITS_PER_BYTE;
    if (rfi_mask_buf->frame_size != _expected_rfi_frame_size)
        FATAL_ERROR("Unexpected frame size for rfi_mask {:d} - expected {:d}",
                    rfi_mask_buf->frame_size, _expected_rfi_frame_size);

    // This is a bit of a misnomer - the lost_samples buffer does not _include_
    // multiple frequencies, but information may be shared by multiple frequencies
    _num_freq_per_lost_samples_buffer = num_n2k_freq / _nbufs;
    // Number of subintegrations - i.e., time dimension of counts
    // Integers, so no remainder
    _num_subintegrations = samples_per_data_set / sub_integration_ntime;
    // Number of instrument elements and the corresponding stride
    // to take in the `counts` array
    _counts_ntiles = (num_elements / (COUNTS_BLOCK_SIZE * COUNTS_ELEMENT_DOWNSAMPLE))
                     * (1 + num_elements / (COUNTS_BLOCK_SIZE * COUNTS_ELEMENT_DOWNSAMPLE)) / 2;
    _counts_stride = _counts_ntiles * COUNTS_BLOCK_SIZE * COUNTS_BLOCK_SIZE;

    // Check counts frame size against expectations. counts has type int32
    size_t _expected_counts_frame_size =
        _num_subintegrations * num_n2k_freq * _counts_stride * sizeof(int32_t);

    if (n2k_counts_buf->frame_size != _expected_counts_frame_size)
        FATAL_ERROR("Unexpected frame sizes for n2_counts {:d} - expected {:d}",
                    n2k_counts_buf->frame_size, _expected_counts_frame_size);

    // Set the frame description
    // The buffer name and axis names are the same as those set in
    // cudaPL1butCorrelator
    n2k_counts_buf->set_frame_desc(
        kotekan::NDArray<kotekan::GetType_t<kotekan::int32>, 5>::describe(
            "n2k_counts",
            {static_cast<long>(_num_subintegrations), static_cast<long>(num_n2k_freq),
             static_cast<long>(_counts_ntiles), COUNTS_BLOCK_SIZE, COUNTS_BLOCK_SIZE},
            {"Tc", "F", "D8Phi", "D8Plo1", "D8Plo2"}));
}

lostSamplesToN2Counts::~lostSamplesToN2Counts() {}

void lostSamplesToN2Counts::main_thread() {

    // Create frame IDs for input and output buffers
    N2::frameID n2k_counts_buf_frame_id(n2k_counts_buf);
    N2::frameID rfi_mask_buf_frame_id(rfi_mask_buf);
    // Get one for each lost_samples buffer
    std::vector<N2::frameID> lost_samples_frame_ids;
    for (size_t iter = 0; iter < lost_samples_buffers.size(); ++iter) {
        lost_samples_frame_ids.emplace_back(lost_samples_buffers.at(iter));
    }

    while (!stop_thread) {
        // Just need one complete frame with all frequencies here
        DEBUG("Waiting on empty n2k_counts frame.");
        int32_t* n2k_count_frame =
            (int32_t*)n2k_counts_buf->wait_for_empty_frame(unique_name, n2k_counts_buf_frame_id);
        if (n2k_count_frame == nullptr)
            break;

        DEBUG("Waiting on full RFI mask frame.");
        uint8_t* rfi_mask_frame =
            (uint8_t*)rfi_mask_buf->wait_for_full_frame(unique_name, rfi_mask_buf_frame_id);
        if (rfi_mask_frame == nullptr)
            break;

        // Get the RFI mask low num samples, which is the last dimension
        std::shared_ptr<chordMetadata> rfi_meta =
            get_chord_metadata(rfi_mask_buf, rfi_mask_buf_frame_id);
        // rfi_mask fine time samples (last dimension). Note that this
        // is in *bytes*, not bits
        size_t _rfi_t_lo = rfi_meta->dim[rfi_meta->dims - 1];
        size_t _rfi_t_lo_bits = BITS_PER_BYTE * _rfi_t_lo;
        // Stride between coarse time samples, also in bytes
        size_t _rfi_f_stride = _rfi_t_lo;
        size_t _rfi_t_hi_stride = num_n2k_freq * _rfi_f_stride;

        // Iterate through the lost sample buffers
        for (size_t fouter = 0; fouter < lost_samples_buffers.size(); ++fouter) {
            auto lost_samples_buf = lost_samples_buffers.at(fouter);
            N2::frameID& lost_samples_frame_id = lost_samples_frame_ids.at(fouter);

            DEBUG("Waiting on full lost_samples frame from buffer {:d}.", fouter);
            uint8_t* lost_samples_frame =
                (uint8_t*)lost_samples_buf->wait_for_full_frame(unique_name, lost_samples_frame_id);
            if (lost_samples_frame == nullptr)
                break;

            // Collect science metadata from lost samples
            const auto lost_samples_meta =
                get_chord_metadata(lost_samples_buf, lost_samples_frame_id);

            if (fouter == 0) {
                n2k_counts_buf->allocate_new_metadata_object(n2k_counts_buf_frame_id);
                std::shared_ptr<chordMetadata> meta =
                    get_chord_metadata(n2k_counts_buf, n2k_counts_buf_frame_id);
                meta->deepCopy(lost_samples_meta);
            } else {
                // Insert per-frequency metadata from the lost samples buffer
                auto meta = get_chord_metadata(n2k_counts_buf, n2k_counts_buf_frame_id);
                const auto lost_samples_coarse_freq = lost_samples_meta->get_coarse_freq();
                auto n2k_counts_coarse_freq = meta->get_coarse_freq();
                n2k_counts_coarse_freq.insert(n2k_counts_coarse_freq.end(),
                                              lost_samples_coarse_freq.begin(),
                                              lost_samples_coarse_freq.end());
                meta->set_coarse_freq(n2k_counts_coarse_freq);

                const auto lost_samples_freq_upchan_factor =
                    lost_samples_meta->get_freq_upchan_factor();
                auto n2k_counts_freq_upchan_factor = meta->get_freq_upchan_factor();
                n2k_counts_freq_upchan_factor.insert(n2k_counts_freq_upchan_factor.end(),
                                                     lost_samples_freq_upchan_factor.begin(),
                                                     lost_samples_freq_upchan_factor.end());
                meta->set_freq_upchan_factor(n2k_counts_freq_upchan_factor);

                const auto lost_samples_freq_upchan_index =
                    lost_samples_meta->get_freq_upchan_index();
                auto n2k_counts_freq_upchan_index = meta->get_freq_upchan_index();
                n2k_counts_freq_upchan_index.insert(n2k_counts_freq_upchan_index.end(),
                                                    lost_samples_freq_upchan_index.begin(),
                                                    lost_samples_freq_upchan_index.end());
                meta->set_freq_upchan_index(n2k_counts_freq_upchan_index);
            }

            // Indices in counts and rfi_mask corresponding to the first
            // frequency in the current lost_samples buffer
            size_t cidx_f = fouter * _num_freq_per_lost_samples_buffer * _counts_stride;
            size_t ridx_f = fouter * _num_freq_per_lost_samples_buffer * _rfi_f_stride;

            // Iterate the frequencies covered by this lost_samples buf
            for (size_t f = 0; f < _num_freq_per_lost_samples_buffer; ++f) {
                // Iterate subintegrations
                for (size_t tau = 0; tau < _num_subintegrations; ++tau) {
                    // Linear time index in the lost_samples buffer
                    size_t t0 = tau * sub_integration_ntime;
                    // Accumulate for this subintegration
                    int32_t _sum = 0;
                    for (size_t ts = t0; ts < t0 + sub_integration_ntime; ++ts) {
                        // base offset (in bytes) + coarse_time_sample * stride (in bytes)
                        // + fine_time_sample (in bytes)
                        size_t tsr = ridx_f + (ts / _rfi_t_lo_bits) * _rfi_t_hi_stride
                                     + (ts % _rfi_t_lo_bits) / BITS_PER_BYTE;
                        // Iterate through individual bits in the rfi mask
                        uint8_t shiftval = ts % BITS_PER_BYTE;
                        // tsr only increments every 8 iterations -
                        // accumulate individual bits
                        _sum += ((lost_samples_frame[ts] & 1u) ^ 1u)
                                & ((rfi_mask_frame[tsr] >> shiftval) & 1u);
                    }
                    // Set the current subintegration and freq in counts. Only
                    // set the 0th value of each stride. If we wanted to set the
                    // entire block, just add an extra loop here
                    size_t idx = cidx_f + tau * num_n2k_freq * _counts_stride;
                    for (size_t ee = 0; ee < _counts_stride; ee++)
                        n2k_count_frame[idx + ee] = _sum;
                }
                // Shift frequency offset
                ridx_f += _rfi_f_stride;
                cidx_f += _counts_stride;
            }
            // Release the current lost_samples frame
            lost_samples_buf->mark_frame_empty(unique_name, lost_samples_frame_id);
            lost_samples_frame_id++;
        }

        // Update metadata from the frame description
        std::shared_ptr<chordMetadata> meta =
            get_chord_metadata(n2k_counts_buf, n2k_counts_buf_frame_id);
        meta->set_from_frame_desc(n2k_counts_buf->get_frame_desc<kotekan::GenericNDArray>());

        // Set additional metadata not handled by the helper, including
        // the name and FPGA time-sample downsampling
        meta->set_name("n2k_counts");
        meta->set_time_downsampling_fpga(sub_integration_ntime);

        // Check that frame desc and metadata match
        meta->check_frame_desc(n2k_counts_buf->get_frame_desc<kotekan::GenericNDArray>());

        // Release the current frames and increment to the next frame ID
        n2k_counts_buf->mark_frame_full(unique_name, n2k_counts_buf_frame_id);
        rfi_mask_buf->mark_frame_empty(unique_name, rfi_mask_buf_frame_id);

        n2k_counts_buf_frame_id++;
        rfi_mask_buf_frame_id++;
    }
}
