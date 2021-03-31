#include "rfiUpdateMetadata.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for mark_frame_empty, register_consumer, wait_for_full_frame
#include "bufferContainer.hpp" // for bufferContainer
#include "chimeMetadata.hpp"   // for atomic_add_lost_timesamples, atomic_add_rfi_flagged_samples
#include "kotekanLogging.hpp"  // for DEBUG2
#include "visUtil.hpp"         // for frameID, modulo

#include <assert.h>   // for assert
#include <atomic>     // for atomic_bool
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <vector>     // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(rfiUpdateMetadata);

rfiUpdateMetadata::rfiUpdateMetadata(Config& config, const std::string& unique_name,
                                     bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&rfiUpdateMetadata::main_thread, this)) {

    // Register on buffers
    rfi_mask_buf = get_buffer("rfi_mask_buf");
    register_consumer(rfi_mask_buf, unique_name.c_str());
    lost_samples_buf = get_buffer("lost_samples_buf");
    register_consumer(lost_samples_buf, unique_name.c_str());

    // We make ourselves a producer of the GPU correlation buffer so that this stage
    // has to release it before other stages can process it.
    gpu_correlation_buf = get_buffer("gpu_correlation_buf");
    register_producer(gpu_correlation_buf, unique_name.c_str());

    // General config parameters
    _sk_step = config.get<uint32_t>(unique_name, "sk_step");
    _num_sub_frames = config.get<uint32_t>(unique_name, "num_sub_frames");

    uint32_t samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");
    _sub_frame_samples = samples_per_data_set / _num_sub_frames;
    _sub_frame_mask_len = _sub_frame_samples / _sk_step;
}

rfiUpdateMetadata::~rfiUpdateMetadata() {}


void rfiUpdateMetadata::main_thread() {
    frameID rfi_mask_frame_id(rfi_mask_buf);
    frameID lost_samples_frame_id(lost_samples_buf);
    frameID gpu_correlation_frame_id(gpu_correlation_buf);

    while (!stop_thread) {
        uint8_t* rfi_mask_frame =
            wait_for_full_frame(rfi_mask_buf, unique_name.c_str(), rfi_mask_frame_id);
        if (rfi_mask_frame == nullptr)
            break;

        uint8_t* lost_samples_frame =
            wait_for_full_frame(lost_samples_buf, unique_name.c_str(), lost_samples_frame_id);
        if (lost_samples_frame == nullptr)
            break;

        for (uint32_t subframe = 0; subframe < _num_sub_frames; ++subframe) {

            // Note this stage expects the output data frame will already
            // have a copy of the metadata at this point.  This is true only if hsaRfiMaskOutput
            // comes after hsaOutputData, this is not really ideal, but I cannot find an easy way
            // around it.
            uint8_t* gpu_correlation_frame = wait_for_empty_frame(
                gpu_correlation_buf, unique_name.c_str(), gpu_correlation_frame_id);
            if (gpu_correlation_frame == nullptr)
                break;

            // Total number of samples flagged as RFI
            uint32_t flagged_samples = 0;

            // The number of samples flagged as RFI less the number of
            // samples flagged as missing (i.e. packet loss/corrupt packet)
            // in the same time block as an RFI flag.
            uint32_t net_lost_samples = 0;

            // Index of the rfi mask subframe
            uint32_t mask_base_idx = subframe * _sub_frame_mask_len;

            for (uint32_t i = 0; i < _sub_frame_mask_len; ++i) {
                uint32_t mask_idx = mask_base_idx + i;
                assert(mask_idx < (uint32_t)rfi_mask_buf->frame_size);
                if (rfi_mask_frame[mask_idx] == 1) {
                    flagged_samples += _sk_step;
                    net_lost_samples += _sk_step;
                    // Remove any samples which were also counted as lost samples
                    // in this block of data.
                    uint32_t lost_samples_base_idx = subframe * _sub_frame_samples + i * _sk_step;
                    for (uint32_t j = 0; j < _sk_step; ++j) {
                        uint32_t lost_samples_idx = lost_samples_base_idx + j;
                        // assert(lost_samples_idx < (uint32_t)lost_samples_buf->frame_size);
                        if (lost_samples_frame[lost_samples_idx] == 1)
                            net_lost_samples--;
                    }
                }
            }
            DEBUG2("flagged_samples: %d, net_lost_samples: %d", flagged_samples, net_lost_samples);
            atomic_add_rfi_flagged_samples(gpu_correlation_buf, gpu_correlation_frame_id,
                                           flagged_samples);

            // Only add the RFI flagged samples to the count of lost samples if
            // we have actively zeroed out the RFI flagged data.
            if (get_rfi_zeroed(gpu_correlation_buf, gpu_correlation_frame_id)) {
                atomic_add_lost_timesamples(gpu_correlation_buf, gpu_correlation_frame_id,
                                            net_lost_samples);
            }

            mark_frame_full(gpu_correlation_buf, unique_name.c_str(), gpu_correlation_frame_id);
            gpu_correlation_frame_id++;
        }

        mark_frame_empty(lost_samples_buf, unique_name.c_str(), lost_samples_frame_id);
        lost_samples_frame_id++;

        mark_frame_empty(rfi_mask_buf, unique_name.c_str(), rfi_mask_frame_id);
        rfi_mask_frame_id++;
    }
}
