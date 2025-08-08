#include "BeamInspect.hpp"

#include "BeamMetadata.hpp"   // for BeamMetadata
#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"      // for Telescope
#include "buffer.hpp"         // for get_metadata, mark_frame_empty, register_consumer, wait_fo...
#include "kotekanLogging.hpp" // for INFO
#include "visUtil.hpp"        // for frameID, modulo

#include "fmt.hpp" // for format

#include <atomic>   // for atomic_bool
#include <stdint.h> // for uint8_t, uint32_t

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(BeamInspect);

STAGE_CONSTRUCTOR(BeamInspect) {
    // Register as consumer on buffer
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
}

BeamInspect::~BeamInspect() {}

void BeamInspect::main_thread() {

    frameID frame_id(in_buf);

    while (!stop_thread) {

        uint8_t* frame = in_buf->wait_for_full_frame(unique_name, frame_id);
        if (frame == nullptr)
            break;

        BeamMetadata* metadata = (BeamMetadata*)(in_buf->get_metadata(frame_id).get());
        const uint32_t num_freq_per_stream = Telescope::instance().num_freq_per_stream();

        std::string frequency_bins = "";
        for (uint32_t f = 0; f < num_freq_per_stream; ++f) {
            frequency_bins +=
                fmt::format("{:d}", Telescope::instance().to_freq_id(metadata->stream_id, f));
            if (f != num_freq_per_stream - 1)
                frequency_bins += ", ";
        }

        INFO("Beam RA: {:f}, Dec: {:f}, scaling: {:d}, freq_bins: {:s}, first value: {:d}+{:d}i",
             metadata->ra, metadata->dec, metadata->scaling, frequency_bins, frame[0] & 0x0F,
             (frame[0] & 0xF0) >> 4);

        // TODO Maybe compute some summary statistics

        in_buf->mark_frame_empty(unique_name, frame_id);
        frame_id++;
    }
}
