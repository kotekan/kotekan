#include "BeamExtract.hpp"
#include "BeamMetadata.hpp"
#include "chimeMetadata.hpp"
#include "buffer.h"
#include "visUtil.hpp"

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(BeamExtract);

BeamExtract::BeamExtract(Config& config_, const std::string& unique_name,
                                     bufferContainer& buffer_container) :
    Stage(config_, unique_name, buffer_container,
          std::bind(&BeamExtract::main_thread, this)) {

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());

    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    _num_beams = config.get<uint32_t>(unique_name, "num_beams");
    _extract_beam = config.get<uint32_t>(unique_name, "extract_beam");
    _samples_per_data_set = config.get<uint32_t>(unique_name, "samples_per_data_set");

    if (_extract_beam >= _num_beams)
        throw std::runtime_error("Cannot extract beam number greater than the number of beams");

    // Check that we have exactly 2 polarizations
    uint32_t num_pol = config.get_default<uint32_t>(unique_name, "num_pol", 2);
    if (num_pol != 2)
        throw std::runtime_error("BeamExtract: Number of polarizations must be 2");
}

void BeamExtract::main_thread() {

    frameID in_frame_id(in_buf);
    frameID out_frame_id(out_buf);
    uint8_t* in_frame;
    uint8_t* out_frame;

    const uint32_t num_pol = 2;

    while(!stop_thread) {

        in_frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_frame_id);
        if (in_frame == nullptr)
            break;

        out_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_id);
        if (out_frame == nullptr)
            break;

        // Copy the single beam we want to the output buffer
        for (int time = 0; time < _samples_per_data_set; ++time) {

            uint32_t out_index = time * num_pol;
            uint32_t in_index = time * _num_beams * num_pol + _extract_beam * 2;

            // Copy both polarizations as 4+4-bit complex numbers.
            out_frame[out_index] = in_frame[in_index];
            out_frame[out_index + 1] = in_frame[in_index + 1];
        }

        // Copy over the relevant metadata
        allocate_new_metadata_object(out_buf, out_frame_id);

        chimeMetadata * in_metadata = (chimeMetadata *)get_metadata(in_frame, in_frame_id);
        BeamMetadata * out_metadata = (BeamMetadata *)get_metadata(out_frame_id, out_frame_id);

        out_metadata->ctime = in_metadata->gps_time;
        out_metadata->fpga_seq_start = in_metadata->fpga_seq_num;
        out_metadata->stream_id = get_stream_id(in_buf, in_frame_id);
        out_metadata->ra = in_metadata->beam_coord.ra[_extract_beam];
        out_metadata->dec = in_metadata->beam_coord.dec[_extract_beam];
        out_metadata->scaling = in_metadata->beam_coord.scaling[_extract_beam];

        mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id);
        in_frame_id++;
        mark_frame_full(out_buf, unique_name.c_str(), out_frame_id);
        out_frame_id++;
    }
}