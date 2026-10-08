#include "testDataGenFewHot.hpp"

#include "Config.hpp"          // for Config
#include "DataType.hpp"        // for int4x2_swapped_withoffset
#include "NDArray.hpp"         // for GenericNDArray
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, get_chord_metadata
#include "kotekanLogging.hpp"  // for FATAL_ERROR

#include <assert.h>   // for assert
#include <stdint.h>   // for uint8_t
#include <string.h>   // for memset
#include <sys/time.h> // for gettimeofday, timeval


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(testDataGenFewHot);

testDataGenFewHot::testDataGenFewHot(Config& config, const std::string& unique_name,
                                     bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&testDataGenFewHot::main_thread, this)),
    type(config.get<std::string>(unique_name, "type")),
    freq_id(config.get<std::vector<freq_id_t>>(unique_name, "freq_id")),
    elemns(config.get<std::vector<int>>(unique_name, "elemns")) {

    buf = get_buffer("out_buf");
    buf->register_producer(unique_name);
    assert(type == "fewhot");

    const auto desc = buf->require_frame_desc<kotekan::GenericNDArray>();
    if (desc->get_rank() != 4 || desc->get_value_datatype() != kotekan::int4x2_swapped_withoffset)
        FATAL_ERROR("Buffer {:s} must be a rank-4 (Tbar, Fbar, P, D) int4x2_swapped_withoffset "
                    "ndarray",
                    buf->buffer_name);
    num_times = desc->get_extent(0);
    num_freq = desc->get_extent(1);
    num_elemns = desc->get_extent(2) * desc->get_extent(3);
    upchan_factor = desc->get_dimscaling(0);
    if (upchan_factor < 1 || num_freq % upchan_factor != 0)
        FATAL_ERROR("Fbar extent {:d} is not a multiple of the Tbar dimscaling {:d}", num_freq,
                    upchan_factor);
    if ((ptrdiff_t)freq_id.size() < num_freq / upchan_factor)
        FATAL_ERROR("freq_id has {:d} entries, need at least {:d}", freq_id.size(),
                    num_freq / upchan_factor);

    bool all_el_good = true;
    for (auto const el : elemns) {
        if (el < 0 || el >= num_elemns) {
            all_el_good = false;
            break;
        }
    }
    if (!all_el_good)
        FATAL_ERROR("Elements {:s} must be in allowed range 0 <= el < {:d}",
                    fmt::format("{:s}", fmt::join(elemns, ", ")), num_elemns);
}


void testDataGenFewHot::main_thread() {

    int seq_num = 0;

    while (!stop_thread) {
        const int frame_id = seq_num % buf->num_frames;
        uint8_t* frame = (uint8_t*)buf->wait_for_empty_frame(unique_name, frame_id);
        if (frame == nullptr)
            break;

        buf->allocate_new_metadata_object(frame_id);
        std::shared_ptr<chordMetadata> chordmeta = get_chord_metadata(buf, frame_id);
        assert(chordmeta && "metadata must be of type chordMetadata");

        chordmeta->set_fpga_seq_num(seq_num * num_times * upchan_factor);
        chordmeta->set_time_downsampling_fpga(1);

        // TODO: Fix this, cannot change from frame to frame (and should not be "now")
        struct timeval now;
        gettimeofday(&now, nullptr);
        chordmeta->set_first_packet_recv_time(now);

        chordmeta->set_from_frame_desc(buf->get_frame_desc<kotekan::GenericNDArray>());

        std::vector<int> coarse_freq(num_freq);
        std::vector<int> freq_upchan_factor(num_freq);
        std::vector<int> freq_upchan_index(num_freq);
        for (ptrdiff_t f = 0; f < num_freq; f++) {
            coarse_freq.at(f) = freq_id.at(f / upchan_factor);
            freq_upchan_factor.at(f) = upchan_factor;
            freq_upchan_index.at(f) = f % upchan_factor;
        }
        chordmeta->set_coarse_freq(coarse_freq);
        chordmeta->set_freq_upchan_factor(freq_upchan_factor);
        chordmeta->set_freq_upchan_index(freq_upchan_index);

        chordmeta->set_frame_counter(seq_num);

        std::memset(frame, 0x88 /* 0 volts */, buf->frame_size);
        for (ptrdiff_t i = 0; i < num_times * num_freq; ++i) {
            for (const auto el : elemns) {
                frame[i * num_elemns + el] += 0x10;
            }
        }

        buf->mark_frame_full(unique_name, frame_id);

        seq_num += 1;
    }
}
