#include "testDataGenFewHot.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata, get_chord_metadata, CHORD_META_MAX_FREQ
#include "kotekanLogging.hpp"  // for INFO, DEBUG, ERROR

#include <assert.h> // for assert
#include <cmath>    // for fmod
#include <stdint.h> // for int8_t, uint32_t, uint8_t, int16_t, int32_t, uint64_t
#include <string.h> // for memset


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(testDataGenFewHot);

#define NUM_ELEMNS 2048
#define NUM_FREQ 1

testDataGenFewHot::testDataGenFewHot(Config& config, const std::string& unique_name,
                                     bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&testDataGenFewHot::main_thread, this)),
    type(config.get<std::string>(unique_name, "type")),
    freq_id(config.get<freq_id_t>(unique_name, "freq_id")),
    elemns(config.get<std::vector<int>>(unique_name, "elemns")),
    samples_per_dataset(config.get<ptrdiff_t>(unique_name, "samples_per_dataset")) {

    buf = get_buffer("out_buf");
    buf->register_producer(unique_name);
    assert(type == "fewhot");

    bool all_el_good = true;
    for (auto const el : elemns) {
        if (!(el < NUM_ELEMNS)) {
            all_el_good = false;
            break;
        }
    }
    if (!all_el_good)
        FATAL_ERROR("Elements {:s} must be in allowed range 0 < el < {:d}",
                    fmt::format("{:s}", fmt::join(elemns, ", ")), NUM_ELEMNS);

    if(buf->frame_size != samples_per_dataset * NUM_ELEMNS * sizeof(kotekan::GetType<kotekan::int4x2_swapped_withoffset>::type))
        FATAL_ERROR("Unexpected frame size: {:d} for {:d} samples of size {:d} for {:d} elements", buf->frame_size, samples_per_dataset, sizeof(kotekan::GetType<kotekan::int4x2_swapped_withoffset>::type), NUM_ELEMNS);


    //_manual_freq_ids = config.get_default<std::vector<uint32_t>>(unique_name, "manual_freq_ids",
    //                                                             std::vector<uint32_t>());
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

        chordmeta->set_fpga_seq_num(seq_num * samples_per_dataset);
        chordmeta->set_time_downsampling_fpga(1);

        // TODO: Fix this, cannot change from frame to frame (and should not be "now")
        struct timeval now;
        gettimeofday(&now, nullptr);
        chordmeta->set_first_packet_recv_time(now);

        chordmeta->set_name("E");
        chordmeta->dims = 3;
        chordmeta->set_array_dimension(0, 1, "F");
        chordmeta->set_array_dimension(1, samples_per_dataset, "T");
        chordmeta->set_array_dimension(2, NUM_ELEMNS, "E");
        chordmeta->set_strides_simple();
        chordmeta->type = kotekan::int4x2_swapped_withoffset;
        assert(NUM_FREQ <= CHORD_META_MAX_FREQ);
        std::vector<int> coarse_freq(NUM_FREQ);
        std::vector<int> freq_upchan_factor(coarse_freq.size());
        std::vector<int> freq_upchan_index(coarse_freq.size());
        assert(NUM_FREQ == 1);
        coarse_freq.at(0) = freq_id;
        freq_upchan_factor.at(0) = 1;
        freq_upchan_index.at(0) = 0;

        chordmeta->set_coarse_freq(coarse_freq);
        chordmeta->set_freq_upchan_factor(freq_upchan_factor);
        chordmeta->set_freq_upchan_index(freq_upchan_index);

        chordmeta->set_frame_counter(seq_num);

        buf->allocate_ndarray_frame_desc<kotekan::GetType<kotekan::int4x2_swapped_withoffset>::type,
                                         3>("E", {1, samples_per_dataset, NUM_ELEMNS},
                                            {"F", "T", "E"});
        /* test that things are consistent */
        chordmeta->check_frame_desc(buf->get_ndarray_frame_desc());

        std::memset(frame, 0x88 /* 0 volts */, samples_per_dataset * NUM_ELEMNS);

        for (int i = 0; i < samples_per_dataset; ++i) {
            for (const auto el : elemns) {
                frame[i * NUM_ELEMNS + el] += 0x10;
            }
        }

        buf->mark_frame_full(unique_name, frame_id);

        seq_num += 1;
    }
}
