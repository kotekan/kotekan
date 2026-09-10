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

// Frame layout: an upchan_U16-like "Ebar" frame, dimensions (Tbar, Fbar, P, D)
constexpr int UPCHAN_FACTOR = 16;
constexpr int NUM_TIMES = 1024;
constexpr int NUM_FREQ = 256;
constexpr int NUM_COARSE_FREQ = NUM_FREQ / UPCHAN_FACTOR;
constexpr int NUM_POL = 2;
constexpr int NUM_ELEMNS = 2048;
constexpr int NUM_DISHES = NUM_ELEMNS / NUM_POL;

testDataGenFewHot::testDataGenFewHot(Config& config, const std::string& unique_name,
                                     bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&testDataGenFewHot::main_thread, this)),
    type(config.get<std::string>(unique_name, "type")),
    freq_id(config.get<std::vector<freq_id_t>>(unique_name, "freq_id")),
    elemns(config.get<std::vector<int>>(unique_name, "elemns")),
    samples_per_dataset(config.get<ptrdiff_t>(unique_name, "samples_per_data_set")) {

    buf = get_buffer("out_buf");
    buf->register_producer(unique_name);
    assert(type == "fewhot");

    bool all_el_good = true;
    for (auto const el : elemns) {
        if (el < 0 || el >= NUM_ELEMNS) {
            all_el_good = false;
            break;
        }
    }
    if (!all_el_good)
        FATAL_ERROR("Elements {:s} must be in allowed range 0 <= el < {:d}",
                    fmt::format("{:s}", fmt::join(elemns, ", ")), NUM_ELEMNS);
    if (freq_id.size() < (size_t)NUM_COARSE_FREQ)
        FATAL_ERROR("freq_id has {:d} entries, need at least {:d}", freq_id.size(),
                    NUM_COARSE_FREQ);

    assert(buf->frame_size
           == NUM_TIMES * NUM_FREQ * NUM_ELEMNS
                  * sizeof(kotekan::GetType<kotekan::int4x2_swapped_withoffset>::type));
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

        chordmeta->set_name("Ebar");
        chordmeta->dims = 4;
        chordmeta->set_array_dimension(0, NUM_TIMES, "Tbar", UPCHAN_FACTOR);
        chordmeta->set_array_dimension(1, NUM_FREQ, "Fbar", 1);
        chordmeta->set_array_dimension(2, NUM_POL, "P", 1);
        chordmeta->set_array_dimension(3, NUM_DISHES, "D", 1);
        chordmeta->set_strides_simple();
        chordmeta->type = kotekan::int4x2_swapped_withoffset;
        std::vector<int> coarse_freq(NUM_FREQ);
        std::vector<int> freq_upchan_factor(coarse_freq.size());
        std::vector<int> freq_upchan_index(coarse_freq.size());
        for (int f = 0; f < NUM_FREQ; f++) {
            coarse_freq.at(f) = freq_id.at(f / UPCHAN_FACTOR);
            freq_upchan_factor.at(f) = UPCHAN_FACTOR;
            freq_upchan_index.at(f) = f % UPCHAN_FACTOR;
        }

        chordmeta->set_coarse_freq(coarse_freq);
        chordmeta->set_freq_upchan_factor(freq_upchan_factor);
        chordmeta->set_freq_upchan_index(freq_upchan_index);

        chordmeta->set_frame_counter(seq_num);

        buf->ensure_frame_desc(kotekan::GenericNDArray::describe(
            kotekan::int4x2_swapped_withoffset, "Ebar",
            std::vector<ptrdiff_t>{NUM_TIMES, NUM_FREQ, NUM_POL, NUM_DISHES},
            std::vector<kotekan::Symbol>{"Tbar", "Fbar", "P", "D"},
            std::vector<ptrdiff_t>{UPCHAN_FACTOR, 1, 1, 1}));
        /* test that things are consistent */
        chordmeta->check_frame_desc(buf->get_frame_desc<kotekan::GenericNDArray>());

        std::memset(frame, 0x88 /* 0 volts */, NUM_TIMES * NUM_FREQ * NUM_ELEMNS);

        for (int i = 0; i < NUM_TIMES * NUM_FREQ; ++i) {
            for (const auto el : elemns) {
                frame[i * NUM_ELEMNS + el] += 0x10;
            }
        }

        buf->mark_frame_full(unique_name, frame_id);

        seq_num += 1;
    }
}
