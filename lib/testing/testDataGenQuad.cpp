#include "testDataGenQuad.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"            // for Buffer, allocate_new_metadata_object, mark_frame_full
#include "bufferContainer.hpp" // for bufferContainer
#include "chimeMetadata.hpp"   // for set_first_packet_recv_time, set_fpga_seq_num, set_stream_id
#include "kotekanLogging.hpp"  // for INFO

#include <assert.h>   // for assert
#include <atomic>     // for atomic_bool
#include <cstdint>    // for int32_t
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <stddef.h>   // for size_t
#include <sys/time.h> // for gettimeofday, timeval
#include <unistd.h>   // for usleep


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(testDataGenQuad);

testDataGenQuad::testDataGenQuad(Config& config, const std::string& unique_name,
                                 bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&testDataGenQuad::main_thread, this)) {

    buf[0] = get_buffer("out_buf0");
    buf[1] = get_buffer("out_buf1");
    buf[2] = get_buffer("out_buf2");
    buf[3] = get_buffer("out_buf3");
    for (int i = 0; i < 4; i++)
        register_producer(buf[i], unique_name.c_str());

    type = config.get<std::string>(unique_name, "type");
    if (type == "const")
        value = config.get<std::vector<int32_t>>(unique_name, "value");
    assert(type == "const" || type == "random");
}

testDataGenQuad::~testDataGenQuad() {}

void testDataGenQuad::main_thread() {

    int frame_id = 0;
    uint8_t* frame[4] = {nullptr, nullptr, nullptr, nullptr};
    uint64_t seq_num = 0;
    static struct timeval now;

    // pre-seed everything!
    INFO("Seeding...");
    gettimeofday(&now, nullptr);
    for (int b = 0; b < 4; b++) {
        for (int f = 0; f < buf[0]->num_frames; f++) {
            uint8_t v = value[f % value.size()];

            frame[b] = wait_for_empty_frame(buf[b], unique_name.c_str(), f);
            if (frame[b] == nullptr)
                break;

            allocate_new_metadata_object(buf[b], f);
            set_fpga_seq_num(buf[b], f, seq_num);
            // TODO This should be dynamic/config controlled.
            set_stream_id(buf[b], f, {0});
            set_first_packet_recv_time(buf[b], f, now);

            for (size_t j = 0; j < buf[b]->frame_size; ++j)
                frame[b][j] = v;
        }
    }

    // mark the first one full to kick things off
    for (int b = 0; b < 4; b++)
        mark_frame_full(buf[b], unique_name.c_str(), 0);
    INFO("Seeded!");


    while (!stop_thread) {
        INFO("Next seed!");

        for (int i = 0; i < 4; i++) {
            frame[i] = wait_for_empty_frame(buf[i], unique_name.c_str(), frame_id);
            if (frame[i] == nullptr)
                break;
        }

        gettimeofday(&now, nullptr);
        for (int i = 0; i < 4; i++) {
            allocate_new_metadata_object(buf[i], frame_id);
            set_fpga_seq_num(buf[i], frame_id, seq_num);
            // TODO This should be dynamic/config controlled.
            set_stream_id(buf[i], frame_id, {0});
            set_first_packet_recv_time(buf[i], frame_id, now);
        }

        // This is nominally the CHIME pace?
        // Todo: Calculate from buffer lengths, etc.
        usleep(125829);

        for (int i = 0; i < 4; i++) {
            INFO("Skipped a {:s} test data set in {:s}[{:d}]", type, buf[i]->buffer_name, frame_id);
            mark_frame_full(buf[i], unique_name.c_str(), frame_id);
        }

        frame_id = (frame_id + 1) % buf[0]->num_frames;
        seq_num += 32768;
    }
}
