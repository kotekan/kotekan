#include "integratePowerStream.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for Buffer, mark_frame_empty, mark_frame_full, register_consumer
#include "bufferContainer.hpp" // for bufferContainer
#include "powerStreamUtil.hpp" // for IntensityPacketHeader

#include <atomic>      // for atomic_bool
#include <exception>   // for exception
#include <functional>  // for _Bind_helper<>::type, bind, function
#include <regex>       // for match_results<>::_Base_type
#include <stdint.h>    // for uint8_t
#include <stdlib.h>    // for malloc, calloc
#include <string.h>    // for memcpy, memset
#include <string>      // for string, allocator
#include <sys/types.h> // for uint
#include <vector>      // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(integratePowerStream);

integratePowerStream::integratePowerStream(Config& config, const std::string& unique_name,
                                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&integratePowerStream::main_thread, this)) {

    in_buf = get_buffer("in_buf");
    out_buf = get_buffer("out_buf");
    register_consumer(in_buf, unique_name.c_str());
    register_producer(out_buf, unique_name.c_str());

    integration_length = config.get<int>(unique_name, "integration");
}

integratePowerStream::~integratePowerStream() {}

void integratePowerStream::main_thread() {
    int in_buf_id = 0;
    uint8_t* in_frame = nullptr;
    int out_buf_id = 0;
    uint8_t* out_frame = nullptr;
    uint packet_length = freqs * sizeof(float) + sizeof(IntensityPacketHeader);
    uint packets_per_buffer = in_buf->frame_size / packet_length;

    void* packet_in = malloc(packet_length);
    IntensityPacketHeader* packet_header = (IntensityPacketHeader*)packet_in;
    float* data_in = (float*)((char*)packet_in + sizeof(IntensityPacketHeader));

    int* integrated_samples = (int*)calloc(elems, sizeof(int));
    void* accum_buffer = malloc(packet_length * elems);
    IntensityPacketHeader* accum_header;
    float* data_out;

    // IntensityPacketHeader *header_in;
    while (!stop_thread) {
        in_frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_buf_id);
        if (in_frame == nullptr)
            break;

        for (uint i = 0; i < packets_per_buffer; i++) {
            memcpy(packet_in, in_frame + packet_length * i, packet_length);
            int e = packet_header->elem_idx;

            integrated_samples[e]++;
            data_out =
                (float*)((char*)accum_buffer + e * packet_length + sizeof(IntensityPacketHeader));
            for (int f = 0; f < freqs; f++)
                data_out[f] += data_in[f];
            accum_header = (IntensityPacketHeader*)((char*)accum_buffer + e * packet_length);
            accum_header->samples_summed += packet_header->samples_summed;

            if (integrated_samples[e] >= integration_length) {
                //                INFO("Integrated sample! {:d}", integrated_samples[e]);
                integrated_samples[e] = 0;
                out_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_buf_id);
                if (out_frame == nullptr)
                    goto end_loop;

                memcpy(out_frame, (char*)accum_buffer + e * packet_length, packet_length);

                mark_frame_full(out_buf, unique_name.c_str(), out_buf_id);
                out_buf_id = (out_buf_id + 1) % out_buf->num_frames;
                memset((char*)accum_buffer + e * packet_length, 0, packet_length);
            }
        }
        mark_frame_empty(in_buf, unique_name.c_str(), in_buf_id);
        in_buf_id = (in_buf_id + 1) % in_buf->num_frames;
    }
end_loop:;
}
