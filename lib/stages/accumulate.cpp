#include "accumulate.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "Telescope.hpp"       // for stream_t
#include "buffer.hpp"            // for Buffer, allocate_new_metadata_object, mark_frame_empty
#include "bufferContainer.hpp" // for bufferContainer
#include "chimeMetadata.hpp"   // for atomic_add_lost_timesamples, get_lost_timesamples, get_fi...

#include <atomic>     // for atomic_bool
#include <cstdint>    // for int32_t
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <sys/time.h> // for timeval
#include <time.h>     // for timespec
#include <vector>     // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(accumulate);

accumulate::accumulate(Config& config, const std::string& unique_name,
                       bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&accumulate::main_thread, this)) {

    in_buf = get_buffer("in_buf");
    register_consumer(in_buf, unique_name.c_str());
    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");
    _num_gpu_frames = config.get<int32_t>(unique_name, "num_gpu_frames");
}

accumulate::~accumulate() {}

void accumulate::main_thread() {

    int in_frame_id = 0;
    int out_frame_id = 0;
    int64_t frame_id = 0;
    int32_t* input;
    int32_t* output = nullptr;
    //    uint64_t seq_num;

    while (!stop_thread) {
        uint8_t* in_frame = wait_for_full_frame(in_buf, unique_name.c_str(), in_frame_id);
        if (in_frame == nullptr)
            break;
        input = (int32_t*)in_frame;

        //        seq_num = get_fpga_seq_num(in_buf, in_frame_id);

        if (frame_id % _num_gpu_frames == 0) {
            uint8_t* out_frame = wait_for_empty_frame(out_buf, unique_name.c_str(), out_frame_id);
            if (out_frame == nullptr)
                break;
            output = (int32_t*)out_frame;

            allocate_new_metadata_object(out_buf, out_frame_id);

            // Copy values for the metadata into the new metadata object.
            uint64_t fpga_seq = get_fpga_seq_num(in_buf, in_frame_id);
            set_fpga_seq_num(out_buf, out_frame_id, fpga_seq);

            stream_t stream_id = get_stream_id(in_buf, in_frame_id);
            set_stream_id(out_buf, out_frame_id, stream_id);

            timeval time_v = get_first_packet_recv_time(in_buf, in_frame_id);
            set_first_packet_recv_time(out_buf, out_frame_id, time_v);

            timespec time_s = get_gps_time(in_buf, in_frame_id);
            set_gps_time(out_buf, out_frame_id, time_s);

            uint64_t lost_samples = get_lost_timesamples(in_buf, in_frame_id);
            atomic_add_lost_timesamples(out_buf, out_frame_id, lost_samples);

            for (uint32_t i = 0; i < in_buf->frame_size / sizeof(int32_t); ++i) {
                output[i] = input[i];
            }

        } else {
            // Add up the number of lost samples from each input frame.
            uint64_t lost_samples = get_lost_timesamples(in_buf, in_frame_id);
            atomic_add_lost_timesamples(out_buf, out_frame_id, lost_samples);

            for (uint32_t i = 0; i < in_buf->frame_size / sizeof(int32_t); ++i) {
                output[i] += input[i];
            }
        }

        mark_frame_empty(in_buf, unique_name.c_str(), in_frame_id);
        in_frame_id = (in_frame_id + 1) % in_buf->num_frames;
        frame_id++;

        if (frame_id % _num_gpu_frames == 0) {
            mark_frame_full(out_buf, unique_name.c_str(), out_frame_id);
            out_frame_id = (out_frame_id + 1) % out_buf->num_frames;
        }
    }
}
