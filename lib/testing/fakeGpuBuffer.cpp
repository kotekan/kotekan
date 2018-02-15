#include "fakeGpuBuffer.hpp"
#include "errors.h"
#include <time.h>
#include <sys/time.h>
#include <unistd.h>
#include "fpga_header_functions.h"
#include "chimeMetadata.h"

fakeGpuBuffer::fakeGpuBuffer(Config& config,
                         const string& unique_name,
                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&fakeGpuBuffer::main_thread, this)) {

    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    freq = config.get_int(unique_name, "freq");
    cadence = config.get_float_default(unique_name, "cadence", 5.0);

    block_size = config.get_int(unique_name, "block_size");
    int nb1 = config.get_int(unique_name, "num_elements") / block_size;
    num_blocks = nb1 * (nb1 + 1) / 2;

    DEBUG("Block size %i, num blocks %i", block_size, num_blocks);
}

fakeGpuBuffer::~fakeGpuBuffer() {
}

void fakeGpuBuffer::apply_config(uint64_t fpga_seq) {
}

void fakeGpuBuffer::main_thread() {

    int frame_id = 0;
    timeval ts;

    uint64_t fpga_seq = 0;

    // This encoding of the stream id should ensure that bin_number_chime gives
    // back the original frequency ID when it is called later
    stream_id_t s = {0, (uint8_t)(freq % 256), 0, (uint8_t)(freq / 256)};

    while(!stop_thread) {
        int32_t * output = (int *)wait_for_empty_frame(
            out_buf, unique_name.c_str(), frame_id
        );
        if (output == NULL) break;

        INFO("Simulating GPU buffer in %s[%d]",
                out_buf->buffer_name, frame_id);

        // Fill the buffer with some data encoding the blocked structure. This
        // can help with some debugging of the task ordering.
        for (int b = 0; b < num_blocks; ++b){
            for (int y = 0; y < block_size; ++y){
                for (int x = 0; x < block_size; ++x) {
                    int ind = b * block_size * block_size + x + y * block_size;
                    output[2 * ind + 0] = block_size * b + y;
                    output[2 * ind + 1] = block_size * b + x;
                }
            }
        }

        allocate_new_metadata_object(out_buf, frame_id);

        // Set the frame metadata
        gettimeofday(&ts, NULL);
        set_fpga_seq_num(out_buf, frame_id, fpga_seq);
        set_first_packet_recv_time(out_buf, frame_id, ts);
        set_stream_id_t(out_buf, frame_id, s);

        mark_frame_full(out_buf, unique_name.c_str(), frame_id);

        fpga_seq++;

        frame_id = (frame_id + 1) % out_buf->num_frames;
        sleep(cadence);
    }
}
