#include "fakeGpuBuffer.hpp"
#include "errors.h"
#include <time.h>
#include <sys/time.h>
#include <csignal>
#include <unistd.h>
#include <random>
#include "fpga_header_functions.h"
#include "chimeMetadata.h"
#include "visUtil.hpp"


fakeGpuBuffer::fakeGpuBuffer(Config& config,
                         const string& unique_name,
                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&fakeGpuBuffer::main_thread, this)) {

    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    freq = config.get_int(unique_name, "freq");
    cadence = config.get_float_default(unique_name, "cadence", 5.0);

    pre_accumulate = config.get_bool_default(unique_name, "pre_accumulate", true);

    if(pre_accumulate) {
        samples_per_data_set = config.get_int(unique_name, "samples_per_data_set");
    }
    block_size = config.get_int(unique_name, "block_size");
    num_elements = config.get_int(unique_name, "num_elements");
    num_frames = config.get_int_default(unique_name, "num_frames", -1);

    wait = config.get_bool_default(unique_name, "wait", true);

    // Fill out the map with the fill patterns
    fill_map["block"] = &fakeGpuBuffer::fill_pattern_block;
    fill_map["accumulate"] = &fakeGpuBuffer::fill_pattern_accumulate;
    fill_map["gaussian"] = &fakeGpuBuffer::fill_pattern_gaussian;

    // Fetch the correct fill function
    std::string pattern = config.get_string(unique_name, "pattern");
    fill = fill_map[pattern];
}

fakeGpuBuffer::~fakeGpuBuffer() {
}

void fakeGpuBuffer::apply_config(uint64_t fpga_seq) {
}

void fakeGpuBuffer::main_thread() {

    int frame_count = 0, frame_id = 0;
    timeval tv;
    timespec ts;

    uint64_t delta_seq, delta_ns;
    uint64_t fpga_seq = 0;

    // This encoding of the stream id should ensure that bin_number_chime gives
    // back the original frequency ID when it is called later.
    // NOTE: all elements must have values < 16 for this encoding to work out.
    stream_id_t s = {0, (uint8_t)(freq % 16), (uint8_t)(freq / 16), (uint8_t)(freq / 256)};

    // Set the start time
    clock_gettime(CLOCK_REALTIME, &ts);

    // Calculate the increment in time between samples
    if(pre_accumulate) {
        delta_seq = samples_per_data_set;
        delta_ns = samples_per_data_set * 2560;
    } else {
        delta_seq = 1;
        delta_ns = (uint64_t)(cadence * 1000000000);
    }

    // Get the amount of time we need to sleep for.
    timespec delta_ts;
    delta_ts.tv_sec = delta_ns / 1000000000;
    delta_ts.tv_nsec = delta_ns % 1000000000;

    while(!stop_thread) {
        int32_t * output = (int *)wait_for_empty_frame(
            out_buf, unique_name.c_str(), frame_id
        );
        if (output == NULL) break;

        DEBUG("Simulating GPU buffer in %s[%d]",
              out_buf->buffer_name, frame_id);

        // Fill the buffer with the specified pattern
        (this->*fill)(output, frame_count);

        allocate_new_metadata_object(out_buf, frame_id);
        set_fpga_seq_num(out_buf, frame_id, fpga_seq);
        set_stream_id_t(out_buf, frame_id, s);

        // Set the two times
        TIMESPEC_TO_TIMEVAL(&tv, &ts);
        set_first_packet_recv_time(out_buf, frame_id, tv);
        set_gps_time(out_buf, frame_id, ts);

        mark_frame_full(out_buf, unique_name.c_str(), frame_id);


        frame_id = (frame_id + 1) % out_buf->num_frames;
        frame_count++;

        // Increment time
        fpga_seq += delta_seq;

        // Increment the timespec
        ts.tv_sec += ((ts.tv_nsec + delta_ns) / 1000000000);
        ts.tv_nsec = (ts.tv_nsec + delta_ns) % 1000000000;

        // Cause kotekan to exit if we've hit the maximum number of frames
        if(num_frames > 0 && frame_count > num_frames) {
            INFO("Reached frame limit [%i frames]. Exiting kotekan...", num_frames);
            std::raise(SIGINT);
            return;
        }
        
        // TODO: only sleep for the extra time required, i.e. account for the
        // elapsed time each loop
        if(this->wait) nanosleep(&delta_ts, nullptr);
    }
}


void fakeGpuBuffer::fill_pattern_block(int32_t* data, int frame_number) {

    int nb1 = num_elements / block_size;
    int num_blocks = nb1 * (nb1 + 1) / 2;

    DEBUG("Block size %i, num blocks %i", block_size, num_blocks);

    for (int b = 0; b < num_blocks; ++b){
        for (int y = 0; y < block_size; ++y){
            for (int x = 0; x < block_size; ++x) {
                int ind = b * block_size * block_size + x + y * block_size;
                data[2 * ind + 0] = y;
                data[2 * ind + 1] = x;
            }
        }
    }
}


void fakeGpuBuffer::fill_pattern_accumulate(int32_t* data, int frame_number) {

    for(int i = 0; i < num_elements; i++) {
        for(int j = i; j < num_elements; j++) {
            uint32_t bi = prod_index(i, j, block_size, num_elements);

            // Every 4th sample the imaginary part is boosted by 4 * samples,
            // but we subtract off a constant to make it average the to be the
            // column index.
            data[2 * bi    ] = (j + 4 * (frame_number % 4 == 0) - 1) * samples_per_data_set;  // Imag

            // ... similar for the real part, except we subtract every 4th
            // frame, and boost by a constant to ensure the average value is the
            // row.
            data[2 * bi + 1] = (i - 4 * ((frame_number + 1) % 4 == 0) + 1) * samples_per_data_set; // Real
        }
    }
}

void fakeGpuBuffer::fill_pattern_gaussian(int32_t* data, int frame_number) {

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> gaussian{0,1};

    float f_auto = pow(samples_per_data_set, 0.5);
    float f_cross = pow(samples_per_data_set / 2, 0.5);

    for(int i = 0; i < num_elements; i++) {
        for(int j = i; j < num_elements; j++) {
            uint32_t bi = prod_index(i, j, block_size, num_elements);

            if(i == j) {
                data[2 * bi + 1] = samples_per_data_set + (int32_t)(f_auto * gaussian(gen));
                data[2 * bi    ] = 0;
            } else {
                data[2 * bi + 1] = (int32_t)(f_cross * gaussian(gen));
                data[2 * bi    ] = (int32_t)(f_cross * gaussian(gen));
            }
        }
    }
}
