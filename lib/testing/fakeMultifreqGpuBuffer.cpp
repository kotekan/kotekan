#include "fakeMultifreqGpuBuffer.hpp"
#include "errors.h"
#include <time.h>
#include <sys/time.h>
#include <csignal>
#include <unistd.h>
#include <random>
#include "fpga_header_functions.h"
#include "chimeMetadata.h"
#include "visUtil.hpp"


REGISTER_KOTEKAN_PROCESS(fakeMultifreqGpuBuffer);

fakeMultifreqGpuBuffer::fakeMultifreqGpuBuffer(Config& config,
                         const string& unique_name,
                         bufferContainer &buffer_container) :
    KotekanProcess(config, unique_name, buffer_container, std::bind(&fakeMultifreqGpuBuffer::main_thread, this)) {

    out_buf = get_buffer("out_buf");
    register_producer(out_buf, unique_name.c_str());

    cadence = config.get_default<float>(unique_name, "cadence", 5.0);

    pre_accumulate = config.get_default<bool>(unique_name, "pre_accumulate", true);

    if(pre_accumulate) {
        samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    }
    block_size = config.get<int>(unique_name, "block_size");
    num_elements = config.get<int>(unique_name, "num_elements");

    if (num_elements < block_size){
        if (block_size % num_elements)
            throw std::runtime_error("num_elements incompatible with block size");
    }

    num_frames = config.get_default<int>(unique_name, "num_frames", -1);
    num_freqs = config.get<int>(unique_name, "num_freqs");
    stream_id = extract_stream_id(0x00ff);//config.get<int>(unique_name,"stream_id"));

    wait = config.get_default<bool>(unique_name, "wait", true);

    // Fill out the map with the fill modes
    fill_map["block"] = &fakeMultifreqGpuBuffer::fill_mode_block;
    fill_map["lostsamples"] = &fakeMultifreqGpuBuffer::fill_mode_lostsamples;
    fill_map["accumulate"] = &fakeMultifreqGpuBuffer::fill_mode_accumulate;
    fill_map["gaussian"] = &fakeMultifreqGpuBuffer::fill_mode_gaussian;

    // Fetch the correct fill function
    std::string mode = config.get<string>(unique_name, "mode");
    fill = fill_map[mode];

    int nb1 = num_elements / block_size;
    int num_blocks = nb1 * (nb1 + 1) / 2;
    blk_x = (int*)malloc(num_blocks * sizeof(int));
    blk_y = (int*)malloc(num_blocks * sizeof(int));
    int i=0;
    for (int y=0; y<num_elements / block_size; y++){
        for (int x=y; x<num_elements / block_size; x++){
            blk_x[i] = x;
            blk_y[i] = y;
            i++;
        }
    }
}

fakeMultifreqGpuBuffer::~fakeMultifreqGpuBuffer() {
    free(blk_x);
    free(blk_y);
}

void fakeMultifreqGpuBuffer::apply_config(uint64_t fpga_seq) {
}

void fakeMultifreqGpuBuffer::main_thread() {

    int frame_count = 0, frame_id = 0;
    timeval tv;
    timespec ts;

    uint64_t delta_seq, delta_ns;
    uint64_t fpga_seq = 0;

    // Set the start time
    clock_gettime(CLOCK_REALTIME, &ts);

    // Calculate the increment in time between samples
    if(pre_accumulate) {
        delta_seq = samples_per_data_set / num_freqs;
        delta_ns = samples_per_data_set * 2560 / num_freqs;
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
        allocate_new_metadata_object(out_buf, frame_id);
        set_fpga_seq_num(out_buf, frame_id, fpga_seq);
        set_stream_id_t(out_buf, frame_id, stream_id);

        // Set the two times
        TIMESPEC_TO_TIMEVAL(&tv, &ts);
        set_first_packet_recv_time(out_buf, frame_id, tv);
        set_gps_time(out_buf, frame_id, ts);

/*KV*/        memset(output,0,out_buf->frame_size);
        // Fill the buffer with the specified mode
        (this->*fill)(output, frame_count,
                      (chimeMetadata *)out_buf->metadata[frame_id]->metadata);

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


void fakeMultifreqGpuBuffer::fill_mode_block(int32_t* data, int frame_number,
                                    chimeMetadata* metadata) {

    int nb1 = num_elements / block_size;
    int num_blocks = nb1 * (nb1 + 1) / 2;

    DEBUG("Block size %i, num blocks %i", block_size, num_blocks);

    if (num_elements < block_size){
        num_blocks = 1;
        for (int f = 0; f < num_freqs; f+=block_size/num_elements){
            for (int b = 0; b < block_size/num_elements; ++b){
                for (int y = 0; y < num_elements; ++y){
                    for (int x = 0; x < num_elements; ++x){
                        int ind = (x+b*num_elements) +
                                  (y+b*num_elements) * block_size +
                                  f/(block_size/num_elements) * block_size * block_size;
                        data[2 * ind + 0] = y;
                        data[2 * ind + 1] = x;
                    }
                }
            }
        }
    }
    else {
        for (int f = 0; f < num_freqs; ++f){
            for (int b = 0; b < num_blocks; ++b){
                for (int y = 0; y < block_size; ++y){
                    for (int x = 0; x < block_size; ++x) {
                        int ind = b * block_size * block_size + x + y * block_size +
                                  f * block_size * block_size * num_blocks;
                        data[2 * ind + 0] = y;
                        data[2 * ind + 1] = x;
                    }
                }
            }
        }
    }
}

void fakeMultifreqGpuBuffer::fill_mode_lostsamples(int32_t* data, int frame_number,
                                          chimeMetadata* metadata) {

    int nb1 = num_elements / block_size;
    int num_blocks = nb1 * (nb1 + 1) / 2;

    uint32_t norm = samples_per_data_set - frame_number;

    if (num_elements < block_size){
        num_blocks = 1;
        for (int f = 0; f < num_freqs; f+=block_size/num_elements){
            for (int b = 0; b < block_size/num_elements; ++b){
                for (int y = 0; y < num_elements; ++y){
                    for (int x = y; x < num_elements; ++x){
                        int ind = (x+b*num_elements) +
                                  (y+b*num_elements) * block_size +
                                  f/(block_size/num_elements) * block_size * block_size;
                        // The visibilities are row + 1j * col, scaled by the total number
                        // of frames.
                        data[2 * ind    ] = x * norm;  // Imag
                        data[2 * ind + 1] = y * norm;  // Real
                    }
                }
            }
        }
    }
    else {
        for (int f = 0; f < num_freqs; ++f){
            for (int b = 0; b < num_blocks; ++b){
                for (int y = 0; y < block_size; ++y){
                    for (int x = 0; x < block_size; ++x) {
                        if ((blk_x[b] == blk_y[b]) && (x<y)) continue;
                        int ind = b * block_size * block_size + x + y * block_size +
                                  f * block_size * block_size * num_blocks;
                        // The visibilities are row + 1j * col, scaled by the total number
                        // of frames.
                        data[2 * ind    ] = x * norm;  // Imag
                        data[2 * ind + 1] = y * norm;  // Real
                    }
                }
            }
        }
    }
    // Every frame has one more lost packet than the last
    metadata->lost_timesamples = frame_number;
}

void fakeMultifreqGpuBuffer::fill_mode_accumulate(int32_t* data, int frame_number,
                                         chimeMetadata* metadata) {

    int nb1 = num_elements / block_size;
    int num_blocks = nb1 * (nb1 + 1) / 2;

    if (num_elements < block_size){
        num_blocks = 1;
        for (int f = 0; f < num_freqs; f+=block_size/num_elements){
            for (int b = 0; b < block_size/num_elements; ++b){
                for (int y = 0; y < num_elements; ++y){
                    for (int x = y; x < num_elements; ++x){
                        int ind = (x+b*num_elements) +
                                  (y+b*num_elements) * block_size +
                                  f/(block_size/num_elements) * block_size * block_size;
                        // Every 4th sample the imaginary part is boosted by 4 * samples,
                        // but we subtract off a constant to make it average the to be the
                        // column index.
                        data[2 * ind    ] = (x + 4 * (frame_number % 4 == 0) - 1) * samples_per_data_set;  // Imag

                        // ... similar for the real part, except we subtract every 4th
                        // frame, and boost by a constant to ensure the average value is the
                        // row.
                        data[2 * ind + 1] = (y - 4 * ((frame_number + 1) % 4 == 0) + 1) * samples_per_data_set; // Real
                    }
                }
            }
        }
    }
    else {
        for (int f = 0; f < num_freqs; ++f){
            for (int b = 0; b < num_blocks; ++b){
                for (int y = 0; y < block_size; ++y){
                    for (int x = 0; x < block_size; ++x) {
                        if ((blk_x[b] == blk_y[b]) && (x<y)) continue;
                        int ind = b * block_size * block_size + x + y * block_size +
                                  f * block_size * block_size * num_blocks;
                        // Every 4th sample the imaginary part is boosted by 4 * samples,
                        // but we subtract off a constant to make it average the to be the
                        // column index.
                        data[2 * ind    ] = (x + 4 * (frame_number % 4 == 0) - 1) * samples_per_data_set;  // Imag

                        // ... similar for the real part, except we subtract every 4th
                        // frame, and boost by a constant to ensure the average value is the
                        // row.
                        data[2 * ind + 1] = (y - 4 * ((frame_number + 1) % 4 == 0) + 1) * samples_per_data_set; // Real
                    }
                }
            }
        }
    }

}

void fakeMultifreqGpuBuffer::fill_mode_gaussian(int32_t* data, int frame_number,
                                       chimeMetadata* metadata) {

    int nb1 = num_elements / block_size;
    int num_blocks = nb1 * (nb1 + 1) / 2;

    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<float> gaussian{0,1};

    float f_auto = pow(samples_per_data_set, 0.5);
    float f_cross = pow(samples_per_data_set / 2, 0.5);


    if (num_elements < block_size){
        num_blocks = 1;
        for (int f = 0; f < num_freqs; f+=block_size/num_elements){
            for (int b = 0; b < block_size/num_elements; ++b){
                for (int y = 0; y < num_elements; ++y){
                    for (int x = y; x < num_elements; ++x){
                        int ind = (x+b*num_elements) +
                                  (y+b*num_elements) * block_size +
                                  f/(block_size/num_elements) * block_size * block_size;
                        if(x == y) {
                            data[2 * ind + 1] = samples_per_data_set + (int32_t)(f_auto * gaussian(gen));
                            data[2 * ind    ] = 0;
                        } else {
                            data[2 * ind + 1] = (int32_t)(f_cross * gaussian(gen));
                            data[2 * ind    ] = (int32_t)(f_cross * gaussian(gen));
                        }
                    }
                }
            }
        }
    }
    else {
        for (int f = 0; f < num_freqs; ++f){
            for (int b = 0; b < num_blocks; ++b){
                for (int y = 0; y < block_size; ++y){
                    for (int x = 0; x < block_size; ++x) {
                        if ((blk_x[b] == blk_y[b]) && (x<y)) continue;
                        int ind = b * block_size * block_size + x + y * block_size +
                                  f * block_size * block_size * num_blocks;
                        if ((blk_x[b] == blk_y[b]) && (x == y)) {
                            data[2 * ind + 1] = samples_per_data_set + (int32_t)(f_auto * gaussian(gen));
                            data[2 * ind    ] = 0;
                        } else {
                            data[2 * ind + 1] = (int32_t)(f_cross * gaussian(gen));
                            data[2 * ind    ] = (int32_t)(f_cross * gaussian(gen));
                        }
                    }
                }
            }
        }
    }
}
