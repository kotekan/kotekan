#include "gpuSimulate.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for Buffer, mark_frame_empty, mark_frame_full, pass_metadata
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for DEBUG, INFO

#include <assert.h>   // for assert
#include <atomic>     // for atomic_bool
#include <cstdint>    // for int32_t
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <stdio.h>    // for printf
#include <stdlib.h>   // for free, malloc
#include <vector>     // for vector


using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(gpuSimulate);

gpuSimulate::gpuSimulate(Config& config, const std::string& unique_name,
                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&gpuSimulate::main_thread, this)) {

    // Apply config.
    _num_elements = config.get<int32_t>(unique_name, "num_elements");
    _num_local_freq = config.get<int32_t>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");
    _num_blocks = config.get<int32_t>(unique_name, "num_blocks");
    _block_size = config.get<int32_t>(unique_name, "block_size");
    _data_format = config.get_default<std::string>(unique_name, "data_format", "4+4b");

    input_buf = get_buffer("network_in_buf");
    register_consumer(input_buf, unique_name.c_str());
    output_buf = get_buffer("corr_out_buf");
    register_producer(output_buf, unique_name.c_str());

    int block_map_len = _num_blocks * 2 * sizeof(uint32_t);
    host_block_map = (uint32_t*)malloc(block_map_len);
    assert(host_block_map != nullptr);
    int block_id = 0;
    for (int y = 0; block_id < _num_blocks; y++) {
        for (int x = y; x < _num_elements / _block_size; x++) {
            host_block_map[2 * block_id + 0] = x;
            host_block_map[2 * block_id + 1] = y;
            block_id++;
        }
    }
}

gpuSimulate::~gpuSimulate() {
    free(host_block_map);
}

int gpuSimulate::dot4b(uint a, uint b) {
    int sum = 0;
    for (int i = 0; i < 8; i++) {
        int a_s = (a >> (4 * i)) & 0x7;
        a_s -= (a >> (4 * i)) & 0x8;
        int b_s = (b >> (4 * i)) & 0x7;
        b_s -= (b >> (4 * i)) & 0x8;
        sum += a_s * b_s;
    }
    return sum;
}

void gpuSimulate::main_thread() {

    int input_frame_id = 0;
    int output_frame_id = 0;

    while (!stop_thread) {
        char* input = (char*)wait_for_full_frame(input_buf, unique_name.c_str(), input_frame_id);
        if (input == nullptr)
            break;
        int* output = (int*)wait_for_empty_frame(output_buf, unique_name.c_str(), output_frame_id);
        if (output == nullptr)
            break;

        // TODO adjust to allow for more than one frequency.
        // TODO remove all the 32's in here with some kind of constant/define
        INFO("Simulating GPU processing for {:s}[{:d}] putting result in {:s}[{:d}]",
             input_buf->buffer_name, input_frame_id, output_buf->buffer_name, output_frame_id);


        if (_data_format == "4+4b") {
            for (int f = 0; f < _num_local_freq; ++f) {
                for (int b = 0; b < _num_blocks; ++b) {
                    for (int y = 0; y < _block_size; ++y) {
                        for (int x = 0; x < _block_size; ++x) {
                            int real = 0;
                            int imag = 0;
                            for (int t = 0; t < _samples_per_data_set; ++t) {
                                int ix = (t * _num_local_freq + f) * _num_elements
                                         + (host_block_map[2 * b + 0]) * _block_size + x;
                                int xi = (input[ix] & 0x0f) - 8;
                                int xr = ((input[ix] & 0xf0) >> 4) - 8;
                                int iy = (t * _num_local_freq + f) * _num_elements
                                         + (host_block_map[2 * b + 1]) * _block_size + y;
                                int yi = (input[iy] & 0x0f) - 8;
                                int yr = ((input[iy] & 0xf0) >> 4) - 8;
                                real += xr * yr + xi * yi;
                                imag += xi * yr - yi * xr;
                            }
                            output[(f * _num_blocks + b) * _block_size * _block_size * 2 + x * 2
                                   + y * _block_size * 2 + 0] = imag;
                            output[(f * _num_blocks + b) * _block_size * _block_size * 2 + x * 2
                                   + y * _block_size * 2 + 1] = real;
                            // INFO("real: {:d}, imag: {:d}", real, imag);
                        }
                    }
                    DEBUG("Done block {:d} of {:d} (freq {:d} of {:d})...", b, _num_blocks, f,
                          _num_local_freq);
                }
            }
        } else if (_data_format == "dot4b") {
            uint32_t* input_u = (uint32_t*)input;
            for (int f = 0; f < _num_local_freq; ++f) {
                for (int b = 0; b < _num_blocks; ++b) {
                    for (int y = 0; y < _block_size; ++y) {
                        for (int x = 0; x < _block_size; ++x) {
                            int real = 0;
                            int imag = 0;
                            for (int t = 0; t < _samples_per_data_set / 8; t++) {
                                int ix = ((t * _num_local_freq + f) * _num_elements) * 2
                                         + ((host_block_map[2 * b + 0]) * _block_size + x) * 2;
                                int xi = input_u[ix + 0];
                                int xr = input_u[ix + 1];
                                int iy = ((t * _num_local_freq + f) * _num_elements) * 2
                                         + ((host_block_map[2 * b + 1]) * _block_size + y) * 2;
                                int yi = input_u[iy + 0];
                                int yr = input_u[iy + 1];
                                real += dot4b(xr, yr) + dot4b(xi, yi);
                                imag += dot4b(xi, yr) - dot4b(xr, yi);
                            }
                            output[(f * _num_blocks + b) * _block_size * _block_size * 2 + x * 2
                                   + y * _block_size * 2 + 0] = imag;
                            output[(f * _num_blocks + b) * _block_size * _block_size * 2 + x * 2
                                   + y * _block_size * 2 + 1] = real;
                            // INFO("real: {:d}, imag: {:d}", real, imag);
                        }
                    }
                    DEBUG("Done block {:d} of {:d} (freq {:d} of {:d})...", b, _num_blocks, f,
                          _num_local_freq);
                }
            }
        } else if (_data_format == "cuda_wmma") {
            printf("CPU Calc:\n");
            uint32_t* input_u = (uint32_t*)input;
            for (int f = 0; f < _num_local_freq; ++f) {
                for (int b = 0; b < _num_blocks; ++b) {
                    for (int y = 0; y < _block_size; ++y) {
                        for (int x = 0; x < _block_size; ++x) {
                            int real = 0;
                            int imag = 0;
                            for (int t = 0; t < _samples_per_data_set / 32; t++) {
                                for (int tt = 0; tt < 4; tt++) {
                                    int ix = tt + 4 * x
                                             + 4 * 2 * _block_size * host_block_map[2 * b + 0]
                                             + 4 * 2 * _num_elements * f
                                             + 4 * 2 * _num_elements * _num_local_freq * t;
                                    int xi = input_u[ix];
                                    int xr = input_u[ix + _block_size * 4];
                                    int iy = tt + 4 * y
                                             + 4 * 2 * _block_size * host_block_map[2 * b + 1]
                                             + 4 * 2 * _num_elements * f
                                             + 4 * 2 * _num_elements * _num_local_freq * t;
                                    int yi = input_u[iy];
                                    int yr = input_u[iy + _block_size * 4];
                                    real += dot4b(xr, yr) + dot4b(xi, yi);
                                    imag += dot4b(xi, yr) + dot4b(xr, yi); // NOTE: THIS IS WRONG!!!
                                }
                            }
                            output[(f * _num_blocks + b) * _block_size * _block_size * 2 + x
                                   + y * _block_size] = real;
                            output[(f * _num_blocks + b) * _block_size * _block_size * 2 + x
                                   + y * _block_size + _block_size * _block_size] = imag;
                        }
                    }
                    DEBUG("Done block %d of %d (freq %d of %d)...", b, _num_blocks, f,
                          _num_local_freq);
                }
            }
        }

        INFO("Simulating GPU processing done for {:s}[{:d}] result is in {:s}[{:d}]",
             input_buf->buffer_name, input_frame_id, output_buf->buffer_name, output_frame_id);

        pass_metadata(input_buf, input_frame_id, output_buf, output_frame_id);
        mark_frame_empty(input_buf, unique_name.c_str(), input_frame_id);
        mark_frame_full(output_buf, unique_name.c_str(), output_frame_id);

        input_frame_id = (input_frame_id + 1) % input_buf->num_frames;
        output_frame_id = (output_frame_id + 1) % output_buf->num_frames;
    }
}
