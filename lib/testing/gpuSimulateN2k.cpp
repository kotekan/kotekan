#include "gpuSimulateN2k.hpp"

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
#include <stdlib.h>   // for free, malloc
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(gpuSimulateN2k);

gpuSimulateN2k::gpuSimulateN2k(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&gpuSimulateN2k::main_thread, this)) {

    // Apply config.
    _num_elements = config.get<int32_t>(unique_name, "num_elements");
    _num_local_freq = config.get<int32_t>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");
    _sub_integration_ntime = config.get<int>(unique_name, "sub_integration_ntime");

    input_buf = get_buffer("network_in_buf");
    register_consumer(input_buf, unique_name.c_str());
    output_buf = get_buffer("corr_out_buf");
    register_producer(output_buf, unique_name.c_str());
}

gpuSimulateN2k::~gpuSimulateN2k() {}

void gpuSimulateN2k::main_thread() {

    int input_frame_id = 0;
    int output_frame_id = 0;

    while (!stop_thread) {
        char* input = (char*)wait_for_full_frame(input_buf, unique_name.c_str(), input_frame_id);
        if (input == nullptr)
            break;
        int* output = (int*)wait_for_empty_frame(output_buf, unique_name.c_str(), output_frame_id);
        if (output == nullptr)
            break;

        INFO("Simulating GPU processing for {:s}[{:d}] putting result in {:s}[{:d}]",
             input_buf->buffer_name, input_frame_id, output_buf->buffer_name, output_frame_id);

        int nt_inner = _sub_integration_ntime;
        int n_outer = _samples_per_data_set / nt_inner;

        for (int tout = 0; tout < n_outer; ++tout) {
            for (int f = 0; f < _num_local_freq; ++f) {
                for (int y = 0; y < _num_elements; ++y) {
                    for (int x = 0; x < _num_elements; ++x) {

                        int real = 0;
                        int imag = 0;

                        if (x <= y)
                            for (int tin = 0; tin < nt_inner; ++tin) {

                                int t = tout * nt_inner + tin;
                                int ix = (t * _num_local_freq + f) * _num_elements + x;
                                int iy = (t * _num_local_freq + f) * _num_elements + y;

                                int xi = ((input[ix] + 8) & 0xf) - 8;
                                int xr = (((input[ix] >> 4) + 8) & 0xf) - 8;
                                int yi = ((input[iy] + 8) & 0xf) - 8;
                                int yr = (((input[iy] >> 4) + 8) & 0xf) - 8;
                                real += xr * yr + xi * yi;
                                imag += xi * yr - yi * xr;
                            }

                        output[(((tout * _num_local_freq + f) * _num_elements + x) * _num_elements
                                + y)
                                   * 2
                               + 0] = +real;
                        output[(((tout * _num_local_freq + f) * _num_elements + x) * _num_elements
                                + y)
                                   * 2
                               + 1] = -imag;
                    }
                }
                DEBUG("Done t_outer {:d} of {:d} (freq {:d} of {:d})...", tout, n_outer, f,
                      _num_local_freq);
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
