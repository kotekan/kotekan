#include "gpuSimulateN2k.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"          // for Buffer, mark_frame_empty, mark_frame_full, pass_metadata
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for INFO, DEBUG

#include <atomic>     // for atomic_bool
#include <cstdint>    // for int32_t
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(gpuSimulateN2k);

gpuSimulateN2k::gpuSimulateN2k(Config& config, const std::string& unique_name,
                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&gpuSimulateN2k::main_thread, this)) {

    // Apply config.
    _num_elements = config.get<int32_t>(unique_name, "num_elements"); // = "2*D"
    _num_local_freq = config.get<int32_t>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int32_t>(unique_name, "samples_per_data_set");
    _sub_integration_ntime = config.get<int>(unique_name, "sub_integration_ntime");

    input_buf = get_buffer("network_in_buf");
    input_buf->register_consumer(unique_name);
    output_buf = get_buffer("corr_out_buf");
    output_buf->register_producer(unique_name);
}

gpuSimulateN2k::~gpuSimulateN2k() {}

void gpuSimulateN2k::main_thread() {

    int input_frame_id = 0;
    int output_frame_id = 0;

    while (!stop_thread) {
        char* input = (char*)input_buf->wait_for_full_frame(unique_name, input_frame_id);
        if (input == nullptr)
            break;
        int* output = (int*)output_buf->wait_for_empty_frame(unique_name, output_frame_id);
        if (output == nullptr)
            break;

        INFO("Simulating GPU processing for {:s}[{:d}] putting result in {:s}[{:d}]",
             input_buf->buffer_name, input_frame_id, output_buf->buffer_name, output_frame_id);

        // number of elements = number of dishes * polarizations
        int nt_inner = _sub_integration_ntime;
        int n_outer = _samples_per_data_set / nt_inner;
        int fstride = 128 * _num_elements/16 * (_num_elements/16 + 1);
        int tstride = _num_local_freq * fstride;

        for (int tout = 0; tout < n_outer; ++tout) {
            for (int f = 0; f < _num_local_freq; ++f) {
                // loop through blocks
                for (int jhi = 0; jhi < _num_elements/16; jhi++) {
                    for (int ihi = jhi; ihi < _num_elements/16; ihi++) {
                        for (int jlo = 0; jlo < 16; jlo++) {
                            for (int ilo = 0; ilo < 16; ilo++) {
                                int real = 0;
                                int imag = 0;

                                for (int tin = 0; tin < nt_inner; ++tin) {

                                    int t = tout * nt_inner + tin;
                                    int ix = (t * _num_local_freq + f) * _num_elements + (16*ihi + ilo);
                                    int iy = (t * _num_local_freq + f) * _num_elements + (16*jhi + jlo);

                                    int xi = ((input[ix] + 8) & 0xf) - 8;
                                    int xr = (((input[ix] >> 4) + 8) & 0xf) - 8;
                                    int yi = ((input[iy] + 8) & 0xf) - 8;
                                    int yr = (((input[iy] >> 4) + 8) & 0xf) - 8;
                                    real += xr * yr + xi * yi;
                                    imag += xi * yr - yi * xr;
                                }

                                // clang-format off
                                int o = tout * tstride + f * fstride + 256*(ihi*(ihi+1)/2 + jhi)
                                        + 16*ilo + jlo;
                                output[o + 0] = +real;
                                output[o + 1] = +imag;
                                // clang-format on

                            } // ilo
                        } // jlo
                    } // iji
                } // jhi

                DEBUG("Done t_outer {:d} of {:d} (freq {:d} of {:d})...", tout, n_outer, f,
                      _num_local_freq);
                if(stop_thread)
                    break;
            } // f
            if(stop_thread)
                break;
        } // tout


        INFO("Simulating GPU processing done for {:s}[{:d}] result is in {:s}[{:d}]",
             input_buf->buffer_name, input_frame_id, output_buf->buffer_name, output_frame_id);

        input_buf->pass_metadata(input_frame_id, output_buf, output_frame_id);
        input_buf->mark_frame_empty(unique_name, input_frame_id);
        output_buf->mark_frame_full(unique_name, output_frame_id);

        input_frame_id = (input_frame_id + 1) % input_buf->num_frames;
        output_frame_id = (output_frame_id + 1) % output_buf->num_frames;
    }
}
