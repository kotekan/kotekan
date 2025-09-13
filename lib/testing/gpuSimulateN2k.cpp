#include "gpuSimulateN2k.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"          // for Buffer, mark_frame_empty, mark_frame_full, pass_metadata
#include "bufferContainer.hpp" // for bufferContainer
#include "chordMetadata.hpp"   // for chordMetadata
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
        int nt_outer = _samples_per_data_set / nt_inner;

        int fstride = 128 * _num_elements / 16 * (_num_elements / 16 + 1);
        int tstride = _num_local_freq * fstride;

        INFO("Running stage with nt_outer={:d}, nt_inner={:d}, _num_local_freq={:d}, "
             "_num_elements={:d}",
             nt_outer, nt_inner, _num_local_freq, _num_elements);

        for (int tout = 0; tout < nt_outer; ++tout) {
            for (int f = 0; f < _num_local_freq; ++f) {
                // loop through blocks
                for (int jhi = 0; jhi < _num_elements / 16; jhi++) {
                    for (int ihi = jhi; ihi < _num_elements / 16; ihi++) {
                        for (int jlo = 0; jlo < 16; jlo++) {
                            for (int ilo = 0; ilo < 16; ilo++) {
                                int real = 0;
                                int imag = 0;

                                for (int tin = 0; tin < nt_inner; ++tin) {
                                    int t = tout * nt_inner + tin;
                                    int ix = (t * _num_local_freq + f) * _num_elements
                                             + (16 * ihi + ilo);
                                    int iy = (t * _num_local_freq + f) * _num_elements
                                             + (16 * jhi + jlo);

                                    /*
                                    int xi = ((input[ix] + 8) & 0xf) - 8;
                                    int xr = (((input[ix] >> 4) + 8) & 0xf) - 8;
                                    int yi = ((input[iy] + 8) & 0xf) - 8;
                                    int yr = (((input[iy] >> 4) + 8) & 0xf) - 8;
                                    */
                                    int xi = (input[ix] & 0x0f) - 8;
                                    int xr = ((input[ix] & 0xf0) >> 4) - 8;
                                    int yi = (input[iy] & 0x0f) - 8;
                                    int yr = ((input[iy] & 0xf0) >> 4) - 8;
                                    real += xr * yr + xi * yi;
                                    imag += xi * yr - yi * xr;
                                }

                                // clang-format off
                                int o = 2*( tout * tstride + f * fstride + 256*(ihi*(ihi+1)/2 + jhi)
                                        + 16*ilo + jlo );
                                output[o + 0] = +real;
                                output[o + 1] = +imag;
                                // clang-format on

                            } // ilo
                        } // jlo
                    } // iji
                } // jhi

                DEBUG("Done t_outer {:d} of {:d} (freq {:d} of {:d}, nt_inner={:d})...", tout,
                      nt_outer, f, _num_local_freq, nt_inner);

                if (stop_thread)
                    break;
            } // f
            if (stop_thread)
                break;
        } // tout

        // input_buf->pass_metadata(input_frame_id, output_buf, output_frame_id);
        output_buf->allocate_new_metadata_object(output_frame_id);
        const std::shared_ptr<metadataObject> mc = output_buf->get_metadata(output_frame_id);
        if (!mc) {
            FATAL_ERROR("Buffer {:s} frame {:d} cannot allocate metadata", output_buf->buffer_name,
                        output_frame_id);
        }
        assert(mc);
        if (!metadata_is_chord(mc)) {
            FATAL_ERROR("Buffer {:s} frame {:d} does not have CHORD metadata",
                        output_buf->buffer_name, output_frame_id);
        }
        assert(metadata_is_chord(mc));
        const std::shared_ptr<chordMetadata> meta_out = get_chord_metadata(mc);
        assert(meta_out);

        meta_out->set_name("cpusim_correlation");
        meta_out->type = kotekan::int32;
        meta_out->dims = 6;
        assert(meta_out->dims <= CHORD_META_MAX_DIM);
        meta_out->set_array_dimension(0, nt_outer, "Tc");
        meta_out->set_array_dimension(1, _num_local_freq, "F");
        meta_out->set_array_dimension(2, (_num_elements / 16) * (_num_elements / 16 + 1) / 2,
                                      "DPhi");
        meta_out->set_array_dimension(3, 16, "DPlo1");
        meta_out->set_array_dimension(4, 16, "DPlo2");
        meta_out->set_array_dimension(5, 2, "C");
        meta_out->set_strides_simple();
        meta_out->nfreq = _num_local_freq;
        assert(meta_out->nfreq <= CHORD_META_MAX_FREQ);
        meta_out->sample0_offset = 0;
        meta_out->offset_downsampling = 1;

        input_buf->mark_frame_empty(unique_name, input_frame_id);

        // Pretend some samples were lost
        // chordMetadata* chord_metadata = (chordMetadata*)
        // output_buf->get_metadata(output_frame_id); chord_metadata->lost_timesamples[0] = 1;

        output_buf->mark_frame_full(unique_name, output_frame_id);

        INFO("Simulating GPU processing done for {:s}[{:d}] result is in {:s}[{:d}]",
             input_buf->buffer_name, input_frame_id, output_buf->buffer_name, output_frame_id);

        input_frame_id = (input_frame_id + 1) % input_buf->num_frames;
        output_frame_id = (output_frame_id + 1) % output_buf->num_frames;
    }
}
