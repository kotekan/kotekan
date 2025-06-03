#include "gpuSimulateCudaBasebandBeamformer.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"          // for Buffer, mark_frame_empty, register_consumer, wait_for_ful...
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for INFO, DEBUG
#include "oneHotMetadata.hpp"  // for metadata_is_onehot, get_onehot_indices, get_onehot_frame_...

#include <array>      // for array
#include <atomic>     // for atomic_bool
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <regex>      // for match_results<>::_Base_type
#include <stddef.h>   // for size_t
#include <stdexcept>  // for runtime_error
#include <vector>     // for vector

using kotekan::bufferContainer;
using kotekan::Config;

REGISTER_KOTEKAN_STAGE(gpuSimulateCudaBasebandBeamformer);

gpuSimulateCudaBasebandBeamformer::gpuSimulateCudaBasebandBeamformer(
    Config& config, const std::string& unique_name, bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&gpuSimulateCudaBasebandBeamformer::main_thread, this)) {
    _num_elements = config.get<int>(unique_name, "num_elements");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _num_beams = config.get<int>(unique_name, "num_beams");
    bool zero_output = config.get_default<bool>(unique_name, "zero_output", false);
    voltage_buf = get_buffer("voltage_in_buf");
    phase_buf = get_buffer("phase_in_buf");
    shift_buf = get_buffer("shift_in_buf");
    voltage_buf->register_consumer(unique_name);
    phase_buf->register_consumer(unique_name);
    shift_buf->register_consumer(unique_name);
    output_buf = get_buffer("beams_out_buf");
    output_buf->register_producer(unique_name);
    if (zero_output)
        output_buf->zero_frames();
}

gpuSimulateCudaBasebandBeamformer::~gpuSimulateCudaBasebandBeamformer() {}

// This code is from Erik's
// https://github.com/eschnett/GPUIndexSpaces.jl/blob/main/kernels/bb.cxx

/**
 CPU implementation of the CUDA Baseband Beamformer kernel.

 This _sub version can handle "one-hot" arrays, where only a single
 value of a given parameter contains non-zero values (eg, a single
 frequency, single time, etc).
 */
void gpuSimulateCudaBasebandBeamformer::bb_simple_sub(
    std::string id_tag, const int8_t* __restrict__ const A, const int4x2_t* __restrict__ const E,
    const int32_t* __restrict__ const s, int4x2_t* __restrict__ const J,
    const int T, // 32768; // number of times
    const int B, // = 96;    // number of beams
    const int D, // = 512;   // number of dishes
    const int F, // = 16;    // frequency channels per GPU
    const int t, const int b, const int d, const int f, const int p) {
    const int f0 = (f == -1 ? 0 : f);
    const int f1 = (f == -1 ? F : f + 1);
    const int p0 = (p == -1 ? 0 : p);
    const int p1 = (p == -1 ? 2 : p + 1);
    const int b0 = (b == -1 ? 0 : b);
    const int b1 = (b == -1 ? B : b + 1);
    const int t0 = (t == -1 ? 0 : t);
    const int t1 = (t == -1 ? T : t + 1);
    const int d0 = (d == -1 ? 0 : d);
    const int d1 = (d == -1 ? D : d + 1);

    int nprint_v = 0;
    int nprint_b = 0;
    int nprint_p = 0;
    const int nprint_max = 2;

    // J[t,p,f,b] = Σ[d] A[d,b,p,f] E[d,p,f,t]
    for (int f = f0; f < f1; ++f) {
        for (int p = p0; p < p1; ++p) {
            for (int t = t0; t < t1; ++t) {
                for (int b = b0; b < b1; ++b) {
                    int Jre = 0, Jim = 0;
                    for (int d = d0; d < d1; ++d) {
                        const int Aim = A[(((f * 2 + p) * B + b) * D + d) * 2 + 0];
                        const int Are = A[(((f * 2 + p) * B + b) * D + d) * 2 + 1];
                        const auto [Eim, Ere] = get4(E[((t * 2 + p) * F + f) * D + d]);
                        Jre += Are * Ere - Aim * Eim;
                        Jim += Are * Eim + Aim * Ere;
                        if (Are || Aim) {
                            size_t indx = (((f * 2 + p) * B + b) * D + d) * 2 + 0;
                            if (nprint_p < nprint_max) {
                                // clang-format off
                                DEBUG("bb_simple: found phase f={:d}, p={:d}, b={:d}, d={:d} = index {:d}=0x{:x} with value {:d} = 0x{:x}",
                                      f, p, b, d, indx, indx, Aim, Are);
                                // clang-format on
                                nprint_p++;
                            }
                        }
                        if (Ere || Eim) {
                            size_t indx = ((t * 2 + p) * F + f) * D + d;
                            if (nprint_v < nprint_max) {
                                // clang-format off
                                DEBUG("bb_simple: found voltage f={:d}, p={:d}, t={:d}, d={:d} = index {:d}=0x{:x} = {:d} = 0x{:x}",
                                      f, p, t, d, indx, indx, E[indx], E[indx]);
                                // clang-format on
                                nprint_v++;
                            }
                        }
                    }
                    int oJre = Jre;
                    int oJim = Jim;
                    int shift = s[(f * 2 + p) * B + b];
                    // rounding
                    Jre += 1 << (shift - 1);
                    Jim += 1 << (shift - 1);
                    Jre >>= shift;
                    Jim >>= shift;
                    if (Jre > 7)
                        Jre = 7;
                    if (Jre < -7)
                        Jre = -7;
                    if (Jim > 7)
                        Jim = 7;
                    if (Jim < -7)
                        Jim = -7;
                    int jindx = ((b * F + f) * 2 + p) * T + t;
                    J[jindx] = set4(Jim, Jre);
                    if (Jre || Jim) {
                        if (nprint_b < nprint_max) {
                            DEBUG("bb_simple: setting b={:d}(0x{:x}), f={:d}(0x{:x}), "
                                  "p={:d}(0x{:x}), t={:d}(0x{:x}) = index 0x{:x}; before shift by "
                                  "{:d} (0x{:x}), re=0x{:x}, im=0x{:x}, after: re=0x{:x}, "
                                  "im=0x{:x}, packed: 0x{:x}",
                                  b, b, f, f, p, p, t, t, jindx, shift, shift, oJre, oJim, Jre, Jim,
                                  set4(Jim, Jre));
                            if (nprint_b == 0)
                                INFO("PY bb[{:s}] = (({:d}, {:d}, {:d}, {:d}, {:d}), ({:d}, {:d}), "
                                     "({:d}, {:d}), 0x{:x})",
                                     id_tag, b, f, p, t, jindx, oJre, oJim, Jre, Jim,
                                     set4(Jim, Jre));
                            nprint_b++;
                        }
                    }
                }
            }
        }
    }
}

void gpuSimulateCudaBasebandBeamformer::bb_simple(
    std::string id_tag, const int8_t* __restrict__ const A, const int4x2_t* __restrict__ const E,
    const int32_t* __restrict__ const s, int4x2_t* __restrict__ const J,
    const int T, // = 32768; // 32768; // number of times
    const int B, // = 96;    // number of beams
    const int D, // = 512;   // number of dishes
    const int F  // = 16;    // frequency channels per GPU
) {
    bb_simple_sub(id_tag, A, E, s, J, T, B, D, F, -1, -1, -1, -1, -1);
}

void gpuSimulateCudaBasebandBeamformer::main_thread() {
    int voltage_frame_id = 0;
    int output_frame_id = 0;

    int phase_frame_id = 0;
    int shift_frame_id = 0;

    while (!stop_thread) {
        int4x2_t* voltage =
            (int4x2_t*)voltage_buf->wait_for_full_frame(unique_name, voltage_frame_id);
        if (voltage == nullptr)
            break;
        int8_t* phase = (int8_t*)phase_buf->wait_for_full_frame(unique_name, phase_frame_id);
        if (phase == nullptr)
            break;
        int32_t* shift = (int32_t*)shift_buf->wait_for_full_frame(unique_name, shift_frame_id);
        if (shift == nullptr)
            break;
        int4x2_t* output =
            (int4x2_t*)output_buf->wait_for_empty_frame(unique_name, output_frame_id);
        if (output == nullptr)
            break;

        INFO("Simulating GPU processing for {:s}[{:d}] {:s}[{:d}] {:s}[{:d}] putting result in "
             "{:s}[{:d}]",
             voltage_buf->buffer_name, voltage_frame_id, phase_buf->buffer_name, phase_frame_id,
             shift_buf->buffer_name, shift_frame_id, output_buf->buffer_name, output_frame_id);

        std::string id_tag = std::to_string(voltage_frame_id);
        if (metadata_is_onehot(voltage_buf, voltage_frame_id)) {
            int frame_counter = get_onehot_frame_counter(voltage_buf, voltage_frame_id);
            if (frame_counter < voltage_frame_id)
                frame_counter = voltage_frame_id;
            id_tag = std::to_string(frame_counter);
        }

        bool done = false;
        DEBUG("Is voltage buffer one-hot? {:}  Phase? {:}",
              metadata_is_onehot(voltage_buf, voltage_frame_id),
              metadata_is_onehot(phase_buf, phase_frame_id));
        if (metadata_is_onehot(voltage_buf, voltage_frame_id)) {
            std::vector<int> inds = get_onehot_indices(voltage_buf, voltage_frame_id);
            if (inds.size() == 0) {
            } else if (inds.size() != 4) {
                INFO("Expected 4 indices in voltage one-hot array, got {:d}", inds.size());
            } else {
                int t = inds[0];
                int p = inds[1];
                int f = inds[2];
                int d = inds[3];
                int b = -1;
                INFO("One-hot voltage buffer: time {:d} pol {:d}, freq {:d}, dish {:d}", t, p, f,
                     d);
                int ndishes = _num_elements / 2;
                bb_simple_sub(id_tag, phase, voltage, shift, output, _samples_per_data_set,
                              _num_beams, ndishes, _num_local_freq, t, b, d, f, p);
                done = true;
            }
        }

        if (!done && metadata_is_onehot(phase_buf, phase_frame_id)) {
            std::vector<int> inds = get_onehot_indices(phase_buf, phase_frame_id);
            if (inds.size() == 0) {
            } else if (inds.size() != 5) {
                INFO("Expected 5 indices in phase one-hot array, got {:d}", inds.size());
            } else {
                int f = inds[0];
                int p = inds[1];
                int b = inds[2];
                int d = inds[3];
                // real/imag = inds[4]
                int t = -1;
                INFO("One-hot phase buffer: freq {:d} pol {:d}, beam {:d}, dish {:d}", f, p, b, d);
                int ndishes = _num_elements / 2;
                bb_simple_sub(id_tag, phase, voltage, shift, output, _samples_per_data_set,
                              _num_beams, ndishes, _num_local_freq, t, b, d, f, p);
                done = true;
            }
        }

        if (!done) {
            int ndishes = _num_elements / 2;
            bb_simple(id_tag, phase, voltage, shift, output, _samples_per_data_set, _num_beams,
                      ndishes, _num_local_freq);
        }

        DEBUG("Simulated GPU processing done for {:s}[{:d}], result is in {:s}[{:d}]",
              voltage_buf->buffer_name, voltage_frame_id, output_buf->buffer_name, output_frame_id);

        voltage_buf->pass_metadata(voltage_frame_id, output_buf, output_frame_id);
        voltage_buf->mark_frame_empty(unique_name, voltage_frame_id);
        output_buf->mark_frame_full(unique_name, output_frame_id);

        voltage_frame_id = (voltage_frame_id + 1) % voltage_buf->num_frames;
        output_frame_id = (output_frame_id + 1) % output_buf->num_frames;

        // Check for available phase & shift frames and advance if they're ready!
        int next_frame = (phase_frame_id + 1) % phase_buf->num_frames;
        if (phase_buf->is_frame_empty(next_frame) == 0) {
            phase_buf->mark_frame_empty(unique_name, phase_frame_id);
            phase_frame_id = next_frame;
        }
        next_frame = (shift_frame_id + 1) % shift_buf->num_frames;
        if (shift_buf->is_frame_empty(next_frame) == 0) {
            shift_buf->mark_frame_empty(unique_name, shift_frame_id);
            shift_frame_id = next_frame;
        }
    }
}
