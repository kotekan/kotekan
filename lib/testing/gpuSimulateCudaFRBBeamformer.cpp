#include "gpuSimulateCudaFRBBeamformer.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.h"            // for Buffer, mark_frame_empty, register_consumer, wait_for_ful...
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

REGISTER_KOTEKAN_STAGE(gpuSimulateCudaFRBBeamformer);

using int4x2_t = uint8_t;
#if KOTEKAN_FLOAT16
static void frb_simple(const int32_t* __restrict__ const S, const float16_t* __restrict__ const W,
                       const int4x2_t* __restrict__ const E, float16_t* __restrict__ const I);
static void frb_simple_sub(const int32_t* __restrict__ const S,
                           const float16_t* __restrict__ const W,
                           const int4x2_t* __restrict__ const E, float16_t* __restrict__ const I,
                           const int t_hot, const int p_hot, const int f_hot, const int d_hot);
#else
#warning No float16 -- cannot simulate FRB beamformer!
#endif

gpuSimulateCudaFRBBeamformer::gpuSimulateCudaFRBBeamformer(Config& config,
                                                           const std::string& unique_name,
                                                           bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&gpuSimulateCudaFRBBeamformer::main_thread, this)) {
    _num_dishes = config.get<int>(unique_name, "num_dishes");
    _dish_grid_size = config.get<int>(unique_name, "dish_grid_size");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _time_downsampling = config.get<int>(unique_name, "time_downsampling");
    _dishlayout = config.get<std::vector<int>>(unique_name, "frb_beamformer_dish_layout");
    bool zero_output = config.get_default<bool>(unique_name, "zero_output", false);

    voltage_buf = get_buffer("voltage_in_buf");
    phase_buf = get_buffer("phase_in_buf");
    register_consumer(voltage_buf, unique_name.c_str());
    register_consumer(phase_buf, unique_name.c_str());
    beamgrid_buf = get_buffer("beams_out_buf");
    register_producer(beamgrid_buf, unique_name.c_str());
    if (zero_output)
        zero_frames(beamgrid_buf);
}

gpuSimulateCudaFRBBeamformer::~gpuSimulateCudaFRBBeamformer() {}

void gpuSimulateCudaFRBBeamformer::main_thread() {
    int voltage_frame_id = 0;
    int beamgrid_frame_id = 0;

    int phase_frame_id = 0;

    int32_t* S = (int32_t*)malloc(_dish_grid_size * _dish_grid_size * sizeof(int32_t));
    for (size_t i = 0; i < _dishlayout.size() / 2; i++)
        S[i] = _dish_grid_size * _dishlayout[i * 2 + 0] + _dishlayout[i * 2 + 1];

    while (!stop_thread) {
        int4x2_t* voltage =
            (int4x2_t*)wait_for_full_frame(voltage_buf, unique_name.c_str(), voltage_frame_id);
        if (voltage == nullptr)
            break;
        float16_t* phase =
            (float16_t*)wait_for_full_frame(phase_buf, unique_name.c_str(), phase_frame_id);
        if (phase == nullptr)
            break;
        float16_t* output =
            (float16_t*)wait_for_empty_frame(beamgrid_buf, unique_name.c_str(), beamgrid_frame_id);
        if (output == nullptr)
            break;

        INFO("Simulating GPU processing for {:s}[{:d}] {:s}[{:d}] putting result in "
             "{:s}[{:d}]",
             voltage_buf->buffer_name, voltage_frame_id, phase_buf->buffer_name, phase_frame_id,
             beamgrid_buf->buffer_name, beamgrid_frame_id);

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
                INFO("One-hot voltage buffer: time {:d} pol {:d}, freq {:d}, dish {:d}", t, p, f,
                     d);
#if KOTEKAN_FLOAT16
                frb_simple_sub(S, phase, voltage, output, t, p, f, d);
                done = true;
#endif
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
                INFO("One-hot phase buffer: freq {:d} pol {:d}, beam {:d}, dish {:d}", f, p, b, d);
                /////// DO WORK
                // done = true;
            }
        }

        if (!done) {
#if KOTEKAN_FLOAT16
            frb_simple(S, phase, voltage, output);
#endif
        }

        DEBUG("Simulated GPU processing done for {:s}[{:d}], result is in {:s}[{:d}]",
              voltage_buf->buffer_name, voltage_frame_id, beamgrid_buf->buffer_name,
              beamgrid_frame_id);

        pass_metadata(voltage_buf, voltage_frame_id, beamgrid_buf, beamgrid_frame_id);
        mark_frame_empty(voltage_buf, unique_name.c_str(), voltage_frame_id);
        mark_frame_full(beamgrid_buf, unique_name.c_str(), beamgrid_frame_id);

        voltage_frame_id = (voltage_frame_id + 1) % voltage_buf->num_frames;
        beamgrid_frame_id = (beamgrid_frame_id + 1) % beamgrid_buf->num_frames;

        // Check for available phase & shift frames and advance if they're ready!
        int next_frame = (phase_frame_id + 1) % phase_buf->num_frames;
        if (is_frame_empty(phase_buf, next_frame) == 0) {
            mark_frame_empty(phase_buf, unique_name.c_str(), phase_frame_id);
            phase_frame_id = next_frame;
        }
    }
    free(S);
}

#if KOTEKAN_FLOAT16
/// This is modified from https://github.com/eschnett/IndexSpaces.jl/blob/main/kernels/frb.cxx

using namespace std::complex_literals;
using std::norm;
using std::polar;

// Kernel parameters

constexpr int C = 2;    // number of complex components
constexpr int T = 2064; // number of times
constexpr int M = 24;   // number of beams
constexpr int N = 24;   // number of beams
constexpr int D = 512;  // number of dishes
constexpr int P = 2;    // number of polarizations
constexpr int F = 84;   // frequency channels per GPU
constexpr int Tds = 40; // time downsampling factor


// Integer divide, rounding up (towards positive infinity)
constexpr int cld(const int x, const int y) {
    return (x + y - 1) / y;
}

// 4-bit integers


constexpr int4x2_t set4(const int8_t lo, const int8_t hi) {
    return (uint8_t(lo) & 0x0f) | ((uint8_t(hi) << 4) & 0xf0);
}

constexpr std::array<int8_t, 2> get4(const int4x2_t i) {
    return {int8_t(int8_t((i + 0x08) & 0x0f) - 0x08),
            int8_t(int8_t(((i >> 4) + 0x08) & 0x0f) - 0x08)};
}

template<typename T>
inline std::complex<T> cispi(const T x) {
    // return exp(T(M_PI) * x * std::complex<T>(0, 1));
    return polar(T(1), T(M_PI) * x);
    // return std::complex<T>(cos(T(M_PI) * x), sin(T(M_PI) * x));
}

static void frb_simple(const int32_t* __restrict__ const S, const float16_t* __restrict__ const W,
                       const int4x2_t* __restrict__ const E, float16_t* __restrict__ const I) {
    frb_simple_sub(S, W, E, I, -1, -1, -1, -1);
}

static void frb_simple_sub(const int32_t* __restrict__ const S,
                           const float16_t* __restrict__ const W,
                           const int4x2_t* __restrict__ const E, float16_t* __restrict__ const I,
                           const int t, const int p, const int f, const int d) {
    const int f0 = (f == -1 ? 0 : f);
    const int f1 = (f == -1 ? F : f + 1);
    const int p0 = (p == -1 ? 0 : p);
    const int p1 = (p == -1 ? 2 : p + 1);
    int t0, t1;
    int tds = 0;
    if (t == -1) {
        t0 = -1;
        t1 = -1;
    } else {
        tds = (t / Tds);
        t0 = tds * Tds;
        t1 = t0 + Tds;
    }
    INFO_NON_OO("frb_simple_sub: t = {:d} -> t0 = {:d}, t1 = {:d}", t, t0, t1);
    const int d0 = (d == -1 ? 0 : d);
    const int d1 = (d == -1 ? D : d + 1);

    // Check consistency of `S`
    {
        std::vector<bool> E1(M * N, false);
        for (int d = 0; d < M * N; ++d)
            E1.at(S[d]) = true;
        for (int d = 0; d < M * N; ++d)
            assert(E1[d]);
    }

    //#pragma omp parallel for
    for (int freq = f0; freq < f1; ++freq) {

        float I1[(2 * M) * (2 * N)];
        int t_running = 0;
        for (int q = 0; q < 2 * N; ++q)
            for (int p = 0; p < 2 * M; ++p)
                I1[p + 2 * M * q] = 0;

        for (int time = t0; time < t1; ++time) {
            for (int polr = p0; polr < p1; ++polr) {

                // grid the dishes
                std::complex<float> E1[M * N];
                for (int d = d0; d < d1; ++d)
                    E1[S[d]] = std::complex<float>(
                        get4(E[d + D * freq + D * F * polr + D * F * P * time])[1],
                        get4(E[d + D * freq + D * F * polr + D * F * P * time])[0]);
                for (int d = D; d < M * N; ++d)
                    E1[S[d]] = 0;

                // FT in n direction
                std::complex<float> G[M * (2 * N)];
                for (int m = 0; m < M; ++m) {
                    for (int q = 0; q < 2 * N; ++q) {
                        std::complex<float> s = 0;
                        for (int n = 0; n < N; ++n) {
                            const std::complex<float> w(
                                W[0 + C * m + C * M * n + C * M * N * freq + C * M * N * F * polr],
                                W[1 + C * m + C * M * n + C * M * N * freq + C * M * N * F * polr]);
                            const std::complex<float> e1 = E1[n * M + m];
                            s += w * e1 * cispi(float(2 * n * q) / float(2 * N));
                        }
                        G[m + M * q] = s;
                    }
                }

                // FT in m direction
                std::complex<float> Et[2 * M * 2 * N];
                for (int q = 0; q < 2 * N; ++q) {
                    for (int p = 0; p < 2 * M; ++p) {
                        std::complex<float> s = 0;
                        for (int m = 0; m < M; ++m) {
                            const std::complex<float> g = G[m + M * q];
                            s += g * cispi(float(2 * m * p) / float(2 * M));
                        }
                        Et[p + 2 * M * q] = s;
                    }
                }

                for (int q = 0; q < 2 * N; ++q)
                    for (int p = 0; p < 2 * M; ++p)
                        I1[p + 2 * M * q] += norm(Et[p + 2 * M * q]);

            } // for polr

            t_running += 1;
            if (t_running == Tds) {
                INFO_NON_OO("frb_simple_sub: hit t_running == Tds, time = {:d}, tds = {:d}", time,
                            tds);
                for (int q = 0; q < 2 * N; ++q)
                    for (int p = 0; p < 2 * M; ++p)
                        I[p + 2 * M * q + 2 * M * 2 * N * freq + 2 * M * 2 * N * F * tds] =
                            I1[p + 2 * M * q];
                tds += 1;
                t_running = 0;
                for (int q = 0; q < 2 * N; ++q)
                    for (int p = 0; p < 2 * M; ++p)
                        I1[p + 2 * M * q] = 0;
            }

        } // for time

        if (t_running != 0) {
            for (int q = 0; q < 2 * N; ++q)
                for (int p = 0; p < 2 * M; ++p)
                    I[p + 2 * M * q + 2 * M * 2 * N * freq + 2 * M * 2 * N * F * tds] =
                        I1[p + 2 * M * q];
        }

    } // for freq
}
#endif
