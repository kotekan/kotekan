#include "gpuSimulateCudaUpchannelize.hpp"

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

REGISTER_KOTEKAN_STAGE(gpuSimulateCudaUpchannelize);

gpuSimulateCudaUpchannelize::gpuSimulateCudaUpchannelize(Config& config,
                                                         const std::string& unique_name,
                                                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&gpuSimulateCudaUpchannelize::main_thread, this)) {
    _num_dishes = config.get<int>(unique_name, "num_dishes");
    _num_local_freq = config.get<int>(unique_name, "num_local_freq");
    _samples_per_data_set = config.get<int>(unique_name, "samples_per_data_set");
    _upchan_factor = config.get<int>(unique_name, "upchan_factor");
    bool zero_output = config.get_default<bool>(unique_name, "zero_output", false);
    voltage_in_buf = get_buffer("voltage_in_buf");
    voltage_out_buf = get_buffer("voltage_out_buf");
    register_consumer(voltage_in_buf, unique_name.c_str());
    register_producer(voltage_out_buf, unique_name.c_str());
    if (zero_output)
        zero_frames(voltage_out_buf);
}

gpuSimulateCudaUpchannelize::~gpuSimulateCudaUpchannelize() {}

// This code is from Erik's
// https://github.com/eschnett/GPUIndexSpaces.jl/blob/main/kernels/upchan.cxx

constexpr int C = 2;     // number of complex components
constexpr int T = 32768; // number of times
constexpr int D = 1;     // TODO 512;   // number of dishes
constexpr int P = 1;     // TODO 2;     // number of polarizations
constexpr int F = 1;     // TODO 16;    // frequency channels per GPU
constexpr int U = 16;    // upchannelization factor
constexpr int M = 4;     // number of taps

// 4-bit integers

using int4x2_t = uint8_t;

constexpr int4x2_t set4(const int8_t lo, const int8_t hi) {
    return (uint8_t(lo) & 0x0f) | ((uint8_t(hi) << 4) & 0xf0);
}
constexpr int4x2_t set4(const std::array<int8_t, 2> a) {
    return set4(a[0], a[1]);
}

constexpr std::array<int8_t, 2> get4(const int4x2_t i) {
    return {int8_t(int8_t((i + 0x08) & 0x0f) - 0x08),
            int8_t(int8_t(((i >> 4) + 0x08) & 0x0f) - 0x08)};
}

constexpr bool test_get4_set4() {
    for (int hi = -8; hi <= 7; ++hi) {
        for (int lo = -8; lo <= 7; ++lo) {
            if (get4(set4(lo, hi))[0] != lo)
                return false;
            if (get4(set4(lo, hi))[1] != hi)
                return false;
        }
    }
    return true;
}

// Storage management

template<typename T, typename I>
constexpr T convert(const I i) {
    return T(i);
}
template<typename T, typename I>
constexpr std::complex<T> convert(const std::complex<I> i) {
    return std::complex<T>(convert<I>(i.real()), convert<I>(i.imag()));
}

#if 1
// Use 4-bit integers for E and Ebar

using storage_t = int4x2_t;
using value_t = int8_t;

constexpr float maxabserr = 0.8f;

constexpr storage_t set_storage(const int8_t lo, const int8_t hi) {
    return set4(lo, hi);
}
constexpr storage_t set_storage(const std::array<int8_t, 2> x) {
    return set_storage(x[0], x[1]);
}
constexpr std::array<int8_t, 2> get_storage(const storage_t x) {
    return get4(x);
}

template<typename I, typename T>
constexpr I quantize(const T x, const I imax) {
    using std::floor;
    const I itmp = I(floor(x + T(0.5)));
    using std::max, std::min;
    const I i = min(imax, max(I(-imax), itmp));
    return i;
}

#else
// Use 32-bit floats for E and Ebar

using storage_t = std::complex<float>;
using value_t = float;

constexpr float maxabserr = 0.0f;

constexpr storage_t set_storage(const float lo, const float hi) {
    return {lo, hi};
}
constexpr storage_t set_storage(const std::array<float, 2> x) {
    return set_storage(x[0], x[1]);
}
constexpr std::array<float, 2> get_storage(const storage_t x) {
    return {real(x), imag(x)};
}

template<typename I, typename T>
constexpr I quantize(const T x, const I imax) {
    return I(x);
}

#endif

template<typename I, typename T>
constexpr std::complex<I> quantize(const std::complex<T> x, const I imax) {
    return std::complex<I>(quantize<I>(x.real(), imax), quantize<I>(x.imag(), imax));
}

// complex numbers

template<typename T>
constexpr std::complex<T> to_complex(const std::array<T, 2> a) {
    return std::complex<T>(a[0], a[1]);
}
template<typename T>
constexpr std::array<T, 2> to_array(const std::complex<T> c) {
    return std::array<T, 2>{c.real(), c.imag()};
}

// functions

template<typename T>
constexpr T linterp(const T x1, const T y1, const T x2, const T y2, const T x) {
    return (x - x2) * y1 / (x1 - x2) + (x - x1) * y2 / (x2 - x1);
}
static_assert(linterp(1.0f, 2.0f, 3.0f, 4.0f, 1.0f) == 2.0f);
static_assert(linterp(1.0f, 2.0f, 3.0f, 4.0f, 3.0f) == 4.0f);
static_assert(linterp(1.0f, 2.0f, 3.0f, 4.0f, 2.0f) == 3.0f);

template<typename T, typename U, std::size_t N>
constexpr U interp(const std::array<std::pair<T, U>, N>& table, const T x) {
    static_assert(N > 0);
    assert(x >= table.front().first);
    assert(x <= table.back().first);
    for (std::size_t n = 0; n < table.size() - 1; ++n)
        if (x <= table[n + 1].first)
            return linterp(table[n].first, table[n].second, table[n + 1].first, table[n + 1].second,
                           x);
    assert(false);
}
namespace {
constexpr std::array<std::pair<float, float>, 3> table{{
    {1.0f, +1.0f},
    {2.0f, -1.0f},
    {3.0f, +3.0f},
}};
static_assert(interp(table, 1.0f) == +1.0f);
static_assert(interp(table, 1.5f) == +0.0f);
static_assert(interp(table, 2.0f) == -1.0f);
static_assert(interp(table, 2.5f) == +1.0f);
static_assert(interp(table, 3.0f) == +3.0f);
} // namespace

template<typename T>
constexpr T sinc(const T x) {
    using std::abs;
    assert(x == T(0) || abs(x) > T(1.0e-10));
    return x == T(0) ? T(1) : sin(x) / x;
}

// array indexing

constexpr int Eidx(int c, int d, int f, int p, int t) {
    assert(c >= 0 && c < C);
    assert(d >= 0 && d < D);
    assert(f >= 0 && f < F);
    assert(p >= 0 && p < P);
    assert(t >= 0 && t < T + M * U - 1);
    return d + D * f + D * F * p + D * F * P * t;
}

constexpr int Ebaridx(int c, int d, int fbar, int p, int tbar) {
    assert(c >= 0 && c < C);
    assert(d >= 0 && d < D);
    assert(fbar >= 0 && fbar < F * U);
    assert(p >= 0 && p < P);
    assert(tbar >= 0 && tbar < T / U);
    return d + D * fbar + D * F * U * p + D * F * U * P * tbar;
}

// kernel
#if KOTEKAN_FLOAT16
void upchan_simple(const float16_t* __restrict__ const W, const float16_t* __restrict__ const G,
                   const storage_t* __restrict__ const E, storage_t* __restrict__ const Ebar) {
#pragma omp parallel for collapse(5)
    for (int f = 0; f < F; ++f) {
        for (int p = 0; p < P; ++p) {
            for (int d = 0; d < D; ++d) {
                for (int u = 0; u < U; ++u) {
                    for (int tbar = 0; tbar < T / U; ++tbar) {

                        const int fbar = u + U * f;

                        std::complex<float> Ebar1 = 0.0f;

                        for (int s = 0; s < M * U; ++s) {
                            const int t = s + U * tbar;

                            const float W1 = W[s];

                            using std::polar;
                            const std::complex<float> phase =
                                polar(1.0f, -2 * float(M_PI) * (u - (U - 1) / 2.0f) / U * s);

                            const std::complex<float> E1 =
                                convert<float>(to_complex(get_storage(E[Eidx(0, d, f, p, t)])));

                            Ebar1 += W1 * phase * E1;

                        } // s

                        const float G1 = G[u];
                        Ebar1 *= G1;
                        Ebar[Ebaridx(0, d, fbar, p, tbar)] =
                            set_storage(to_array(quantize<value_t>(Ebar1, 7)));

                    } // tbar
                }     // u
            }         // d
        }             // p
    }                 // f
}


///////////////////////////////////////////////////////////////////////

void gpuSimulateCudaUpchannelize::upchan_simple(
    std::string tag, const float16_t* __restrict__ const W, const float16_t* __restrict__ const G,
    // const storage_t *__restrict__ const E,
    // storage_t *__restrict__ const Ebar,
    const void* __restrict__ const E, void* __restrict__ const Ebar,
    const int T, // 32768; // number of times
    const int D, // = 512;   // number of dishes
    const int F, // = 16;    // input frequency channels per GPU
    const int U  // = 16;    // upchannelization factor
) {}
void gpuSimulateCudaUpchannelize::upchan_simple_sub(
    std::string tag, const float16_t* __restrict__ const W, const float16_t* __restrict__ const G,
    // const storage_t *__restrict__ const E,
    // storage_t *__restrict__ const Ebar,
    const void* __restrict__ const E, void* __restrict__ const Ebar,
    const int T, // 32768; // number of times
    const int D, // = 512;   // number of dishes
    const int F, // = 16;    // input frequency channels per GPU
    const int U, // = 16;    // upchannelization factor
    int t, int d, int p, int f) {}
#endif

/*
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
*/

void gpuSimulateCudaUpchannelize::main_thread() {
    int voltage_frame_id = 0;
    int output_frame_id = 0;

    while (!stop_thread) {
        int4x2_t* voltage_in =
            (int4x2_t*)wait_for_full_frame(voltage_in_buf, unique_name.c_str(), voltage_frame_id);
        if (voltage_in == nullptr)
            break;
        int4x2_t* voltage_out =
            (int4x2_t*)wait_for_empty_frame(voltage_out_buf, unique_name.c_str(), output_frame_id);
        if (voltage_out == nullptr)
            break;

        INFO("Simulating GPU processing for {:s}[{:d}] putting result in {:s}[{:d}]",
             voltage_in_buf->buffer_name, voltage_frame_id, voltage_out_buf->buffer_name,
             output_frame_id);

        std::string id_tag = std::to_string(voltage_frame_id);
        if (metadata_is_onehot(voltage_in_buf, voltage_frame_id)) {
            int frame_counter = get_onehot_frame_counter(voltage_in_buf, voltage_frame_id);
            if (frame_counter < voltage_frame_id)
                frame_counter = voltage_frame_id;
            id_tag = std::to_string(frame_counter);
        }

        bool done = false;
        if (metadata_is_onehot(voltage_in_buf, voltage_frame_id)) {
            std::vector<int> inds = get_onehot_indices(voltage_in_buf, voltage_frame_id);
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
                // upchan_simple_sub(id_tag, phase, voltage, shift, output, _samples_per_data_set,
                //_num_beams, ndishes, _num_local_freq, t, b, d, f, p);
                done = true;
            }
        }

        if (!done) {
            // upchan_simple(id_tag, phase, voltage, shift, output, _samples_per_data_set,
            // _num_beams, ndishes, _num_local_freq);
        }

        DEBUG("Simulated GPU processing done for {:s}[{:d}], result is in {:s}[{:d}]",
              voltage_in_buf->buffer_name, voltage_frame_id, voltage_out_buf->buffer_name,
              output_frame_id);

        pass_metadata(voltage_in_buf, voltage_frame_id, voltage_out_buf, output_frame_id);
        mark_frame_empty(voltage_in_buf, unique_name.c_str(), voltage_frame_id);
        mark_frame_full(voltage_out_buf, unique_name.c_str(), output_frame_id);

        voltage_frame_id = (voltage_frame_id + 1) % voltage_in_buf->num_frames;
        output_frame_id = (output_frame_id + 1) % voltage_out_buf->num_frames;
    }
}
