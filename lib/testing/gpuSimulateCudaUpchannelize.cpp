#include "gpuSimulateCudaUpchannelize.hpp"

#include "Config.hpp"          // for Config
#include "StageFactory.hpp"    // for REGISTER_KOTEKAN_STAGE, StageMakerTemplate
#include "buffer.hpp"          // for Buffer, mark_frame_empty, register_consumer, wait_for_ful...
#include "bufferContainer.hpp" // for bufferContainer
#include "kotekanLogging.hpp"  // for INFO, DEBUG
#include "oneHotMetadata.hpp"  // for metadata_is_onehot, get_onehot_indices, get_onehot_frame_...

#include <algorithm>  // for max, min
#include <array>      // for array
#include <assert.h>   // for assert
#include <atomic>     // for atomic_bool
#include <complex>    // for complex
#include <cstdint>    // for int8_t
#include <exception>  // for exception
#include <functional> // for _Bind_helper<>::type, bind, function
#include <iosfwd>     // for size_t
#include <regex>      // for match_results<>::_Base_type
#include <stdexcept>  // for runtime_error
#include <utility>    // for pair
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
    // Try config value "freq_gains" as either a scalar float or a vector of floats
#if KOTEKAN_FLOAT16
    float gain0 = config.get_default<float>(unique_name, "freq_gains", 1.);
    std::vector<float> gains =
        config.get_default<std::vector<float>>(unique_name, "freq_gains", std::vector<float>());
    size_t ngains = _num_local_freq * _upchan_factor;
    if (gains.size() == 0) {
        for (size_t i = 0; i < ngains; i++)
            gains.push_back(gain0);
    }
    if (gains.size() != ngains)
        throw std::runtime_error(
            fmt::format("The number of elements in the 'freq_gains' config setting array must be "
                        "{:d} for gpuSimulateCudaUpchannelize",
                        ngains));
    gains16.resize(gains.size());
    for (size_t i = 0; i < gains.size(); i++)
        gains16[i] = (float16_t) gains[i];
#endif

    bool zero_output = config.get_default<bool>(unique_name, "zero_output", false);
    voltage_in_buf = get_buffer("voltage_in_buf");
    voltage_out_buf = get_buffer("voltage_out_buf");
    voltage_in_buf->register_consumer(unique_name);
    voltage_out_buf->register_producer(unique_name);
    if (zero_output)
        voltage_out_buf->zero_frames();
}

gpuSimulateCudaUpchannelize::~gpuSimulateCudaUpchannelize() {}

// This code is from Erik's
// https://github.com/eschnett/GPUIndexSpaces.jl/blob/main/kernels/upchan.cxx

constexpr int C = 2;     // number of complex components
constexpr int T = 32768; // number of times
constexpr int D = 512;   // number of dishes
constexpr int P = 2;     // number of polarizations
constexpr int F = 16;    // frequency channels per GPU
constexpr int U = 16;    // upchannelization factor
constexpr int M = 4;     // number of taps

// Storage management

template<typename T, typename I>
constexpr T convert(const I i) {
    return T(i);
}
template<typename T, typename I>
constexpr std::complex<T> convert(const std::complex<I> i) {
    return std::complex<T>(convert<I>(i.real()), convert<I>(i.imag()));
}

// Use 4-bit integers for E and Ebar

using storage_t = int4x2_t;
using value_t = int8_t;

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
void upchan_simple_cxx(const float16_t* __restrict__ const W, const float16_t* __restrict__ const G,
                       const storage_t* __restrict__ const E, storage_t* __restrict__ const Ebar,
                       int t, int p, int f, int d) {
    const int t0 = (t == -1 ? 0 : t);
    // const int t1 = (t == -1 ? T : t + 1);
    const int p0 = (p == -1 ? 0 : p);
    const int p1 = (p == -1 ? 2 : p + 1);
    const int f0 = (f == -1 ? 0 : f);
    const int f1 = (f == -1 ? F : f + 1);
    const int d0 = (d == -1 ? 0 : d);
    const int d1 = (d == -1 ? D : d + 1);

    const int tbar0 = t0 / U;
    const int tbar1 = (t == -1 ? T / U : (tbar0 + 1));

#ifdef _OPENMP
#pragma omp parallel for collapse(5)
#endif
    for (int f = f0; f < f1; ++f) {
        for (int p = p0; p < p1; ++p) {
            for (int d = d0; d < d1; ++d) {
                for (int u = 0; u < U; ++u) {
                    // for (int tbar = 0; tbar < T / U; ++tbar) {
                    for (int tbar = tbar0; tbar < tbar1; ++tbar) {

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

void gpuSimulateCudaUpchannelize::upchan_simple_sub(std::string tag,
                                                    const void* __restrict__ const E,
                                                    void* __restrict__ const Ebar, int t, int p,
                                                    int f, int d) {
    (void)tag;

    std::vector<float16_t> W(M * U); // PFB weight function

    // Set up window function
    using std::cos, std::pow, std::sin;
    float sumW = 0;
    for (int s = 0; s < M * U; ++s) {
        // sinc-Hanning window function, eqn. (11), with `N=U`
        W.at(s) = (float16_t) ( pow(cos(float(M_PI) * (s - (M * U - 1) / 2.0f) / (M * U + 1)), 2)
                  * sinc((s - (M * U - 1) / 2.0f) / U) );
        sumW += (float)W.at(s);
    }
    float16_t sumW16 = (float16_t)sumW;
    // Normalize the window function
    for (int s = 0; s < M * U; ++s)
        // W.at(s) /= (float16_t)sumW;
        W[s] = W[s] / sumW16;

    upchan_simple_cxx(W.data(), gains16.data(), (const storage_t*)E, (storage_t*)Ebar, t, p, f, d);
}

void gpuSimulateCudaUpchannelize::upchan_simple(std::string tag, const void* __restrict__ const E,
                                                void* __restrict__ const Ebar) {
    upchan_simple_sub(tag, E, Ebar, -1, -1, -1, -1);
}
#endif

void gpuSimulateCudaUpchannelize::main_thread() {
    int voltage_frame_id = 0;
    int output_frame_id = 0;

    while (!stop_thread) {
        int4x2_t* voltage_in =
            (int4x2_t*)voltage_in_buf->wait_for_full_frame(unique_name, voltage_frame_id);
        if (voltage_in == nullptr)
            break;
        int4x2_t* voltage_out =
            (int4x2_t*)voltage_out_buf->wait_for_empty_frame(unique_name, output_frame_id);
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
                INFO("One-hot voltage buffer: time {:d} pol {:d}, freq {:d}, dish {:d}", t, p, f,
                     d);
#if KOTEKAN_FLOAT16
                upchan_simple_sub(id_tag, voltage_in, voltage_out, t, p, f, d);
#else
                WARN("No Float16 support, so no gpuSimulateCudaUpchannelize!");
#endif
                done = true;
            }
        }

        if (!done) {
#if KOTEKAN_FLOAT16
            upchan_simple(id_tag, voltage_in, voltage_out);
#else
            WARN("No Float16 support, so no gpuSimulateCudaUpchannelize!");
#endif
        }

        DEBUG("Simulated GPU processing done for {:s}[{:d}], result is in {:s}[{:d}]",
              voltage_in_buf->buffer_name, voltage_frame_id, voltage_out_buf->buffer_name,
              output_frame_id);

        voltage_in_buf->pass_metadata(voltage_frame_id, voltage_out_buf, output_frame_id);
        voltage_in_buf->mark_frame_empty(unique_name, voltage_frame_id);
        voltage_out_buf->mark_frame_full(unique_name, output_frame_id);

        voltage_frame_id = (voltage_frame_id + 1) % voltage_in_buf->num_frames;
        output_frame_id = (output_frame_id + 1) % voltage_out_buf->num_frames;
    }
}
