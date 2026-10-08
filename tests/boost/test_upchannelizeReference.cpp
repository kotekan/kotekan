#define BOOST_TEST_MODULE "test_upchannelizeReference"

#include "upchannelizeReference.hpp"

#include <algorithm>
#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <vector>

using kotekan::upchan_decode_int4;
using kotekan::upchan_decode_int8;
using kotekan::upchan_encode_int4;
using kotekan::upchan_encode_int8;
using kotekan::upchan_window;
using kotekan::upchannelize_reference;

namespace {

constexpr int M = kotekan::upchan_default_num_taps; // 4

// The upchannelization factors the generated kernels are built for.
const std::vector<int> all_U = {2, 4, 8, 16, 32, 64, 128};

// Run the reference on a single (P=1, D=1) timestream produced by `signal(t)`.
// Returns the output as [tbar][u].
std::vector<std::vector<std::complex<float>>>
run(const int U, const int num_times_out, const std::function<std::complex<float>(int)>& signal,
    const float gain_value = 1.0f) {
    const int num_times = U * num_times_out + M * U;
    std::vector<std::complex<float>> E(num_times);
    for (int t = 0; t < num_times; ++t)
        E.at(t) = signal(t);

    const std::vector<float> window = upchan_window(M, U);
    const std::vector<float> gain(U, gain_value);
    std::vector<std::complex<float>> Ebar(std::size_t(num_times_out) * U);

    upchannelize_reference(E.data(), window.data(), gain.data(), Ebar.data(), M, U, 1, 1,
                           num_times_out);

    std::vector<std::vector<std::complex<float>>> out(num_times_out,
                                                      std::vector<std::complex<float>>(U));
    for (int tbar = 0; tbar < num_times_out; ++tbar)
        for (int u = 0; u < U; ++u)
            out.at(tbar).at(u) = Ebar.at(std::size_t(tbar) * U + u);
    return out;
}

} // namespace

////////////////////////////////////////////////////////////////////////////////
// The window function
////////////////////////////////////////////////////////////////////////////////

// Re-derive the window straight from eqn. (11) / `Wkernel` and require an exact match.
BOOST_AUTO_TEST_CASE(window_matches_the_formula) {
    for (const int U : all_U) {
        const std::vector<float> window = upchan_window(M, U);
        BOOST_REQUIRE_EQUAL(window.size(), std::size_t(M * U));
        for (int s = 0; s < M * U; ++s) {
            const double sp = double(2 * s - (M * U - 1)) / double(2 * M * U);
            const double x = double(M) * sp;
            const double sinc = x == 0.0 ? 1.0 : std::sin(M_PI * x) / (M_PI * x);
            const double expected = std::pow(std::cos(M_PI * sp), 2) * sinc / double(U);
            BOOST_CHECK_SMALL(double(window.at(s)) - expected, 1.0e-6);
        }
    }
}

// The window is symmetric about the centre of its M*U samples. This is the (MU-1)/2
// centring; centring on MU/2 instead would break it.
BOOST_AUTO_TEST_CASE(window_is_symmetric) {
    for (const int U : all_U) {
        const std::vector<float> window = upchan_window(M, U);
        for (int s = 0; s < M * U; ++s)
            BOOST_CHECK_SMALL(window.at(s) - window.at(M * U - 1 - s), 1.0e-6f);
    }
}

// The window has sinc zeros: it changes sign. An unnormalized sinc(x) = sin(x)/x over this
// range is a single positive lobe, so this is what distinguishes the two.
BOOST_AUTO_TEST_CASE(window_changes_sign) {
    for (const int U : all_U) {
        const std::vector<float> window = upchan_window(M, U);
        bool has_positive = false, has_negative = false;
        for (const float w : window) {
            has_positive = has_positive || w > 0.0f;
            has_negative = has_negative || w < 0.0f;
        }
        BOOST_CHECK(has_positive);
        BOOST_CHECK_MESSAGE(has_negative, "window has no negative lobe for U = " << U);
    }
}

// The sub-bin frequency response, (1/U) sum_s W(s) exp(-2 pi i s delta / U), normalized so
// that the 1/U is already in W. Measured values: 1.0127 at delta = 0, 0.5008 at delta = 1/2.
BOOST_AUTO_TEST_CASE(window_subbin_response) {
    const int U = 16;
    const std::vector<float> window = upchan_window(M, U);

    const auto response = [&](const double delta) {
        std::complex<double> sum(0.0, 0.0);
        for (int s = 0; s < M * U; ++s) {
            const double angle = -2.0 * M_PI * double(s) * delta / double(U);
            sum += double(window.at(s)) * std::complex<double>(std::cos(angle), std::sin(angle));
        }
        return std::abs(sum);
    };

    BOOST_CHECK_CLOSE(response(0.0), 1.0127, 0.5);
    BOOST_CHECK_CLOSE(response(0.5), 0.5008, 0.5);
}

////////////////////////////////////////////////////////////////////////////////
// The transform
////////////////////////////////////////////////////////////////////////////////

// A constant (DC) input must land in fine channels U/2 - 1 and U/2, split equally, and be
// time-independent. This needs no convention reasoning and has no quantization noise; it is
// the cheapest structural check on the transform, and the one that caught two real FFT bugs
// in `upchan.jl` (U = 4 and U = 128).
//
// DC sits exactly half a bin from each of the two centre channels, so each picks up the
// window's half-bin response, 0.5008. The remaining channels are not exactly zero -- they see
// the window's response at 3/2, 5/2, ... bins -- but they stay below 1.4e-3 of the peak.
BOOST_AUTO_TEST_CASE(constant_input_splits_between_the_two_centre_bins) {
    constexpr float half_bin_response = 0.5008f;
    constexpr float max_sidelobe = 2.0e-3f; // relative to the peak; measured worst is 1.37e-3

    for (const int U : all_U) {
        const int num_times_out = 3;
        const auto out = run(U, num_times_out, [](int) { return std::complex<float>(1.0f, 0.0f); });

        const float ref = std::abs(out.at(0).at(U / 2 - 1));
        BOOST_CHECK_MESSAGE(std::abs(ref - half_bin_response) < 1.0e-3f,
                            "U = " << U << ": half-bin response is " << ref << ", expected "
                                   << half_bin_response);

        for (int tbar = 0; tbar < num_times_out; ++tbar) {
            for (int u = 0; u < U; ++u) {
                const float mag = std::abs(out.at(tbar).at(u));
                if (u == U / 2 - 1 || u == U / 2) {
                    // Equal split, and the same at every output time.
                    BOOST_CHECK_MESSAGE(std::abs(mag - ref) < 1.0e-4f,
                                        "U = " << U << " tbar = " << tbar << " u = " << u
                                               << ": got " << mag << ", expected " << ref);
                } else {
                    BOOST_CHECK_MESSAGE(mag < max_sidelobe * ref,
                                        "U = " << U << " tbar = " << tbar << " u = " << u
                                               << ": leaked " << mag / ref << " of the peak");
                }
            }
        }
    }
}

// A tone at normalized frequency (bin - (U-1)/2)/U must land in fine bin `bin`, and the
// output must advance in phase as exp(2 pi i tbar nu U) with no time-origin offset.
//
// The response is not a delta: a 4-tap PFB leaks a fixed fraction into its neighbours. Those
// fractions are a property of the window alone, so they are the same for every U, and pinning
// them (i.e. expecting certain vaues) is a much sharper test than merely requiring "not much"
// leakage.
//
// The fine-frequency axis is cyclic, so bin 1 is a *neighbour* of bin U-1. Where +k and -k
// land on the same channel (k == U/2) the two contributions cancel instead of adding, so the
// fixed fractions only apply for k < U/2.
BOOST_AUTO_TEST_CASE(single_tone_lands_in_its_bin) {
    // On-bin response, and the leakage at cyclic distance 1 and 2, as fractions of the peak.
    constexpr double on_bin = 1.0127;
    // The constants characterizing the leakage were measured, and confirmed by a double
    // precision calculation, and are consistent with the U -> \infty limit.
    constexpr double leak[3] = {1.0, 6.0414e-3, 1.8051e-4};

    for (const int U : all_U) {
        for (const int bin : {1, U / 2, U - 2}) {
            if (bin < 0 || bin >= U)
                continue;
            const double nu = (double(bin) - double(U - 1) / 2.0) / double(U);
            const int num_times_out = 4;
            const auto out = run(U, num_times_out, [&](const int t) {
                const double angle = 2.0 * M_PI * nu * double(t);
                return std::complex<float>(float(std::cos(angle)), float(std::sin(angle)));
            });

            // A unit-amplitude tone on the bin centre comes back with the window's on-bin
            // response, because the 1/U normalization is already folded into W.
            const double peak = std::abs(out.at(0).at(bin));
            BOOST_CHECK_MESSAGE(std::abs(peak - on_bin) < 1.0e-3,
                                "U = " << U << " bin = " << bin << ": on-bin response is " << peak
                                       << ", expected " << on_bin);

            for (int u = 0; u < U; ++u) {
                const int offset = std::abs(u - bin);
                const int distance = std::min(offset, U - offset); // the axis is cyclic
                if (distance == 0)
                    continue;
                const double ratio = std::abs(out.at(0).at(u)) / peak;
                if (2 * distance == U) {
                    // +distance and -distance are the same channel; they cancel.
                    BOOST_CHECK_MESSAGE(ratio < 1.0e-6, "U = " << U << " bin = " << bin
                                                               << ": self-aliased bin " << u
                                                               << " holds " << ratio);
                } else if (distance <= 2) {
                    // 5%, not tighter: at U = 128 the sum runs over M*U = 512 terms, and
                    // float32 accumulation moves the second sidelobe by a few percent.
                    BOOST_CHECK_MESSAGE(std::abs(ratio - leak[distance]) < 0.05 * leak[distance],
                                        "U = " << U << " bin = " << bin << ": bin " << u
                                               << " at cyclic distance " << distance << " holds "
                                               << ratio << ", expected " << leak[distance]);
                } else {
                    BOOST_CHECK_MESSAGE(ratio < 1.0e-4, "U = " << U << " bin = " << bin << ": bin "
                                                               << u << " holds " << ratio);
                }
            }

            // Phase advance per output sample, with no offset: Ebar[tbar] / Ebar[0] must be
            // exp(2 pi i tbar nu U).
            for (int tbar = 1; tbar < num_times_out; ++tbar) {
                const double angle = 2.0 * M_PI * nu * double(U) * double(tbar);
                const std::complex<double> expected(std::cos(angle), std::sin(angle));
                const std::complex<double> got = std::complex<double>(out.at(tbar).at(bin))
                                                 / std::complex<double>(out.at(0).at(bin));
                BOOST_CHECK_SMALL(std::abs(got - expected), 1.0e-3);
            }
        }
    }
}

// The gain is applied per fine channel, after the sum.
BOOST_AUTO_TEST_CASE(gain_scales_the_output) {
    const int U = 8;
    const auto plain = run(U, 1, [](int) { return std::complex<float>(1.0f, 0.0f); }, 1.0f);
    const auto scaled = run(U, 1, [](int) { return std::complex<float>(1.0f, 0.0f); }, 3.0f);
    for (int u = 0; u < U; ++u)
        BOOST_CHECK_SMALL(std::abs(scaled.at(0).at(u) - 3.0f * plain.at(0).at(u)), 1.0e-5f);
}

// The output at tbar reads inputs [U*tbar, U*tbar + M*U), and nothing outside that range.
// An impulse in the part of tbar=1's window that tbar=0's window does not cover must show up
// in tbar=1 and be completely absent from tbar=0.
BOOST_AUTO_TEST_CASE(output_reads_its_own_input_window) {
    const int U = 8;
    const int num_times_out = 2;
    // tbar=0 covers [0, M*U); tbar=1 covers [U, U + M*U). This impulse sits in the second
    // but not in the first.
    const int impulse = M * U;
    const auto out = run(U, num_times_out, [&](const int t) {
        return t == impulse ? std::complex<float>(1.0f, 0.0f) : std::complex<float>(0.0f, 0.0f);
    });

    float sum0 = 0.0f, sum1 = 0.0f;
    for (int u = 0; u < U; ++u) {
        sum0 += std::abs(out.at(0).at(u));
        sum1 += std::abs(out.at(1).at(u));
    }
    BOOST_CHECK_EQUAL(sum0, 0.0f);
    BOOST_CHECK_GT(sum1, 1.0e-3f);
}

////////////////////////////////////////////////////////////////////////////////
// Sample encodings
////////////////////////////////////////////////////////////////////////////////

// Real in the HIGH nibble, imaginary in the LOW one, each offset-encoded by +8.
// See `external/n2k/include/n2k/Correlator.hpp:61` (`real_part_in_low_bits = false`).
BOOST_AUTO_TEST_CASE(int4_places_real_in_the_high_nibble) {
    BOOST_CHECK_EQUAL(int(upchan_encode_int4({0.0f, 0.0f})), 0x88);
    BOOST_CHECK_EQUAL(int(upchan_encode_int4({1.0f, 0.0f})), 0x98);
    BOOST_CHECK_EQUAL(int(upchan_encode_int4({0.0f, 1.0f})), 0x89);
    BOOST_CHECK_EQUAL(int(upchan_encode_int4({-1.0f, 2.0f})), 0x7a);

    BOOST_CHECK_EQUAL(upchan_decode_int4(0x88), std::complex<float>(0.0f, 0.0f));
    BOOST_CHECK_EQUAL(upchan_decode_int4(0x98), std::complex<float>(1.0f, 0.0f));
    BOOST_CHECK_EQUAL(upchan_decode_int4(0x89), std::complex<float>(0.0f, 1.0f));
}

BOOST_AUTO_TEST_CASE(int4_round_trips) {
    for (int real = -7; real <= 7; ++real) {
        for (int imag = -7; imag <= 7; ++imag) {
            const std::complex<float> value{float(real), float(imag)};
            BOOST_CHECK_EQUAL(upchan_decode_int4(upchan_encode_int4(value)), value);
        }
    }
}

// Clamped to +-7, not [-8, +7]: -8 is the poison / missing-data marker.
BOOST_AUTO_TEST_CASE(int4_saturates_symmetrically) {
    BOOST_CHECK_EQUAL(upchan_decode_int4(upchan_encode_int4({100.0f, -100.0f})),
                      std::complex<float>(7.0f, -7.0f));
    BOOST_CHECK_EQUAL(upchan_decode_int4(upchan_encode_int4({-8.0f, -7.6f})),
                      std::complex<float>(-7.0f, -7.0f));
}

// Ties round to even, matching the `cvt.rni` in the generated PTX. A straightforward
// `floor(x + 0.5)` would give 1, 2, 3, 4 here instead.
BOOST_AUTO_TEST_CASE(int4_rounds_half_to_even) {
    BOOST_CHECK_EQUAL(upchan_decode_int4(upchan_encode_int4({0.5f, 1.5f})).real(), 0.0f);
    BOOST_CHECK_EQUAL(upchan_decode_int4(upchan_encode_int4({0.5f, 1.5f})).imag(), 2.0f);
    BOOST_CHECK_EQUAL(upchan_decode_int4(upchan_encode_int4({2.5f, 3.5f})).real(), 2.0f);
    BOOST_CHECK_EQUAL(upchan_decode_int4(upchan_encode_int4({2.5f, 3.5f})).imag(), 4.0f);
}

// K = 8 is plain two's complement with the real component first -- no swap, no offset.
BOOST_AUTO_TEST_CASE(int8_is_plain_twos_complement) {
    const std::complex<std::int8_t> encoded = upchan_encode_int8({5.0f, -3.0f});
    BOOST_CHECK_EQUAL(int(encoded.real()), 5);
    BOOST_CHECK_EQUAL(int(encoded.imag()), -3);
    BOOST_CHECK_EQUAL(upchan_decode_int8(encoded), std::complex<float>(5.0f, -3.0f));
}

BOOST_AUTO_TEST_CASE(int8_saturates_symmetrically) {
    BOOST_CHECK_EQUAL(upchan_decode_int8(upchan_encode_int8({1000.0f, -1000.0f})),
                      std::complex<float>(127.0f, -127.0f));
}
