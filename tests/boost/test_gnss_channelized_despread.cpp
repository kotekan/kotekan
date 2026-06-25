#define BOOST_TEST_MODULE "test_gnss_channelized_despread"

#include "gnssChannelizedDespread.hpp" // for channelized_despread
#include "gnssSimSignal.hpp"           // for sim_bpsk_baseband
#include "gpsCACode.hpp"               // for generate_ca_code
#include "pfbPrototype.hpp"            // for pfb_prototype, pfb_push, pfb_fold

#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <complex>
#include <vector>

using cf = std::complex<float>;
using cd = std::complex<double>;
using gnss::channelized_despread;
using gnss::sim_bpsk_baseband;

namespace {

constexpr int N = 31; // channels (divides the 1023-chip code * 2 samp/chip)
constexpr int P = 4;  // polyphase taps
constexpr int SP = 2; // samples per chip

std::vector<int8_t> ca_code(int prn) {
    auto a = gps::generate_ca_code(prn);
    return std::vector<int8_t>(a.begin(), a.end());
}

// PFB-analyze a complex signal into N channels via the PFB core + a reference
// DFT (no FFTW). Returns chan[c][m]. Mirrors fftwEngine's PFB path.
std::vector<std::vector<cf>> analyze(const std::vector<cf>& sig, const std::vector<float>& proto) {
    const int L = N * P;
    const int M = static_cast<int>(sig.size()) / N;
    std::vector<cf> hist(L, cf(0.0f, 0.0f)), block(N), v(N);
    std::vector<std::vector<cf>> chan(N, std::vector<cf>(M));
    for (int m = 0; m < M; ++m) {
        for (int i = 0; i < N; ++i)
            block[i] = sig[m * N + i];
        dsp::pfb_push(hist.data(), block.data(), N, P);
        dsp::pfb_fold(hist.data(), proto.data(), N, P, v.data());
        for (int k = 0; k < N; ++k) {
            cf x(0.0f, 0.0f);
            for (int p = 0; p < N; ++p) {
                const double a = -2.0 * M_PI * k * p / N;
                x += v[p] * cf(std::cos(a), std::sin(a));
            }
            chan[k][m] = x;
        }
    }
    return chan;
}

// Full-band matched-filter amplitude estimate: <x, r> / <r, r>.
cd full_band_amplitude(const std::vector<cf>& x, const std::vector<cf>& r) {
    cd num(0.0, 0.0);
    double den = 0.0;
    for (size_t n = 0; n < x.size(); ++n) {
        num += cd(x[n]) * std::conj(cd(r[n]));
        den += std::norm(cd(r[n]));
    }
    return num / den;
}

// Channels carrying replica energy above a fraction of the peak (the occupied,
// band-plan-selected subset); returns parallel data/replica channel lists.
void occupied_subset(const std::vector<std::vector<cf>>& data_ch,
                     const std::vector<std::vector<cf>>& repl_ch, double frac,
                     std::vector<std::vector<cf>>& data_sub,
                     std::vector<std::vector<cf>>& repl_sub) {
    std::vector<double> e(repl_ch.size(), 0.0);
    double emax = 0.0;
    for (size_t c = 0; c < repl_ch.size(); ++c) {
        for (const cf& v : repl_ch[c])
            e[c] += std::norm(v);
        emax = std::max(emax, e[c]);
    }
    for (size_t c = 0; c < repl_ch.size(); ++c)
        if (e[c] > frac * emax) {
            data_sub.push_back(data_ch[c]);
            repl_sub.push_back(repl_ch[c]);
        }
}

const std::vector<float>& proto() {
    static const std::vector<float> h = dsp::pfb_prototype(N, P, dsp::Window::Hamming);
    return h;
}

} // namespace

BOOST_AUTO_TEST_CASE(noiseless_recovers_amplitude_exactly) {
    const cf A(0.8f, 0.6f);
    auto code = ca_code(1);
    auto data = sim_bpsk_baseband(code, SP, 0.0, A);
    auto repl = sim_bpsk_baseband(code, SP, 0.0, cf(1.0f, 0.0f));

    auto dch = analyze(data, proto());
    auto rch = analyze(repl, proto());
    auto res = channelized_despread(dch, rch);

    // data == A*replica sample-wise, analysis is linear -> exact recovery.
    BOOST_CHECK_CLOSE(res.amplitude.real(), A.real(), 0.01);
    BOOST_CHECK_CLOSE(res.amplitude.imag(), A.imag(), 0.01);
}

BOOST_AUTO_TEST_CASE(occupied_channels_suffice) {
    const cf A(0.8f, 0.6f);
    auto code = ca_code(1);
    auto dch = analyze(sim_bpsk_baseband(code, SP, 0.0, A), proto());
    auto rch = analyze(sim_bpsk_baseband(code, SP, 0.0, cf(1.0f, 0.0f)), proto());

    std::vector<std::vector<cf>> ds, rs;
    occupied_subset(dch, rch, 0.01, ds, rs); // drop near-empty channels
    BOOST_REQUIRE_LT(ds.size(), static_cast<size_t>(N)); // some really were dropped
    BOOST_REQUIRE_GT(ds.size(), 0u);

    auto res = channelized_despread(ds, rs);
    BOOST_CHECK_CLOSE(res.amplitude.real(), A.real(), 0.5);
    BOOST_CHECK_CLOSE(res.amplitude.imag(), A.imag(), 0.5);
}

BOOST_AUTO_TEST_CASE(wrong_prn_does_not_correlate) {
    const cf A(0.8f, 0.6f);
    auto dch = analyze(sim_bpsk_baseband(ca_code(1), SP, 0.0, A), proto());
    auto rch = analyze(sim_bpsk_baseband(ca_code(7), SP, 0.0, cf(1.0f, 0.0f)), proto());

    auto res = channelized_despread(dch, rch);
    BOOST_CHECK_LT(std::abs(res.amplitude), 0.1 * std::abs(A)); // cross-corr floor
}

BOOST_AUTO_TEST_CASE(channelized_matches_full_band_under_noise) {
    const cf A(0.8f, 0.6f);
    auto code = ca_code(1);
    auto data = sim_bpsk_baseband(code, SP, 0.0, A, /*noise=*/0.2f, /*seed=*/42);
    auto repl = sim_bpsk_baseband(code, SP, 0.0, cf(1.0f, 0.0f));

    auto chan = channelized_despread(analyze(data, proto()), analyze(repl, proto())).amplitude;
    auto full = full_band_amplitude(data, repl);

    BOOST_CHECK_LT(std::abs(chan - cd(A)), 0.1 * std::abs(A)); // near truth
    BOOST_CHECK_LT(std::abs(chan - full), 0.1 * std::abs(A));  // and tracks full-band
}

// Centerpiece of the distributed combine: split the covering channels across two
// disjoint subbands, despread each independently, and recombine. The raw
// correlation G and replica energy are sums over channels, so they are additive
// across the partition, and (sum G)/(sum energy) reproduces the full despread
// amplitude exactly -- this is the invariant the coherent combiner stage relies on.
BOOST_AUTO_TEST_CASE(channel_partition_recombines_coherently) {
    const cf A(0.8f, 0.6f);
    auto code = ca_code(1);
    auto dch = analyze(sim_bpsk_baseband(code, SP, 0.0, A), proto());
    auto rch = analyze(sim_bpsk_baseband(code, SP, 0.0, cf(1.0f, 0.0f)), proto());

    auto full = channelized_despread(dch, rch);

    std::vector<std::vector<cf>> dA, rA, dB, rB;
    for (int c = 0; c < N; ++c) {
        if (c < N / 2) {
            dA.push_back(dch[c]);
            rA.push_back(rch[c]);
        } else {
            dB.push_back(dch[c]);
            rB.push_back(rch[c]);
        }
    }
    auto resA = channelized_despread(dA, rA);
    auto resB = channelized_despread(dB, rB);

    // Raw correlation + replica energy are additive across the channel partition...
    const cd corr = resA.correlation + resB.correlation;
    const double energy = resA.replica_energy + resB.replica_energy;
    BOOST_CHECK_CLOSE(corr.real(), full.correlation.real(), 1e-3);
    BOOST_CHECK_CLOSE(corr.imag(), full.correlation.imag(), 1e-3);
    BOOST_CHECK_CLOSE(energy, full.replica_energy, 1e-3);

    // ...so the recombined amplitude (sum G)/(sum energy) equals the full despread.
    const cd recombined = corr / energy;
    BOOST_CHECK_CLOSE(recombined.real(), full.amplitude.real(), 1e-3);
    BOOST_CHECK_CLOSE(recombined.imag(), full.amplitude.imag(), 1e-3);
}
