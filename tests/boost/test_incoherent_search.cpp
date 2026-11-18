#define BOOST_TEST_MODULE "test_incoherent_search"

#include "IncoherentSearch.hpp"

#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <random>
#include <vector>

// 256 channels over 400-800 MHz at 1.31 ms: DM 150 sweeps 2226 samples.
static IncoherentSearchParams test_params() {
    IncoherentSearchParams p;
    p.num_freq = 256;
    p.freq0_mhz = 400;
    p.dfreq_mhz = 400.0 / 256;
    p.sample_s = 256 * 5.12e-6;
    for (double dm = 50; dm <= 150; dm += 1)
        p.dms.push_back(dm);
    p.thermal = 0.01;
    p.holdoff_s = 3;
    p.num_threads = 2;
    return p;
}

// Runs noise of unit mean through the search. A pulse of the given amplitude (in sigma per
// channel; 0 for none) arrives at the top channel at sample 6000 with DM 100, in the channels
// below last_channel.
static std::vector<IncoherentCandidate> run(const IncoherentSearchParams& p, double amplitude,
                                            int last_channel) {
    const double mean = 1000;
    const int num_samples = 12000, pulse_sample = 6000;
    const double nu_top = (p.freq0_mhz + p.dfreq_mhz * (p.num_freq - 1)) / 1e3;
    std::vector<int> delay(p.num_freq);
    for (int c = 0; c < p.num_freq; ++c) {
        const double nu = (p.freq0_mhz + p.dfreq_mhz * c) / 1e3;
        delay[c] = (int)std::lround(4.148808 * 100 * (1 / (nu * nu) - 1 / (nu_top * nu_top))
                                    / (p.sample_s * 1e3));
    }
    IncoherentSearch search(p);
    std::mt19937 rng(1);
    std::normal_distribution<float> gauss(0, 1);
    std::vector<float> power(p.num_freq);
    std::vector<IncoherentCandidate> out;
    for (int t = 0; t < num_samples; ++t) {
        for (int c = 0; c < p.num_freq; ++c) {
            const bool on = amplitude > 0 && c < last_channel && t == pulse_sample + delay[c];
            power[c] = (float)(mean * (1 + p.thermal * (gauss(rng) + (on ? amplitude : 0))));
        }
        search.add_sample(power.data());
        for (const IncoherentCandidate& c : search.take_candidates())
            out.push_back(c);
    }
    return out;
}

BOOST_AUTO_TEST_CASE(noise_gives_no_candidates) {
    BOOST_CHECK(run(test_params(), 0, 0).empty());
}

BOOST_AUTO_TEST_CASE(pulse_is_found_at_its_dm_and_time) {
    const IncoherentSearchParams p = test_params();
    const std::vector<IncoherentCandidate> cands = run(p, 1.5, p.num_freq);
    BOOST_REQUIRE_EQUAL(cands.size(), 1);
    const IncoherentCandidate& c = cands[0];
    // 1.5 sigma in each of 256 channels is 24 sigma in the sum.
    BOOST_CHECK_GT(c.snr, 18);
    BOOST_CHECK_LT(c.snr, 30);
    BOOST_CHECK_EQUAL(c.dm, 100);
    BOOST_CHECK_EQUAL(c.sample, 6000);
    BOOST_CHECK_EQUAL(c.width, 1);
    int filled = 0;
    for (float b : c.block_snr)
        filled += b > 3;
    BOOST_CHECK_GE(filled, 14);
}

BOOST_AUTO_TEST_CASE(band_occupancy_shows_a_narrow_signal) {
    // The same sweep in the lowest 96 channels only (6 of the 16 blocks). Each sample is limited
    // to 3 sigma, so this is about 17 sigma in the sum.
    const IncoherentSearchParams p = test_params();
    const std::vector<IncoherentCandidate> cands = run(p, 4, 96);
    BOOST_REQUIRE(!cands.empty());
    const IncoherentCandidate* best = &cands[0];
    for (const IncoherentCandidate& c : cands)
        if (c.snr > best->snr)
            best = &c;
    BOOST_CHECK_GT(best->snr, 12);
    int filled = 0;
    for (float b : best->block_snr)
        filled += b > 3;
    BOOST_CHECK_GE(filled, 5);
    BOOST_CHECK_LE(filled, 7);
}

BOOST_AUTO_TEST_CASE(burst_in_part_of_the_band_is_cut) {
    // Interference at DM 0 in part of the band: 8 sigma in the top 56 channels for 8 samples,
    // searched at low DMs. Every sample is limited to 3 sigma, but 56 channels at the limit for
    // 8 samples would still add up to about 20 sigma. The search leaves their blocks out of
    // those samples instead.
    IncoherentSearchParams p = test_params();
    p.dms.clear();
    for (double dm = 3; dm <= 30; dm += 0.5)
        p.dms.push_back(dm);
    IncoherentSearch search(p);
    std::mt19937 rng(1);
    std::normal_distribution<float> gauss(0, 1);
    std::vector<float> power(p.num_freq);
    size_t num_candidates = 0;
    for (int t = 0; t < 12000; ++t) {
        for (int c = 0; c < p.num_freq; ++c) {
            const bool on = c >= 200 && t >= 6000 && t < 6008;
            power[c] = (float)(1000 * (1 + p.thermal * (gauss(rng) + (on ? 8 : 0))));
        }
        search.add_sample(power.data());
        num_candidates += search.take_candidates().size();
    }
    BOOST_CHECK_EQUAL(num_candidates, 0);
}

BOOST_AUTO_TEST_CASE(masked_band_is_not_searched) {
    // Channels 0-96 are at 400-550 MHz.
    IncoherentSearchParams p = test_params();
    p.mask_bands_mhz = {{400, 550}};
    BOOST_CHECK(run(p, 4, 96).empty());
}
