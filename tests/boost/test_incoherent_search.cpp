#define BOOST_TEST_MODULE "test_incoherent_search"

#include "IncoherentSearch.hpp"

#include <algorithm>
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

// A pulse of one sample that arrives at the top channel at `sample`, in the channels below
// last_channel. The amplitude is in sigma per channel.
struct Pulse {
    double amplitude;
    double dm;
    int sample;
    int last_channel;
};

// Runs noise of unit mean with the pulses through the search.
static std::vector<IncoherentCandidate> run(const IncoherentSearchParams& p,
                                            const std::vector<Pulse>& pulses) {
    const double mean = 1000;
    const int num_samples = 12000;
    const double nu_top = (p.freq0_mhz + p.dfreq_mhz * (p.num_freq - 1)) / 1e3;
    // Delay in samples per unit DM
    std::vector<double> delay(p.num_freq);
    for (int c = 0; c < p.num_freq; ++c) {
        const double nu = (p.freq0_mhz + p.dfreq_mhz * c) / 1e3;
        delay[c] = 4.148808 * (1 / (nu * nu) - 1 / (nu_top * nu_top)) / (p.sample_s * 1e3);
    }
    IncoherentSearch search(p);
    std::mt19937 rng(1);
    std::normal_distribution<float> gauss(0, 1);
    std::vector<float> power(p.num_freq);
    std::vector<IncoherentCandidate> out;
    for (int t = 0; t < num_samples; ++t) {
        for (int c = 0; c < p.num_freq; ++c) {
            double amplitude = 0;
            for (const Pulse& pulse : pulses)
                if (c < pulse.last_channel && t == pulse.sample + std::lround(pulse.dm * delay[c]))
                    amplitude += pulse.amplitude;
            power[c] = (float)(mean * (1 + p.thermal * (gauss(rng) + amplitude)));
        }
        search.add_sample(power.data());
        for (const IncoherentCandidate& c : search.take_candidates())
            out.push_back(c);
    }
    return out;
}

BOOST_AUTO_TEST_CASE(noise_gives_no_candidates) {
    BOOST_CHECK(run(test_params(), {}).empty());
}

BOOST_AUTO_TEST_CASE(pulse_is_found_at_its_dm_and_time) {
    const IncoherentSearchParams p = test_params();
    const std::vector<IncoherentCandidate> cands = run(p, {{1.5, 100, 6000, p.num_freq}});
    BOOST_REQUIRE_EQUAL(cands.size(), 1);
    const IncoherentCandidate& c = cands[0];
    // 1.5 sigma in each of 256 channels is 24 sigma in the sum.
    BOOST_CHECK_GT(c.snr, 18);
    BOOST_CHECK_LT(c.snr, 30);
    BOOST_CHECK_EQUAL(c.dm, 100);
    BOOST_CHECK_EQUAL(c.sample, 6000);
    BOOST_CHECK_EQUAL(c.width, 1);
    BOOST_CHECK(!c.undispersed);
    int filled = 0;
    for (float b : c.block_snr)
        filled += b > 3;
    BOOST_CHECK_GE(filled, 14);
}

BOOST_AUTO_TEST_CASE(band_occupancy_shows_a_narrow_signal) {
    // The same sweep in the lowest 96 channels only (6 of the 16 blocks). Each sample is limited
    // to 3 sigma, so this is about 17 sigma in the sum.
    const IncoherentSearchParams p = test_params();
    const std::vector<IncoherentCandidate> cands = run(p, {{4, 100, 6000, 96}});
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

BOOST_AUTO_TEST_CASE(burst_without_dispersion_is_marked_at_every_dm) {
    // Interference at DM 0 that the limit per block lets through: 14 channels across the edge
    // of two blocks, for 96 samples. Their delays differ by less than the burst lasts, so every
    // DM from 50 to 250 adds them up, to about 9 sigma at the widest window.
    IncoherentSearchParams p = test_params();
    p.threshold = 6;
    p.dms.clear();
    for (double dm = 50; dm <= 250; dm += 2)
        p.dms.push_back(dm);
    IncoherentSearch search(p);
    std::mt19937 rng(1);
    std::normal_distribution<float> gauss(0, 1);
    std::vector<float> power(p.num_freq);
    std::vector<IncoherentCandidate> cands;
    for (int t = 0; t < 13000; ++t) {
        for (int c = 0; c < p.num_freq; ++c) {
            const bool on = c >= 233 && c < 247 && t >= 5000 && t < 5096;
            power[c] = (float)(1000 * (1 + p.thermal * (gauss(rng) + (on ? 8 : 0))));
        }
        search.add_sample(power.data());
        for (const IncoherentCandidate& c : search.take_candidates())
            cands.push_back(c);
    }
    BOOST_REQUIRE_GE(cands.size(), 10);
    double dm_low = cands[0].dm, dm_high = cands[0].dm;
    for (const IncoherentCandidate& c : cands) {
        BOOST_CHECK(c.undispersed);
        dm_low = std::min(dm_low, c.dm);
        dm_high = std::max(dm_high, c.dm);
    }
    BOOST_CHECK_GT(dm_high - dm_low, 150);
}

BOOST_AUTO_TEST_CASE(channels_back_from_no_data_start_again) {
    // After a frame of no data every channel has another level, by up to 1 sigma and smooth
    // over the band, as when the set of feeds in the sum changes. Against the statistics from
    // before, the low DMs would give candidates of 20 sigma.
    IncoherentSearchParams p = test_params();
    p.dms.clear();
    for (double dm = 3; dm <= 30; dm += 1)
        p.dms.push_back(dm);
    IncoherentSearch search(p);
    std::mt19937 rng(1);
    std::normal_distribution<float> gauss(0, 1);
    std::vector<float> power(p.num_freq);
    size_t num_candidates = 0;
    for (int t = 0; t < 12000; ++t) {
        for (int c = 0; c < p.num_freq; ++c) {
            const double level =
                t < 6000 ? 1000 : 1000 * (1 + p.thermal * std::sin(2 * M_PI * c / p.num_freq));
            power[c] = t >= 6000 && t < 6032 ? 0 : (float)(level * (1 + p.thermal * gauss(rng)));
        }
        search.add_sample(power.data());
        num_candidates += search.take_candidates().size();
    }
    BOOST_CHECK_EQUAL(num_candidates, 0);
    BOOST_CHECK_EQUAL(search.channels_used(), p.num_freq);
}

BOOST_AUTO_TEST_CASE(masked_band_is_not_searched) {
    // Channels 0-96 are at 400-550 MHz.
    IncoherentSearchParams p = test_params();
    p.mask_bands_mhz = {{400, 550}};
    BOOST_CHECK(run(p, {{4, 100, 6000, 96}}).empty());
}

BOOST_AUTO_TEST_CASE(faint_pulse_is_found_beside_a_bright_one_at_another_dm) {
    // 2.5 sigma per channel is about 35 sigma in the sum, and 0.5 is 8. The faint pulse lies
    // within the sweep between the two DMs, where the bright pulse has no residue this strong.
    IncoherentSearchParams p = test_params();
    p.threshold = 6;
    const std::vector<IncoherentCandidate> cands =
        run(p, {{2.5, 100, 6000, p.num_freq}, {0.5, 140, 6200, p.num_freq}});
    BOOST_REQUIRE_EQUAL(cands.size(), 2);
    const IncoherentCandidate& faint = cands[0].snr < cands[1].snr ? cands[0] : cands[1];
    BOOST_CHECK_EQUAL(faint.dm, 140);
    BOOST_CHECK_EQUAL(faint.sample, 6200);
    BOOST_CHECK_LT(faint.snr, 11);
}
