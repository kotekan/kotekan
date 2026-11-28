#define BOOST_TEST_MODULE "test_incoherent_search"

#include "IncoherentSearch.hpp"

#include <algorithm>
#include <boost/test/included/unit_test.hpp>
#include <cmath>
#include <cstdint>
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

// Runs noise of unit mean with the pulses through the search. With `burst`, interference at DM 0
// that the limit per block lets through is added: 8 sigma in 14 channels across the edge of the
// top two blocks, for 96 samples from sample 4000. Their delays differ by less than the burst
// lasts, so every DM up to 250 adds them up, to about 9 sigma at the widest window. `cutout`
// gets the cutout of the last candidate.
static std::vector<IncoherentCandidate> run(const IncoherentSearchParams& p,
                                            const std::vector<Pulse>& pulses, bool burst = false,
                                            std::vector<int8_t>* cutout = nullptr) {
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
            double amplitude = burst && c >= 233 && c < 247 && t >= 4000 && t < 4096 ? 8 : 0;
            for (const Pulse& pulse : pulses)
                if (c < pulse.last_channel && t == pulse.sample + std::lround(pulse.dm * delay[c]))
                    amplitude += pulse.amplitude;
            power[c] = (float)(mean * (1 + p.thermal * (gauss(rng) + amplitude)));
        }
        search.add_sample(power.data());
        for (const IncoherentCandidate& c : search.take_candidates()) {
            out.push_back(c);
            if (cutout != nullptr) {
                cutout->resize((size_t)p.num_freq * IncoherentSearch::cutout_samples);
                search.cutout(c, cutout->data());
            }
        }
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
    BOOST_CHECK_EQUAL(c.num_partners, 0);
    BOOST_CHECK(!c.one_band);
    // The pulse fills the band: no part of it has much more S/N than the whole.
    BOOST_CHECK_GT(c.part_ratio, 0.9);
    BOOST_CHECK_LT(c.part_ratio, 1.15);
    BOOST_CHECK_GT(c.num_channels, 240);
    int filled = 0;
    for (float b : c.block_snr)
        filled += b > 3;
    BOOST_CHECK_GE(filled, 14);
}

BOOST_AUTO_TEST_CASE(cutout_has_the_pulse_in_its_middle_column) {
    const IncoherentSearchParams p = test_params();
    std::vector<int8_t> cutout;
    const std::vector<IncoherentCandidate> cands =
        run(p, {{1.5, 100, 6000, p.num_freq}}, false, &cutout);
    BOOST_REQUIRE_EQUAL(cands.size(), 1);
    const int T = IncoherentSearch::cutout_samples;
    BOOST_REQUIRE_EQUAL(cutout.size(), (size_t)p.num_freq * T);
    // 1.5 sigma in each of 256 channels in the pulse's column, noise of 16 sigma in the others.
    for (int k = 0; k < T; ++k) {
        double sum = 0;
        for (int c = 0; c < p.num_freq; ++c)
            sum += cutout[(size_t)c * T + k] / (double)IncoherentSearch::sample_scale;
        if (k == T / 2)
            BOOST_CHECK_GT(sum, 300);
        else
            BOOST_CHECK_LT(std::abs(sum), 100);
    }
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

// Trial DMs 50 to 250, for the burst of run(). The burst's channels are masked 32 samples after
// it begins, and a margin before the mask would take those too.
static IncoherentSearchParams burst_params() {
    IncoherentSearchParams p = test_params();
    p.threshold = 6;
    p.mask_margin_s = 0;
    p.dms.clear();
    for (double dm = 50; dm <= 250; dm += 2)
        p.dms.push_back(dm);
    return p;
}

BOOST_AUTO_TEST_CASE(burst_without_dispersion_is_marked_at_every_dm) {
    const std::vector<IncoherentCandidate> cands = run(burst_params(), {}, true);
    BOOST_REQUIRE_GE(cands.size(), 10);
    double dm_low = cands[0].dm, dm_high = cands[0].dm;
    int one_band = 0;
    for (const IncoherentCandidate& c : cands) {
        BOOST_CHECK(c.undispersed);
        BOOST_CHECK_GE(c.num_partners, 2);
        one_band += c.one_band;
        dm_low = std::min(dm_low, c.dm);
        dm_high = std::max(dm_high, c.dm);
    }
    BOOST_CHECK_GT(dm_high - dm_low, 150);
    // The burst is 16 samples wide in the search and has its S/N in the top two blocks.
    BOOST_CHECK_GT(one_band, (int)cands.size() / 2);
}

BOOST_AUTO_TEST_CASE(burst_is_masked_from_its_start_with_the_margin) {
    IncoherentSearchParams p = burst_params();
    BOOST_CHECK(!run(p, {}, true).empty());
    p.mask_margin_s = IncoherentSearchParams().mask_margin_s;
    BOOST_CHECK(run(p, {}, true).empty());
}

BOOST_AUTO_TEST_CASE(burst_over_a_narrow_range_of_dm_is_marked_in_the_middle) {
    // With trial DMs 50 to 140 no candidate near DM 95 has another more than 50 away. All are
    // marked because each has partners, and with them covers more than 50.
    IncoherentSearchParams p = burst_params();
    while (p.dms.back() > 140)
        p.dms.pop_back();
    const std::vector<IncoherentCandidate> cands = run(p, {}, true);
    BOOST_REQUIRE_GE(cands.size(), 8);
    int middle = 0;
    for (const IncoherentCandidate& c : cands) {
        BOOST_CHECK(c.undispersed);
        middle += c.dm > 90 && c.dm < 100;
    }
    BOOST_CHECK_GE(middle, 1);
}

BOOST_AUTO_TEST_CASE(pulse_beside_a_burst_is_not_marked) {
    // A pulse of about 12 sigma in the lower half of the band, 200 samples after the burst
    // began. The burst's candidates above DM 150 are placed before it, within the sweep
    // between the DMs, but their S/N is in the top two blocks and the pulse's is not.
    const std::vector<IncoherentCandidate> cands =
        run(burst_params(), {{1.5, 100, 4200, 128}}, true);
    int pulses = 0, marked = 0;
    for (const IncoherentCandidate& c : cands) {
        if (c.dm == 100 && c.sample == 4200) {
            BOOST_CHECK(!c.undispersed);
            // The pulse is in half of the band.
            BOOST_CHECK_GT(c.part_ratio, 1.25);
            ++pulses;
        } else {
            marked += c.undispersed;
        }
    }
    BOOST_CHECK_EQUAL(pulses, 1);
    BOOST_CHECK_GE(marked, 10);
}

BOOST_AUTO_TEST_CASE(candidates_from_noise_are_not_partners) {
    // At 4 sigma the noise gives many candidates, and several of them fit the arrival of a
    // pulse of about 8 sigma without dispersion.
    IncoherentSearchParams p = burst_params();
    p.threshold = 4;
    const std::vector<IncoherentCandidate> cands = run(p, {{0.5, 150, 4200, p.num_freq}});
    BOOST_CHECK_GE(cands.size(), 50);
    int pulses = 0;
    for (const IncoherentCandidate& c : cands) {
        BOOST_CHECK(!c.undispersed);
        if (c.dm == 150 && c.sample == 4200) {
            BOOST_CHECK_GT(c.snr, 6);
            ++pulses;
        }
    }
    BOOST_CHECK_EQUAL(pulses, 1);
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

BOOST_AUTO_TEST_CASE(waterfall_bins_the_samples_and_marks_what_is_masked) {
    // Channels 0-96 are at 400-550 MHz: columns 0-11 of 8 channels are masked whole.
    IncoherentSearchParams p = test_params();
    p.mask_bands_mhz = {{400, 550}};
    IncoherentSearch search(p);
    std::mt19937 rng(1);
    std::normal_distribution<float> gauss(0, 1);
    std::vector<float> power(p.num_freq);
    for (int t = 0; t < 8000; ++t) {
        for (int c = 0; c < p.num_freq; ++c)
            power[c] = (float)(1000 * (1 + p.thermal * gauss(rng)));
        search.add_sample(power.data());
    }
    const int rows = 16, columns = 32;
    std::vector<int8_t> image((size_t)rows * columns);
    search.waterfall(7000, rows, 32, 8, image.data());
    double square = 0;
    int used = 0;
    for (int r = 0; r < rows; ++r)
        for (int b = 0; b < columns; ++b) {
            const int8_t v = image[(size_t)r * columns + b];
            if (b < 12) {
                BOOST_CHECK_EQUAL(v, IncoherentSearch::waterfall_no_data);
            } else if (b > 12) {
                BOOST_CHECK_NE(v, IncoherentSearch::waterfall_no_data);
                square += (double)v * v;
                ++used;
            }
        }
    // Thermal noise is about 1 sigma in a bin.
    const double rms = std::sqrt(square / used) / IncoherentSearch::waterfall_scale;
    BOOST_CHECK_GT(rms, 0.8);
    BOOST_CHECK_LT(rms, 1.1);

    // Rows past the newest sample have no data.
    search.waterfall(8000 - 32, 2, 32, 8, image.data());
    BOOST_CHECK_NE(image[columns - 1], IncoherentSearch::waterfall_no_data);
    BOOST_CHECK_EQUAL(image[2 * columns - 1], IncoherentSearch::waterfall_no_data);
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
