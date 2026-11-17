#include "IncoherentSearch.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace {
// Dispersion delay in ms = dispersion_ms * DM * (difference of GHz^-2). basebandApiManager::K_DM
// is the same constant in Hz^2 s.
const double dispersion_ms = 4.148808;
// Pulse widths tried, in samples
const std::array<int, 5> widths = {1, 2, 4, 8, 16};
// Samples on each side of a pulse window that give its baseline
const int baseline = 128;
// A normalized sample is limited to this many sigma
const double sample_clip = 3.0;
// Stored samples are in units of 1 / sample_scale sigma: int8 reaches 4 sigma
const double sample_scale = 32.0;
// A channel is masked when its sigma/mean exceeds mask_threshold times the lower quartile of the
// channels within mask_neighbours on each side
const double mask_threshold = 1.25;
const int mask_neighbours = 256;
// A channel is masked when the variance of its block means exceeds this times that of white noise
const double drift_threshold = 2.0;
// Samples per block of the drift test
const int drift_block = 64;
// Updates of a running statistic before the update is clipped
const int64_t unclipped_updates = 64;
// An update of a running statistic is limited to this many sigma
const double update_clip = 5.0;
// A sample merges into a candidate at another DM only below this fraction of its S/N
const double residue_fraction = 0.3;
// A sample merges into a candidate within this many samples of it, plus the sweep between the
// two DMs
const int merge_slack = 20;
// A running statistic is used after this many seconds of updates
const double settle_s = 1.0;
// A candidate is reported this many seconds after its first sample above threshold
const double close_after_s = 2.0;
// Seconds between updates of the neighbours' reference
const double ref_interval_s = 0.5;
} // namespace

IncoherentSearch::IncoherentSearch(const IncoherentSearchParams& params) :
    p(params), F(params.num_freq), D((int)params.dms.size()), W((int)widths.size()),
    a(params.sample_s / params.tau_s) {
    if (F <= 0 || D == 0 || p.sample_s <= 0 || p.tau_s <= 0 || p.thermal <= 0 || p.dfreq_mhz <= 0
        || p.freq0_mhz <= 0)
        throw std::invalid_argument("IncoherentSearch: invalid parameters");

    holdoff = std::llround(p.holdoff_s / p.sample_s);
    min_count = std::llround(settle_s / p.sample_s);
    close_after = std::llround(close_after_s / p.sample_s);
    ref_interval = std::max<int64_t>(1, std::llround(ref_interval_s / p.sample_s));

    mean.assign(F, 0.0);
    var.assign(F, 0.0);
    count.assign(F, 0);
    ref.assign(F, 1e9f);
    block_sum.assign(F, 0.0);
    drift.assign(F, 0.0);
    scratch.assign(F, 0.0f);
    state.assign(F, ChannelState::missing);
    band_ok.assign(F, 1);
    for (int c = 0; c < F; ++c) {
        const double f = p.freq0_mhz + p.dfreq_mhz * c;
        for (const auto& band : p.mask_bands_mhz)
            if (f >= band.first && f <= band.second)
                band_ok[c] = 0;
    }

    const double sample_ms = p.sample_s * 1e3;
    const double nu_top = (p.freq0_mhz + p.dfreq_mhz * (F - 1)) / 1e3;
    const double nu_bottom = p.freq0_mhz / 1e3;
    sweep_per_dm =
        dispersion_ms * (1 / (nu_bottom * nu_bottom) - 1 / (nu_top * nu_top)) / sample_ms;
    delay.resize((size_t)D * F);
    max_delay.assign(D, 0);
    for (int d = 0; d < D; ++d) {
        for (int c = 0; c < F; ++c) {
            const double nu = (p.freq0_mhz + p.dfreq_mhz * c) / 1e3;
            const double ms = dispersion_ms * p.dms[d] * (1 / (nu * nu) - 1 / (nu_top * nu_top));
            const int32_t n = (int32_t)std::lround(ms / sample_ms);
            delay[(size_t)d * F + c] = n;
            max_delay[d] = std::max<int64_t>(max_delay[d], n);
        }
        max_delay_all = std::max(max_delay_all, max_delay[d]);
    }

    const int wmax = widths.back();
    lookahead = 2 * wmax + baseline;
    history_len = lookahead + wmax + baseline;
    history.assign((size_t)D * (history_len + chunk), 0.0f);
    history_n.assign(D, 0);
    next_sum.assign(D, 0);
    next_eval.assign(D, wmax + baseline);
    sum_var.assign((size_t)D * W, 0.0);
    sum_count.assign((size_t)D * W, 0);
    sigma_floor.assign(W, 0.0);
    if (p.keep_sums)
        pass_sums.assign((size_t)D * chunk, 0.0f);

    // A candidate's samples are still in the buffer when it is closed: band_occupancy relies on it.
    keep = max_delay_all + lookahead + wmax + 2 * chunk + close_after;
    buffer_len = keep + std::max<int64_t>(4096, keep / 4);
    w.assign((size_t)F * buffer_len, 0);
}

void IncoherentSearch::update_reference() {
    std::vector<float> ratio(F), window(2 * mask_neighbours + 1);
    for (int c = 0; c < F; ++c) {
        const bool known = count[c] >= min_count && mean[c] > 0 && var[c] > 0;
        ratio[c] = known ? (float)(std::sqrt(var[c]) / mean[c] / p.thermal) : 1e9f;
    }
    for (int c = 0; c < F; ++c) {
        const int lo = std::max(0, c - mask_neighbours);
        const int hi = std::min(F - 1, c + mask_neighbours);
        const int n = hi - lo + 1;
        std::copy(ratio.begin() + lo, ratio.begin() + hi + 1, window.begin());
        std::nth_element(window.begin(), window.begin() + n / 4, window.begin() + n);
        // A bright source raises the noise of every channel, so the reference is what the
        // neighbours do, but never below thermal.
        ref[c] = std::max(window[n / 4], 1.0f);
    }
}

bool IncoherentSearch::add_sample(const float* power) {
    // When the buffer is full, shift it down to its newest `keep` samples.
    if (ns - base == buffer_len) {
        const int64_t drop = buffer_len - keep;
        for (int c = 0; c < F; ++c)
            std::memmove(&w[(size_t)c * buffer_len], &w[(size_t)c * buffer_len + drop], keep);
        base += drop;
    }
    if (ns % ref_interval == 0)
        update_reference();

    // Normalize each channel by its statistics before this sample, decide whether it is used,
    // then update the statistics.
    double zero_dm = 0;
    int n_used = 0;
    for (int c = 0; c < F; ++c) {
        const double x = power[c];
        state[c] = ChannelState::missing;
        scratch[c] = 0;
        if (!(x > 0))
            continue;
        if (count[c] == 0) {
            mean[c] = x;
            count[c] = 1;
            continue;
        }
        const double d = x - mean[c];
        const double sd = std::sqrt(var[c]);
        if (count[c] >= min_count && sd > 0 && mean[c] > 0) {
            scratch[c] = (float)std::clamp(d / sd, -sample_clip, sample_clip);
            const bool quiet = sd <= mask_threshold * ref[c] * p.thermal * mean[c];
            const bool steady = drift[c] <= drift_threshold;
            state[c] = band_ok[c] && quiet && steady ? ChannelState::used : ChannelState::masked;
            if (state[c] == ChannelState::used) {
                zero_dm += scratch[c];
                ++n_used;
            }
        }
        // A variance of zero must stay free to grow.
        const double dc = count[c] >= unclipped_updates && sd > 0
                              ? std::clamp(d, -update_clip * sd, update_clip * sd)
                              : d;
        const double weight = std::max(a, 1.0 / (count[c] + 1));
        mean[c] += weight * dc;
        var[c] += weight * (dc * dc - var[c]);
        ++count[c];
    }
    if (n_used)
        zero_dm /= n_used;
    num_used = n_used;

    // Store the sample less the average of the used channels (zero DM). A masked channel stores
    // 0 but stays in the drift test, so that it can come back.
    const int64_t col = ns - base;
    for (int c = 0; c < F; ++c) {
        const double v =
            state[c] == ChannelState::used ? (scratch[c] - zero_dm) * sample_scale : 0.0;
        w[(size_t)c * buffer_len + col] = (int8_t)std::lround(std::clamp(v, -127.0, 127.0));
        if (state[c] != ChannelState::missing)
            block_sum[c] += scratch[c] - zero_dm;
    }
    ++ns;

    // Drift test: the running variance of the means of drift_block samples, scaled so that
    // white noise gives 1.
    if (ns % drift_block == 0) {
        const int64_t blocks = ns / drift_block;
        const double weight = std::max(std::min(1.0, a * drift_block), 1.0 / blocks);
        for (int c = 0; c < F; ++c) {
            const double m = block_sum[c] / drift_block;
            drift[c] += weight * (drift_block * m * m - drift[c]);
            block_sum[c] = 0;
        }
    }
    // Search once per chunk of samples.
    if (ns % chunk != 0)
        return false;
    dedisperse_chunk();
    close_events();
    return true;
}

void IncoherentSearch::dedisperse_chunk() {
    std::vector<Hit> hits;
    double max_snr = 0;
    std::fill(pass_sums.begin(), pass_sums.end(), 0.0f);
#pragma omp parallel num_threads(p.num_threads)
    {
        std::vector<Hit> my_hits;
        std::vector<float> sums(chunk);
        std::vector<int32_t> totals(chunk);
        std::vector<int16_t> parts(chunk);
        std::vector<double> prefix(history_len + chunk + 1);
        double my_max = 0;
#pragma omp for schedule(dynamic, 8) nowait
        for (int d = 0; d < D; ++d) {
            // The sum for top-channel time t needs the samples up to t + max_delay.
            const int64_t t0 = next_sum[d];
            const int64_t n = ns - max_delay[d] - t0;
            if (n <= 0)
                continue;
            // 16-bit partial sums hold up to 258 samples of 127; they are added up in 32 bits.
            int32_t* const total = totals.data();
            int16_t* const part = parts.data();
            std::fill(total, total + n, 0);
            std::fill(part, part + n, 0);
            const int32_t* const dl = &delay[(size_t)d * F];
            int in_part = 0;
            for (int c = 0; c < F; ++c) {
                if (!band_ok[c])
                    continue; // always zero
                const int8_t* const h = row(c, t0 + dl[c]);
#pragma omp simd
                for (int64_t k = 0; k < n; ++k)
                    part[k] += h[k];
                if (++in_part == 256) {
#pragma omp simd
                    for (int64_t k = 0; k < n; ++k)
                        total[k] += part[k];
                    std::fill(part, part + n, 0);
                    in_part = 0;
                }
            }
            float* const s = sums.data();
            for (int64_t k = 0; k < n; ++k)
                s[k] = (total[k] + part[k]) / (float)sample_scale;
            next_sum[d] = t0 + n;
            // The last of these sums reaches the bottom channel at the newest sample.
            if (p.keep_sums)
                std::copy(s, s + n, &pass_sums[(size_t)d * chunk + chunk - n]);
            search_dm(d, s, n, prefix, my_hits, my_max);
        }
#pragma omp critical
        {
            hits.insert(hits.end(), my_hits.begin(), my_hits.end());
            max_snr = std::max(max_snr, my_max);
        }
    }
    max_snr_seen = std::max(max_snr_seen, max_snr);

    // Near DM 0 the zero-DM subtraction removes the sum itself; do not divide by what is left.
    std::vector<double> sigma;
    for (int wi = 0; wi < W; ++wi) {
        sigma.clear();
        for (int d = 0; d < D; ++d)
            if (sum_count[(size_t)d * W + wi] > 0)
                sigma.push_back(std::sqrt(sum_var[(size_t)d * W + wi]));
        if (sigma.empty())
            continue;
        std::nth_element(sigma.begin(), sigma.begin() + sigma.size() / 2, sigma.end());
        sigma_floor[wi] = 0.5 * sigma[sigma.size() / 2];
    }

    std::sort(hits.begin(), hits.end(), [](const Hit& x, const Hit& y) { return x.snr > y.snr; });
    merge_hits(hits);
}

void IncoherentSearch::search_dm(int d, const float* sums, int64_t n, std::vector<double>& prefix,
                                 std::vector<Hit>& hits, double& max_snr) {
    float* const h = &history[(size_t)d * (history_len + chunk)];
    std::memcpy(h + history_n[d], sums, n * sizeof(float));
    history_n[d] += n;
    const int64_t hn = history_n[d];
    const int64_t start = next_sum[d] - hn; // top-channel time of h[0]
    prefix[0] = 0;
    for (int64_t i = 0; i < hn; ++i)
        prefix[i + 1] = prefix[i] + h[i];

    const int nb = baseline;
    int64_t t = next_eval[d];
    for (; t + lookahead <= next_sum[d]; ++t) {
        const int64_t i = t - start;
        for (int wi = 0; wi < W; ++wi) {
            // The window of wd samples from t, less the mean of nb samples on each side, which
            // start wd samples clear of the window.
            const int wd = widths[wi];
            const double on = prefix[i + wd] - prefix[i];
            const double sides = (prefix[i - wd] - prefix[i - wd - nb])
                                 + (prefix[i + 2 * wd + nb] - prefix[i + 2 * wd]);
            const double v = on - sides * wd / (2.0 * nb);
            // Exactly zero means no data (start-up, or every channel missing): leave the
            // statistics as they are.
            if (v == 0)
                continue;

            double& variance = sum_var[(size_t)d * W + wi];
            int64_t& updates = sum_count[(size_t)d * W + wi];
            const double sd = std::sqrt(variance);
            const double sigma = std::max(sd, sigma_floor[wi]);
            if (updates >= min_count && t >= holdoff && sigma > 0) {
                const double snr = v / sigma;
                max_snr = std::max(max_snr, snr);
                if (snr > p.threshold)
                    hits.push_back({(float)snr, d, wi, t});
            }
            const double vc = updates >= unclipped_updates && sd > 0
                                  ? std::clamp(v, -update_clip * sd, update_clip * sd)
                                  : v;
            variance += std::max(a, 1.0 / (updates + 1)) * (vc * vc - variance);
            ++updates;
        }
    }
    next_eval[d] = t;

    // Keep what the next windows' left baselines need.
    const int64_t drop = (t - widths.back() - nb) - start;
    if (drop > 0) {
        std::memmove(h, h + drop, (hn - drop) * sizeof(float));
        history_n[d] = hn - drop;
    }
}

void IncoherentSearch::merge_hits(const std::vector<Hit>& hits) {
    for (const Hit& hit : hits) {
        const double dm = p.dms[hit.dm_index];
        bool merged = false;
        for (Event& e : events) {
            const double ddm = std::abs(dm - p.dms[e.best.dm_index]);
            if (std::abs((double)(hit.sample - e.best.sample)) > merge_slack + sweep_per_dm * ddm)
                continue;
            // The same pulse while the two sweeps differ by a few of the widest window
            const bool same_pulse = sweep_per_dm * ddm <= 4.0 * widths.back();
            if (!same_pulse && hit.snr >= residue_fraction * e.best.snr)
                continue;
            ++e.num_hits;
            if (!e.reported && hit.snr > e.best.snr)
                e.best = hit;
            merged = true;
            break;
        }
        if (!merged)
            events.push_back({hit, 1, ns, false});
    }
}

void IncoherentSearch::close_events() {
    for (Event& e : events) {
        if (e.reported || ns - e.created < close_after)
            continue;
        IncoherentCandidate cand;
        cand.snr = e.best.snr;
        cand.dm = p.dms[e.best.dm_index];
        cand.width = widths[e.best.width_index];
        cand.sample = e.best.sample;
        cand.num_hits = e.num_hits;
        band_occupancy(cand, e.best);
        candidates.push_back(cand);
        e.reported = true;
    }
    // A reported event stays until its residue at the highest DM has been searched.
    const int64_t lifetime = max_delay_all + lookahead + 2 * chunk + close_after;
    events.erase(std::remove_if(events.begin(), events.end(),
                                [&](const Event& e) { return ns - e.created > lifetime; }),
                 events.end());
}

// S/N of the candidate's window in each block of channels: the sum of the stored samples along
// the dispersion curve over the root of their number. A channel counts when its first sample is
// not 0, which leaves out the masked channels (and the rare used sample that rounds to 0).
void IncoherentSearch::band_occupancy(IncoherentCandidate& cand, const Hit& hit) const {
    const int32_t* const dl = &delay[(size_t)hit.dm_index * F];
    const int wd = widths[hit.width_index];
    for (int b = 0; b < IncoherentCandidate::num_blocks; ++b) {
        const int c0 = (int)((int64_t)b * F / IncoherentCandidate::num_blocks);
        const int c1 = (int)((int64_t)(b + 1) * F / IncoherentCandidate::num_blocks);
        double sum = 0;
        int n = 0;
        for (int c = c0; c < c1; ++c) {
            const int8_t* const h = row(c, hit.sample + dl[c]);
            n += h[0] != 0;
            for (int k = 0; k < wd; ++k)
                sum += h[k] / sample_scale;
        }
        cand.block_snr[b] = n ? (float)(sum / std::sqrt((double)n * wd)) : 0.0f;
    }
}

std::vector<IncoherentCandidate> IncoherentSearch::take_candidates() {
    std::vector<IncoherentCandidate> out;
    out.swap(candidates);
    return out;
}

double IncoherentSearch::take_max_snr() {
    const double out = max_snr_seen;
    max_snr_seen = 0;
    return out;
}
