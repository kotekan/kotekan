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
// A channel is masked when its sigma/mean exceeds mask_threshold times the lower quartile of the
// channels within mask_neighbours on each side
const double mask_threshold = 1.25;
const int mask_neighbours = 256;
// A channel is masked when the variance of its block means exceeds this times that of white noise
const double drift_threshold = 2.0;
// Samples per block of the drift test
const int drift_block = 64;
// The tests of a channel see interference a few MHz wide only after tens of ms, and in its
// strongest channels first. So a channel that fails a test takes mask_margin_channels on each
// side with it, and a channel that becomes masked loses its last mask_margin_s too.
const int mask_margin_channels = 2;
// Updates of a running statistic before the update is clipped
const int64_t unclipped_updates = 64;
// An update of a running statistic is limited to this many sigma
const double update_clip = 5.0;
// A sample merges into a candidate at another DM only below this fraction of its S/N. The
// fraction holds where the two sweeps stop being the same pulse and falls from there as one over
// the difference of DM, as the residue of a pulse does.
const double residue_fraction = 0.3;
// A block of channels is left out of a time sample when the sum of its normalized samples
// exceeds this many sigma
const double block_clip = 6.0;
// A sample merges into a candidate within this many samples of it, plus the sweep between the
// two DMs
const int merge_slack = 20;
// A partner of a candidate is another one more than partner_dm from it in DM, of at least
// far_fraction of its S/N and of partner_min_snr, that fits the same arrival without dispersion
// and has its strongest block of channels where the candidate is strong too: within far_blocks
// of a block in which the candidate has block_fraction of its own highest S/N. Two sources of
// interference at once can put a candidate's highest S/N in another block than its partners'.
// A candidate is undispersed when it has far_candidates partners and they cover, with it, more
// than far_dm in DM. A pulse's own candidates at neighbouring DMs lie within partner_dm, and two
// pulsars in the beam within far_dm of each other. With a threshold below partner_min_snr the
// candidates from noise are many, and several fit any arrival by chance.
const double partner_dm = 20.0;
const double far_dm = 50.0;
const double far_fraction = 0.5;
const double partner_min_snr = 6.0;
const int far_candidates = 2;
const int far_blocks = 1;
const double block_fraction = 0.5;
// A candidate is described as one_band when it is at least one_band_width samples wide and one
// block of channels has more than one_band_ratio times its S/N. A pulse that fills the band has
// a quarter of its S/N in each of 16 blocks. On the pathfinder, interference a few MHz wide
// that lasts tens of ms has a ratio of 0.9 to 2; pulses of the Crab have about 1 but are narrower
// than one_band_width.
const int one_band_width = 8;
const double one_band_ratio = 0.8;
// part_snr: adjacent channels with data in a bin, and the limit on a bin's sum in sigma
const int part_bin = 8;
const double part_clip = 3.0;
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
        || p.freq0_mhz <= 0 || p.mask_margin_s < 0)
        throw std::invalid_argument("IncoherentSearch: invalid parameters");

    holdoff = std::llround(p.holdoff_s / p.sample_s);
    min_count = std::llround(settle_s / p.sample_s);
    close_after = std::llround(close_after_s / p.sample_s);
    mask_margin = std::llround(p.mask_margin_s / p.sample_s);
    ref_interval = std::max<int64_t>(1, std::llround(ref_interval_s / p.sample_s));

    mean.assign(F, 0.0);
    var.assign(F, 0.0);
    count.assign(F, 0);
    ref.assign(F, 0.0f);
    block_sum.assign(F, 0.0);
    drift.assign(F, 0.0);
    scratch.assign(F, 0.0f);
    noisy.assign(F, 0);
    was_used.assign(F, 0);
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
    pass_max.assign(D, 0.0f);
    if (p.keep_sums)
        pass_sums.assign((size_t)D * chunk, 0.0f);

    // A candidate's samples are still in the buffer when it is closed, with half a cutout
    // before them: band_occupancy and cutout rely on it.
    keep = max_delay_all + lookahead + wmax + 4 * chunk + close_after + cutout_samples / 2;
    buffer_len = keep + std::max<int64_t>(4096, keep / 4);
    w.assign((size_t)F * buffer_len, 0);
}

void IncoherentSearch::update_reference() {
    std::vector<float> ratio(F), window(2 * mask_neighbours + 1);
    for (int c = 0; c < F; ++c) {
        const bool known = count[c] >= min_count && mean[c] > 0 && var[c] > 0;
        ratio[c] = known ? (float)(std::sqrt(var[c]) / mean[c] / p.thermal) : 0.0f;
    }
    for (int c = 0; c < F; ++c) {
        const int lo = std::max(0, c - mask_neighbours);
        const int hi = std::min(F - 1, c + mask_neighbours);
        int n = 0;
        for (int k = lo; k <= hi; ++k)
            if (ratio[k] > 0)
                window[n++] = ratio[k];
        // Until a neighbour is known the channel is not used.
        if (n == 0) {
            ref[c] = 0;
            continue;
        }
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
        noisy[c] = 0;
        // A channel that comes back from no data starts its statistics again: its level may
        // have changed (another set of feeds, a restarted node).
        if (!(x > 0)) {
            count[c] = 0;
            continue;
        }
        if (count[c] == 0) {
            mean[c] = x;
            var[c] = 0;
            count[c] = 1;
            continue;
        }
        const double d = x - mean[c];
        const double sd = std::sqrt(var[c]);
        if (count[c] >= min_count && sd > 0 && mean[c] > 0) {
            scratch[c] = (float)std::clamp(d / sd, -sample_clip, sample_clip);
            const bool quiet = sd <= mask_threshold * ref[c] * p.thermal * mean[c];
            const bool steady = drift[c] <= drift_threshold;
            noisy[c] = band_ok[c] && !(quiet && steady);
            state[c] = band_ok[c] && quiet && steady ? ChannelState::used : ChannelState::masked;
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
    for (int c = 0; c < F; ++c) {
        if (!noisy[c])
            continue;
        const int lo = std::max(0, c - mask_margin_channels);
        const int hi = std::min(F - 1, c + mask_margin_channels);
        for (int k = lo; k <= hi; ++k)
            if (state[k] == ChannelState::used)
                state[k] = ChannelState::masked;
    }
    // The stored samples that a sum has already taken stay in that sum.
    const int64_t held = ns - base;
    const int64_t margin = std::min(mask_margin, held);
    for (int c = 0; c < F; ++c) {
        if (was_used[c] && state[c] == ChannelState::masked)
            std::memset(&w[(size_t)c * buffer_len + held - margin], 0, margin);
        was_used[c] = state[c] == ChannelState::used;
        if (was_used[c]) {
            zero_dm += scratch[c];
            ++n_used;
        }
    }
    if (n_used)
        zero_dm /= n_used;

    // Interference in part of the band arrives in many channels of a block at once, and each
    // is already at its limit. A dispersed pulse does not: its channels come at different
    // times. Take such a block out of this sample and average the rest again.
    bool block_cut = false;
    for (int b = 0; b < IncoherentCandidate::num_blocks; ++b) {
        double sum = 0;
        int n = 0;
        for (int c = block_start(b); c < block_start(b + 1); ++c)
            if (state[c] == ChannelState::used) {
                sum += scratch[c] - zero_dm;
                ++n;
            }
        if (n == 0 || std::abs(sum) <= block_clip * std::sqrt((double)n))
            continue;
        for (int c = block_start(b); c < block_start(b + 1); ++c)
            if (state[c] == ChannelState::used)
                state[c] = ChannelState::masked;
        block_cut = true;
    }
    if (block_cut) {
        zero_dm = 0;
        n_used = 0;
        for (int c = 0; c < F; ++c)
            if (state[c] == ChannelState::used) {
                zero_dm += scratch[c];
                ++n_used;
            }
        if (n_used)
            zero_dm /= n_used;
    }
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
    std::fill(pass_max.begin(), pass_max.end(), 0.0f);
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
                pass_max[d] = std::max(pass_max[d], (float)snr);
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
            const double same_sweep = 4.0 * widths.back();
            const double sweep = sweep_per_dm * ddm;
            if (sweep > same_sweep && hit.snr >= residue_fraction * e.best.snr * same_sweep / sweep)
                continue;
            ++e.num_hits;
            if (!e.reported && hit.snr > e.best.snr) {
                e.best = hit;
                e.block = strongest_block(hit);
            }
            merged = true;
            break;
        }
        if (!merged)
            events.push_back({hit, 1, ns, false, strongest_block(hit)});
    }
}

void IncoherentSearch::close_events() {
    for (Event& e : events) {
        // An arrival without dispersion is found at the higher DMs up to the difference of the
        // sweeps later: wait for those candidates.
        const int64_t wait = close_after + max_delay_all - max_delay[e.best.dm_index] + 2 * chunk;
        if (e.reported || ns - e.created < wait)
            continue;
        IncoherentCandidate cand;
        cand.snr = e.best.snr;
        cand.dm = p.dms[e.best.dm_index];
        cand.dm_index = e.best.dm_index;
        cand.width = widths[e.best.width_index];
        cand.sample = e.best.sample;
        cand.num_hits = e.num_hits;
        band_occupancy(cand, e.best);
        double dm_range = 0;
        cand.num_partners = count_partners(e, cand.block_snr, dm_range);
        cand.undispersed = cand.num_partners >= far_candidates && dm_range > far_dm;
        cand.one_band = cand.width >= one_band_width
                        && *std::max_element(cand.block_snr.begin(), cand.block_snr.end())
                               > one_band_ratio * cand.snr;
        band_parts(cand);
        candidates.push_back(cand);
        e.reported = true;
    }
    // A reported event stays until its residue at the highest DM has been searched, and until
    // the candidates at higher DMs that it may mark undispersed are reported.
    const int64_t lifetime = max_delay_all + lookahead + 6 * chunk + close_after;
    events.erase(std::remove_if(events.begin(), events.end(),
                                [&](const Event& e) { return ns - e.created > lifetime; }),
                 events.end());
}

// The other events that fit the same arrival without dispersion, and in dm_range the range of
// DM that they and this event cover. The search places such an arrival earlier at the top
// channel the higher the DM: by nothing when it is at the top of the band, by the whole sweep
// between the two DMs when it is at the bottom. The arrival is in one part of the band, so the
// others have their S/N where this event, with block_snr along its curve, has S/N too.
int IncoherentSearch::count_partners(
    const Event& e, const std::array<float, IncoherentCandidate::num_blocks>& block_snr,
    double& dm_range) const {
    const double dm = p.dms[e.best.dm_index];
    const float strong = block_fraction * *std::max_element(block_snr.begin(), block_snr.end());
    double lowest = 0, highest = 0; // of the partners' DMs, relative to this event's
    int n = 0;
    for (const Event& other : events) {
        const double ddm = p.dms[other.best.dm_index] - dm;
        if (std::abs(ddm) <= partner_dm
            || other.best.snr < std::max(far_fraction * e.best.snr, partner_min_snr))
            continue;
        bool shared = false;
        for (int b = std::max(0, other.block - far_blocks);
             b <= std::min(IncoherentCandidate::num_blocks - 1, other.block + far_blocks); ++b)
            shared = shared || block_snr[b] >= strong;
        if (!shared)
            continue;
        const int64_t earlier =
            ddm > 0 ? e.best.sample - other.best.sample : other.best.sample - e.best.sample;
        if (earlier < -merge_slack || earlier > merge_slack + sweep_per_dm * std::abs(ddm))
            continue;
        ++n;
        lowest = std::min(lowest, ddm);
        highest = std::max(highest, ddm);
    }
    dm_range = highest - lowest;
    return n;
}

// S/N of the candidate's window in each block of channels: the sum of the stored samples along
// the dispersion curve over the root of their number. A channel counts when its first sample is
// not 0, which leaves out the masked channels (and the rare used sample that rounds to 0); their
// number is the candidate's num_channels.
void IncoherentSearch::band_occupancy(IncoherentCandidate& cand, const Hit& hit) const {
    const int32_t* const dl = &delay[(size_t)hit.dm_index * F];
    const int wd = widths[hit.width_index];
    cand.num_channels = 0;
    for (int b = 0; b < IncoherentCandidate::num_blocks; ++b) {
        double sum = 0;
        int n = 0;
        for (int c = block_start(b); c < block_start(b + 1); ++c) {
            const int8_t* const h = row(c, hit.sample + dl[c]);
            n += h[0] != 0;
            for (int k = 0; k < wd; ++k)
                sum += h[k] / sample_scale;
        }
        cand.block_snr[b] = n ? (float)(sum / std::sqrt((double)n * wd)) : 0.0f;
        cand.num_channels += n;
    }
}

// part_snr and part_ratio of a candidate with dm_index and sample set. Windows of every width
// that start up to the width and 2 samples before the candidate and up to 2 after are tried.
void IncoherentSearch::band_parts(IncoherentCandidate& cand) const {
    const int32_t* const dl = &delay[(size_t)cand.dm_index * F];
    const int wmax = widths.back();
    const int span = 2 * wmax + 4;
    const int64_t first = cand.sample - wmax - 2;
    // For each bin of part_bin channels with data, the running sums over the span of the
    // samples and of their number.
    std::vector<int32_t> sum, num;
    std::vector<int8_t> v(span);
    int in_bin = 0;
    for (int c = 0; c < F; ++c) {
        bool data = false;
        for (int k = 0; k < span; ++k) {
            const int64_t s = first + k + dl[c];
            v[k] = s >= base && s < ns ? *row(c, s) : 0;
            data = data || v[k] != 0;
        }
        if (!data)
            continue;
        if (in_bin == 0) {
            sum.resize(sum.size() + span + 1, 0);
            num.resize(num.size() + span + 1, 0);
        }
        int32_t* const bs = &sum[sum.size() - span - 1];
        int32_t* const bn = &num[num.size() - span - 1];
        int32_t run = 0, count = 0;
        for (int k = 0; k < span; ++k) {
            run += v[k];
            count += v[k] != 0;
            bs[k + 1] += run;
            bn[k + 1] += count;
        }
        in_bin = (in_bin + 1) % part_bin;
    }
    // A last bin with fewer channels is left out.
    const int bins = (int)(sum.size() / (span + 1)) - (in_bin != 0);
    const int parts = IncoherentCandidate::num_blocks;
    cand.part_snr = 0;
    cand.part_ratio = 0;
    if (bins < parts)
        return;
    double full = 0, part = 0;
    std::vector<double> cy(parts + 1, 0.0), cn(parts + 1, 0.0);
    for (const int wd : widths) {
        for (int o = wmax - wd; o <= wmax + 4; ++o) {
            double total = 0, total_n = 0;
            for (int b = 0; b < bins; ++b) {
                const size_t at = (size_t)b * (span + 1) + o;
                total += sum[at + wd] - sum[at];
                total_n += num[at + wd] - num[at];
            }
            if (total_n == 0)
                continue;
            full = std::max(full, total / std::sqrt(total_n));
            // A pulse with all its S/N in two of the parts has parts / 2 over the root of the
            // number of bins times the S/N of the whole band in each bin: a bright pulse is not
            // limited below that.
            const double share = parts / 2 * total / std::sqrt(total_n * bins);
            const double clip = std::max(part_clip * sample_scale, share);
            double y = 0, yn = 0;
            int edge = 1;
            for (int b = 0; b < bins; ++b) {
                const size_t at = (size_t)b * (span + 1) + o;
                const double x = sum[at + wd] - sum[at];
                const double n = num[at + wd] - num[at];
                const double limit = clip * std::sqrt(n);
                yn += n;
                y += std::clamp(x, -limit, limit);
                if (b + 1 == (int)((int64_t)edge * bins / parts)) {
                    cy[edge] = y;
                    cn[edge] = yn;
                    ++edge;
                }
            }
            for (int i = 0; i < parts; ++i)
                for (int j = i + 2; j <= parts; ++j)
                    if (cn[j] > cn[i])
                        part = std::max(part, (cy[j] - cy[i]) / std::sqrt(cn[j] - cn[i]));
        }
    }
    cand.part_snr = (float)(part / sample_scale);
    cand.part_ratio = full > 0 ? (float)(part / full) : 0.0f;
}

int IncoherentSearch::strongest_block(const Hit& hit) const {
    IncoherentCandidate cand;
    band_occupancy(cand, hit);
    return (int)(std::max_element(cand.block_snr.begin(), cand.block_snr.end())
                 - cand.block_snr.begin());
}

void IncoherentSearch::cutout(const IncoherentCandidate& cand, int8_t* out) const {
    const int32_t* const dl = &delay[(size_t)cand.dm_index * F];
    for (int c = 0; c < F; ++c) {
        const int64_t first = cand.sample - cutout_samples / 2 + dl[c];
        for (int k = 0; k < cutout_samples; ++k) {
            const int64_t s = first + k;
            out[(size_t)c * cutout_samples + k] = s >= base && s < ns ? *row(c, s) : 0;
        }
    }
}

void IncoherentSearch::waterfall(int64_t first, int rows, int time_bin, int freq_bin,
                                 int8_t* out) const {
    const int columns = (F + freq_bin - 1) / freq_bin;
    std::vector<int32_t> sum(columns), used(columns);
    for (int r = 0; r < rows; ++r) {
        const int64_t s = first + (int64_t)r * time_bin;
        std::fill(sum.begin(), sum.end(), 0);
        std::fill(used.begin(), used.end(), 0);
        if (s >= base && s + time_bin <= ns)
            for (int c = 0; c < F; ++c) {
                const int8_t* const v = row(c, s);
                for (int k = 0; k < time_bin; ++k) {
                    sum[c / freq_bin] += v[k];
                    used[c / freq_bin] += v[k] != 0;
                }
            }
        for (int b = 0; b < columns; ++b) {
            const double value =
                used[b] ? waterfall_scale * sum[b] / (sample_scale * std::sqrt((double)used[b]))
                        : 0.0;
            out[(size_t)r * columns + b] =
                used[b] ? (int8_t)std::lround(std::clamp(value, -127.0, 127.0)) : waterfall_no_data;
        }
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
