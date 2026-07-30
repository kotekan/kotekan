#include "GnssChannelizedSearch.hpp"

#include "StageFactory.hpp"            // for REGISTER_KOTEKAN_STAGE
#include "GnssChanMetadata.hpp"        // for get_gnss_chan_metadata, metadata_is_gnss_chan
#include "clockProfile.hpp"           // for resolve_clock_profile, clock_doppler_half_range_hz
#include "gnssChannelizedDespread.hpp" // for channelized_despread
#include "gnssSignal.hpp"              // for SignalDescriptor, signal_by_name
#include "kotekanLogging.hpp"          // for INFO, FATAL_ERROR
#include "pfbPrototype.hpp"            // for window_from_string

#include <algorithm>  // for min, max
#include <chrono>     // for steady_clock (hint TTL)
#include <cmath>      // for fabs, fmod, norm
#include <cstring>    // for memcpy
#include <functional> // for bind
#include "json.hpp"   // for json

// Monotonic seconds, for the Doppler-hint time-to-live (a hint the broker stops refreshing expires).
static inline double steady_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

using kotekan::Config;
using kotekan::bufferContainer;
using kotekan::Stage;
using cf = std::complex<float>;

REGISTER_KOTEKAN_STAGE(GnssChannelizedSearch);

GnssChannelizedSearch::GnssChannelizedSearch(Config& config, const std::string& unique_name,
                                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssChannelizedSearch::main_thread, this)),
    _snap_ready(false), _worker_busy(false), _worker_stop(false) {

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    const std::string signal = config.get_default<std::string>(unique_name, "signal", "GPS_L1CA");
    const gnss::SignalDescriptor* sig = gnss::signal_by_name(signal);
    if (sig == nullptr) {
        FATAL_ERROR("GnssChannelizedSearch: unknown signal '{:s}'", signal);
        return;
    }

    _sample_rate = config.get_default<double>(unique_name, "sample_rate", 5e6);
    const double f_offset = config.get_default<double>(unique_name, "f_offset", 0.0);
    _doppler_margin_hz = config.get_default<double>(unique_name, "doppler_margin_hz", 5000.0);

    _N = config.get<int>(unique_name, "spectrum_length");
    _fft_len = 2 * _N;
    _chan_offset = config.get_default<int>(unique_name, "channel_offset", 0);
    // SPARSE SUBBANDS. The airspy node owns a CONTIGUOUS run of channels, so a single offset
    // describes it. CHORD does not: the corner-turn gives a node every 16th bin, so its
    // covering channels for one carrier are a stride-16 comb spanning ~20 MHz. Without an
    // explicit list the search would build replicas for `channel_offset + 0..n-1` -- the wrong
    // seven bins -- and correlate real data against a replica for a different part of the
    // spectrum. That fails SILENTLY: the surface is noise, so it looks like "no satellite"
    // rather than like a bug. Empty (the default) keeps the contiguous behaviour exactly.
    _chan_ids = config.get_default<std::vector<int>>(unique_name, "channel_ids", {});
    _n_chan = config.get<int>(unique_name, "n_channels");
    if (_chan_ids.empty()) {
        if (_chan_offset < 0 || _n_chan <= 0 || _chan_offset + _n_chan > _N) {
            FATAL_ERROR("GnssChannelizedSearch: channel slice [{:d},{:d}) invalid for N={:d}",
                        _chan_offset, _chan_offset + _n_chan, _N);
            return;
        }
    } else {
        if ((int)_chan_ids.size() != _n_chan) {
            FATAL_ERROR("GnssChannelizedSearch: channel_ids has {:d} entries but n_channels is "
                        "{:d}",
                        _chan_ids.size(), _n_chan);
            return;
        }
        for (int c : _chan_ids)
            if (c < 0 || c >= _N) {
                FATAL_ERROR("GnssChannelizedSearch: channel_id {:d} outside [0,{:d})", c, _N);
                return;
            }
    }
    const int num_taps = config.get_default<int>(unique_name, "num_taps", 4);
    const std::string win = config.get_default<std::string>(unique_name, "pfb_window", "hamming");

    _prns = config.get<std::vector<int>>(unique_name, "prns");
    if (_prns.empty()) {
        FATAL_ERROR("GnssChannelizedSearch: 'prns' is empty");
        return;
    }
    _acquire_snr = config.get_default<double>(unique_name, "acquire_snr", 12.0);
    _acquire_windows = config.get_default<int>(unique_name, "acquire_windows", 64);
    _hold_snapshots = config.get_default<int>(unique_name, "hold_snapshots", 5);
    // require_hint: scan only broker-hinted (visible) PRNs, skip the rest -> `prns` can list the
    // whole constellation and the active-scan set follows the sky (mid-run PRN swap) at a cost that
    // tracks the visible count, never blind-gridding a below-horizon sat. hint_ttl_s expires a hint
    // the broker stopped refreshing (a set sat) so it drops out. Off by default (blind-grid legacy).
    // Threads for the aggregate half of the acquire (parallel over Doppler bins). 1 -- the
    // exact serial path -- unless configured: the airspy chains are sized for one core, while
    // the aggregator's 27-channel surface is ~10 s/window serial and owns several cores.
    _acquire_threads = config.get_default<int>(unique_name, "acquire_threads", 1);
    _require_hint = config.get_default<bool>(unique_name, "require_hint", false);
    _hint_ttl_s = config.get_default<double>(unique_name, "hint_ttl_s", 8.0);
    double dmin = config.get_default<double>(unique_name, "doppler_min", -6000.0);
    double dmax = config.get_default<double>(unique_name, "doppler_max", 6000.0);
    _doppler_step = config.get_default<double>(unique_name, "doppler_step", 500.0);
    // Clock-profile Doppler sizing: the top-level /clock_profile block (shared by every stage) sets
    // the receiver clock quality; when present, the carrier search extent is DERIVED from its
    // frequency-accuracy bound + the band's max sky Doppler, so one knob sizes every band and clock
    // (airspy TCXO ... GPSDO ... maser). Absent -> legacy explicit doppler_min/max. See clockProfile.hpp.
    const std::string clk_name =
        config.get_default<std::string>("/clock_profile", "name", std::string(""));
    const double clk_acc = config.get_default<double>("/clock_profile", "accuracy_ppm", std::nan(""));
    const double clk_coh = config.get_default<double>("/clock_profile", "coherence_s", std::nan(""));
    if (!clk_name.empty() || !std::isnan(clk_acc)) {
        const gnss::ClockProfile cp =
            gnss::resolve_clock_profile(clk_name.empty() ? "auto" : clk_name, clk_acc, clk_coh);
        const double half = gnss::clock_doppler_half_range_hz(sig->carrier_hz, cp.accuracy_ppm);
        dmin = -half;
        dmax = half;
        INFO("GnssChannelizedSearch: clock_profile '{:s}' ({:.3g} ppm) -> Doppler search +-{:.0f} Hz",
             clk_name.empty() ? "auto" : clk_name, cp.accuracy_ppm, half);
    }
    for (double f = dmin; f <= dmax + 1e-6; f += _doppler_step)
        _doppler_grid.push_back(f);

    try {
        _replica = std::make_unique<gnss::ChannelizedReplicaBank>(
            *sig, _sample_rate, f_offset, _N, num_taps, dsp::window_from_string(win), _prns);
    } catch (const std::exception& e) {
        FATAL_ERROR("GnssChannelizedSearch: {:s}", e.what());
        return;
    }
    _hops_per_record =
        config.get_default<int>(unique_name, "hops_per_record", _replica->repl_period_hops());
    _replica->code_doppler_sign = config.get_default<double>(unique_name, "code_doppler_sign", 1.0);

    // Fine-refine geometry. The span is one hop each side (the acquire's coarse-lag quantum);
    // the step defaults to 1 sample, which is what the narrow-bank (airspy) case wants and has
    // always done. `refine_step` exists because the cost is one exact replica build PER STEP:
    // at CHORD's fft_len 16384 the default is 32769 builds, and see the loop for why they buy
    // nothing. Set it in config for a wide bank.
    _refine_span = config.get_default<int>(unique_name, "refine_span", _fft_len);
    _refine_step = std::max(1, config.get_default<int>(unique_name, "refine_step", 1));

    // SECONDARY-CODE (Neuman-Hofman) ALIGNMENT SEARCH.
    //
    // The acquire correlates coherently over hops_per_record, which defaults to the replica
    // period -- 3125 hops = 16 ms on CHORD L5, i.e. SIXTEEN code periods. L5 Q5 is a dataless
    // pilot, but "dataless" does not mean unmodulated: it carries the NH20 overlay, one +-1 chip
    // per 1 ms code period. Correlating 16 code periods against a replica built with the overlay
    // OFF therefore sums 16 consecutive NH chips of a near-balanced sequence:
    //     measured over all 20 alignments -- 12.7 dB rms loss, best case 8.5 dB,
    //     and EXACTLY ZERO for three of the twenty.
    // Worse, it is not a fixed penalty. Consecutive 16 ms windows step the alignment by
    // 16 mod 20, so a snapshot only ever visits phases {0,4,8,12,16} -- and phase 4 is one of
    // the nulls, so a fixed fraction of every snapshot contributes nothing at all.
    //
    // With the alignment applied the same window integrates at FULL coherent gain (the
    // l5q_nh20_overlay test pins this: aligned overlay = matched filter over 20 periods, one
    // chip off = decorrelated). The alignment is a single unknown in [0, secondary_length), so
    // search it. 20 acquires per PRN buys 12.7 dB COHERENTLY, which incoherent integration would
    // need ~350x more windows to match -- so this is strongly favourable even before trading
    // acquire_windows down to pay for it.
    _nh_search = config.get_default<bool>(unique_name, "nh_search", false);
    _n_nh = (_nh_search && _replica->secondary_length() > 0) ? _replica->secondary_length() : 1;
    if (_nh_search && _n_nh == 1)
        INFO("GnssChannelizedSearch[{:s}]: nh_search requested but {:s} has no secondary code "
             "-- ignored",
             unique_name, sig->name);
    else if (_n_nh > 1)
        INFO("GnssChannelizedSearch[{:s}]: searching {:d} secondary-code alignments over a "
             "{:d}-hop ({:.1f} ms) coherent window",
             unique_name, _n_nh, _hops_per_record,
             1e3 * (double)_hops_per_record * _fft_len / _sample_rate);

    _detections.assign(_prns.size(), Detection{});
    _dop_hints.assign(_prns.size(), DopHint{});
    _repl0.assign((size_t)_n_nh, std::vector<std::vector<std::vector<cf>>>(_prns.size()));
    _snap_hops = (size_t)_acquire_windows * _hops_per_record;
    _snapshot.assign(_snap_hops * (size_t)_n_chan, cf(0.0f, 0.0f));
}

GnssChannelizedSearch::~GnssChannelizedSearch() {
    {
        std::lock_guard<std::mutex> lk(_m);
        _worker_stop = true;
        _cv.notify_one();
    }
    if (_worker.joinable())
        _worker.join();
}

void GnssChannelizedSearch::search_snapshot() {
    const int n_prn = (int)_prns.size();
    const int Mp = _replica->repl_period_hops();
    const int hpr = _hops_per_record;
    const long long anchor = (long long)Mp * _fft_len; // warm-up reads periodic code
    const double cps = _replica->chip_rate_hz() / _sample_rate;
    const double dmax = _doppler_grid.empty()
                            ? 0.0
                            : std::max(std::fabs(_doppler_grid.front()),
                                       std::fabs(_doppler_grid.back()));
    const int nwin = std::min(_acquire_windows, (int)(_snap_hops / (size_t)hpr));

    // Covering channels (global) for this carrier that fall in this subband.
    const auto cover = _replica->covering_bins(dmax, _doppler_margin_hz);
    std::vector<int> cov_local, cov_global;
    for (int c : cover) {
        const int lc = local_of_global(c);
        if (lc >= 0) {
            cov_local.push_back(lc);
            cov_global.push_back(c);
        }
    }

    // The empty-cover path below is a SILENT no-op: it publishes an invalid detection for every
    // PRN, which is indistinguishable from "no satellites overhead". Say so once per snapshot
    // instead, because the usual cause is a misconfigured f_offset (the replica placed at the
    // wrong sky frequency), and nothing else in the output distinguishes that from bad luck.
    if (cov_local.empty())
        INFO("GnssChannelizedSearch[{:s}]: NO covering channels overlap this subband -- the "
             "replica's covering bins do not intersect channels [{:d},{:d}). Check f_offset "
             "(currently placing the carrier at bin {:.1f}).",
             unique_name, _chan_offset, _chan_offset + _n_chan,
             _replica->f_offset() / (_sample_rate / _fft_len));
    else
        INFO("GnssChannelizedSearch[{:s}]: {:d} covering channels in this subband (global {:d}..{:d})",
             unique_name, (int)cov_local.size(), cov_global.front(), cov_global.back());

    // Build (once) the banded repl0 for every PRN. This used to be a per-PRN, per-snapshot
    // channels() call returning the ENTIRE [N][Mp] spectrum, of which channelized_accumulate
    // reads only the covering rows: at CHORD's N = 8192 that is 204.8 MB and ~1170x the needed
    // work for 7 useful rows, per PRN, per pass -- the whole reason the search could not keep up
    // here. Two independent fixes, both correctness-neutral:
    //   BANDED -- channels_hoprate() generates only `want`, per chip instead of per sample, and
    //     is numerically equal to channels() (gated at BOTH airspy and CHORD scale in
    //     test_gnss_channelized_replica; the CHORD case is the one this depends on).
    //   CACHED -- repl0 is the code at Doppler 0 and code phase 0, so it depends only on
    //     (PRN, covering set, Mp). None of those move between snapshots. The covering set is
    //     derived from the fixed blind grid, not from the per-PRN hint, so the Doppler hints
    //     changing every broker cycle does NOT invalidate it -- but key on it anyway rather
    //     than asserting that invariant from a distance.
    // Rows are indexed by LOCAL channel to match channelized_accumulate's repl0_ch[covering[ci]];
    // the non-covering rows stay empty vectors (24 B each) and are never read.
    if (!cov_local.empty() && cov_global != _repl0_cover) {
        const double t_build = steady_s();
        for (int nh = 0; nh < _n_nh; ++nh) {
            // nh_phase < 0 leaves the overlay OFF, which is the single-alignment default and
            // what the airspy chain has always used; >= 0 applies that alignment.
            const int nh_phase = (_n_nh > 1) ? nh : -1;
            for (int p = 0; p < n_prn; ++p) {
                const auto banded = _replica->channels_hoprate(p, anchor, 0.0, 0.0, Mp, cov_global,
                                                               {}, nh_phase);
                _repl0[(size_t)nh][p].assign((size_t)_n_chan, {});
                for (size_t i = 0; i < cov_local.size(); ++i)
                    _repl0[(size_t)nh][p][(size_t)cov_local[i]] = banded[i];
            }
        }
        _repl0_cover = cov_global;
        INFO("GnssChannelizedSearch[{:s}]: precomputed banded repl0 for {:d} PRNs x {:d} channels "
             "x {:d} hops x {:d} nh alignment(s) in {:.2f} s ({:.1f} MB; the full-spectrum form "
             "would be {:.1f} MB and is rebuilt no further)",
             unique_name, n_prn, (int)cov_local.size(), Mp, _n_nh, steady_s() - t_build,
             1e-6 * (double)_n_nh * n_prn * cov_local.size() * Mp * sizeof(cf),
             1e-6 * (double)_n_nh * n_prn * _replica->spectrum_length() * Mp * sizeof(cf));
        // The refine cost is one exact replica build per step, so say the number out loud: at
        // fft_len 16384 with the default step of 1 this is 32769 builds on the FIRST detection,
        // which looks exactly like a hang. See the refine loop for the resolution argument.
        const long evals = 2L * _refine_span / _refine_step + 1;
        const double res_samp = (double)_fft_len / (double)cov_local.size();
        INFO("GnssChannelizedSearch[{:s}]: code-phase refine +-{:d} samples step {:d} = {:d} exact "
             "despreads; the {:d}-channel covering set band-limits the despread to {:.2f} MHz, so "
             "its intrinsic resolution is ~{:.0f} samples ({:.1f} chips){:s}",
             unique_name, _refine_span, _refine_step, evals, (int)cov_local.size(),
             1e-6 * cov_local.size() * _sample_rate / _fft_len, res_samp,
             res_samp * _replica->chip_rate_hz() / _sample_rate,
             (double)_refine_step < res_samp / 8.0
                 ? " -- OVERSAMPLED, consider raising refine_step"
                 : "");
    }

    double best_any = 0.0;
    int best_any_prn = -1;
    int best_any_nh = -1;
    for (int p = 0; p < n_prn; ++p) {
        if (cov_local.empty()) { // carrier not in this subband: nothing to find
            std::lock_guard<std::mutex> lk(_det_mtx);
            _detections[p] = Detection{};
            continue;
        }

        // Almanac-narrowed grid: if the broker pushed a Doppler hint for this PRN, scan only
        // hint +- margin (the orbit fixes Doppler; only the common clock drift remains) -- far
        // cheaper + lower false-alarm. The hint is the PHYSICAL (reported) Doppler; the internal
        // grid runs in the r2c-flipped convention (det.doppler_hz = -grid value below), so centre
        // the window at -hint. A hint older than _hint_ttl_s (the broker stopped refreshing it =
        // the sat set) counts as absent. No fresh hint -> the full blind grid, EXCEPT in
        // require_hint mode where the PRN is SKIPPED entirely (no replica build, no scan): that
        // lets `prns` list the whole constellation while cost tracks only the visible/hinted set,
        // so PRNs swap in/out as the sky rotates without ever sweeping a below-horizon sat.
        std::vector<double> pgrid;
        bool hinted = false;
        {
            std::lock_guard<std::mutex> lk(_hint_mtx);
            const DopHint& hh = _dop_hints[p];
            if (hh.valid && (_hint_ttl_s <= 0.0 || steady_s() - hh.t_recv < _hint_ttl_s)) {
                hinted = true;
                const double c = -hh.doppler, m = std::max(0.0, hh.margin);
                // Anchor the hinted window to the ABSOLUTE Doppler grid (integer multiples
                // of _doppler_step), NOT to the hint. A hint-anchored (sliding-origin) grid
                // made the reported dop a function of the hint's continuous drift: a static
                // peak's detection could hop a full bin between scans, and a re-seed then
                // retuned the NCO exactly one grid step off truth. Measured 2026-07-18
                // (G6, t=371 in the 3-band leg): seed DOP STEP +100.0 with the matching
                // -24.03-chip currency translation, records rotating at +100.000 Hz after
                // -- the railed-sat / L2C-B1C sinc-null disease (one step x T_rec = 1 cycle
                // kills 20/10 ms records outright). The blind grid always had a fixed
                // origin; this restores the same property to the narrowed scan.
                const double lo = std::ceil((c - m) / _doppler_step) * _doppler_step;
                for (double f = lo; f <= c + m + 1e-6; f += _doppler_step)
                    pgrid.push_back(f);
            }
        }
        if (_require_hint && !hinted) { // below horizon / stale -> not in the active set this cycle
            std::lock_guard<std::mutex> lk(_det_mtx);
            _detections[p] = Detection{};
            continue;
        }

        const std::vector<double>& grid = pgrid.empty() ? _doppler_grid : pgrid;

        // Integrate the |D|^2 surface over the snapshot windows (incoherent), once per
        // secondary-code alignment, and keep the best peak. The alignments are processed
        // SEQUENTIALLY and only the winning result is retained, so peak memory is one surface
        // (128 MB at CHORD dimensions) rather than _n_nh of them.
        gnss::AcquisitionResult a{};
        int best_nh = -1;
        std::vector<double> surf;
        std::vector<std::vector<cf>> dch(_n_chan, std::vector<cf>(hpr));
        for (int nh = 0; nh < _n_nh; ++nh) {
            // repl0 (code 0, Doppler 0) on the covering channels -- precomputed, not rebuilt.
            const std::vector<std::vector<cf>>& repl0 = _repl0[(size_t)nh][p];
            gnss::AcquisitionSurface dims{};
            surf.assign(surf.size(), 0.0); // reuse the allocation, drop the previous alignment
            for (int w = 0; w < nwin; ++w) {
                for (int lc = 0; lc < _n_chan; ++lc)
                    for (int m = 0; m < hpr; ++m)
                        dch[lc][m] = _snapshot[((size_t)(w * hpr + m)) * _n_chan + lc];
                dims = gnss::channelized_accumulate(dch, repl0, cov_local, grid, _sample_rate,
                                                    _n_chan, surf, _acq_ws, cov_global, _fft_len,
                                                    _acquire_threads);
            }
            const auto ai = gnss::channelized_peak(surf, dims, grid, _sample_rate,
                                                   _replica->chip_rate_hz(),
                                                   _replica->code_length());
            if (best_nh < 0 || ai.snr > a.snr) {
                a = ai;
                best_nh = nh;
            }
        }

        const bool detected = a.snr >= _acquire_snr;
        if (a.snr > best_any) {
            best_any = a.snr;
            best_any_prn = _prns[p];
            best_any_nh = best_nh;
        }
        Detection det;
        det.snr = (float)a.snr;
        if (detected) {
            // r2c fold conjugates the channel frequency axis -> flip Doppler sign.
            const double dop = -a.doppler_hz;
            // Refine code phase within the acquire's coarse (hop) lag quantum: an exact-despread
            // scan over +-_refine_span samples in _refine_step steps, on window 0.
            //
            // Banded, with the filter built ONCE. The Doppler is fixed across the scan, so the
            // slow half of the hop-rate generator (the cumulative channel filter, O(num_taps *
            // fft_len) per channel) is Doppler-only and can be hoisted; only the per-chip stream
            // varies with cp. channels_hoprate() would rebuild the filter every step. This
            // replaced a full-spectrum channels() call per step, which at CHORD scale was 204.8 MB
            // and ~1170x the needed work, 32769 times over -- i.e. the first detection would have
            // looked like a hang rather than a result.
            //
            // On _refine_step: only the covering channels enter the despread, so it is
            // band-limited to n_cover * bin_width and cannot resolve code phase better than
            // ~fft_len/n_cover samples (7.5 chips for 7 CHORD channels). Stepping finer than that
            // costs a replica build per step and buys nothing but interpolation of a peak whose
            // width is set elsewhere. The default step is 1, which is right for a narrow bank
            // (airspy: fft_len 20, so 41 evaluations either way) and wrong for a wide one.
            const gnss::ChannelizedReplicaBank::HopRateFilter rf =
                _replica->hoprate_filter(cov_global, dop);
            // Hoist the data columns too: they do not depend on the trial cp, and rebuilding them
            // per step was copying hpr * n_cover samples for nothing.
            std::vector<std::vector<cf>> d(cov_local.size(), std::vector<cf>(hpr));
            for (size_t i = 0; i < cov_local.size(); ++i)
                for (int m = 0; m < hpr; ++m)
                    d[i][m] = _snapshot[((size_t)m) * _n_chan + cov_local[i]];
            double best_cp = a.code_phase_chips, best_pw = -1.0;
            for (int off = -_refine_span; off <= _refine_span; off += _refine_step) {
                const double cp = a.code_phase_chips + off * cps;
                // Same secondary-code alignment the peak was found at, or the refine despreads
                // an overlay-blind replica against an overlay-aligned peak and walks off it.
                const auto r = _replica->hoprate_stream(rf, p, anchor, cp, dop, hpr, {},
                                                        (_n_nh > 1) ? best_nh : -1);
                const double pw = std::norm(gnss::channelized_despread(d, r).amplitude);
                if (pw > best_pw) {
                    best_pw = pw;
                    best_cp = cp;
                }
            }
            // best_cp is the satellite code phase at the snapshot's absolute start
            // (S_snap = snap_start_hop * fft_len). Reference it to absolute sample 0
            // so the tracker (on the same hop*2N grid) can seed directly:
            // cp0 = best_cp - S_snap*chip_per_sample (mod L). Reduce the hop offset
            // modulo the replica period first to keep float precision.
            const double L = (double)_replica->code_length();
            const double off = std::fmod((double)(_snap_start_hop % Mp) * (double)_fft_len * cps, L);
            // Code-Doppler drift of the reference over the FULL absolute snapshot start
            // (NOT Mp-periodic, so use the full snap_start_hop): makes cp0 the true
            // sample-0 phase, matching the feed-forward in ChannelizedReplicaBank, so
            // the seed is latency-invariant (a stale seed no longer drifts the cp).
            const double drift = std::fmod((double)_snap_start_hop * (double)_fft_len * cps
                                               * (_replica->code_doppler_sign * dop
                                                  / _replica->carrier_hz()),
                                           L);
            double cp = std::fmod(best_cp - off - drift, L);
            if (cp < 0.0)
                cp += L;
            det.doppler_hz = dop;
            det.code_phase_chips = cp;
            det.ref_hop = _snap_start_hop; // capture-time anchor for cp0 (for the slope fit)
            det.valid = true;
            // DIAG: decompose where cp comes from, to localize any instability:
            //   hop   = absolute snapshot reference (sample_seq/fft_len)
            //   coarse= cross-channel acquire cp (pre-refine)   refine= +- from refine
            //   off/drift = nominal + code-Doppler back-reference to sample 0
            //   cp0   = final reported code phase
            INFO("GnssChannelizedSearch[{:s}]: PRN {:d} snr {:.1f} dop {:+.0f} | hop {:d} "
                 "coarse {:.2f} refine {:+.2f} off {:.2f} drift {:.2f} -> cp0 {:.2f}",
                 unique_name, _prns[p], a.snr, dop, _snap_start_hop, a.code_phase_chips,
                 best_cp - a.code_phase_chips, off, drift, cp);
        }
        // Latch: a fresh detection replaces the held one; a miss keeps the last valid
        // detection for _hold_snapshots snapshots (a momentary miss must not drop the
        // seed), then expires it.
        std::lock_guard<std::mutex> lk(_det_mtx);
        if (detected) {
            _detections[p] = det;
        } else {
            _detections[p].snr = det.snr;
            if (_detections[p].valid && ++_detections[p].misses > _hold_snapshots)
                _detections[p].valid = false;
        }
    }
    // The peak SNR is computed for every PRN and then THROWN AWAY unless it clears acquire_snr,
    // which leaves "found nothing" and "found 8 sigma with the threshold at 12" looking
    // identical from outside. Report the best of the pass so a near-miss is visible.
    if (best_any_prn >= 0)
        INFO("GnssChannelizedSearch[{:s}]: pass best snr {:.2f} (PRN {:d}{:s}), threshold {:.2f}",
             unique_name, best_any, best_any_prn,
             (_n_nh > 1) ? fmt::format(", nh {:d}/{:d}", best_any_nh, _n_nh) : std::string(),
             _acquire_snr);
}

void GnssChannelizedSearch::search_worker() {
    while (true) {
        {
            std::unique_lock<std::mutex> lk(_m);
            _cv.wait(lk, [&] { return _snap_ready || _worker_stop; });
            if (_worker_stop && !_snap_ready)
                return;
            _snap_ready = false;
        }
        search_snapshot(); // snapshot stable: _worker_busy keeps the drain off it
        {
            std::lock_guard<std::mutex> lk(_m);
            _worker_busy = false;
        }
    }
}

int GnssChannelizedSearch::global_of_local(int lc) const {
    return _chan_ids.empty() ? (_chan_offset + lc) : _chan_ids[(size_t)lc];
}

int GnssChannelizedSearch::local_of_global(int c) const {
    if (_chan_ids.empty())
        return (c >= _chan_offset && c < _chan_offset + _n_chan) ? (c - _chan_offset) : -1;
    for (size_t i = 0; i < _chan_ids.size(); ++i)
        if (_chan_ids[i] == c)
            return (int)i;
    return -1;
}

void GnssChannelizedSearch::main_thread() {
    using namespace std::placeholders;
    kotekan::restServer::instance().register_get_callback(
        unique_name + "/get_detections",
        std::bind(&GnssChannelizedSearch::get_detections_callback, this, _1));
    kotekan::restServer::instance().register_post_callback(
        unique_name + "/set_doppler_hints",
        std::bind(&GnssChannelizedSearch::set_doppler_hints_callback, this, _1, _2));

    _worker = std::thread(&GnssChannelizedSearch::search_worker, this);

    int frame_in = 0;
    bool filling = false;
    size_t filled_hops = 0;  // hops actually copied into the current snapshot (for the drop log)
    long long abs_hops = 0;  // total hops consumed from the stream (absolute reference)

    while (!stop_thread) {
        auto* in_local = (cf*)in_buf->wait_for_full_frame(unique_name, frame_in);
        if (in_local == nullptr)
            break;
        const int frame_hops = in_buf->frame_size / (int)sizeof(cf) / _n_chan;

        // Absolute hop index of this frame's first hop, from the F-engine's sample_seq
        // metadata (shared across nodes); fall back to a local count if absent.
        long long frame_first_hop = abs_hops;
        if (metadata_is_gnss_chan(in_buf)) {
            auto* mi = get_gnss_chan_metadata(in_buf, frame_in);
            if (mi && mi->sample_seq >= 0)
                frame_first_hop = mi->sample_seq / _fft_len;
        }

        if (!filling) {
            std::lock_guard<std::mutex> lk(_m);
            if (!_worker_busy) {
                filling = true;
                filled_hops = 0;
                _snap_start_hop = frame_first_hop; // snapshot hop 0 is this frame's hop 0
                // Zero first so DROPPED frames (the search arm uses drop_frames, so gather_buf
                // has gaps) leave signal-free holes -- each frame is then placed at its TRUE
                // sample_seq offset below, keeping the code phase aligned across drops. Without
                // this, a dropped frame shifts every later window's code phase and smears the
                // |D|^2 peak away -- which is why a longer (40-frame) snapshot found nothing.
                std::fill(_snapshot.begin(), _snapshot.end(), cf(0.0f, 0.0f));
            }
        }
        if (filling) {
            // Place this frame at its absolute time position in the snapshot (drop-tolerant).
            const long long off = frame_first_hop - _snap_start_hop;
            if (off >= 0 && off < (long long)_snap_hops) {
                const size_t take =
                    std::min((size_t)frame_hops, _snap_hops - (size_t)off);
                std::memcpy(&_snapshot[(size_t)off * (size_t)_n_chan], in_local,
                            take * (size_t)_n_chan * sizeof(cf));
                filled_hops += take;
            }
            // Finalize once this frame reaches the end of the snapshot's time window (gaps stay
            // zero). A frame entirely past the window (off >= _snap_hops) also finalizes here.
            if (off + frame_hops >= (long long)_snap_hops) {
                filling = false;
                if (filled_hops < _snap_hops) // dropped frames left holes; integration is shallower
                    INFO("GnssChannelizedSearch[{:s}]: snapshot {:.0f}% filled ({:d}/{:d} hops; "
                         "rest dropped upstream -> zero-filled, peak stays aligned)",
                         unique_name, 100.0 * (double)filled_hops / (double)_snap_hops,
                         (long)filled_hops, (long)_snap_hops);
                std::lock_guard<std::mutex> lk(_m);
                _snap_ready = true;
                _worker_busy = true;
                _cv.notify_one();
            }
        }

        // Always release the frame immediately -- never backpressure.
        in_buf->mark_frame_empty(unique_name, frame_in);
        frame_in = (frame_in + 1) % in_buf->num_frames;
        abs_hops += frame_hops;
    }

    {
        std::lock_guard<std::mutex> lk(_m);
        _worker_stop = true;
        _cv.notify_one();
    }
    if (_worker.joinable())
        _worker.join();
}

void GnssChannelizedSearch::get_detections_callback(kotekan::connectionInstance& conn) {
    nlohmann::json reply = nlohmann::json::array();
    std::lock_guard<std::mutex> lk(_det_mtx);
    for (size_t p = 0; p < _prns.size(); ++p) {
        const Detection& d = _detections[p];
        if (!d.valid)
            continue;
        reply.push_back({{"prn", _prns[p]},
                         {"doppler_hz", d.doppler_hz},
                         {"code_phase_chips", d.code_phase_chips},
                         {"ref_hop", d.ref_hop},
                         {"snr", d.snr}});
    }
    conn.send_json_reply(reply);
}

void GnssChannelizedSearch::set_doppler_hints_callback(kotekan::connectionInstance& conn,
                                                       nlohmann::json& json_request) {
    // Body: [{prn, doppler_hz, margin_hz}, ...] -- narrow each listed PRN's search to
    // doppler_hz +- margin_hz (the broker's orbit prediction + clock-freq bias). margin_hz < 0
    // CLEARS the hint (revert that PRN to the blind grid). PRNs not listed keep their current
    // hint (the broker resends visible sats every cycle). Unknown PRNs are ignored.
    try {
        std::lock_guard<std::mutex> lk(_hint_mtx);
        for (const auto& h : json_request) {
            const int prn = h.at("prn").get<int>();
            for (size_t i = 0; i < _prns.size(); ++i) {
                if (_prns[i] != prn)
                    continue;
                const double margin = h.value("margin_hz", 0.0);
                if (margin < 0.0)
                    _dop_hints[i] = DopHint{}; // clear -> blind grid (or skip, if require_hint)
                else
                    _dop_hints[i] =
                        DopHint{true, h.at("doppler_hz").get<double>(), margin, steady_s()};
                break;
            }
        }
    } catch (const std::exception& e) {
        conn.send_error(e.what(), kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
}
