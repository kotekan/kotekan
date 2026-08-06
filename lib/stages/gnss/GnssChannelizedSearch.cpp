#include "GnssChannelizedSearch.hpp"

#include "StageFactory.hpp"            // for REGISTER_KOTEKAN_STAGE
#include "GnssChanMetadata.hpp"        // for get_gnss_chan_metadata, metadata_is_gnss_chan
#include "clockProfile.hpp"           // for resolve_clock_profile, clock_doppler_half_range_hz
#include "gnssSeedTransport.hpp"       // for detection_phase
#include "gnssChannelizedDespread.hpp" // for channelized_despread
#include "gnssSignal.hpp"              // for SignalDescriptor, signal_by_name
#include "kotekanLogging.hpp"          // for INFO, FATAL_ERROR
#include "pfbPrototype.hpp"            // for window_from_string
#ifdef GNSS_CUDA
#include "GnssCudaAcquire.hpp" // device-resident acquire (docs/gnss_gpu_search.md A2)
#include "GnssCudaDespread.hpp" // A5: the refine reuses the tracker's despread driver
#endif

#include <algorithm>  // for min, max
#include <chrono>     // for steady_clock (hint TTL)
#include <cmath>      // for fabs, fmod, norm
#include <cstdio>     // for fopen/fread/fwrite (the replica disk cache)
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
    // Fine-lag decimation. 1 = every sample (the old behaviour). The fine axis resolves a lobe
    // sph/(covering comb span) samples wide -- 157 at CHORD -- so storing it per sample is
    // ~157x oversampled, and the surface is the entire cost of a pass. See AcquisitionSurface.
    _acquire_fine_step = std::max(1, config.get_default<int>(unique_name, "acquire_fine_step", 1));
    // How many ELIGIBLE PRNs to search per snapshot, round-robin. 0 = all of them, which is the
    // historical behaviour and what airspy runs.
    //
    // A detection's ref_hop is the SNAPSHOT's start hop, and one snapshot used to serve the whole
    // pass -- so a PRN searched last carried an epoch as old as the pass itself (285 s at CHORD
    // even after the 2026-08-01 speedups). Seed error is Doppler error x anchor age at 0.0087
    // chips per Hz per second, so that alone put every seed tens of chips off before the tracker
    // ever saw it: the seed was never accurate at ANY moment, which is what a tracker needs in
    // order to pull in. Searching a few PRNs against a FRESH snapshot each time bounds the epoch
    // at emit by (pass time)/(PRNs per pass) instead, and the cursor makes the visits fair --
    // a fixed order would keep re-detecting whichever PRNs sit at the front of the list.
    _prns_per_pass = std::max(0, config.get_default<int>(unique_name, "prns_per_pass", 0));
    _hint_dop_sign = config.get_default<double>(unique_name, "hint_doppler_sign", -1.0);
    _require_hint = config.get_default<bool>(unique_name, "require_hint", false);
    _nh_hint_span = std::max(0, config.get_default<int>(unique_name, "nh_hint_span", 1));
    _hint_ttl_s = config.get_default<double>(unique_name, "hint_ttl_s", 8.0);
    // Bounded cold-start: with require_hint false, EVERY unhinted PRN would otherwise get a full
    // blind grid (~30x a hinted one), so a sky full of below-horizon satellites dominates the
    // pass. 0 keeps the historical behaviour exactly.
    _blind_prns_per_pass = std::max(0, config.get_default<int>(unique_name, "blind_prns_per_pass", 0));
#ifdef GNSS_CUDA
    _cuda_acq_wanted = config.get_default<bool>(unique_name, "use_cuda_acquire", false);
    // Defaults to following use_cuda_acquire: with the acquire on the GPU the refine is 98% of
    // the pass, so turning one on without the other leaves the win on the table.
    _cuda_ref_wanted = config.get_default<bool>(unique_name, "use_cuda_refine", _cuda_acq_wanted);
#else
    if (config.get_default<bool>(unique_name, "use_cuda_acquire", false))
        WARN("GnssChannelizedSearch[{:s}]: use_cuda_acquire set but this binary was built without "
             "CUDA -- running the CPU acquire",
             unique_name);
#endif
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
    // FDMA (GLONASS L1OF/L2OF): every satellite shares one code and is separated by CARRIER, so
    // each PRN needs its own offset from band centre. The table is built by the config generator
    // from the live GLONASS frequency plan and travels in the yaml, where it is readable next to
    // the PRN list. Absent/empty -> CDMA, i.e. every other signal, unchanged.
    _replica->set_prn_freq_offsets(
        config.get_default<std::vector<double>>(unique_name, "prn_freq_offset_hz", {}));
    if (_replica->prn_freq_offsets_set() > 0)
        INFO("GnssChannelizedSearch: FDMA carrier offsets applied to {:d} of {:d} PRNs",
             _replica->prn_freq_offsets_set(), (int)_prns.size());

    _hops_per_record =
        config.get_default<int>(unique_name, "hops_per_record", _replica->repl_period_hops());
    _replica->code_doppler_sign = config.get_default<double>(unique_name, "code_doppler_sign", 1.0);

    // Fine-refine geometry. The span is one hop each side (the acquire's coarse-lag quantum);
    // the step defaults to 1 sample, which is what the narrow-bank (airspy) case wants and has
    // always done. `refine_step` exists because the cost is one exact replica build PER STEP:
    // at CHORD's fft_len 16384 the default is 32769 builds, and see the loop for why they buy
    // nothing. Set it in config for a wide bank.
    _refine_span = config.get_default<int>(unique_name, "refine_span", _fft_len);
    // REFINE INTEGRATION LENGTH, hops. 0 = the full record (hpr), which is what it has always
    // used. The refine only has to LOCALISE a peak that already cleared the detection
    // threshold, and it costs one full 106-channel x hpr replica build per trial phase -- ~3.5 s
    // at CHORD dimensions, which is 82% of all search compute. The airspy prototype refines
    // over 10 channels x 1000 hops = 1/33 of the work, because its record IS one code period;
    // CHORD's spans 16. Shortening this to a period or two is therefore matching the prototype
    // rather than cutting below it -- but it trades SNR for speed, so measure the phase error
    // before trusting it (scripts/gnss/e2e --refine-hops).
    _refine_hops = std::max(0, config.get_default<int>(unique_name, "refine_hops", 0));
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
    _nh_hints.assign(_prns.size(), NhHint{});
    _nh_snap_hints.assign(_prns.size(), NhHint{});
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
    // ★ UNION over PRNs, because this set is sliced ONCE and then reused for every PRN below.
    // Under FDMA (GLONASS) each satellite sits on its own carrier, so a single PRN's covering
    // set would send the search looking for every OTHER satellite in channels it does not
    // occupy -- which finds nothing, while the front end, the codes and the hints all look
    // healthy. Identical to the old call for CDMA, where every PRN shares one carrier.
    const auto cover = _replica->covering_bins_union(dmax, _doppler_margin_hz);
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
    else if ((int)cov_local.size() != _last_cov_n || cov_global.front() != _last_cov_first
             || cov_global.back() != _last_cov_last) { // once per SET, not per snapshot
        INFO("GnssChannelizedSearch[{:s}]: {:d} covering channels in this subband (global {:d}..{:d})",
             unique_name, (int)cov_local.size(), cov_global.front(), cov_global.back());
        _last_cov_n = (int)cov_local.size();
        _last_cov_first = cov_global.front();
        _last_cov_last = cov_global.back();
    }

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
        const long Lc = _replica->eff_code_length();
        // Period index of each hop's FIRST chip. cp = 0 and Doppler = 0 here (repl0 is the
        // zero-phase, zero-Doppler bank), so chip(m) = (anchor + m*fft_len) * chip_rate/Fs.
        _repl_k0.assign((size_t)Mp, 0);
        for (int m = 0; m < Mp; ++m) {
            const double chip = ((double)anchor + (double)m * (double)_fft_len)
                                * _replica->chip_rate_hz() / _sample_rate;
            _repl_k0[(size_t)m] = (long long)std::floor(chip / (double)Lc);
        }
        // DISK CACHE. The head/tail bank is a pure function of the signal, the rates, the PFB,
        // the PRN list, the covering comb, Mp and the anchor -- and the anchor is Mp*fft_len, a
        // derived constant, not a per-run value. So it is bit-identical on every restart and
        // there is no reason to spend two minutes rebuilding it. Node-LOCAL path: /home is NFS
        // here, and 170 MB over NFS would give back what the cache saves.
        //
        // The key hashes every input, so a changed comb / PRN set / signal / format simply
        // misses and rebuilds -- a stale file can never be silently used. Bump CACHE_VERSION to
        // invalidate everywhere.
        constexpr uint32_t CACHE_VERSION = 1;
        uint64_t ckey = 1469598103934665603ull;
        const auto mix = [&ckey](const void* d, size_t n) {
            const auto* b = static_cast<const unsigned char*>(d);
            for (size_t i = 0; i < n; ++i) { ckey ^= b[i]; ckey *= 1099511628211ull; }
        };
        {
            const uint32_t ver = CACHE_VERSION;
            const long cl = _replica->code_length(), ecl = _replica->eff_code_length();
            const double cr = _replica->chip_rate_hz();
            const int sl = _replica->secondary_length();
            mix(&ver, sizeof ver); mix(&cl, sizeof cl); mix(&ecl, sizeof ecl);
            mix(&cr, sizeof cr);   mix(&sl, sizeof sl);
            mix(&_sample_rate, sizeof _sample_rate);
            mix(&_fft_len, sizeof _fft_len);
            mix(&Mp, sizeof Mp);
            mix(&anchor, sizeof anchor);
            mix(&n_prn, sizeof n_prn);
            mix(_prns.data(), _prns.size() * sizeof(int));
            mix(cov_global.data(), cov_global.size() * sizeof(int));
            mix(cov_local.data(), cov_local.size() * sizeof(int));
        }
        char cpath[128];
        std::snprintf(cpath, sizeof cpath, "/tmp/gnss_repl_%016llx.bin",
                      (unsigned long long)ckey);
        const size_t per_bytes = (size_t)Mp * sizeof(cf);
        const size_t want_bytes = 2u * (size_t)n_prn * cov_local.size() * per_bytes;
        // EVERY local channel index must be inside the row before anything indexes by it. The
        // previous attempt at this cache indexed without checking and corrupted the heap.
        bool cov_ok = true;
        for (int lc : cov_local)
            if (lc < 0 || lc >= _n_chan)
                cov_ok = false;
        if (!cov_ok)
            FATAL_ERROR("GnssChannelizedSearch[{:s}]: covering channel index outside [0,{:d})",
                        unique_name, _n_chan);
        bool from_cache = false;
        if (cov_ok) {
            if (FILE* fp = std::fopen(cpath, "rb")) {
                std::fseek(fp, 0, SEEK_END);
                const long sz = std::ftell(fp);
                std::fseek(fp, 0, SEEK_SET);
                if (sz > 0 && (size_t)sz == want_bytes) {
                    _repl_head.assign((size_t)n_prn, {});
                    _repl_tail.assign((size_t)n_prn, {});
                    from_cache = true;
                    for (int p = 0; p < n_prn && from_cache; ++p) {
                        _repl_head[(size_t)p].assign((size_t)_n_chan, {});
                        _repl_tail[(size_t)p].assign((size_t)_n_chan, {});
                        for (int lc : cov_local) {
                            auto& h = _repl_head[(size_t)p][(size_t)lc];
                            auto& t = _repl_tail[(size_t)p][(size_t)lc];
                            h.assign((size_t)Mp, cf(0.0f, 0.0f));
                            t.assign((size_t)Mp, cf(0.0f, 0.0f));
                            if (std::fread(h.data(), 1, per_bytes, fp) != per_bytes
                                || std::fread(t.data(), 1, per_bytes, fp) != per_bytes) {
                                from_cache = false; // truncated or racing writer -> rebuild
                                break;
                            }
                        }
                    }
                }
                std::fclose(fp);
            }
        }
        if (!from_cache) {
        _repl_head.assign((size_t)n_prn, {});
        _repl_tail.assign((size_t)n_prn, {});
        for (int p = 0; p < n_prn; ++p) {
            // A: no overlay -> head + tail.   B: (-1)^period -> (-1)^k0 * (head - tail).
            const auto A = _replica->channels_hoprate(p, anchor, 0.0, 0.0, Mp, cov_global, {}, -1);
            const auto B = _replica->channels_hoprate(
                p, anchor, 0.0, 0.0, Mp, cov_global,
                [Lc](long long chip) {
                    const long long k = (long long)std::floor((double)chip / (double)Lc);
                    return ((k % 2 + 2) % 2 == 0) ? 1.0f : -1.0f;
                },
                -1);
            _repl_head[(size_t)p].assign((size_t)_n_chan, {});
            _repl_tail[(size_t)p].assign((size_t)_n_chan, {});
            for (size_t i = 0; i < cov_local.size(); ++i) {
                const size_t lc = (size_t)cov_local[i];
                _repl_head[(size_t)p][lc].assign((size_t)Mp, cf(0.0f, 0.0f));
                _repl_tail[(size_t)p][lc].assign((size_t)Mp, cf(0.0f, 0.0f));
                for (int m = 0; m < Mp; ++m) {
                    const float sgn = ((_repl_k0[(size_t)m] % 2 + 2) % 2 == 0) ? 1.0f : -1.0f;
                    const cf d = B[i][(size_t)m] * sgn; // = head - tail
                    _repl_head[(size_t)p][lc][(size_t)m] = 0.5f * (A[i][(size_t)m] + d);
                    _repl_tail[(size_t)p][lc][(size_t)m] = 0.5f * (A[i][(size_t)m] - d);
                }
            }
        }
        }
        if (!from_cache) { // persist for every future restart; a partial file simply misses above
            if (FILE* fp = std::fopen(cpath, "wb")) {
                for (int p = 0; p < n_prn; ++p)
                    for (int lc : cov_local) {
                        std::fwrite(_repl_head[(size_t)p][(size_t)lc].data(), 1, per_bytes, fp);
                        std::fwrite(_repl_tail[(size_t)p][(size_t)lc].data(), 1, per_bytes, fp);
                    }
                std::fclose(fp);
            }
        }
        _repl_scratch.assign((size_t)_n_chan, {});
        for (size_t i = 0; i < cov_local.size(); ++i)
            _repl_scratch[(size_t)cov_local[i]].assign((size_t)Mp, cf(0.0f, 0.0f));
        INFO("GnssChannelizedSearch[{:s}]: repl0 head/tail built for {:d} PRNs x {:d} channels x "
             "{:d} hops in {:.2f} s ({:.1f} MB, {:s}) -- the {:d} nh alignments are a per-hop "
             "sign pair, 2 generator calls per PRN instead of {:d}",
             unique_name, n_prn, (int)cov_local.size(), Mp, steady_s() - t_build,
             2e-6 * (double)n_prn * cov_local.size() * Mp * sizeof(cf),
             from_cache ? "from disk cache" : cpath, _n_nh, _n_nh);
        _repl0_cover = cov_global;
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

#ifdef GNSS_CUDA
        // Build the device engine once the covering set and Mp are known. Anything it cannot
        // represent EXACTLY (a non-power-of-two fine axis, a geometry where the fold's
        // precondition fails) throws here, and we log it and stay on the CPU -- the GPU path is
        // an optimization, never a change of answer.
        if (_cuda_acq_wanted && !cov_local.empty()) {
            try {
                const int s_cols = (_fft_len + _acquire_fine_step - 1) / _acquire_fine_step;
                _cuda_acq.reset(new GnssCudaAcquire((int)cov_local.size(), Mp, hpr, _n_chan,
                                                    (int)_doppler_grid.size(), s_cols));
                _cuda_acq->set_channels(cov_local, cov_global, _fft_len, _acquire_fine_step);
                INFO("GnssChannelizedSearch[{:s}]: CUDA acquire ENABLED -- {:.2f} GB on device, "
                     "surface stays there ({:d} Doppler x {:d} lags x {:d} fine)",
                     unique_name, 1e-9 * (double)_cuda_acq->device_bytes(),
                     (int)_doppler_grid.size(), Mp, s_cols);
            } catch (const std::exception& e) {
                _cuda_acq.reset();
                WARN("GnssChannelizedSearch[{:s}]: CUDA acquire unavailable, using the CPU path: "
                     "{:s}",
                     unique_name, e.what());
            }
        }
        // A5: the refine engines, ONE PER <=64-CHANNEL GROUP. chan_mask is a uint64_t, so a
        // single engine cannot cover the aggregator's 79-channel union comb; the split is exact
        // because correlation and replica_energy are both additive over channels.
        if (_cuda_ref_wanted && !cov_local.empty()) {
            try {
                const int rh = (_refine_hops > 0 && _refine_hops < hpr) ? _refine_hops : hpr;
                _cuda_ref.clear();
                _cuda_ref_groups.clear();
                const int GMAX = 64;
                for (size_t i0 = 0; i0 < cov_local.size(); i0 += GMAX) {
                    const size_t ng = std::min((size_t)GMAX, cov_local.size() - i0);
                    std::vector<int> gl(cov_local.begin() + i0, cov_local.begin() + i0 + ng);
                    std::vector<int> gg(cov_global.begin() + i0, cov_global.begin() + i0 + ng);
                    _cuda_ref.emplace_back(
                        new GnssCudaDespread(*_replica, n_prn, gg, rh, _sample_rate,
                                             _replica->f_offset()));
                    gnss::CudaRefineGroup grp;
                    grp.gpu = _cuda_ref.back().get();
                    grp.local = gl;
                    grp.n_hops = rh;
                    _cuda_ref_groups.push_back(std::move(grp));
                }
                INFO("GnssChannelizedSearch[{:s}]: CUDA refine ENABLED -- {:d} trial phases as "
                     "{:d} E/P/L specs, {:d} channel group(s) of <=64, {:d} hops",
                     unique_name, 2 * _refine_span / _refine_step + 1,
                     (2 * _refine_span / _refine_step + 1 + 2) / 3, (int)_cuda_ref.size(), rh);
            } catch (const std::exception& e) {
                _cuda_ref.clear();
                _cuda_ref_groups.clear();
                WARN("GnssChannelizedSearch[{:s}]: CUDA refine unavailable, using the CPU path: "
                     "{:s}",
                     unique_name, e.what());
            }
        }
#endif
    }

    // PASS PROFILE, opt-in via GNSS_SEARCH_PROFILE=1. The pass runs at ~30% CPU across 11
    // cores, so it is not saturating and the interesting question is what the 2.1 s is made of
    // -- guessing from CPU percentages has already misled twice today. Split it into the parts
    // we can actually act on: materialising each alignment's replica, the acquire surface, and
    // the refine.
    const bool prof = std::getenv("GNSS_SEARCH_PROFILE") != nullptr;
    double t_mat = 0.0, t_acq = 0.0, t_ref = 0.0;
    long n_align = 0;
    const double t_pass0 = steady_s();

    double best_any = 0.0;
    int best_any_prn = -1;
    int best_any_nh = -1;
    int n_unhinted = 0;
    int n_deferred = 0;

    // Pick this pass's slice of the eligible set, before the search loop so the cursor advances
    // once per snapshot. The hint-validity test here MUST match the one in the loop below (it is
    // only the validity predicate, not the grid construction); if they ever disagree the cursor
    // walks a different set than the loop searches.
    // Blind slot: which UNHINTED PRNs may be blind-scanned this pass. Uses the same hint-validity
    // predicate as the loop below, and its own cursor so it rotates independently of _prn_cursor.
    std::vector<char> blind_selected((size_t)n_prn, 0);
    if (_blind_prns_per_pass > 0 && !cov_local.empty()) {
        std::vector<int> unhinted;
        {
            std::lock_guard<std::mutex> lk(_hint_mtx);
            for (int p = 0; p < n_prn; ++p) {
                const DopHint& hh =
                    _snap_hints.size() == _prns.size() ? _snap_hints[(size_t)p] : _dop_hints[(size_t)p];
                const bool hinted =
                    hh.valid && (_hint_ttl_s <= 0.0 || _snap_taken_s - hh.t_recv < _hint_ttl_s);
                if (!hinted)
                    unhinted.push_back(p);
            }
        }
        const int nu = (int)unhinted.size();
        if (nu > 0) {
            if (_blind_cursor >= nu)
                _blind_cursor = 0;
            for (int i = 0; i < _blind_prns_per_pass && i < nu; ++i)
                blind_selected[(size_t)unhinted[(size_t)((_blind_cursor + i) % nu)]] = 1;
            _blind_cursor = (_blind_cursor + _blind_prns_per_pass) % nu;
        }
    }

    std::vector<char> selected((size_t)n_prn, 1);
    if (_prns_per_pass > 0 && !cov_local.empty()) {
        std::vector<int> elig;
        {
            std::lock_guard<std::mutex> lk(_hint_mtx);
            for (int p = 0; p < n_prn; ++p) {
                const DopHint& hh =
                    _snap_hints.size() == _prns.size() ? _snap_hints[(size_t)p] : _dop_hints[(size_t)p];
                const bool hinted =
                    hh.valid && (_hint_ttl_s <= 0.0 || _snap_taken_s - hh.t_recv < _hint_ttl_s);
                if (!_require_hint || hinted)
                    elig.push_back(p);
            }
        }
        const int ne = (int)elig.size();
        if (ne > _prns_per_pass) {
            std::fill(selected.begin(), selected.end(), (char)0);
            if (_prn_cursor >= ne)
                _prn_cursor = 0;
            for (int i = 0; i < _prns_per_pass; ++i)
                selected[(size_t)elig[(size_t)((_prn_cursor + i) % ne)]] = 1;
            _prn_cursor = (_prn_cursor + _prns_per_pass) % ne;
        }
    }

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
            // The SNAPSHOT-epoch hints, TTL judged against the snapshot time -- not the live
            // table and not now(): the data is frozen, so the hints must be too.
            const DopHint& hh = _snap_hints.size() == _prns.size() ? _snap_hints[p] : _dop_hints[p];
            if (hh.valid && (_hint_ttl_s <= 0.0 || _snap_taken_s - hh.t_recv < _hint_ttl_s)) {
                hinted = true;
                // _hint_dop_sign: the map from PHYSICAL (hinted) Doppler to the internal r2c
                // grid. -1 is the convention validated on airspy end to end. It is a CONFIG
                // because a front end whose FFT sign convention is opposite to the model's
                // conjugates every channel -- which leaves the correlation peak height alone
                // and NEGATES the internal Doppler, so a hinted scan centered at -pred misses
                // a real satellite at +pred by 2x its Doppler, for every satellite, forever.
                // A symmetric blind grid would catch that; require_hint never scans one. An
                // A/B run of the two signs is the direct hardware-convention test.
                const double c = _hint_dop_sign * hh.doppler, m = std::max(0.0, hh.margin);
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
            ++n_unhinted;
            std::lock_guard<std::mutex> lk(_det_mtx);
            _detections[p] = Detection{};
            continue;
        }
        // BOUNDED COLD START. An unhinted PRN gets the full blind grid, ~30x a hinted one, so
        // with require_hint false a sky of below-horizon satellites would spend the whole pass on
        // PRNs that cannot be found. Give the blind scan a rotating slot instead: every hinted
        // PRN is searched every pass (cheap, and what keeps cp_hist dense enough for the broker's
        // code_phase_rate fit), while reacquisition sweeps the rest a few at a time.
        if (!hinted && _blind_prns_per_pass > 0 && !blind_selected[(size_t)p]) {
            ++n_deferred;
            continue;
        }

        // ROUND-ROBIN: search only this pass's slice of the eligible set, and leave everyone
        // else's HELD detection untouched (they are not misses -- they simply were not looked
        // at, so neither the latch nor the miss counter may fire). Selection is over the
        // eligible PRNs, not the raw list, or a sky full of below-horizon PRNs would spend the
        // pass's slots on satellites that get skipped anyway.
        if (!selected[(size_t)p]) {
            ++n_deferred;
            continue;
        }

        const std::vector<double>& grid = pgrid.empty() ? _doppler_grid : pgrid;

        // Per-PRN device setup: the Doppler grid (this PRN's, hinted or blind) and the replica
        // tables. Both are PRN-scoped, so they happen once here rather than per alignment.
        bool gpu_ok = false;
#ifdef GNSS_CUDA
        if (_cuda_acq) {
            gpu_ok = _cuda_acq->set_doppler_grid(grid, _sample_rate, _fft_len);
            if (!gpu_ok) {
                // Almost always doppler_step being half the transform's bin spacing. The engine
                // refuses rather than computing a DIFFERENT thing, so say why, once in a while.
                if ((_cuda_acq_fallbacks++ % 512) == 0)
                    WARN("GnssChannelizedSearch[{:s}]: CUDA acquire declined this Doppler grid "
                         "(step {:.4f} Hz is not a whole multiple of the transform bin {:.4f} Hz) "
                         "-- running the CPU acquire. Set doppler_step to the bin spacing.",
                         unique_name, _doppler_step,
                         _sample_rate / ((double)Mp * (double)_fft_len));
            } else {
                // Covering-order head/tail for this PRN. A2 re-uploads per PRN (~4 MB); caching
                // the whole bank on the device is A3's follow-on, not correctness.
                std::vector<std::vector<cf>> hh(cov_local.size()), tt(cov_local.size());
                for (size_t ii = 0; ii < cov_local.size(); ++ii) {
                    hh[ii] = _repl_head[(size_t)p][(size_t)cov_local[ii]];
                    tt[ii] = _repl_tail[(size_t)p][(size_t)cov_local[ii]];
                }
                _cuda_acq->set_replica(hh, tt);
            }
        }
#endif

        // Integrate the |D|^2 surface over the snapshot windows (incoherent), once per
        // secondary-code alignment, and keep the best peak. The alignments are processed
        // SEQUENTIALLY and only the winning result is retained, so peak memory is one surface
        // (128 MB at CHORD dimensions) rather than _n_nh of them.
        gnss::AcquisitionResult a{};
        int best_nh = -1;
        std::vector<double> surf;
        std::vector<std::vector<cf>> dch(_n_chan, std::vector<cf>(hpr));
        gnss::AcquisitionSurface dims{}; // hoisted: the refine needs s_stored to de-alias
        // WHICH ALIGNMENTS TO SCAN. Each one costs a FULL acquisition surface, so with a
        // 20-chip overlay this loop is ~92% of a pass (measured: acquire 40.00 s vs refine
        // 3.62 s at live parameters). The alignment advances exactly one index per primary code
        // period and a period is an exact hop count, so a hint from OUR OWN last detection
        // propagates by counter arithmetic alone -- no clock, nothing to bootstrap. See NhHint.
        std::vector<int> nh_scan;
        {
            std::lock_guard<std::mutex> lk(_hint_mtx);
            const NhHint& nhh = (_nh_snap_hints.size() == _prns.size())
                                    ? _nh_snap_hints[(size_t)p] : NhHint{};
            const bool fresh = nhh.valid && _n_nh > 1
                               && (_hint_ttl_s <= 0.0 || _snap_taken_s - nhh.t_recv < _hint_ttl_s);
            if (fresh) {
                // hops per primary code period; exact for this signal's geometry
                const double per_hops = (_replica->code_length() / _replica->chip_rate_hz())
                                        * _sample_rate / (double)_fft_len;
                const long long dperiods =
                    (per_hops > 0.0)
                        ? (long long)std::llround((double)(_snap_start_hop - nhh.ref_hop) / per_hops)
                        : 0;
                const int pred = (int)(((nhh.nh + dperiods) % _n_nh + _n_nh) % _n_nh);
                for (int d = -_nh_hint_span; d <= _nh_hint_span; ++d)
                    nh_scan.push_back(((pred + d) % _n_nh + _n_nh) % _n_nh);
            }
        }
        if (nh_scan.empty())                       // no fresh hint -> the full blind scan
            for (int i = 0; i < _n_nh; ++i)
                nh_scan.push_back(i);
        for (int nh : nh_scan) {
            // repl0 (code 0, Doppler 0) on the covering channels -- precomputed, not rebuilt.
            // Materialise this alignment: R[m] = s(k0[m]+nh)*head[m] + s(k0[m]+1+nh)*tail[m].
            // A per-hop complex scale over ~331k elements (~1 ms) against a correlation costing
            // seconds -- so twenty alignments cost one build, not twenty.
            const double _t_m0 = prof ? steady_s() : 0.0;
            ++n_align;
            // The sign pair depends on the HOP and the alignment, not on the channel, so hoist
            // it: computing it inside the channel loop evaluated overlay_sign 106x per hop and
            // made a pass SLOWER than the eager bank it replaced (7.14 s vs 4.4 s, measured).
            _sgn0.resize((size_t)Mp);
            _sgn1.resize((size_t)Mp);
            for (int m = 0; m < Mp; ++m) {
                _sgn0[(size_t)m] = (float)_replica->overlay_sign(_repl_k0[(size_t)m], nh);
                _sgn1[(size_t)m] = (float)_replica->overlay_sign(_repl_k0[(size_t)m] + 1, nh);
            }
            for (size_t ii = 0; ii < cov_local.size(); ++ii) {
                const size_t lc = (size_t)cov_local[ii];
                const cf* __restrict hd = _repl_head[(size_t)p][lc].data();
                const cf* __restrict tl = _repl_tail[(size_t)p][lc].data();
                cf* __restrict outv = _repl_scratch[lc].data();
                const float* __restrict a0 = _sgn0.data();
                const float* __restrict a1 = _sgn1.data();
                for (int m = 0; m < Mp; ++m)
                    outv[m] = hd[m] * a0[m] + tl[m] * a1[m];
            }
            const std::vector<std::vector<cf>>& repl0 = _repl_scratch;
            if (prof) t_mat += steady_s() - _t_m0;
            const double _t_a0 = prof ? steady_s() : 0.0;
            gnss::AcquisitionResult ai{};
#ifdef GNSS_CUDA
            if (gpu_ok) {
                // Device path. The replica MATERIALISE happens on the GPU too (from the sign
                // pair), so _repl_scratch above is redundant here -- left in place because the
                // CPU fallback shares this loop and the materialise is 0.002 s per pass.
                dims = gnss::surface_dims(cov_global, _fft_len, (int)grid.size(), Mp,
                                          _acquire_fine_step);
                _cuda_acq->clear_surface();
                for (int w = 0; w < nwin; ++w) {
                    // FFT(data) depends on neither PRN nor alignment; re-uploading per window is
                    // the price of the (PRN -> nh -> window) loop order, and with the shipped
                    // acquire_windows: 1 it happens once per alignment. Hoisting it out of the
                    // PRN loop entirely is A4.
                    _cuda_acq->set_snapshot(_snapshot.data() + (size_t)w * hpr * _n_chan);
                    _cuda_acq->accumulate(_sgn0.data(), _sgn1.data());
                }
                _last_surface_cells = dims.size();
                if (prof) t_acq += steady_s() - _t_a0;
                const double _t_p0 = prof ? steady_s() : 0.0;
                ai = _cuda_acq->peak_result(dims, grid, _sample_rate, _replica->chip_rate_hz(),
                                            _replica->code_length());
                if (prof) t_acq += steady_s() - _t_p0;
            } else
#endif
            {
                surf.assign(surf.size(), 0.0); // reuse the allocation, drop the previous alignment
                for (int w = 0; w < nwin; ++w) {
                    for (int lc = 0; lc < _n_chan; ++lc)
                        for (int m = 0; m < hpr; ++m)
                            dch[lc][m] = _snapshot[((size_t)(w * hpr + m)) * _n_chan + lc];
                    dims = gnss::channelized_accumulate(dch, repl0, cov_local, grid, _sample_rate,
                                                        _n_chan, surf, _acq_ws, cov_global,
                                                        _fft_len, _acquire_threads,
                                                        _acquire_fine_step);
                }
                _last_surface_cells = dims.size();
                if (prof) t_acq += steady_s() - _t_a0;
                const double _t_p0 = prof ? steady_s() : 0.0;
                ai = gnss::channelized_peak(surf, dims, grid, _sample_rate,
                                            _replica->chip_rate_hz(), _replica->code_length());
                if (prof) t_acq += steady_s() - _t_p0;
            }
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
            // Refine the code phase: de-alias the fine lag, then localize within the surviving
            // ambiguity. Lives in gnssSeedTransport (with the full argument for why it is an
            // alias SCAN and not a wider span) so the offline harness drives THIS code.
            //
            // Hoist the data columns: they do not depend on the trial cp, and rebuilding them
            // per step was copying hpr * n_cover samples for nothing.
            std::vector<std::vector<cf>> d(cov_local.size(), std::vector<cf>(hpr));
            for (size_t i = 0; i < cov_local.size(); ++i)
                for (int m = 0; m < hpr; ++m)
                    d[i][m] = _snapshot[((size_t)m) * _n_chan + cov_local[i]];
            const double _t_r0 = prof ? steady_s() : 0.0;
            const int rh = (_refine_hops > 0 && _refine_hops < hpr) ? _refine_hops : hpr;
            double best_cp;
#ifdef GNSS_CUDA
            if (!_cuda_ref_groups.empty()) {
                // The GPU refine reads the snapshot in its native [hop][n_chan] layout, so it
                // does NOT need `d` -- the per-channel copy above is CPU-path scratch.
                best_cp = gnss::refine_peak_cuda(_cuda_ref_groups, *_replica, p, _snapshot.data(),
                                                 _n_chan, anchor, a.code_phase_chips, dop,
                                                 _sample_rate, _refine_span, _refine_step);
            } else
#endif
            {
                best_cp = gnss::refine_peak(*_replica, p, d, cov_global, a.code_phase_chips, dims,
                                            (_n_nh > 1) ? best_nh : -1, dop, anchor, rh,
                                            _sample_rate, _refine_span, _refine_step,
                                            _acquire_threads);
            }
            if (prof) t_ref += steady_s() - _t_r0;
            // Peak -> reported phases. The arithmetic lives in gnssSeedTransport so the
            // offline end-to-end harness (scripts/e2e.cpp) drives THIS code rather than a
            // second copy of it -- see that header for why.
            const gnss::DetectionPhase dp = gnss::detection_phase(
                *_replica, best_cp, best_nh, _n_nh, dop, _snap_start_hop, anchor, _sample_rate,
                a.peak_tau_samples);
            det.doppler_hz = dop;
            det.code_phase_chips = dp.cp0;
            det.ref_hop = _snap_start_hop; // capture-time anchor for cp0 (for the slope fit)
            det.nh = (_n_nh > 1) ? best_nh : -1; // MEASURED overlay alignment, not reconstructed
            det.code_phase_long_chips = dp.cp_long;
            det.code_phase_at_ref_chips = dp.cp_at_ref;
            det.valid = true;
            // DIAG: decompose where the reported phases come from, to localize instability:
            //   hop   = absolute snapshot reference (sample_seq/fft_len)
            //   coarse= cross-channel acquire cp (pre-refine)   refine= +- from refine
            //   nh    = MEASURED overlay alignment
            //   cp0   = argument at sample 0 (mod L); ph@ref = physical phase at hop (mod 20L)
            // DEBUG, not INFO (50c4a0563): unthrottled once per PRN per snapshot, which is spam
            // at the prototype's cadence. CHORD re-enables it per stage (log_level: debug on
            // gps_search) rather than globally -- it is the seeding investigation's instrument.
            DEBUG("GnssChannelizedSearch[{:s}]: PRN {:d} snr {:.1f} dop {:+.0f} | hop {:d} "
                 "coarse {:.2f} refine {:+.2f} nh {:d} -> cp0 {:.2f} cp_long {:.2f} "
                 "ph@ref {:.2f}",
                 unique_name, _prns[p], a.snr, dop, _snap_start_hop, a.code_phase_chips,
                 best_cp - a.code_phase_chips, det.nh, dp.cp0, dp.cp_long, dp.cp_at_ref);
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
    if (best_any_prn >= 0) {
        // The PURE-NOISE expectation of this pass's best SNR, so a threshold below it is
        // visibly meaningless in the log it has to be read from. The surface cell is a sum of
        // nwin |D|^2 draws (Gamma(nwin) up to channel-count effects, unit mean), and the pass
        // maximum over N cells solves N * P(X > x) = 1 with the true Gamma tail:
        //     k*x - (k-1)*ln(k*x) + ln((k-1)!) = ln(N),  k = windows integrated.
        // The Gaussian-ish shortcut 1 + c/sqrt(k) badly UNDERESTIMATES this at small k (the
        // Gamma tail is heavy): at k = 4 it said ~5.0 where the truth is ~7.5, which is how a
        // 20-pass "detection plateau" of 5.9-7.3 got waved through on 2026-07-30 -- flat
        // through an assumed main-beam crossing, because it was noise the whole time. The
        // model reproduces the measured noise maxima at k=16 (3.1-3.4 obs, 3.4 pred) and k=64
        // (1.86-1.92 obs, 1.94 pred). Correlations between cells make it a mild OVERestimate,
        // which is the safe side for a ceiling.
        const int kwin = std::min(_acquire_windows, (int)(_snap_hops / (size_t)_hops_per_record));
        double ceiling = 0.0;
        if (kwin > 0) {
            double lgamma_k = std::lgamma((double)kwin); // ln((k-1)!)
            const double lnN = std::log((double)_last_surface_cells * std::max(1, _n_nh));
            double kx = std::max((double)kwin, lnN); // iterate kx = lnN + (k-1)ln(kx) - ln((k-1)!)
            for (int it = 0; it < 50; ++it)
                kx = lnN + (kwin - 1) * std::log(kx) - lgamma_k;
            ceiling = kx / (double)kwin;
        }
        if (prof) {
            const double tot = steady_s() - t_pass0;
            INFO("GnssChannelizedSearch[{:s}]: [profile] pass {:.3f}s = materialise {:.3f} + "
                 "acquire {:.3f} + refine {:.3f} (+{:.3f} other) over {:d} alignment(s)",
                 unique_name, tot, t_mat, t_acq, t_ref, tot - t_mat - t_acq - t_ref, n_align);
        }
        INFO("GnssChannelizedSearch[{:s}]: pass best snr {:.2f} (PRN {:d}{:s}), threshold {:.2f}, "
             "pure-noise ceiling ~{:.2f}{:s}{:s}",
             unique_name, best_any, best_any_prn,
             (_n_nh > 1) ? fmt::format(", nh {:d}/{:d}", best_any_nh, _n_nh) : std::string(),
             _acquire_snr, ceiling,
             (n_deferred > 0)
                 ? fmt::format(" | round-robin: {:d} searched, {:d} deferred to later passes",
                               _prns_per_pass, n_deferred)
                 : std::string(),
             (_acquire_snr < ceiling) ? "  [THRESHOLD BELOW NOISE CEILING -- every \"detection\" "
                                        "is meaningless]"
                                      : "");
    } else if (_require_hint && n_unhinted == n_prn && n_prn > 0) {
        // Every PRN skipped for lack of a fresh hint: the pass did NOTHING, at input frame
        // rate. Without this line the only symptom is the covering-channels line spamming --
        // 43k of them on 2026-07-31, 100 minutes of "searching" after the hint broker died,
        // indistinguishable from an empty sky. Say the real cause, rate-limited.
        if ((_empty_hint_passes++ % 256) == 0)
            WARN("GnssChannelizedSearch[{:s}]: 0/{:d} PRNs hinted -- require_hint is set and "
                 "the broker's hints are absent or older than hint_ttl_s ({:.0f} s). The "
                 "search is scanning NOTHING. (repeat {:d})",
                 unique_name, n_prn, _hint_ttl_s, _empty_hint_passes - 1);
    }
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
    kotekan::restServer::instance().register_post_callback(
        unique_name + "/set_nh_hint",
        std::bind(&GnssChannelizedSearch::set_nh_hint_callback, this, _1, _2));

    _worker = std::thread(&GnssChannelizedSearch::search_worker, this);

    // CONSUMER PROFILE (GNSS_SEARCH_PROFILE=1). The worker profile showed the pass computing
    // for 0.25 s on a 1.57 s period -- 82% idle -- so the interesting time is on THIS side.
    // Split it: blocked in wait_for_full_frame (upstream is slow), vs frames arriving but
    // discarded because the worker was still busy (we are throwing away sky we could search).
    const bool cprof = std::getenv("GNSS_SEARCH_PROFILE") != nullptr;
    double t_wait = 0.0, t_cyc0 = steady_s();
    long n_frames = 0, n_skipped_busy = 0, n_snaps = 0;

    int frame_in = 0;
    bool filling = false;
    size_t filled_hops = 0;  // hops actually copied into the current snapshot (for the drop log)
    long long abs_hops = 0;  // total hops consumed from the stream (absolute reference)

    while (!stop_thread) {
        const double _t_w0 = cprof ? steady_s() : 0.0;
        auto* in_local = (cf*)in_buf->wait_for_full_frame(unique_name, frame_in);
        if (in_local == nullptr)
            break;
        if (cprof) { t_wait += steady_s() - _t_w0; ++n_frames; }
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
            if (_worker_busy && cprof)
                ++n_skipped_busy; // sky arriving while the searcher is occupied -> discarded
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
                {
                    // Freeze the hint table at the data's epoch -- see _snap_hints.
                    std::lock_guard<std::mutex> hk(_hint_mtx);
                    _snap_hints = _dop_hints;
                    _nh_snap_hints = _nh_hints; // freeze with the data, same reason as the Doppler hints
                    _snap_taken_s = steady_s();
                }
                std::lock_guard<std::mutex> lk(_m);
                _snap_ready = true;
                _worker_busy = true;
                _cv.notify_one();
                if (cprof && ++n_snaps % 20 == 0) {
                    const double el = steady_s() - t_cyc0;
                    INFO("GnssChannelizedSearch[{:s}]: [consumer] {:.1f}s: {:d} frames "
                         "({:.1f}/s), blocked {:.1f}s ({:.0f}%), {:d} frames discarded while "
                         "the worker was busy ({:.0f}%), {:d} snapshots",
                         unique_name, el, n_frames, n_frames / el, t_wait, 100.0 * t_wait / el,
                         n_skipped_busy, 100.0 * n_skipped_busy / std::max(1L, n_frames), n_snaps);
                    t_cyc0 = steady_s(); t_wait = 0.0; n_frames = 0; n_skipped_busy = 0;
                }
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
                         {"nh", d.nh},
                         {"code_phase_long_chips", d.code_phase_long_chips},
                         {"code_phase_at_ref_chips", d.code_phase_at_ref_chips},
                         {"snr", d.snr}});
    }
    conn.send_json_reply(reply);
}

void GnssChannelizedSearch::set_nh_hint_callback(kotekan::connectionInstance& conn,
                                                 nlohmann::json& json_request) {
    // Body: [{prn, nh, ref_hop}, ...] -- the alignment index THIS STAGE last reported for that
    // PRN, and the absolute hop it was measured at. nh < 0 CLEARS (revert to the full scan).
    // PRNs not listed keep their hint. Unknown PRNs ignored. Deliberately self-referential:
    // the broker echoes back our own measurement, so there is no clock to bootstrap.
    try {
        std::lock_guard<std::mutex> lk(_hint_mtx);
        for (const auto& h : json_request) {
            const int prn = h.at("prn").get<int>();
            for (size_t i = 0; i < _prns.size(); ++i) {
                if (_prns[i] != prn)
                    continue;
                const int nh = h.value("nh", -1);
                if (nh < 0)
                    _nh_hints[i] = NhHint{};
                else
                    _nh_hints[i] = NhHint{true, nh, h.value("ref_hop", (long long)0), steady_s()};
                break;
            }
        }
    } catch (const std::exception& e) {
        conn.send_error(e.what(), kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
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
