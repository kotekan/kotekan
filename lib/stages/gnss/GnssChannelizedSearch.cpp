#include "GnssChannelizedSearch.hpp"

#include "GnssChanMetadata.hpp"  // for get_gnss_chan_metadata, metadata_is_gnss_chan
#include "StageFactory.hpp"      // for REGISTER_KOTEKAN_STAGE
#include "clockProfile.hpp"      // for clock_profile_preset, clock_doppler_half_range_hz
#include "gnssSeedTransport.hpp" // for detection_phase
#include "gnssSignal.hpp"        // for SignalDescriptor, signal_by_name
#include "kotekanLogging.hpp"    // for INFO, FATAL_ERROR
#include "pfbPrototype.hpp"      // for window_from_string
#ifdef GNSS_CUDA
#include "GnssCudaAcquire.hpp"  // for GnssCudaAcquire
#include "GnssCudaDespread.hpp" // for GnssCudaDespread
#endif

#include "json.hpp" // for json

#include <algorithm>  // for min, max
#include <chrono>     // for steady_clock (hint TTL)
#include <cmath>      // for fabs, fmod, norm
#include <cstdio>     // for fopen, fread, fwrite, snprintf (the replica disk cache)
#include <cstring>    // for memcpy
#include <functional> // for bind
#include <map>        // for map (the period-label vote)

// Monotonic seconds, for the hint time-to-live.
static inline double steady_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

using kotekan::bufferContainer;
using kotekan::Config;
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
    // Frames are placed in the snapshot by the F-engine's sample_seq, which only GnssChanMetadata
    // carries.
    if (!metadata_is_gnss_chan(in_buf)) {
        FATAL_ERROR("GnssChannelizedSearch[{:s}]: in_buf has no GnssChanMetadata, so frames "
                    "carry no sample_seq to place them by",
                    unique_name);
        return;
    }

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
    // The global spectrum index of every local channel (a strided comb, or a contiguous run). A
    // wrong channel list correlates the data against the replica for another part of the
    // spectrum, which gives a noise surface rather than an error.
    _chan_ids = config.get<std::vector<int>>(unique_name, "channel_ids");
    _n_chan = config.get<int>(unique_name, "n_channels");
    if ((int)_chan_ids.size() != _n_chan) {
        FATAL_ERROR("GnssChannelizedSearch: channel_ids has {:d} entries but n_channels is {:d}",
                    _chan_ids.size(), _n_chan);
        return;
    }
    for (int c : _chan_ids)
        if (c < 0 || c >= _N) {
            FATAL_ERROR("GnssChannelizedSearch: channel_id {:d} outside [0,{:d})", c, _N);
            return;
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
    // Threads for the CPU acquire (the per-channel correlations, then the aggregate over
    // Doppler x lag rows) and the CPU refine's trial phases; 1 is the serial path.
    _acquire_threads = config.get_default<int>(unique_name, "acquire_threads", 1);
    // Fine-lag decimation (1 = every sample). The fine-lag lobe is fft_len / (comb span in
    // bins) samples wide, so a per-sample axis oversamples it; see AcquisitionSurface.
    _acquire_fine_step = std::max(1, config.get_default<int>(unique_name, "acquire_fine_step", 1));
    _nh_hint_span = std::max(0, config.get_default<int>(unique_name, "nh_hint_span", 1));
    _hint_ttl_s = config.get_default<double>(unique_name, "hint_ttl_s", 8.0);
    // Retired knobs. The generated config still sets each one to the only behaviour left (every
    // hinted PRN on every pass, unhinted PRNs skipped, the pair-sum plane and the period-label
    // vote, the bank's +1 code-Doppler sign), so another value is refused rather than ignored.
    if (config.get_default<int>(unique_name, "prns_per_pass", 0) != 0
        || config.get_default<int>(unique_name, "blind_prns_per_pass", 0) != 0
        || !config.get_default<bool>(unique_name, "require_hint", true)
        || !config.get_default<bool>(unique_name, "acq_pairsum_select", true)
        || !config.get_default<bool>(unique_name, "nh_label_consensus", true)
        || config.get_default<double>(unique_name, "code_doppler_sign", 1.0) != 1.0) {
        FATAL_ERROR("GnssChannelizedSearch[{:s}]: prns_per_pass, blind_prns_per_pass, "
                    "require_hint, acq_pairsum_select, nh_label_consensus and code_doppler_sign "
                    "are retired: leave them unset or at 0, 0, true, true, true and 1",
                    unique_name);
        return;
    }
#ifdef GNSS_CUDA
    // The refine follows the acquire: with the acquire on the GPU, a CPU refine dominates the pass.
    _cuda_acq_wanted = config.get_default<bool>(unique_name, "use_cuda_acquire", false);
#else
    if (config.get_default<bool>(unique_name, "use_cuda_acquire", false))
        WARN("GnssChannelizedSearch[{:s}]: use_cuda_acquire set but this binary was built without "
             "CUDA -- running the CPU acquire",
             unique_name);
#endif
    double dmin = config.get_default<double>(unique_name, "doppler_min", -6000.0);
    double dmax = config.get_default<double>(unique_name, "doppler_max", 6000.0);
    _doppler_step = config.get_default<double>(unique_name, "doppler_step", 500.0);
    // A top-level /clock_profile block, when present, sets the Doppler extent from the named
    // clock's frequency accuracy and the band's maximum sky Doppler, overriding doppler_min/max.
    // See clockProfile.hpp.
    const std::string clk_name =
        config.get_default<std::string>("/clock_profile", "name", std::string(""));
    if (!clk_name.empty()) {
        const gnss::ClockProfile cp = gnss::clock_profile_preset(clk_name);
        const double half = gnss::clock_doppler_half_range_hz(sig->carrier_hz, cp.accuracy_ppm);
        dmin = -half;
        dmax = half;
        INFO("GnssChannelizedSearch: clock_profile '{:s}' ({:.3g} ppm) -> Doppler search +-{:.0f} "
             "Hz",
             clk_name, cp.accuracy_ppm, half);
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
    // FDMA (GLONASS L1OF/L2OF): satellites share one code and are separated by carrier, so each
    // slot needs its own offset from band centre. Empty for CDMA.
    _replica->set_prn_freq_offsets(
        config.get_default<std::vector<double>>(unique_name, "prn_freq_offset_hz", {}));
    if (_replica->prn_freq_offsets_set() > 0)
        INFO("GnssChannelizedSearch: FDMA carrier offsets applied to {:d} of {:d} PRNs",
             _replica->prn_freq_offsets_set(), (int)_prns.size());

    // The coherent window is one replica period: hpr == Mp, so the data window is never
    // zero-padded.
    _hops_per_record = _replica->repl_period_hops();

    // Refine geometry. The span defaults to one hop each side, the acquire's coarse-lag quantum.
    // Each step costs one exact despread, so a wide bank needs a refine_step well above the
    // default of 1 sample (2 * fft_len + 1 despreads per detection).
    _refine_span = config.get_default<int>(unique_name, "refine_span", _fft_len);
    // Refine integration length, hops (0 = the full record). The refine only localises a peak
    // that already cleared detection, so a shorter integration trades SNR for speed;
    // scripts/gnss/e2e.cpp (--refine-hops) measures the resulting phase error.
    _refine_hops = std::max(0, config.get_default<int>(unique_name, "refine_hops", 0));
    _refine_step = std::max(1, config.get_default<int>(unique_name, "refine_step", 1));

    // Secondary-code (Neuman-Hofman) alignment search. The coherent window is the replica
    // period: 3125 hops = 16 ms = 16 code periods on CHORD L5. The L5 Q5 pilot carries the NH20
    // overlay, one +-1 chip per 1 ms period, so a replica without it sums 16 chips of a
    // near-balanced sequence: 12.7 dB rms loss over the 20 alignments, and a null for some. With
    // the right alignment the window integrates at full coherent gain (test l5q_nh20_overlay).
    // The alignment is one unknown in [0, secondary_length), so each value is searched.
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
    _snap_hints.assign(_prns.size(), DopHint{});
    _nh_snap_hints.assign(_prns.size(), NhHint{});
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
    // Replica window start: one replica period, a whole number of code periods, so the PFB's
    // warm-up taps before it fall at positive samples.
    const long long anchor = (long long)Mp * _fft_len;
    const double dmax = _doppler_grid.empty() ? 0.0
                                              : std::max(std::fabs(_doppler_grid.front()),
                                                         std::fabs(_doppler_grid.back()));
    const int nwin = std::min(_acquire_windows, (int)(_snap_hops / (size_t)hpr));

    // Covering channels (global) that fall in this subband: the union over PRNs, since one set
    // serves every PRN below. Under FDMA each satellite has its own carrier, so one PRN's set
    // would miss the others; under CDMA every PRN has the same set.
    const auto cover = _replica->covering_bins_union(dmax, _doppler_margin_hz);
    std::vector<int> cov_local, cov_global;
    for (int c : cover) {
        const int lc = local_of_global(c);
        if (lc >= 0) {
            cov_local.push_back(lc);
            cov_global.push_back(c);
        }
    }

    // With no covering channels every PRN gets an invalid detection, which looks like an empty
    // sky. Log it every snapshot: the usual cause is an f_offset that puts the carrier outside
    // this subband.
    if (cov_local.empty())
        INFO("GnssChannelizedSearch[{:s}]: NO covering channels overlap this subband -- the "
             "replica's covering bins do not intersect channels [{:d},{:d}). Check f_offset "
             "(currently placing the carrier at bin {:.1f}).",
             unique_name, _chan_offset, _chan_offset + _n_chan,
             _replica->f_offset() / (_sample_rate / _fft_len));
    else if ((int)cov_local.size() != _last_cov_n || cov_global.front() != _last_cov_first
             || cov_global.back() != _last_cov_last) { // once per set, not per snapshot
        INFO("GnssChannelizedSearch[{:s}]: {:d} covering channels in this subband (global "
             "{:d}..{:d})",
             unique_name, (int)cov_local.size(), cov_global.front(), cov_global.back());
        _last_cov_n = (int)cov_local.size();
        _last_cov_first = cov_global.front();
        _last_cov_last = cov_global.back();
    }

    // Build the head/tail bank (see _repl_head) when the covering set changes. channels_hoprate()
    // generates only the covering channels and equals channels() (checked at CHORD scale in
    // test_gnss_channelized_replica). At code phase 0 and Doppler 0 the bank depends only on
    // the PRNs, the covering set and Mp, but only a change of covering set (which comes from
    // the blind grid, not the hints) triggers a rebuild. Rows are indexed by local channel, as
    // channelized_accumulate reads repl0_ch[covering[ci]]; non-covering rows stay empty.
    if (!cov_local.empty() && cov_global != _repl0_cover) {
        const double t_build = steady_s();
        const long Lc = _replica->eff_code_length();
        // k0[m], the period index at sample anchor + m*fft_len. With code phase 0 and Doppler 0
        // the chip there is (anchor + m*fft_len) * chip_rate / Fs.
        _repl_k0.assign((size_t)Mp, 0);
        for (int m = 0; m < Mp; ++m) {
            const double chip = ((double)anchor + (double)m * (double)_fft_len)
                                * _replica->chip_rate_hz() / _sample_rate;
            _repl_k0[(size_t)m] = (long long)std::floor(chip / (double)Lc);
        }
        // Disk cache of the bank, which is the same on every restart (the anchor is derived, not
        // per run). The file name is a 64-bit hash of the code and secondary lengths, chip rate,
        // sample rate, fft_len, Mp, anchor, PRN list and covering set; the signal name and codes,
        // f_offset, FDMA offsets, num_taps and window are not hashed. /tmp keeps the file
        // (~170 MB at CHORD dimensions) node-local, off NFS. Bump CACHE_VERSION to invalidate
        // every file.
        constexpr uint32_t CACHE_VERSION = 1;
        uint64_t ckey = 1469598103934665603ull;
        const auto mix = [&ckey](const void* d, size_t n) {
            const auto* b = static_cast<const unsigned char*>(d);
            for (size_t i = 0; i < n; ++i) {
                ckey ^= b[i];
                ckey *= 1099511628211ull;
            }
        };
        {
            const uint32_t ver = CACHE_VERSION;
            const long cl = _replica->code_length(), ecl = _replica->eff_code_length();
            const double cr = _replica->chip_rate_hz();
            const int sl = _replica->secondary_length();
            mix(&ver, sizeof ver);
            mix(&cl, sizeof cl);
            mix(&ecl, sizeof ecl);
            mix(&cr, sizeof cr);
            mix(&sl, sizeof sl);
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
        std::snprintf(cpath, sizeof cpath, "/tmp/gnss_repl_%016llx.bin", (unsigned long long)ckey);
        const size_t per_bytes = (size_t)Mp * sizeof(cf);
        const size_t want_bytes = 2u * (size_t)n_prn * cov_local.size() * per_bytes;
        // Every local channel index must be inside the row before anything indexes by it.
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
                const auto A =
                    _replica->channels_hoprate(p, anchor, 0.0, 0.0, Mp, cov_global, {}, -1);
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
        if (!from_cache) { // a partial file fails the size check above
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
        // Log the refine cost (one exact despread per step, per detection) against
        // fft_len / (covering channels) samples, the resolution of a contiguous covering set. A
        // comb's fine-lag lobe is narrower, fft_len / (comb span in bins); see refine_peak.
        const long evals = 2L * _refine_span / _refine_step + 1;
        const double res_samp = (double)_fft_len / (double)cov_local.size();
        INFO("GnssChannelizedSearch[{:s}]: code-phase refine +-{:d} samples step {:d} = {:d} exact "
             "despreads; the {:d}-channel covering set band-limits the despread to {:.2f} MHz, so "
             "its intrinsic resolution is ~{:.0f} samples ({:.1f} chips){:s}",
             unique_name, _refine_span, _refine_step, evals, (int)cov_local.size(),
             1e-6 * cov_local.size() * _sample_rate / _fft_len, res_samp,
             res_samp * _replica->chip_rate_hz() / _sample_rate,
             (double)_refine_step < res_samp / 8.0 ? " -- OVERSAMPLED, consider raising refine_step"
                                                   : "");

#ifdef GNSS_CUDA
        // Build the device engine once the covering set and Mp are known. A geometry it cannot
        // represent exactly throws, and the CPU acquire runs instead.
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
        // Refine engines, one per group of <= 64 covering channels (chan_mask is a uint64_t).
        // The split is exact: correlation and replica_energy add over channels.
        if (_cuda_acq_wanted && !cov_local.empty()) {
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
                        new GnssCudaDespread(*_replica, n_prn, gg, rh, _sample_rate));
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

    double best_any = 0.0;
    int best_any_prn = -1;
    int best_any_nh = -1;
    int n_unhinted = 0;

    for (int p = 0; p < n_prn; ++p) {
        if (cov_local.empty()) { // carrier not in this subband: nothing to find
            std::lock_guard<std::mutex> lk(_det_mtx);
            _detections[p] = Detection{};
            continue;
        }

        // Hinted grid: with a fresh Doppler hint, scan only hint +- margin. The hint is the
        // physical Doppler, while the internal grid has the r2c-flipped sign (det.doppler_hz =
        // -grid value below). Without a fresh hint the PRN is skipped this pass.
        std::vector<double> pgrid;
        bool hinted = false;
        {
            std::lock_guard<std::mutex> lk(_hint_mtx);
            // Snapshot-epoch hints, aged at the snapshot time: the data is frozen, so the hints
            // must be too.
            const DopHint& hh = _snap_hints[(size_t)p];
            if (hh.valid && (_hint_ttl_s <= 0.0 || _snap_taken_s - hh.t_recv < _hint_ttl_s)) {
                hinted = true;
                const double c = -hh.doppler, m = std::max(0.0, hh.margin);
                // The window sits on integer multiples of _doppler_step, not on the hint, so
                // a static peak gets the same grid point from scan to scan. A grid that slid
                // with the hint would let the reported Doppler hop a step between scans, and a
                // seed one step off rotates step * T_rec cycles per record (a whole cycle, a
                // correlation null, at 62.5 Hz over 16 ms).
                const double lo = std::ceil((c - m) / _doppler_step) * _doppler_step;
                for (double f = lo; f <= c + m + 1e-6; f += _doppler_step)
                    pgrid.push_back(f);
            }
        }
        if (!hinted) { // not in the active set this pass
            ++n_unhinted;
            std::lock_guard<std::mutex> lk(_det_mtx);
            _detections[p] = Detection{};
            continue;
        }

        const std::vector<double>& grid = pgrid.empty() ? _doppler_grid : pgrid;

        // The sub-grid Doppler refine (peak_from_reduction) inverts r = |g(1-d)|/|g(d)| with
        // the argument in transform bins, so the +-1 grid neighbour must be the +-1 bin. On a
        // sub-bin grid the ratio is ~1 whatever the offset and the result biases half a step
        // toward the neighbour. Checked here because both the CPU and GPU paths pass this point;
        // the check flags only a step that is not a whole number of bins.
        if (grid.size() >= 2) {
            const double bin_hz = _sample_rate / ((double)Mp * (double)_fft_len);
            const double q = (grid[1] - grid[0]) / bin_hz;
            if (std::fabs(q - std::round(q)) > 1e-6 && (_grid_bin_warns++ % 512) == 0)
                WARN("GnssChannelizedSearch[{:s}]: Doppler grid step {:.4f} Hz is not a whole "
                     "multiple of the transform bin {:.4f} Hz ({:.3f} bins) -- the sub-grid "
                     "refine is outside its domain and will bias toward the neighbouring cell. "
                     "Set doppler_step to the bin spacing.",
                     unique_name, grid[1] - grid[0], bin_hz, q);
        }

        // Per-PRN device setup, once rather than per alignment: this PRN's Doppler grid and
        // replica tables.
#ifdef GNSS_CUDA
        bool gpu_ok = false;
        if (_cuda_acq) {
            gpu_ok = _cuda_acq->set_doppler_grid(grid, _sample_rate, _fft_len);
            if (!gpu_ok) {
                // The engine refuses a grid with an entry off the transform's bin lattice, or
                // more entries than it was sized for, rather than computing something else.
                if ((_cuda_acq_fallbacks++ % 512) == 0)
                    WARN("GnssChannelizedSearch[{:s}]: CUDA acquire declined this Doppler grid "
                         "(step {:.4f} Hz is not a whole multiple of the transform bin {:.4f} Hz) "
                         "-- running the CPU acquire. Set doppler_step to the bin spacing.",
                         unique_name, _doppler_step,
                         _sample_rate / ((double)Mp * (double)_fft_len));
            } else {
                // This PRN's head/tail in covering order, uploaded per PRN.
                std::vector<std::vector<cf>> hh(cov_local.size()), tt(cov_local.size());
                for (size_t ii = 0; ii < cov_local.size(); ++ii) {
                    hh[ii] = _repl_head[(size_t)p][(size_t)cov_local[ii]];
                    tt[ii] = _repl_tail[(size_t)p][(size_t)cov_local[ii]];
                }
                _cuda_acq->set_replica(hh, tt);
            }
        }
#endif

        // Integrate the |D|^2 surface incoherently over the snapshot windows, once per
        // alignment, and keep the best peak. Alignments run one at a time and only the winner
        // is kept, so peak memory is one surface.
        gnss::AcquisitionResult a{};
        int best_nh = -1;
        std::vector<double> surf;
        std::vector<std::vector<cf>> dch; // the CPU acquire's window, sized on first use
        gnss::AcquisitionSurface dims{};
        // Alignments to scan: pred +- _nh_hint_span with a fresh hint, else all of them. Each
        // costs a full surface. pred rounds a non-integer period count, so it can be off by one
        // (see NhHint).
        std::vector<int> nh_scan;
        {
            std::lock_guard<std::mutex> lk(_hint_mtx);
            const NhHint& nhh = _nh_snap_hints[(size_t)p];
            const bool fresh = nhh.valid && _n_nh > 1
                               && (_hint_ttl_s <= 0.0 || _snap_taken_s - nhh.t_recv < _hint_ttl_s);
            if (fresh) {
                // hops per primary code period, not an integer in general
                const double per_hops = (_replica->code_length() / _replica->chip_rate_hz())
                                        * _sample_rate / (double)_fft_len;
                const long long dperiods =
                    (per_hops > 0.0) ? (long long)std::llround(
                                           (double)(_snap_start_hop - nhh.ref_hop) / per_hops)
                                     : 0;
                const int pred = (int)(((nhh.nh + dperiods) % _n_nh + _n_nh) % _n_nh);
                for (int d = -_nh_hint_span; d <= _nh_hint_span; ++d)
                    nh_scan.push_back(((pred + d) % _n_nh + _n_nh) % _n_nh);
            }
        }
        if (nh_scan.empty()) // no fresh hint -> the full blind scan
            for (int i = 0; i < _n_nh; ++i)
                nh_scan.push_back(i);
        for (int nh : nh_scan) {
            // This alignment's replica is R[m] = s(k0[m]+nh)*head[m] + s(k0[m]+1+nh)*tail[m]. The
            // sign pair depends only on the hop and the alignment, so it is computed once per
            // hop rather than per channel.
            _sgn0.resize((size_t)Mp);
            _sgn1.resize((size_t)Mp);
            for (int m = 0; m < Mp; ++m) {
                _sgn0[(size_t)m] = (float)_replica->overlay_sign(_repl_k0[(size_t)m], nh);
                _sgn1[(size_t)m] = (float)_replica->overlay_sign(_repl_k0[(size_t)m] + 1, nh);
            }
            gnss::AcquisitionResult ai{};
#ifdef GNSS_CUDA
            if (gpu_ok) {
                // Device path: the GPU applies the sign pair itself.
                dims = gnss::surface_dims(cov_global, _fft_len, (int)grid.size(), Mp,
                                          _acquire_fine_step);
                _cuda_acq->clear_surface();
                for (int w = 0; w < nwin; ++w) {
                    // The data FFT depends on neither PRN nor alignment, but this loop order
                    // uploads and transforms each window again for every (PRN, alignment).
                    _cuda_acq->set_snapshot(_snapshot.data() + (size_t)w * hpr * _n_chan);
                    _cuda_acq->accumulate(_sgn0.data(), _sgn1.data());
                }
                _last_surface_cells = dims.size();
                ai = _cuda_acq->peak_result(dims, grid, _sample_rate, _replica->chip_rate_hz(),
                                            _replica->code_length(), true);
            } else
#endif
            {
                // Materialise R into _repl_scratch on the covering rows.
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
                if (dch.empty())
                    dch.assign(_n_chan, std::vector<cf>(hpr));
                surf.assign(surf.size(), 0.0); // reuse the allocation, drop the previous alignment
                for (int w = 0; w < nwin; ++w) {
                    for (int lc = 0; lc < _n_chan; ++lc)
                        for (int m = 0; m < hpr; ++m)
                            dch[lc][m] = _snapshot[((size_t)(w * hpr + m)) * _n_chan + lc];
                    dims = gnss::channelized_accumulate(
                        dch, _repl_scratch, cov_local, grid, _sample_rate, _n_chan, surf, _acq_ws,
                        cov_global, _fft_len, _acquire_threads, _acquire_fine_step);
                }
                _last_surface_cells = dims.size();
                ai =
                    gnss::channelized_peak(surf, dims, grid, _sample_rate, _replica->chip_rate_hz(),
                                           _replica->code_length(), true, hpr);
            }
            // Both paths pick each surface's Doppler plane on the mainlobe pair-sum (the `true`
            // above), so an on-grid overlay Doppler sideband cannot beat a scalloped true peak;
            // the served snr stays the chosen cell's raw peak/mean.
            //
            // Alignments are compared on the scalloping-corrected peak_interp, not the raw grid
            // sample: on a critically sampled grid the true alignment's peak can fall between
            // Doppler bins (down to 0.405 of its power) while an overlay-mismatch sideband lands
            // on a bin. The served det.snr stays the raw peak/mean, on which the downstream
            // thresholds are set.
            if (best_nh < 0 || ai.peak_interp > a.peak_interp) {
                a = ai;
                best_nh = nh;
            }
            _nh_prof.push_back({nh, ai.snr, ai.peak_tau_samples, ai.doppler_hz});
        }

        const bool detected = a.snr >= _acquire_snr;
        // NHPROF: one line per strong detection, each alignment as nh:snr@tau then d and its
        // Doppler (internal sign). When healthy, tau steps by about one code period per
        // alignment step and snr peaks at the winner; with overlay discrimination broken, snr
        // is flat across alignments at one tau.
        if (detected && a.snr >= 100.0 && !_nh_prof.empty()) {
            std::string prof_s;
            for (const auto& e : _nh_prof)
                prof_s +=
                    fmt::format(" {:d}:{:.0f}@{:d}d{:+.0f}", e.nh, e.snr, (long long)e.tau, e.dop);
            INFO("GnssChannelizedSearch[{:s}]: PRN {:d} NHPROF snr {:.0f} best_nh {:d} "
                 "|{:s}",
                 unique_name, _prns[p], a.snr, best_nh, prof_s);
        }
        // Label consensus. The scanned alignments are hypotheses about one signal ((nh + 1,
        // tau - one period) is the same physical phase), so the period label nh + lag should
        // agree across them. An alignment's argmax can land on its neighbour's
        // peak cell (same tau, wrong nh), and peak_interp can then make that cell the winner.
        // The vote moves only the label; tau and the refined fine phase stay the winner's. It
        // needs a strict majority and at least three voters, and a tie keeps the winner's label.
        if (detected && _n_nh > 1 && best_nh >= 0 && _nh_prof.size() >= 3) {
            const double per_samp =
                (_replica->code_length() / _replica->chip_rate_hz()) * _sample_rate;
            const int span = (int)std::llround((double)Mp * (double)_fft_len / per_samp);
            auto label_of = [&](int nh, long long tau) {
                int lag = (int)((double)tau / per_samp) % span;
                if (lag > span / 2)
                    lag -= span;
                return ((nh + lag) % _n_nh + _n_nh) % _n_nh;
            };
            std::map<int, int> votes;
            for (const auto& e : _nh_prof)
                ++votes[label_of(e.nh, (long long)e.tau)];
            const int mine = label_of(best_nh, (long long)a.peak_tau_samples);
            int maj = mine, maj_n = 0;
            for (const auto& kv : votes)
                if (kv.second > maj_n || (kv.second == maj_n && kv.first == mine)) {
                    maj = kv.first;
                    maj_n = kv.second;
                }
            if (maj != mine && maj_n > (int)_nh_prof.size() / 2) {
                const int shift = ((maj - mine) % _n_nh + _n_nh) % _n_nh;
                const int nh_corr = (best_nh + shift) % _n_nh;
                INFO("GnssChannelizedSearch[{:s}]: PRN {:d} LABEL CONSENSUS: winner nh {:d} "
                     "labels the period {:d} but {:d}/{:d} scanned alignments say {:d} -- "
                     "relabelled (nh {:d} -> {:d}, tau kept)",
                     unique_name, _prns[p], best_nh, mine, maj_n, (int)_nh_prof.size(), maj,
                     best_nh, nh_corr);
                best_nh = nh_corr;
            }
        }
        _nh_prof.clear();
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
            // Refine the code phase with refine_peak or refine_peak_cuda (gnssSeedTransport.hpp),
            // which scripts/gnss/e2e.cpp also drives.
            double best_cp;
#ifdef GNSS_CUDA
            if (!_cuda_ref_groups.empty()) {
                // Reads the snapshot in its [hop][n_chan] layout. Unlike refine_peak it takes no
                // nh_phase (see refine_peak_cuda).
                best_cp = gnss::refine_peak_cuda(_cuda_ref_groups, *_replica, p, _snapshot.data(),
                                                 _n_chan, anchor, a.code_phase_chips, dop,
                                                 _sample_rate, _refine_span, _refine_step);
            } else
#endif
            {
                // The covering channels' data columns do not depend on the trial phase, so they
                // are copied once.
                std::vector<std::vector<cf>> d(cov_local.size(), std::vector<cf>(hpr));
                for (size_t i = 0; i < cov_local.size(); ++i)
                    for (int m = 0; m < hpr; ++m)
                        d[i][m] = _snapshot[((size_t)m) * _n_chan + cov_local[i]];
                const int rh = (_refine_hops > 0 && _refine_hops < hpr) ? _refine_hops : hpr;
                best_cp = gnss::refine_peak(
                    *_replica, p, d, cov_global, a.code_phase_chips, (_n_nh > 1) ? best_nh : -1,
                    dop, anchor, rh, _sample_rate, _refine_span, _refine_step, _acquire_threads);
            }
            // Peak to reported phases (gnssSeedTransport.hpp, also driven by scripts/gnss/e2e.cpp).
            const gnss::DetectionPhase dp =
                gnss::detection_phase(*_replica, best_cp, best_nh, _n_nh, dop, _snap_start_hop,
                                      anchor, _sample_rate, a.peak_tau_samples);
            det.doppler_hz = dop;
            det.code_phase_chips = dp.cp0;
            det.ref_hop = _snap_start_hop;       // capture epoch (the broker's slope fit)
            det.nh = (_n_nh > 1) ? best_nh : -1; // measured overlay alignment
            det.code_phase_long_chips = dp.cp_long;
            det.code_phase_at_ref_chips = dp.cp_at_ref;
            det.valid = true;
            // Where the reported phases come from, one line per detection (unthrottled):
            //   hop    = snapshot start hop          coarse = acquire cp, before the refine
            //   refine = refine offset               nh     = measured overlay alignment
            //   cp0    = argument at sample 0, mod L
            //   ph@ref = physical phase at hop, mod n_nh * L
            INFO("GnssChannelizedSearch[{:s}]: PRN {:d} snr {:.1f} dop {:+.0f} | hop {:d} "
                 "coarse {:.2f} refine {:+.2f} nh {:d} -> cp0 {:.2f} cp_long {:.2f} "
                 "ph@ref {:.2f}",
                 unique_name, _prns[p], a.snr, dop, _snap_start_hop, a.code_phase_chips,
                 best_cp - a.code_phase_chips, det.nh, dp.cp0, dp.cp_long, dp.cp_at_ref);
        }
        // Latch: a fresh detection replaces the held one; a miss keeps the last valid one for
        // _hold_snapshots snapshots, so a momentary miss does not drop the seed.
        std::lock_guard<std::mutex> lk(_det_mtx);
        if (detected) {
            _detections[p] = det;
        } else {
            _detections[p].snr = det.snr;
            if (_detections[p].valid && ++_detections[p].misses > _hold_snapshots)
                _detections[p].valid = false;
        }
    }
    // Log the pass's best SNR whether or not it cleared acquire_snr, so a near-miss is visible.
    if (best_any_prn >= 0) {
        // Pure-noise expectation of the best SNR, so a threshold below it shows in the log. A
        // surface cell is a sum of k = nwin unit-mean |D|^2 draws, Gamma(k) up to channel-count
        // effects. The largest of N = (surface cells) x n_nh cells, as x = peak / mean, solves
        // N * P(cell > k*x) = 1 with the Gamma tail:
        //     k*x - (k-1)*ln(k*x) + ln((k-1)!) = ln(N).
        // The Gaussian shortcut 1 + c/sqrt(k) underestimates it badly at small k (~5.0 against
        // ~7.5 at k = 4). Correlation between cells makes this a mild overestimate.
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
        INFO("GnssChannelizedSearch[{:s}]: pass best snr {:.2f} (PRN {:d}{:s}), threshold {:.2f}, "
             "pure-noise ceiling ~{:.2f}{:s}",
             unique_name, best_any, best_any_prn,
             (_n_nh > 1) ? fmt::format(", nh {:d}/{:d}", best_any_nh, _n_nh) : std::string(),
             _acquire_snr, ceiling,
             (_acquire_snr < ceiling) ? "  [THRESHOLD BELOW NOISE CEILING -- every \"detection\" "
                                        "is meaningless]"
                                      : "");
    } else if (n_unhinted == n_prn && n_prn > 0) {
        // Every PRN was skipped for lack of a fresh hint, so the pass searched nothing. Say so,
        // rate-limited; otherwise this looks like an empty sky.
        if ((_empty_hint_passes++ % 256) == 0)
            WARN("GnssChannelizedSearch[{:s}]: 0/{:d} PRNs hinted -- the broker's hints are "
                 "absent or older than hint_ttl_s ({:.0f} s). The search is scanning NOTHING. "
                 "(repeat {:d})",
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
        // Apply staged PRN swaps between passes, when no per-pass state is in use.
        apply_prn_swaps();
        search_snapshot(); // _worker_busy keeps the consumer loop off the snapshot
        {
            std::lock_guard<std::mutex> lk(_m);
            _worker_busy = false;
        }
    }
}

int GnssChannelizedSearch::local_of_global(int c) const {
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
    // Slot membership, with the same payload as the tracker's set_prns.
    kotekan::restServer::instance().register_post_callback(
        unique_name + "/set_prns",
        std::bind(&GnssChannelizedSearch::set_prns_callback, this, _1, _2));

    _worker = std::thread(&GnssChannelizedSearch::search_worker, this);

    int frame_in = 0;
    bool filling = false;
    size_t filled_hops = 0; // hops actually copied into the current snapshot (for the drop log)

    while (!stop_thread) {
        auto* in_local = (cf*)in_buf->wait_for_full_frame(unique_name, frame_in);
        if (in_local == nullptr)
            break;
        const int frame_hops = in_buf->frame_size / (int)sizeof(cf) / _n_chan;

        // Absolute hop index of this frame's first hop, from the F-engine's sample_seq metadata
        // (shared across nodes).
        const GnssChanMetadata* mi = get_gnss_chan_metadata(in_buf, frame_in);
        if (mi == nullptr || mi->sample_seq < 0) {
            FATAL_ERROR("GnssChannelizedSearch[{:s}]: frame {:d} has no sample_seq, so it cannot "
                        "be placed in time",
                        unique_name, frame_in);
            break;
        }
        const long long frame_first_hop = mi->sample_seq / _fft_len;

        if (!filling) {
            std::lock_guard<std::mutex> lk(_m);
            if (!_worker_busy) {
                filling = true;
                filled_hops = 0;
                _snap_start_hop = frame_first_hop; // snapshot hop 0 is this frame's hop 0
                // Zero first so dropped frames leave signal-free holes. Each frame is placed at
                // its sample_seq offset below, so a drop does not shift the code phase of the
                // windows after it.
                std::fill(_snapshot.begin(), _snapshot.end(), cf(0.0f, 0.0f));
            }
        }
        if (filling) {
            // Place this frame at its absolute time position in the snapshot (drop-tolerant).
            const long long off = frame_first_hop - _snap_start_hop;
            if (off >= 0 && off < (long long)_snap_hops) {
                const size_t take = std::min((size_t)frame_hops, _snap_hops - (size_t)off);
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
                    // Freeze both hint tables at the data's epoch; see _snap_hints.
                    std::lock_guard<std::mutex> hk(_hint_mtx);
                    _snap_hints = _dop_hints;
                    _nh_snap_hints = _nh_hints;
                    _snap_taken_s = steady_s();
                }
                std::lock_guard<std::mutex> lk(_m);
                _snap_ready = true;
                _worker_busy = true;
                _cv.notify_one();
            }
        }

        // Always release the frame at once: never backpressure.
        in_buf->mark_frame_empty(unique_name, frame_in);
        frame_in = (frame_in + 1) % in_buf->num_frames;
    }

    {
        std::lock_guard<std::mutex> lk(_m);
        _worker_stop = true;
        _cv.notify_one();
    }
    if (_worker.joinable())
        _worker.join();
}

void GnssChannelizedSearch::set_prns_callback(kotekan::connectionInstance& conn,
                                              nlohmann::json& request) {
    std::vector<int> want;
    std::string err;
    try {
        const nlohmann::json& arr = request.is_object() ? request.at("prns") : request;
        if (!arr.is_array())
            throw std::runtime_error("expected an array of PRNs, or {\"prns\": [...]}");
        for (const auto& v : arr)
            want.push_back(v.get<int>());
    } catch (const std::exception& e) {
        err = std::string("bad prns payload: ") + e.what();
    }
    // Same length rule as the tracker: per-slot tables are sized once, so a resize needs a
    // restart.
    if (err.empty() && want.size() != _prns.size())
        err = "prns has " + std::to_string(want.size()) + " entries but this search has "
              + std::to_string(_prns.size()) + " slots; the count is not live";
    if (err.empty())
        for (size_t i = 0; i < want.size() && err.empty(); ++i)
            for (size_t j = 0; j < i; ++j)
                if (want[j] == want[i]) {
                    err = "PRN " + std::to_string(want[i]) + " appears twice";
                    break;
                }
    if (!err.empty()) {
        conn.send_error(err, kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(_prn_mtx);
        if (want != _prns) {
            _pending_prns = want;
            _prn_pending = true;
        }
    }
    conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
}

void GnssChannelizedSearch::apply_prn_swaps() {
    std::vector<std::pair<int, int>> changed; // (slot, old prn)
    {
        std::lock_guard<std::mutex> lk(_prn_mtx);
        if (!_prn_pending)
            return;
        _prn_pending = false;
        for (size_t p = 0; p < _prns.size(); ++p) {
            const int np = _pending_prns[p];
            if (np == _prns[p])
                continue;
            // The bank first: a PRN with no code for this signal must leave the slot exactly
            // as it was rather than half-swapped.
            if (!_replica->set_prn((int)p, np)) {
                WARN("GnssChannelizedSearch[{:s}]: slot {:d}: no code for PRN {:d} on this "
                     "signal; the slot keeps PRN {:d}",
                     unique_name, p, np, _prns[p]);
                continue;
            }
            changed.emplace_back((int)p, _prns[p]);
            _prns[p] = np;
        }
        _pending_prns.clear();
    }
    if (changed.empty())
        return;
    for (const auto& c : changed) {
        const size_t p = (size_t)c.first;
#ifdef GNSS_CUDA
        // Every refine engine holds its own device copy of the code table over the shared
        // bank (see GnssCudaDespread::set_prn).
        for (auto& e : _cuda_ref)
            if (e)
                e->set_prn((int)p, _prns[p], nullptr);
#endif
        // The head/tail bank the acquire uses (_repl_head, _repl_tail) is not rebuilt here: it
        // is rebuilt only when the covering set changes, so the acquire keeps the old PRN's
        // code for this slot.
        {
            std::lock_guard<std::mutex> lk(_det_mtx);
            _detections[p] = Detection{};
        }
        {
            // Hints are per satellite: carried across a swap they would narrow the new
            // satellite's scan to the old one's Doppler and alignment.
            std::lock_guard<std::mutex> lk(_hint_mtx);
            _dop_hints[p] = DopHint{};
            _nh_hints[p] = NhHint{};
            _snap_hints[p] = DopHint{};
            _nh_snap_hints[p] = NhHint{};
        }
        WARN("GnssChannelizedSearch[{:s}]: slot {:d} PRN {:d} -> {:d}; refine code tables, "
             "detection and hints all cleared for that slot.",
             unique_name, c.first, c.second, _prns[p]);
    }
}

void GnssChannelizedSearch::get_detections_callback(kotekan::connectionInstance& conn) {
    nlohmann::json reply = nlohmann::json::array();
    std::lock_guard<std::mutex> pk(_prn_mtx); // before _det_mtx; see set_doppler_hints_callback
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
    // Body: [{prn, nh, ref_hop}, ...]: the overlay alignment for that PRN at absolute hop
    // ref_hop (see NhHint). nh < 0 clears the hint (full scan). PRNs not listed keep their
    // hint; unknown PRNs are ignored.
    try {
        // _prn_mtx first: the worker rewrites _prns between passes, and resolving a PRN against
        // a half-written map would hint the wrong slot.
        std::lock_guard<std::mutex> pk(_prn_mtx);
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
    // Body: [{prn, doppler_hz, margin_hz}, ...]: narrow each listed PRN's search to
    // doppler_hz +- margin_hz (physical Doppler). margin_hz < 0 clears the hint. PRNs not
    // listed keep their hint; unknown PRNs are ignored.
    try {
        // _prn_mtx first: the worker rewrites _prns between passes, and resolving a PRN against
        // a half-written map would hint the wrong slot.
        std::lock_guard<std::mutex> pk(_prn_mtx);
        std::lock_guard<std::mutex> lk(_hint_mtx);
        for (const auto& h : json_request) {
            const int prn = h.at("prn").get<int>();
            for (size_t i = 0; i < _prns.size(); ++i) {
                if (_prns[i] != prn)
                    continue;
                const double margin = h.value("margin_hz", 0.0);
                if (margin < 0.0)
                    _dop_hints[i] = DopHint{}; // clear -> the PRN is skipped
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
