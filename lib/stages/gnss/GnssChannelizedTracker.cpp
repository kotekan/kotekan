#include "GnssChannelizedTracker.hpp"

#ifdef GNSS_CUDA
#include "GnssCudaDespread.hpp" // GPU batched despread (use_cuda: true)
#endif

#include "StageFactory.hpp"            // for REGISTER_KOTEKAN_STAGE
#include "GnssChanMetadata.hpp"        // for get_gnss_chan_metadata, metadata_is_gnss_chan
#include "gnssChannelizedDespread.hpp" // for channelized_despread
#include "gnssSignal.hpp"              // for SignalDescriptor, signal_by_name
#include "kotekanLogging.hpp"          // for FATAL_ERROR
#include "pfbPrototype.hpp"            // for window_from_string

#include <algorithm> // for fill
#include <array>     // for array (GPU despread results)
#include <chrono>    // for system_clock
#include <cmath>
#include <numeric>
#include <complex>   // for complex, arg, norm
#include <cstring>   // for memcpy
#include <set>       // for set

using kotekan::Config;
using kotekan::bufferContainer;
using kotekan::Stage;
using cf = std::complex<float>;

REGISTER_KOTEKAN_STAGE(GnssChannelizedTracker);

GnssChannelizedTracker::GnssChannelizedTracker(Config& config, const std::string& unique_name,
                                               bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssChannelizedTracker::main_thread, this)) {

    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    const std::string signal = config.get_default<std::string>(unique_name, "signal", "GPS_L1CA");
    const gnss::SignalDescriptor* sig = gnss::signal_by_name(signal);
    if (sig == nullptr) {
        FATAL_ERROR("GnssChannelizedTracker: unknown signal '{:s}'", signal);
        return;
    }

    _sample_rate = config.get_default<double>(unique_name, "sample_rate", 5e6);
    const double f_offset = config.get_default<double>(unique_name, "f_offset", 0.0);
    _capture_utc0 = config.get_default<double>(unique_name, "capture_utc0", 0.0);
    _doppler_margin_hz = config.get_default<double>(unique_name, "doppler_margin_hz", 5000.0);

    _N = config.get<int>(unique_name, "spectrum_length");
    _fft_len = 2 * _N;
    _chan_offset = config.get_default<int>(unique_name, "channel_offset", 0);
    _n_chan = config.get<int>(unique_name, "n_channels");
    if (_chan_offset < 0 || _n_chan <= 0 || _chan_offset + _n_chan > _N) {
        FATAL_ERROR("GnssChannelizedTracker: channel slice [{:d},{:d}) invalid for N={:d}",
                    _chan_offset, _chan_offset + _n_chan, _N);
        return;
    }
    const int num_taps = config.get_default<int>(unique_name, "num_taps", 4);
    const std::string win = config.get_default<std::string>(unique_name, "pfb_window", "hamming");

    _prns = config.get<std::vector<int>>(unique_name, "prns");
    if (_prns.empty()) {
        FATAL_ERROR("GnssChannelizedTracker: 'prns' is empty");
        return;
    }
    const int n_prn = (int)_prns.size();
    // The record WIDTH is a C++ constant (gnssRecord.hpp) but the frame is sized in yaml as
    // n_prn * record_floats * sizeof_float, and nothing links the two. A config left behind on a
    // schema bump under-sizes the frame and this stage writes past its end -- silent memory
    // corruption. Die at construction instead.
    const size_t rec_bytes_need = (size_t)n_prn * RECORD_FLOATS * sizeof(float);
    if ((size_t)out_buf->frame_size < rec_bytes_need) {
        FATAL_ERROR("GnssChannelizedTracker: out_buf frame {:d} B < {:d} PRN x {:d} floats "
                    "({:d} B). Bump 'record_floats' in the config to {:d}.",
                    (size_t)out_buf->frame_size, n_prn, RECORD_FLOATS, rec_bytes_need,
                    RECORD_FLOATS);
        return;
    }

    auto seed = [&](const std::string& key, std::vector<double>& out) {
        const auto v = config.get_default<std::vector<double>>(unique_name, key, {});
        out.assign(n_prn, 0.0);
        for (int i = 0; i < n_prn; ++i)
            out[i] = v.empty() ? 0.0 : (v.size() == 1 ? v[0] : v.at(i));
    };
    seed("doppler_hz", _doppler);
    seed("code_phase_chips", _code_phase);
    _code_phase_rate.assign(n_prn, 0.0);
    _doppler_rate.assign(n_prn, 0.0);
    _carrier_trim.assign(n_prn, 0.0);
    _ref_hop.assign(n_prn, 0);

    try {
        _replica = std::make_unique<gnss::ChannelizedReplicaBank>(
            *sig, _sample_rate, f_offset, _N, num_taps, dsp::window_from_string(win), _prns);
    } catch (const std::exception& e) {
        FATAL_ERROR("GnssChannelizedTracker: {:s}", e.what());
        return;
    }
    _hops_per_record =
        config.get_default<int>(unique_name, "hops_per_record", _replica->repl_period_hops());
    _replica->code_doppler_sign = config.get_default<double>(unique_name, "code_doppler_sign", 1.0);
    // EFFECTIVE (post-comb) chips -> component chips; see cudaGnssTrack for the BOC false-lock
    // analysis (power disc at +-0.5 component chips on BOC(1,1) is sign-inverted -> -12 dB).
    _dll_spacing = config.get_default<double>(unique_name, "dll_spacing_chips", 0.5)
                   / (double)_replica->comb_mult();
    _fll_gain = config.get_default<double>(unique_name, "fll_gain", 0.0);
    // f_ref fence: the replica carrier is FIXED between re-anchors, so a stale f_ref
    // decoheres the despread WITHIN one record. The tolerable staleness therefore scales with
    // the RECORD PERIOD, not with a constant: 0.1 cycle per record -> 100 Hz at GPS's 1 ms,
    // 25 Hz at E1C's 4 ms, 10 Hz at B1C's 10 ms. The old fixed 200 Hz default was 0.2 cycle
    // for GPS (survivable) but TWO WHOLE CYCLES at a 10 ms B1C record -- total loss. The
    // tri-constellation config already carries the right per-chain values (25/10); deriving
    // the DEFAULT means the next band gets it right without anyone remembering to. Same
    // reasoning, same 0.1-cycle rule, as the broker's hold_max_dop_hz.
    const double t_rec_s = (double)_hops_per_record * (double)_fft_len / _sample_rate;
    _fll_reacq_hz = config.get_default<double>(unique_name, "fll_reacq_hz",
                                               (t_rec_s > 0.0) ? 0.1 / t_rec_s : 200.0);
    // Carrier-anchor age cap: see cudaGnssTrack (quadratic ff error ~0.5*rate_dot*age^2).
    _max_anchor_age_s = config.get_default<double>(unique_name, "max_anchor_age_s", 30.0);
    _fll_max_gap = config.get_default<double>(unique_name, "fll_max_gap_s", 0.005);
    // Freeze the FLL discriminator when the despread |A| drops below this -- so a signal dropout
    // (radar sweep, the broker coasting the seed) doesn't drive f_track on noise; the NCO keeps
    // coasting at the held frequency and the loop resumes cleanly on return. 0 = never freeze.
    _fll_lock_amp = config.get_default<double>(unique_name, "fll_lock_amp", 0.0);
    _carrier_shared = config.get_default<bool>(unique_name, "carrier_shared", false);
    // use_cuda: route the E/P/L despread through the GPU kernel (identical semantics to the
    // hoprate replica path; validated 3.6e-9 vs CPU). Requires a USE_CUDA build.
    _use_cuda = config.get_default<bool>(unique_name, "use_cuda", false);
    if (_use_cuda) {
#ifdef GNSS_CUDA
        // Contiguous subband here -- stated explicitly, since the despread now takes an
        // arbitrary channel list (GnssCudaDespread.hpp).
        std::vector<int> chan_ids((size_t)_n_chan);
        std::iota(chan_ids.begin(), chan_ids.end(), _chan_offset);
        _cuda = std::make_shared<GnssCudaDespread>(*_replica, (int)_prns.size(), chan_ids,
                                                   _hops_per_record, _sample_rate,
                                                   _replica->f_offset());
        INFO("GNSS despread on GPU (use_cuda): {:d} PRN slots x E/P/L x {:d} channels",
             (int)_prns.size(), _n_chan);
#else
        FATAL_ERROR("use_cuda: true but kotekan was built without USE_CUDA");
#endif
    }

    const auto active_prns = config.get_default<std::vector<int>>(unique_name, "active_prns", {});
    _active.assign(n_prn, 1);
    if (!active_prns.empty()) {
        const std::set<int> on(active_prns.begin(), active_prns.end());
        for (int i = 0; i < n_prn; ++i)
            _active[i] = on.count(_prns[i]) ? 1 : 0;
    }
}

void GnssChannelizedTracker::set_seeds_callback(kotekan::connectionInstance& conn,
                                                nlohmann::json& request) {
    try {
        std::lock_guard<std::mutex> lk(_seed_mtx);
        std::fill(_active.begin(), _active.end(), (uint8_t)0); // list is authoritative
        for (const auto& s : request) {
            const int prn = s.at("prn").get<int>();
            for (size_t i = 0; i < _prns.size(); ++i)
                if (_prns[i] == prn) {
                    _doppler[i] = s.at("doppler_hz").get<double>();
                    _code_phase[i] = s.at("code_phase_chips").get<double>();
                    // Optional first-order code-phase model: cp0 drifts linearly with the
                    // capture hop (residual code-rate error), so the broker can hand a
                    // slope + anchor hop and the tracker extrapolates to its own window
                    // hop -- removing the seed-staleness bias rather than searching it out.
                    _code_phase_rate[i] = s.value("code_phase_rate", 0.0);
                    // Optional 2nd-order carrier model: the almanac Doppler rate (Hz/s), used to
                    // ramp the replica frequency so the coherent-integration residual stays flat.
                    _doppler_rate[i] = s.value("doppler_rate_hz_s", 0.0);
                    // Shared carrier loop: broker-commanded NCO frequency (Hz). Phase-continuous
                    // by construction -- the NCO integrates it, so a trim change alters the slope
                    // of phi, never jumps it.
                    _carrier_trim[i] = s.value("carrier_trim_hz", 0.0);
                    _ref_hop[i] = s.value("ref_hop", (long long)0);
                    _active[i] = 1;
                }
        }
    } catch (const std::exception& e) {
        conn.send_error(e.what(), kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
}

void GnssChannelizedTracker::main_thread() {
    using namespace std::placeholders;
    kotekan::restServer::instance().register_post_callback(
        unique_name + "/set_seeds",
        std::bind(&GnssChannelizedTracker::set_seeds_callback, this, _1, _2));

    int frame_in = 0;
    int frame_out = 0;
    const int n_prn = (int)_prns.size();

    // Carrier FLL state (per PRN). The replica stays at a FIXED reference Doppler f_ref
    // (NaN = not yet acquired) -- it is anchored to the absolute sample index, so retuning
    // it mid-stream would jump the phase by 2*pi*df*t_abs (huge). Instead an NCO accumulates
    // a phase correction phi_track (= integral of 2*pi*f_track) that derotates the despread
    // output, and the discriminator drives the residual frequency f_track. a_prev holds the
    // previous (derotated) amplitude for the record-to-record discriminator.
    std::vector<double> f_ref(n_prn, std::nan(""));
    std::vector<double> f_track(n_prn, 0.0);   // FLL residual-frequency estimate (Hz)
    std::vector<double> phi_track(n_prn, 0.0); // NCO accumulated carrier phase (rad)
    // Carrier-phase continuity offset (cycles): absorbs the replica-phase step that an f_ref
    // re-pin puts into the ABSOLUTELY-anchored commanded phase, so the exported carrier-phase
    // observable runs continuously through re-pins instead of breaking its arc at every one.
    std::vector<double> phi_cyc(n_prn, 0.0);  // NCO phase UNWRAPPED (cycles): the export's base.
                                              // phi_track is the same phase WRAPPED for the
                                              // rotation, and its wraps would otherwise inject a
                                              // spurious +-1 cycle into every phase increment.
    std::vector<double> phi_cmd_prev(n_prn, 0.0);   // previous record's commanded phase (cycles)
    std::vector<uint8_t> phi_cmd_ok(n_prn, 0);      // ... and whether there is one
    std::vector<cf> a_prev(n_prn, cf(0.0f, 0.0f));
    std::vector<uint8_t> a_prev_ok(n_prn, 0);
    std::vector<long long> hop_prev(n_prn, 0);
    std::vector<long long> reacq_hop(n_prn, 0); // hop where f_ref was last (re)fixed: anchor for the Doppler-rate feed-forward

    // Rolling measurement window of [hop][subband-channel] voltages.
    std::vector<cf> window((size_t)_hops_per_record * _n_chan);
    int hops_filled = 0;
    long long seen_hops = 0;        // fallback absolute hop counter
    long long window_start_hop = 0; // absolute hop index of the current window's first hop

    while (!stop_thread) {
        auto* in_local = (cf*)in_buf->wait_for_full_frame(unique_name, frame_in);
        if (in_local == nullptr)
            break;
        const int frame_hops = in_buf->frame_size / (int)sizeof(cf) / _n_chan;

        // Absolute hop index of this frame's first hop, from sample_seq (shared across
        // nodes so every tracker's window_start agrees); fall back to a local count.
        long long frame_first_hop = seen_hops;
        if (metadata_is_gnss_chan(in_buf)) {
            auto* mi = get_gnss_chan_metadata(in_buf, frame_in);
            if (mi && mi->sample_seq >= 0)
                frame_first_hop = mi->sample_seq / _fft_len;
        }

        for (int in_pos = 0; in_pos < frame_hops; ++in_pos) {
            if (hops_filled == 0)
                window_start_hop = frame_first_hop + in_pos;
            std::memcpy(&window[(size_t)hops_filled * _n_chan], &in_local[(size_t)in_pos * _n_chan],
                        sizeof(cf) * _n_chan);
            if (++hops_filled < _hops_per_record)
                continue;
            hops_filled = 0;

            const long long window_start = window_start_hop * (long long)_fft_len;
#ifdef GNSS_CUDA
            if (_use_cuda) // one upload per record; all PRNs' despreads read it
                ((GnssCudaDespread*)_cuda.get())->upload_window(window.data(), window_start);
#endif

            // Snapshot the (REST-updatable) seeds + active set for this window.
            std::vector<double> dop, cp, cp_rate, dop_rate, ctrim;
            std::vector<long long> ref_hop;
            std::vector<uint8_t> active;
            {
                std::lock_guard<std::mutex> lk(_seed_mtx);
                dop = _doppler;
                cp = _code_phase;
                cp_rate = _code_phase_rate;
                dop_rate = _doppler_rate;
                ctrim = _carrier_trim;
                ref_hop = _ref_hop;
                active = _active;
            }

            float* out = (float*)out_buf->wait_for_empty_frame(unique_name, frame_out);
            if (out == nullptr)
                return;
            const double utc = (_capture_utc0 > 0.0)
                                   ? _capture_utc0 + (double)window_start / _sample_rate
                                   : std::chrono::duration<double>(
                                         std::chrono::system_clock::now().time_since_epoch())
                                         .count();

            // ---- Pass 1: per-PRN CONTROL (seeds -> commanded cp/carrier + covering set),
            // stashed per PRN so the despread can run as ONE batched GPU launch across all
            // PRNs (G1c). The CPU path consumes the same stash inline in pass 2 -- the
            // control logic is single-sourced for both paths.
            struct PrnCmd {
                bool run = false;
                double cp_seed = 0.0, fcar_eff = 0.0, ff_hz = 0.0;
                std::vector<int> owned; // GLOBAL covering channel ids in this subband
            };
            std::vector<PrnCmd> cmds(n_prn);
            const bool nco_mode = (_fll_gain > 0.0) || _carrier_shared;
            for (int p = 0; p < n_prn; ++p) {
                float* rec = out + (size_t)p * RECORD_FLOATS;
                for (int f = 0; f < RECORD_FLOATS; ++f)
                    rec[f] = 0.0f;
                rec[0] = (float)_prns[p];
                *reinterpret_cast<double*>(rec + RECORD_UTC_SLOT) = utc;
                if (!active[p]) {
                    f_ref[p] = std::nan(""); // forget the loop; re-acquire when it returns
                    a_prev_ok[p] = 0;
                    phi_cyc[p] = 0.0;
                    phi_cmd_ok[p] = 0;
                    continue;
                }

                // Effective carrier Doppler for the replica = the broker seed. Covering
                // channels follow it.
                const double dop_eff = dop[p];
                const auto cover = _replica->covering_bins(dop_eff, _doppler_margin_hz);
                for (int c : cover)
                    if (c >= _chan_offset && c < _chan_offset + _n_chan)
                        cmds[p].owned.push_back(c);
                // Code phase at this window: first-order extrapolation of the seeded cp0 from
                // its anchor hop (removes seed staleness), PLUS the known QUADRATIC code-
                // Doppler term. The broker's hold-on-lock freezes the seed's LINEAR model, but
                // the sky's code rate changes at dop_rate*f_chip/f_c (~3e-4 chips/s^2 at L1),
                // so a frozen linear model walks off by 0.5*(dop_rate*f_chip/f_c)*dt^2 -- 0.5
                // chips at 60 s, 2.3 at 120 s: the ~2 min C/N0 sawtooth (roll-off then
                // re-anchor reset) observed live 2026-07-11. dop_rate is the almanac rate
                // (physical-signed, refreshed every seed) -- a deterministic feed-forward, so
                // it respects the piecewise-constant currency rule; the code twin of the NCO's
                // ff_hz carrier ramp. Fresh anchors (dt~0) make it a no-op outside holds.
                // The broker's DLL trims the residual at ~Hz from the shipped E/L correlators
                // -- the tracker itself never moves it.
                const double L = (double)_replica->code_length();
                const double dt_anchor =
                    (double)(window_start_hop - ref_hop[p]) * (double)_fft_len / _sample_rate;
                double cp_seed = cp[p] + cp_rate[p] * (double)(window_start_hop - ref_hop[p])
                                 + 0.5 * _replica->code_doppler_sign * dop_rate[p]
                                       * (_replica->eff_chip_rate()
                                          / (double)_replica->comb_mult())
                                       / _replica->carrier_hz() * dt_anchor * dt_anchor;
                cp_seed = std::fmod(cp_seed, L);
                if (cp_seed < 0.0)
                    cp_seed += L;

                // FLL reference Doppler: (re)acquire from the broker seed when uninitialised
                // or far adrift (loss of lock / sat change), else hold it FIXED so the
                // absolute-anchored replica phase stays continuous; the NCO tracks the rest.
                // The adrift test uses the TOTAL tracked Doppler (f_ref + f_track) vs the seed,
                // NOT f_ref alone: f_ref is pinned and the seed is stable, so f_ref-only was
                // blind to f_track diverging underneath it -- and the bit-robust FLL
                // discriminator aliases past ~1/(4*dt) (~250 Hz at 1 ms records), so once noise
                // (a drooping off-peak |A|) kicked f_track past that it ran away unbounded
                // (observed -1158 Hz, deep dead). Resetting at fll_reacq_hz < the alias limit
                // catches it before it aliases (the 2026-07-07 L1 deep-decay root cause).
                const double f_tracked = f_ref[p] + ((_fll_gain > 0.0) ? f_track[p] : 0.0);
                const double anchor_age =
                    (double)(window_start_hop - reacq_hop[p]) * (double)_fft_len / _sample_rate;
                const bool fresh_or_fence = std::isnan(f_ref[p])
                                            || std::fabs(f_tracked - dop[p]) > _fll_reacq_hz;
                if (nco_mode && (fresh_or_fence || anchor_age > _max_anchor_age_s)) {
                    // PHASE-CONTINUOUS RE-PIN (the carrier-phase twin of the frequency fold
                    // below). Re-pinning f_ref moves the ABSOLUTELY-ANCHORED replica, so the
                    // commanded phase f_ref*t_abs steps by df*t_abs -- thousands of cycles at
                    // soak age, which arg(A) can only report mod 1 cycle.
                    //
                    // Until 2026-07-14 that step was folded into an EXPORT-ONLY offset
                    // (phi_fix) and the NCO was then ZEROED. That kept the ADR arc alive but
                    // left the step IN THE DATA: the rotation applied to A jumped by the step,
                    // so every re-pin punched an effectively random phase into the correlation
                    // the combiner deep-integrates. That was the residual C/N0 sawtooth, and it
                    // showed exactly where this says it must -- in the COHERENT estimator at
                    // precisely max_anchor_age_s (30.0 s on every strong satellite, 0.2 dB on
                    // GPS rising to 1.6 dB on B1C) and NOT in the phase-blind incoherent one.
                    //
                    // The step is FOLDABLE: an NCO absorbs an arbitrary constant phase. Folding
                    // it into the NCO instead of the export leaves f_ref*t_abs - phi_cyc exactly
                    // as continuous as before (the fold is invariant in that combination, so the
                    // validated ADR arcs are untouched) AND makes the despread output continuous
                    // through the re-pin. phi_fix is gone: it is identically zero now.
                    //
                    // The prize is that a re-pin becomes CHEAP. With the phase step folded and
                    // the code-currency step already translated below, the replica can be
                    // re-pinned as often as we like -- max_anchor_age_s: 0 re-pins EVERY record,
                    // so f_ref never goes stale and the OTHER half of the sawtooth (within-record
                    // decoherence ~(dop_rate*age*t_rec)^2, negligible on GPS's 1 ms record and
                    // ~1 dB on B1C's 10 ms) never accumulates at all.
                    const double t_abs_pin =
                        (double)window_start_hop * (double)_fft_len / _sample_rate;
                    const bool fresh = std::isnan(f_ref[p]);
                    const double f_ref_before = f_ref[p];
                    if (fresh_or_fence) {
                        f_ref[p] = dop[p]; // genuine (re)acquisition / the seed left the fence
                        f_track[p] = 0.0;  // ... so the frequency residual is re-acquired too
                    } else {
                        // AGE re-pin, FREQUENCY-CONTINUOUS (2026-07-13): the old := dop reset
                        // discarded the accumulated FF ramp (rate*age, ~12 Hz per 30 s at
                        // MEO rates) and the broker trim re-absorbed it as a ~5 s transient
                        // every cap period -- the GPS wave that excised 53% of emits and, on
                        // BOC's sharp peak, the E/C dip episodes. Folding the ramp INTO f_ref
                        // restarts it at zero with the total command unchanged: no frequency
                        // step at all (the code-currency translation below absorbs the cp side
                        // of the f_ref change automatically). Nothing physical moved, so the
                        // FLL residual and the |A| history are KEPT.
                        f_ref[p] += dop_rate[p] * anchor_age; // PHYSICAL frame: no NCO-side negation
                    }
                    if (fresh) {
                        phi_track[p] = 0.0; // nothing to preserve; the arc starts here
                        phi_cyc[p] = 0.0;
                        a_prev_ok[p] = 0;
                        phi_cmd_ok[p] = 0;
                    } else {
                        // fold the replica-phase step into the NCO, where the DATA sees it
                        const double dcyc = (f_ref[p] - f_ref_before) * t_abs_pin;
                        phi_cyc[p] += dcyc;
                        phi_track[p] =
                            std::remainder(phi_track[p] + 2.0 * M_PI * dcyc, 2.0 * M_PI);
                        if (fresh_or_fence)
                            a_prev_ok[p] = 0; // f_track was just zeroed: the FLL restarts
                    }
                    reacq_hop[p] = window_start_hop; // anchor the Doppler-rate feed-forward here
                }
                const double fcar = nco_mode ? f_ref[p] : dop[p];
                // 2nd-order carrier feed-forward -- applied in the NCO, NEVER as a replica retune.
                // The replica's carrier AND code are anchored to ABSOLUTE sample 0 (phase = w*n,
                // code = cp0 + n*chip_per_sample(f)), so retuning its frequency by df mid-stream
                // rotates the whole phase history by 2*pi*df*t_abs and walks the code off-peak by
                // chip_rate*t_abs*df/f_c (~0.65 chips per Hz per 1000 s of capture) -- a RAMPED
                // replica frequency therefore destroys the correlation as the anchor ages
                // (measured: |A| dies in ~10 s under an injected 5 Hz/s ramp; this was the root
                // cause of the L1 deep decay, 2026-07-10). The NCO is the time-local corrector:
                // integrating the ramp there (phi += 2*pi*f_nco*dt) yields exactly the intended
                // quadratic phase with no absolute-anchor coupling. The replica frequency stays
                // FIXED at f_ref between re-anchors; the fence above re-anchors it (one phase
                // glitch) only when the true carrier walks far enough (~200 Hz) to decohere the
                // intra-record despread itself. Non-NCO mode (plain correlator) needs no ramp at
                // all: its dop refreshes from the broker every window (staleness < 0.2 s).
                // Sign: the seed's doppler_rate_hz_s arrives PHYSICAL-signed (same convention as
                // doppler_hz), but the despread residual / NCO / trim run in the r2c-flipped
                // INTERNAL convention (residual = -(physical); same flip as the search's
                // det.doppler_hz = -grid). Negate on entry -- measured both ways on the 07-04
                // replay: the internal +0.34 Hz/s flattens the trim that the physical -0.31
                // doubles, and live all trims crept to the rail before this negation.
                const double ff_hz = nco_mode ? -dop_rate[p]
                                                    * (double)(window_start_hop - reacq_hop[p])
                                                    * (double)_fft_len / _sample_rate
                                              : 0.0;
                // CODE-CURRENCY TRANSLATION: the broker expresses cp0 for a replica whose
                // code rate follows the CURRENT seed Doppler (the search's convention: its
                // sample-0 drift back-reference uses the measured/seeded Doppler), but in NCO
                // mode the replica actually runs at the PINNED f_ref. Same cp0 number +
                // different code rate FROM ABSOLUTE SAMPLE 0 = a different code phase at this
                // window: off by t_abs*f_chip*(dop-f_ref)/f_c (~0.65 chips/Hz per 1000 s at
                // L1) -- it grows as the seed Doppler walks away from f_ref, and JUMPS by
                // tens-to-hundreds of chips at soak age when the fence re-pins f_ref (the
                // zombie-lock bomb: the code teleports off-peak, E/P/L see noise, the DLL
                // disc reads 'centered' on noise, and nothing ever recovers -- measured
                // 2026-07-11: tracker commanding 180 chips vs search truth 403 at t~1000 s).
                // Translating cp0 into the replica's own currency makes the code phase AT THE
                // WINDOW invariant, and exactly continuous across fences, where (dop - fcar)
                // -> 0 by construction. Non-NCO mode: fcar == dop, exact no-op.
                cp_seed += (double)window_start_hop * (double)_fft_len / _sample_rate
                           * (_replica->eff_chip_rate() / (double)_replica->comb_mult())
                           * _replica->code_doppler_sign * (dop[p] - fcar)
                           / _replica->carrier_hz();
                cp_seed = std::fmod(cp_seed, L);
                if (cp_seed < 0.0)
                    cp_seed += L;
                cmds[p].cp_seed = cp_seed;
                cmds[p].fcar_eff = fcar; // replica: FIXED between re-anchors (see above)
                cmds[p].ff_hz = ff_hz;
                rec[1] = (float)(fcar - ff_hz // report the ramp physical-signed
                                 + ((_fll_gain > 0.0) ? f_track[p] : 0.0) + ctrim[p]);
                // EXPORT CURRENCY (2026-07-31; see cudaGnssTrack for the full story): the
                // command above is in the f_ref currency, but downstream reconstructions
                // propagate rec[2] at the REPORTED carrier rec[1]. Re-express the export into
                // rec[1]'s currency (same algebra as the translation above, target rec[1] =
                // fcar - ff + f_track? + ctrim) so the reconstruction is exact by
                // construction; the despread command (cmds[p]) keeps f_ref.
                double cp_export =
                    cp_seed
                    + (double)window_start_hop * (double)_fft_len / _sample_rate
                          * (_replica->eff_chip_rate() / (double)_replica->comb_mult())
                          * _replica->code_doppler_sign
                          * (ff_hz - ((_fll_gain > 0.0) ? f_track[p] : 0.0) - ctrim[p])
                          / _replica->carrier_hz();
                cp_export = std::fmod(cp_export, L);
                if (cp_export < 0.0)
                    cp_export += L;
                rec[2] = (float)cp_export;
                rec[6] = (float)cmds[p].owned.size();
                cmds[p].run = !cmds[p].owned.empty(); // empty = carrier not in this subband
            }

#ifdef GNSS_CUDA
            // ---- Batched GPU despread: every runnable PRN's E/P/L triple in ONE launch (each
            // job carries its own Doppler-bucket Phi tables), one stream sync per record.
            std::vector<std::array<gnss::DespreadResult, 3>> gpu_res;
            std::vector<int> gpu_slot(n_prn, -1);
            if (_use_cuda) {
                std::vector<GnssCudaDespread::Spec> specs;
                for (int p = 0; p < n_prn; ++p) {
                    if (!cmds[p].run)
                        continue;
                    std::vector<int> local; // wrapper masks by LOCAL channel index
                    local.reserve(cmds[p].owned.size());
                    for (int c : cmds[p].owned)
                        local.push_back(c - _chan_offset);
                    gpu_slot[p] = (int)specs.size();
                    specs.push_back(GnssCudaDespread::Spec{p, cmds[p].cp_seed, _dll_spacing,
                                                           cmds[p].fcar_eff, std::move(local)});
                }
                gpu_res = ((GnssCudaDespread*)_cuda.get())->despread_batch(specs);
            }
#endif

            // ---- Pass 2: despread results -> record slots + the carrier NCO/FLL.
            for (int p = 0; p < n_prn; ++p) {
                if (!cmds[p].run)
                    continue;
                float* rec = out + (size_t)p * RECORD_FLOATS;
                const double cp_seed = cmds[p].cp_seed;
                const double fcar_eff = cmds[p].fcar_eff;
                const double ff_hz = cmds[p].ff_hz;
                const std::vector<int>& owned = cmds[p].owned;

                // Early/Prompt/Late despread -- three DUMB correlators at the COMMANDED
                // code phase (R1, docs/gnss_architecture_audit.md). The Prompt despreads
                // exactly at cp_seed: it is the science output and is phase-continuous by
                // construction, because the tracker makes NO alignment decisions (the
                // retired per-record max-power pull-in re-phased the despread and killed
                // deep coherence; the retired hold_lock false-armed on noise statistics).
                // Early/Late sit at -+_dll_spacing chips on the correlation triangle and
                // exist ONLY so the low-rate DLL downstream can measure the sub-chip
                // tracking error: the combiner aggregates |E|^2 / |L|^2 across subbands and
                // windows, and the BROKER closes the code loop at ~Hz through set_seeds.
                // Only E/L powers are consumed, so they ship un-derotated.
                gnss::DespreadResult early, best, late;
#ifdef GNSS_CUDA
                if (_use_cuda) {
                    const auto& r3 = gpu_res[(size_t)gpu_slot[p]];
                    early = r3[0];
                    best = r3[1]; // prompt
                    late = r3[2];
                } else
#endif
                {
                    // The window data is fixed; only the replica varies with the cp trial.
                    std::vector<std::vector<cf>> data_ch;
                    data_ch.reserve(owned.size());
                    for (int c : owned) {
                        std::vector<cf> col(_hops_per_record);
                        const int local = c - _chan_offset;
                        for (int m = 0; m < _hops_per_record; ++m)
                            col[m] = window[(size_t)m * _n_chan + local];
                        data_ch.push_back(std::move(col));
                    }
                    auto despread_at = [&](double cpx) {
                        // hoprate (per-chip closed form) not channels() (per-sample fold+FFT):
                        // numerically equal to ~machine precision (the GPU gate validates the
                        // fused port of THIS path at 5e-7) and ~10x cheaper -- the difference
                        // between a saturated tracker shedding frames and ~20% of a core at
                        // the 20 MSPS wide front end (1000-hop L1 records, 4000-hop E1C).
                        auto repl_ch = _replica->channels_hoprate(p, window_start, cpx,
                                                                  fcar_eff, _hops_per_record,
                                                                  owned);
                        // rows come back in `owned` order, parallel to data_ch above.
                        return gnss::channelized_despread(data_ch, repl_ch);
                    };
                    early = despread_at(cp_seed - _dll_spacing);
                    best = despread_at(cp_seed); // prompt
                    late = despread_at(cp_seed + _dll_spacing);
                }
                rec[gnss::REC_E_RE] = (float)early.correlation.real();
                rec[gnss::REC_E_IM] = (float)early.correlation.imag();
                rec[gnss::REC_E_ENERGY] = (float)early.replica_energy;
                rec[gnss::REC_L_RE] = (float)late.correlation.real();
                rec[gnss::REC_L_IM] = (float)late.correlation.imag();
                rec[gnss::REC_L_ENERGY] = (float)late.replica_energy;

                if (!nco_mode) {
                    rec[3] = (float)best.correlation.real();
                    rec[4] = (float)best.correlation.imag();
                    rec[5] = (float)best.replica_energy;
                } else {
                    // Carrier NCO + FLL. The replica is at the fixed f_ref, so the despread
                    // carries the residual carrier (true - f_ref). Advance the phase NCO by
                    // the tracked residual over the inter-record gap and derotate the output
                    // -- this corrects frequency WITHOUT retuning the absolute-anchored
                    // replica. The bit-robust discriminator (square the phase product to
                    // cancel the +-1 nav-bit pi flip, valid while |err*dt|<1/4) then drives
                    // f_track toward the residual; skip it across a frame gap (it would alias).
                    const double dt = (double)(window_start_hop - hop_prev[p]) * (double)_fft_len
                                      / _sample_rate;
                    // NCO frequency: local FLL estimate (if enabled) + the broker-commanded
                    // shared-loop trim + the almanac Doppler-rate feed-forward (ff_hz -- see the
                    // fcar comment above for why the ramp lives HERE and not in the replica).
                    // All change the slope of phi, never jump it.
                    const double f_nco = ((_fll_gain > 0.0) ? f_track[p] : 0.0) + ctrim[p] + ff_hz;
                    if (a_prev_ok[p] && dt > 0.0) {
                        phi_track[p] += 2.0 * M_PI * f_nco * dt; // NCO (advances over gaps too)
                        phi_track[p] = std::remainder(phi_track[p], 2.0 * M_PI); // keep bounded
                        phi_cyc[p] += f_nco * dt; // ... and UNWRAPPED, in cycles, for the export
                    }
                    const std::complex<double> rot = std::polar(1.0, -phi_track[p]);
                    const std::complex<double> g_corr = std::complex<double>(best.correlation) * rot;
                    const cf a_corr = (cf)(std::complex<double>(best.amplitude) * rot);
                    rec[3] = (float)g_corr.real();
                    rec[4] = (float)g_corr.imag();
                    rec[5] = (float)best.replica_energy;

                    if (a_prev_ok[p] && dt > 0.0 && dt <= _fll_max_gap
                        && std::abs(a_corr) > (float)_fll_lock_amp
                        && std::abs(a_prev[p]) > (float)_fll_lock_amp) {
                        const std::complex<double> prod =
                            std::complex<double>(a_corr) * std::conj(std::complex<double>(a_prev[p]));
                        const double dphi = std::arg(prod * prod) / 2.0; // bit-robust phase walk
                        f_track[p] += _fll_gain * dphi / (2.0 * M_PI * dt); // += gain * (residual Hz)
                    }
                    a_prev[p] = a_corr;
                    a_prev_ok[p] = 1;
                    hop_prev[p] = window_start_hop;
                }
                // Head segment (gnssRecord.hpp slots 16-18): this CPU tracker does not
                // segment the despread at the code-period boundary yet -- write PH == P,
                // the compatibility default (tail = 0), so every segmented consumer
                // reduces exactly to the whole-record behaviour. The GPU chain
                // (cudaGnssTrack) segments for real; port the split here when a CPU band
                // needs a secondary overlay (L5 NH) -- see the bistable fix, 2026-07-15.
                rec[gnss::REC_PH_RE] = rec[3];
                rec[gnss::REC_PH_IM] = rec[4];
                rec[gnss::REC_PH_ENERGY] = rec[5];

                // COMMANDED CARRIER PHASE (cycles mod 1) -- the carrier-phase observable's
                // other half (gnssRecord.hpp): the replica's absolutely-anchored
                // f_ref*t_abs plus the NCO's phi. The combiner adds the measured arg(A)
                // to get the received carrier phase, so the f_ref re-pin's phase step
                // (df*t_abs -- huge at soak age, and invisible in the reported Doppler,
                // which is continuous by design) cancels exactly instead of slipping the
                // arc. Non-NCO mode: phi == 0 and fcar == the per-record Doppler, same
                // identity. mod 1 keeps float32 at ~1e-7 cycle resolution.
                // SIGN: f_ref is PHYSICAL-signed (it comes from the seed Doppler), but the NCO
                // phase -- like the trim and the despread residual it corrects -- lives in the
                // r2c-flipped INTERNAL convention (the same flip as the search's
                // det.doppler_hz = -grid). So phi enters the physical commanded phase NEGATED.
                // Measured on sky, 2026-07-13: with phi added, every satellite's ADR rate was
                // off by 2*(its mean NCO frequency) -- invisible on sats whose trim hovers near
                // zero, but PRN10, whose trim sat pinned at the +100 Hz clamp, read +212 Hz off
                // its ephemeris-predicted Doppler. One satellite at the rail exposed the sign
                // that six near-zero ones had hidden.
                const double t_abs_w =
                    (double)window_start_hop * (double)_fft_len / _sample_rate;
                const double phi_cmd_cyc =
                    fcar_eff * t_abs_w
                    - (nco_mode ? phi_cyc[p] : 0.0); // UNWRAPPED NCO phase (see phi_cyc).
                    // No phi_fix term: the re-pin step is folded into phi_cyc itself, which
                    // leaves f_ref*t_abs - phi_cyc continuous on its own.
                // Ship the INCREMENT, not the phase: small, bounded, exactly representable in
                // a float, and nothing downstream has to unwrap anything (gnssRecord.hpp).
                rec[gnss::REC_CPHASE] =
                    phi_cmd_ok[p] ? (float)(phi_cmd_cyc - phi_cmd_prev[p]) : 0.0f;
                phi_cmd_prev[p] = phi_cmd_cyc;
                phi_cmd_ok[p] = 1;
            }

            // Stamp the window's absolute sample on the record frame: it survives
            // bufferSend/Recv (records ship back to the combiner across nodes) and lets
            // the combiner verify per-subband window alignment before summing.
            if (out_buf->metadata_pool) {
                out_buf->allocate_new_metadata_object(frame_out);
                get_gnss_chan_metadata(out_buf, frame_out)->sample_seq = window_start;
            }
            out_buf->mark_frame_full(unique_name, frame_out);
            frame_out = (frame_out + 1) % out_buf->num_frames;
        }

        seen_hops += frame_hops;
        in_buf->mark_frame_empty(unique_name, frame_in);
        frame_in = (frame_in + 1) % in_buf->num_frames;
    }
}
