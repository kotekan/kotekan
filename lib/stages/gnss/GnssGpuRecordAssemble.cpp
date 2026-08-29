#include "GnssGpuRecordAssemble.hpp"

#include "GnssChanMetadata.hpp"
#include "StageFactory.hpp"
#include "visUtil.hpp" // for frameID
#include "gnssGpuChain.hpp"
#include "gnssRecord.hpp"
#include "json.hpp" // for the /get_spectrum reply

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssGpuRecordAssemble);

GnssGpuRecordAssemble::GnssGpuRecordAssemble(Config& config, const std::string& unique_name,
                                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssGpuRecordAssemble::main_thread, this)) {
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);
    _prns = config.get<std::vector<int>>(unique_name, "prns");
    _sample_rate = config.get_default<double>(unique_name, "sample_rate", 5e6);
    // Per-channel prompt-phase dump (see the hpp note; diagnostic, default off).
    _chan_dump_prn = config.get_default<int>(unique_name, "chan_dump_prn", -1);
    _chan_dump_decim = std::max(1, config.get_default<int>(unique_name, "chan_dump_decim", 10));
    if (_chan_dump_prn >= 0) {
        const std::string path = config.get_default<std::string>(
            unique_name, "chan_dump_path", "/tmp/gnss_chan_phase_dump.txt");
        _chan_dump = std::fopen(path.c_str(), "a");
    }
    // ELEMENT AXIS (CHORD). 0 = the single-antenna airspy layout, byte-for-byte -- the default,
    // so nothing existing changes. >0 appends n_elements per-antenna blocks to every PRN record
    // (gnssRecord.hpp). The GPU correlation array grows the same axis: [rows][n_chan][n_elem].
    _n_elements = config.get_default<int>(unique_name, "n_elements", 0);
    // Which element the record HEADER's correlation slots carry. The broker closes the DLL and
    // carrier loops off those slots, so this must be a phase-coherent single-antenna view (see
    // gnssRecord.hpp) -- pick a healthy, high-gain feed. Ignored when n_elements == 0.
    _reference_element = config.get_default<int>(unique_name, "reference_element", 0);
    if (_n_elements > 0 && (_reference_element < 0 || _reference_element >= _n_elements)) {
        FATAL_ERROR("GnssGpuRecordAssemble: reference_element {:d} outside [0, n_elements={:d})",
                    _reference_element, _n_elements);
        return;
    }
    // SELF-CALIBRATED ELEMENT SUM (hpp note / gnssElemCal.hpp). Off by default; no-op unless
    // the element axis exists.
    _elem_sum = config.get_default<bool>(unique_name, "elem_sum", false) && _n_elements > 0;
    _elem_sum_tau_s = config.get_default<double>(unique_name, "elem_sum_tau_s", 5.0);
    _elem_hold_on_reanchor =
        config.get_default<bool>(unique_name, "elem_sum_hold_on_reanchor", true);
    _elem_sum_min_w = config.get_default<double>(unique_name, "elem_sum_min_w", 0.02);
    // ── #102 ELEMENT STEERING: geometric per-(sat, channel, element) phasors ─────────
    // OFF unless elem_positions_enu is provided (flat [n_elements][3], metres ENU of any
    // fixed array point -- re-referenced to the reference element below so the header's
    // slots and the ADR observable are UNTOUCHED by steering: the ref phasor is unity by
    // construction). The SIGN is a measured config parameter (gnssElemSteer.hpp note).
    {
        auto pos = config.get_default<std::vector<double>>(unique_name, "elem_positions_enu", {});
        const double st_sign = config.get_default<double>(unique_name, "elem_steer_sign", 1.0);
        const double st_hold = config.get_default<double>(unique_name, "elem_steer_hold_s", 120.0);
        if (!pos.empty() && _n_elements > 0) {
            if ((int)pos.size() != 3 * _n_elements)
                FATAL_ERROR("GnssGpuRecordAssemble: elem_positions_enu has {:d} values, "
                            "want 3*n_elements = {:d}",
                            (int)pos.size(), 3 * _n_elements);
            // channel_ids is read into _spec_freq_ids LATER in this constructor (the
            // chan-export block) -- read it locally here; init order bit the fleet once
            // (crash-loop, 2026-08-29 22:5x).
            auto _st_fids =
                config.get_default<std::vector<int>>(unique_name, "channel_ids", {});
            if (_st_fids.empty())
                FATAL_ERROR("GnssGpuRecordAssemble: elem steering needs channel_ids (the "
                            "per-channel RF frequencies)");
            for (int i = 0; i < 3; ++i) {
                const double r0 = pos[(size_t)_reference_element * 3 + i];
                for (int e = 0; e < _n_elements; ++e)
                    pos[(size_t)e * 3 + i] -= r0;
            }
            std::vector<double> f_mhz;
            for (int id : _st_fids)
                f_mhz.push_back(id * 0.1953125);
            _steer = gnss::ElemSteer(std::move(pos), std::move(f_mhz), (int)_prns.size(),
                                     st_sign, st_hold);
            INFO("GnssGpuRecordAssemble[{:s}]: #102 element steering READY ({:d} elements, "
                 "{:d} channels, sign {:+.0f}) -- awaiting /set_sat_geometry posts",
                 unique_name, _n_elements, (int)_st_fids.size(), st_sign);
        }
    }
    const int n = (int)_prns.size();
    // Record width is a C++ constant; the frame is sized in yaml (n_prn * record_floats *
    // sizeof_float) with nothing linking the two. A stale config under-sizes the frame and this
    // stage writes past its end -- die at construction rather than corrupt memory.
    const int rec_stride = gnss::record_stride(_n_elements);
    const size_t rec_bytes_need = (size_t)n * rec_stride * sizeof(float);
    if ((size_t)out_buf->frame_size < rec_bytes_need) {
        FATAL_ERROR("GnssGpuRecordAssemble: out_buf frame {:d} B < {:d} PRN x {:d} floats "
                    "({:d} B). Bump 'record_floats' in the config to {:d}.",
                    (size_t)out_buf->frame_size, n, rec_stride, rec_bytes_need, rec_stride);
        return;
    }
    if (_elem_sum) {
        _cal.assign((size_t)n, gnss::ElemCal(_n_elements, _reference_element, _elem_sum_tau_s,
                                             _elem_sum_min_w));
        _anchor_warned.assign((size_t)n, 0);
    }
    _phi.assign(n, 0.0);
    _phi_cyc.assign(n, 0.0);
    _phi_cmd_prev.assign(n, 0.0);
    _phi_cmd_ok.assign(n, 0);
    _fcar_prev.assign(n, 0.0);
    _fnco_prev.assign(n, 0.0);
    _fcar_prev_ok.assign(n, 0);
    _a_prev.assign(n, {0.0, 0.0});
    _a_prev_ok.assign(n, 0);
    _elem_prev_ok.assign(n, 0);
    _wstart_prev.assign(n, 0);
    // PER-CHANNEL PROMPT SPECTRUM (hpp note; task #32). Keyed on `channel_ids` being present:
    // the generator wires the same per-GPU freq_id list the despread runs, so the export knows
    // which sky frequency each channel index is. No channel_ids (airspy, legacy configs) ->
    // fully inert: no accumulation, and the endpoint answers with an explanatory error rather
    // than not existing, so a mis-wired poller sees WHY instead of a bare 404.
    _spec_freq_ids = config.get_default<std::vector<int>>(unique_name, "channel_ids", {});

    // PER-CHANNEL COMB EXPORT (gnssRecord.hpp, 2026-08-14). ⚠️ THE CROSS-CHANNEL SUM BELOW IS A
    // LOSS WE SHOULD NEVER HAVE TAKEN -- it destroys the frequency axis a delay lives on, so
    // the broker can only FIT a per-instance constant where it should DERIVE one from the ramp.
    // KV: "purge the idea of summing across channels in each instance, that's *never* what we
    // want to do." This appends the UNSUMMED comb after the PRN records; the summed slots stay
    // for now so nothing downstream has to change on the same day.
    //
    // channel_ids is REQUIRED: the broker must know which SKY FREQUENCY each column is. A comb
    // whose columns are unlabelled is worse than no comb -- a delay fit across mislabelled
    // frequencies returns a confident wrong tau.
    _chan_export = config.get_default<bool>(unique_name, "chan_export", false);
    if (_chan_export && _spec_freq_ids.empty()) {
        FATAL_ERROR("GnssGpuRecordAssemble[{:s}]: chan_export needs channel_ids -- an unlabelled "
                    "frequency axis makes every delay fit downstream confidently wrong",
                    unique_name);
        return;
    }
    if (_chan_export) {
        const size_t need =
            gnss::frame_floats(n, _n_elements, (int)_spec_freq_ids.size()) * sizeof(float);
        if ((size_t)out_buf->frame_size < need) {
            FATAL_ERROR("GnssGpuRecordAssemble[{:s}]: chan_export needs a {:d} B frame ({:d} PRN "
                        "x {:d} record floats, then {:d} PRN x {:d} chan x {:d}); out_buf is "
                        "{:d} B. The generator sizes this -- regenerate the config.",
                        unique_name, need, n, rec_stride, n, (int)_spec_freq_ids.size(),
                        gnss::CHAN_FLOATS, (size_t)out_buf->frame_size);
            return;
        }
        INFO("GnssGpuRecordAssemble[{:s}]: exporting the UNSUMMED comb -- {:d} channels x {:d} "
             "floats per PRN per record, appended after the records ({:d} B frame)",
             unique_name, (int)_spec_freq_ids.size(), gnss::CHAN_FLOATS, need);
    }

    if (!_spec_freq_ids.empty()) {
        const size_t nc = _spec_freq_ids.size();
        // WINDOW QUANTISATION (task #53). The window index is floor(wstart / win_samples) on
        // the F-engine sample clock, so every instance assigns a record to the same window
        // WITHOUT talking to any other instance. Configure in SAMPLES, not seconds: the
        // division must be exact integer arithmetic or two instances can straddle a boundary
        // differently, which is the whole failure this replaces. Every instance must be given
        // the SAME value -- the generator writes it, and a mismatch shows up immediately as
        // disjoint window indices in the broker's first poll.
        //
        // Whole records land in exactly one window regardless of whether win_samples is a
        // record multiple (a record is assigned by its wstart, never split), so alignment does
        // not depend on that -- but making it a multiple keeps the per-window record count
        // constant, which is one less thing to explain in a residual.
        _spec_win_samples =
            (int64_t)config.get_default<double>(unique_name, "spectrum_window_samples", 0.0);
        const int depth = config.get_default<int>(unique_name, "spectrum_ring_depth", 8);
        if (_spec_win_samples > 0) {
            _spec_ring.resize((size_t)std::max(2, depth));
            for (auto& w : _spec_ring) {
                w.re.assign((size_t)n * nc, 0.0);
                w.im.assign((size_t)n * nc, 0.0);
                w.energy.assign((size_t)n * nc, 0.0);
                w.nrec.assign(n, 0);
                w.phi0.assign(n, 0.0);
                w.nreanchor.assign(n, 0);
            }
            INFO("per-channel spectrum export ON: {:d} channels, freq_id {:d}..{:d}; "
                 "ADDRESSABLE windows of {:d} samples ({:.3f} s), ring depth {:d}",
                 (int)nc, _spec_freq_ids.front(), _spec_freq_ids.back(),
                 (long long)_spec_win_samples, (double)_spec_win_samples / _sample_rate,
                 (int)_spec_ring.size());
        } else {
            // LEGACY: one open accumulator, reset on read. Kept so a node whose config predates
            // spectrum_window_samples keeps working until it is regenerated -- but it CANNOT be
            // aligned across instances, so the broker treats a legacy reply as unaddressable.
            _spec_ring.resize(1);
            _spec_ring[0].re.assign((size_t)n * nc, 0.0);
            _spec_ring[0].im.assign((size_t)n * nc, 0.0);
            _spec_ring[0].energy.assign((size_t)n * nc, 0.0);
            _spec_ring[0].nrec.assign(n, 0);
            _spec_ring[0].phi0.assign(n, 0.0);
            _spec_ring[0].nreanchor.assign(n, 0);
            WARN("per-channel spectrum export ON: {:d} channels -- but LEGACY reset-on-read "
                 "windows (no 'spectrum_window_samples' in this config). Instances CANNOT be "
                 "aligned; regenerate this node's config (task #53).", (int)nc);
        }
        _spec_scratch.assign((size_t)std::max(1, _n_elements), {0.0, 0.0});
    }
    using namespace std::placeholders;
    kotekan::restServer::instance().register_get_callback(
        unique_name + "/get_spectrum",
        std::bind(&GnssGpuRecordAssemble::spectrum_callback, this, _1));
    if (_elem_sum)
        kotekan::restServer::instance().register_post_callback(
            unique_name + "/set_elem_gain",
            std::bind(&GnssGpuRecordAssemble::set_elem_gain_callback, this, _1, _2));
    if (_steer.enabled())
        kotekan::restServer::instance().register_post_callback(
            unique_name + "/set_sat_geometry",
            std::bind(&GnssGpuRecordAssemble::set_sat_geometry_callback, this, _1, _2));
    // LIVE REFERENCE SWAP (KV, 2026-08-20). Registered whenever the element axis exists --
    // the header's correlation slots carry the reference even with elem_sum off, so the
    // swap is meaningful either way. See set_reference_element_callback.
    if (_n_elements > 0)
        kotekan::restServer::instance().register_post_callback(
            unique_name + "/set_reference_element",
            std::bind(&GnssGpuRecordAssemble::set_reference_element_callback, this, _1, _2));
}

GnssGpuRecordAssemble::~GnssGpuRecordAssemble() {
    if (_chan_dump)
        std::fclose(_chan_dump);
}

void GnssGpuRecordAssemble::set_sat_geometry_callback(kotekan::connectionInstance& conn,
                                                      nlohmann::json& request) {
    // #102: {"<prn>": [az_deg, el_deg], ...}. Slow-moving (~0.5 deg/min): a 30 s post
    // cadence keeps the steering within ~1 mm of true. Unknown PRNs are skipped (the
    // broker posts its whole sky; this stage steers the slots it owns).
    int n_up = 0;
    try {
        const double now_s = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now().time_since_epoch())
                                 .count();
        std::lock_guard<std::mutex> lk(_steer_mtx);
        for (auto it = request.begin(); it != request.end(); ++it) {
            const int prn = std::atoi(it.key().c_str());
            if (prn <= 0 || !it.value().is_array() || it.value().size() < 2)
                continue;
            for (size_t p = 0; p < _prns.size(); ++p)
                if (_prns[p] == prn) {
                    _steer.update((int)p, it.value()[0].get<double>(),
                                  it.value()[1].get<double>(), now_s);
                    ++n_up;
                    break;
                }
        }
    } catch (const std::exception& e) {
        conn.send_error(e.what(), kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    conn.send_json_reply(nlohmann::json{{"updated", n_up}});
}

int GnssGpuRecordAssemble::follow_frame_prns(const void* pctl_v, int n_prn) {
    using namespace gnss_gpu;
    const PrnCtl* pctl = (const PrnCtl*)pctl_v;
    int n_changed = 0;
    for (int p = 0; p < n_prn && p < (int)_prns.size(); ++p) {
        // Record 0 speaks for the frame: the producer applies swaps at a FRAME boundary, so
        // every record in this frame carries the same slot->PRN map. A producer that predates
        // the field leaves prn = 0, and 0 is not a PRN -- read it as "no claim" and keep the
        // config's value, which is exactly the old behaviour.
        const int claimed = (int)pctl[(size_t)p].prn;
        if (claimed <= 0 || claimed == _prns[(size_t)p])
            continue;
        const int was = _prns[(size_t)p];
        _prns[(size_t)p] = claimed;
        ++n_changed;

        // COLD RESET, everything keyed to this slot, in ONE place. Each of these describes the
        // satellite that just left; carried across, each becomes the old satellite's state
        // attributed to the new one -- the accumulator-identity trap, and every one of them
        // would be silent (a warm cal is a plausible cal, a phase history is a plausible
        // history). The reference-element swap above resets the same set for the same reason.
        if (_elem_sum && (size_t)p < _cal.size()) {
            _cal[(size_t)p] = gnss::ElemCal(_n_elements, _reference_element, _elem_sum_tau_s,
                                            _elem_sum_min_w);
            if ((size_t)p < _anchor_warned.size())
                _anchor_warned[(size_t)p] = 0;
        }
        _phi[(size_t)p] = 0.0;
        _phi_cyc[(size_t)p] = 0.0;
        _phi_cmd_prev[(size_t)p] = 0.0;
        _phi_cmd_ok[(size_t)p] = 0;
        _fcar_prev[(size_t)p] = 0.0;
        _fnco_prev[(size_t)p] = 0.0;
        _fcar_prev_ok[(size_t)p] = 0;
        _a_prev[(size_t)p] = {0.0, 0.0};
        _a_prev_ok[(size_t)p] = 0;
        _elem_prev_ok[(size_t)p] = 0;
        _wstart_prev[(size_t)p] = 0;
        // The spectrum ring accumulates PER SLOT across a window. A window straddling the swap
        // would otherwise sum two satellites into one row and report the total as one PRN's
        // spectrum -- so drop this slot's partial accumulation in every open window. nrec = 0
        // makes the row read as absent rather than as a quiet satellite.
        {
            std::lock_guard<std::mutex> lk(_spec_mtx);
            for (auto& W : _spec_ring) {
                if (W.idx < 0 || (size_t)p >= W.nrec.size())
                    continue;
                const size_t nc = W.re.size() / std::max<size_t>(1, _prns.size());
                for (size_t c = 0; c < nc; ++c) {
                    W.re[(size_t)p * nc + c] = 0.0;
                    W.im[(size_t)p * nc + c] = 0.0;
                    W.energy[(size_t)p * nc + c] = 0.0;
                }
                W.nrec[(size_t)p] = 0;
                if ((size_t)p < W.phi0.size())
                    W.phi0[(size_t)p] = 0.0;
                if ((size_t)p < W.nreanchor.size())
                    W.nreanchor[(size_t)p] = 0;
            }
        }
        WARN("GnssGpuRecordAssemble: slot {:d} PRN {:d} -> {:d} (the producer swapped it). "
             "Every per-slot accumulator for this slot reset COLD -- element cal, NCO phase, "
             "arc continuity and any open spectrum window. Downstream sees a fresh "
             "acquisition on this slot, which is what it is.",
             p, was, claimed);
    }
    return n_changed;
}

void GnssGpuRecordAssemble::main_thread() {
    using namespace gnss_gpu;
    frameID frame_in(in_buf), frame_out(out_buf);
    const int n_prn = (int)_prns.size();

    while (!stop_thread) {
        const uint8_t* in = in_buf->wait_for_full_frame(unique_name, frame_in);
        if (in == nullptr)
            return;
        FrameHdr hdr;
        std::memcpy(&hdr, in, sizeof(hdr));
        if (hdr.n_prn != n_prn) {
            FATAL_ERROR("GnssGpuRecordAssemble: frame n_prn {:d} != config {:d}", hdr.n_prn,
                        n_prn);
            return;
        }
        // LIVE REFERENCE SWAP, applied at the frame boundary so every record in a frame
        // sees ONE reference. Rebuilds the cal set COLD: every learned per-element gain
        // (and any stored path-B prior) is phase-anchored to the OLD reference element and
        // does not transfer -- carrying them across would hand the combine a wrong phase
        // convention with full confidence. The header rides the new bare reference while
        // the cal re-warms (~3 tau); downstream sees one phase step, exactly as it does on
        // any fresh acquisition, and the WARN below is the operator's receipt.
        {
            int newref = -1;
            {
                std::lock_guard<std::mutex> lk(_gain_mtx);
                if (_pending_ref >= 0) {
                    newref = _pending_ref;
                    _pending_ref = -1;
                }
            }
            if (newref >= 0 && newref != _reference_element) {
                const int oldref = _reference_element;
                _reference_element = newref;
                if (_elem_sum) {
                    _cal.assign(_prns.size(),
                                gnss::ElemCal(_n_elements, _reference_element, _elem_sum_tau_s,
                                              _elem_sum_min_w));
                    std::fill(_anchor_warned.begin(), _anchor_warned.end(), 0);
                    std::fill(_elem_prev_ok.begin(), _elem_prev_ok.end(), 0);
                }
                WARN("set_reference_element: header reference {:d} -> {:d} at the frame "
                     "boundary. Element cal rebuilt COLD ({} PRNs, re-warm ~{:.1f} s); "
                     "downstream sees one phase step on every PRN (fresh-acquisition class).",
                     oldref, newref, _cal.size(), 3.0 * _elem_sum_tau_s);
            }
        }
        // PATH B: a newly-injected per-element gain prior seeds every PRN's cal here, once,
        // before this frame's records are combined. seed() also stores it, so each subsequent
        // reset() (re-anchor) re-applies it rather than going cold -- the whole point.
        if (_elem_sum) {
            std::vector<std::complex<double>> pend;
            {
                std::lock_guard<std::mutex> lk(_gain_mtx);
                if (_pending_gain_set) {
                    pend.swap(_pending_gain);
                    _pending_gain_set = false;
                }
            }
            if (!pend.empty())
                WARN("set_elem_gain: consuming {:d} gains (n_elem={:d}, {:d} cals)",
                     (int)pend.size(), _n_elements, (int)_cal.size());
            if ((int)pend.size() == _n_elements) {
                for (auto& ec : _cal)
                    ec.seed(pend.data(), _n_elements);
                int nwarm = 0;
                for (auto& ec : _cal)
                    if (ec.warm())
                        ++nwarm;
                WARN("set_elem_gain: seeded, {:d}/{:d} cals warm right after seed",
                     nwarm, (int)_cal.size());
            }
        }
        const int n_chan = hdr.n_chan;
        // Output rows per spec: 4 normally, 6 when the writer peels (rows 4/5 = the peel
        // residual). 0 means a writer that predates the field -- read it as 4.
        const int n_rows_spec = (hdr.n_rows_spec > 0) ? hdr.n_rows_spec : ROWS_PLAIN;
        const int64_t* winstart = (const int64_t*)(in + off_winstart());
        const PrnCtl* pctl = (const PrnCtl*)(in + off_prnctl());
        // LIVE SLOT MEMBERSHIP: adopt the producer's map before anything in this frame is
        // labelled or accumulated. Costs one integer compare per slot per frame in steady
        // state.
        follow_frame_prns(pctl, n_prn);
        const double* corr = (const double*)(in + off_corr(n_prn));      // double2 rows
        // The energy block sits AFTER the corr block, whose size scales with the ELEMENT axis
        // -- the writer passes its n_elem (cudaGnssChordTrack: 32), so the reader must too, or
        // every "energy" lands inside the corr block: garbage when enough PRNs run to fill
        // that far (which is why sparse days flickered and broker-full days "worked"), zeros
        // when few do -- and a zero P_ENERGY makes the combiner treat the record as inactive,
        // so 2-PRN seeding produced structurally-perfect numerically-zero records
        // (found 2026-07-31, first trim-port test). Airspy configs set n_elements 0 -> 1,
        // matching cudaGnssTrack's element-free layout.
        const double* energy = (const double*)(in
                                               + off_energy(n_prn, n_chan, n_rows_spec,
                                                            (_n_elements > 0) ? _n_elements : 1));

        // NO ABSOLUTE ANCHOR? SAY SO, AND STILL EMIT A UNIFORM GRID. The old fallback stamped
        // system_clock::now() PER RECORD, and that silently broke every cross-record estimator
        // downstream: this stage emits hdr.n_rec records back to back, so on CHORD the four
        // sub-records of a frame were stamped microseconds apart instead of 10.49 ms apart.
        // GnssCoherentCombiner's rate search takes dt = the MINIMUM consecutive spacing and
        // builds an integer grid from it, so those microseconds became the grid and the records
        // were scattered across the transform. Cost, measured on cx19 2026-08-07: path B's
        // deep_rate_q read 4.5-11.4 against path A's 27-38 on IDENTICAL per-record data, failing
        // the q >= 10 gate, and coh_frac sat at 0.02-0.10 against 0.93-0.97 -- diagnosed for a
        // day and a half as a sensitivity problem because amp_snr, which uses no time base at
        // all, was a healthy 71-83% of path A throughout.
        //
        // Latching the anchor once and extrapolating by wstart keeps the same host-clock origin
        // (no worse than before for anything reading absolute time) and makes the grid exactly
        // uniform (no jitter at all for anything reading differences). The warning still fires,
        // because a host clock is not a GPS clock and the config should carry frame0_utc.
        if (hdr.utc0 <= 0.0 && hdr.n_rec > 0) {
            if (_wall_anchor == 0.0) {
                const double now = std::chrono::duration<double>(
                                       std::chrono::system_clock::now().time_since_epoch())
                                       .count();
                _wall_anchor = now - (double)winstart[0] / _sample_rate;
            }
            if ((_no_utc0_frames++ % 2000) == 0)
                WARN("GnssGpuRecordAssemble: producer sent no frame0_utc -- record UTC is the "
                     "HOST clock anchored once at startup, not GPS time ({:d} frames). Absolute "
                     "time is only as good as this host's clock; add frame0_utc to the producer "
                     "stage's config. ({:d} records/frame, so a per-record stamp would not even "
                     "be uniform.)",
                     _no_utc0_frames, hdr.n_rec);
        }

        for (int r = 0; r < hdr.n_rec && !stop_thread; ++r) {
            float* out = (float*)out_buf->wait_for_empty_frame(unique_name, frame_out);
            // ZERO THE WHOLE COMB BLOCK FIRST. The per-PRN record zeroing below covers only
            // record_stride floats; a PRN that does not run this window, or a channel outside
            // its covering mask, would otherwise hand the broker the PREVIOUS record's value as
            // if it were current -- the same "stale is not neutral" trap GnssChanAlignMerge
            // zeroes absent feeds for.
            if (_chan_export && out != nullptr)
                std::fill(out + gnss::chan_block_offset(n_prn, _n_elements),
                          out + gnss::frame_floats(n_prn, _n_elements, n_chan), 0.0f);
            if (out == nullptr)
                return;
            const int64_t wstart = winstart[r];
            const double utc = ((hdr.utc0 > 0.0) ? hdr.utc0 : _wall_anchor)
                               + (double)wstart / _sample_rate;

            for (int p = 0; p < n_prn; ++p) {
                const int rec_stride = gnss::record_stride(_n_elements);
                float* rec = out + (size_t)p * rec_stride;
                // Zero the WHOLE record, element blocks included: a PRN that does not run this
                // window must not leave a previous window's per-antenna correlations behind.
                for (int f = 0; f < rec_stride; ++f)
                    rec[f] = 0.0f;
                const PrnCtl& c = pctl[(size_t)r * n_prn + p];
                // THE FRAME'S PRN WINS. _prns was reconciled to it at the top of this frame,
                // so the two agree -- but reading it from the record's own control word means
                // record slot 0 cannot be wrong even if a future producer moves a slot
                // mid-frame. 0 = a producer that does not stamp it; fall back to the list.
                rec[0] = (float)((c.prn > 0) ? (int)c.prn : _prns[p]);
                *reinterpret_cast<double*>(rec + gnss::RECORD_UTC_SLOT) = utc;
                if (!c.run) {
                    _a_prev_ok[p] = 0;
                    _elem_prev_ok[p] = 0;
                    _fcar_prev_ok[p] = 0;
                    _phi_cyc[p] = 0.0;
                    _phi_cmd_ok[p] = 0;
                    continue;
                }
                // Cross-channel sum over the covering mask, per correlator trial
                // (E, P, L, P_HEAD -- the prompt's head segment, gnssRecord.hpp slots 16-18),
                // plus, when the chain peels, the residual prompt and its head (rows 4/5 ->
                // gnssRecord.hpp slots 20-23). PrnCtl::job0 already carries the per-spec row
                // stride, so indexing is job0 + t either way.
                //
                // NB rows 0-3 hold the FULL, un-peeled correlation even when peeling: the
                // analytic add-back was applied on the device (docs/gnss_voltage_peel_live.md).
                // That is what leaves everything below this stage -- combiner, broker, viewer,
                // TEC -- unable to tell whether a peel happened.
                // With an element axis the correlation array is [rows][n_chan][n_elem] and the
                // covering-mask sum runs PER ANTENNA. The ENERGIES do not: one replica is
                // correlated against every antenna, so energy stays [rows][n_chan] and is summed
                // once (gnssRecord.hpp -- this is why energies live in the record header and only
                // the correlations go in the element blocks).
                //
                // n_e == 1 with n_elements == 0 makes the index expression below collapse to
                // exactly the single-antenna one, so that path is unchanged.
                const int n_e = (_n_elements > 0) ? _n_elements : 1;
                const int ref_e = (_n_elements > 0) ? _reference_element : 0;
                std::complex<double> g3[6];  // reference element, for the header + NCO/gain state
                double e3[6];                // element-independent
                _g_elem.assign((size_t)n_rows_spec * n_e, std::complex<double>(0.0, 0.0));
                // #102: steer this satellite's elements if geometry is fresh. The lock is
                // cheap here (one take per record row-set); the REST writer holds it only
                // while rebuilding one slot's table.
                const gnss::ElemSteer::cf* steer_rows[64] = {nullptr};
                {
                    std::lock_guard<std::mutex> lk(_steer_mtx);
                    const double now_s =
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();
                    if (_steer.warm((int)p, now_s))
                        for (int ch = 0; ch < n_chan && ch < 64; ++ch)
                            steer_rows[ch] = _steer.row((int)p, ch);
                }
                for (int t = 0; t < n_rows_spec; ++t) {
                    const size_t row = (size_t)(c.job0 + t) * n_chan;
                    double e = 0.0;
                    for (int ch = 0; ch < n_chan; ++ch)
                        if ((c.chan_mask >> ch) & 1ULL) {
                            e += energy[row + ch];
                            const size_t base = (row + ch) * n_e;
                            const gnss::ElemSteer::cf* st = steer_rows[ch];
                            for (int el = 0; el < n_e; ++el) {
                                std::complex<double> v(corr[2 * (base + el)],
                                                       corr[2 * (base + el) + 1]);
                                if (st)
                                    v *= std::complex<double>(st[el].real(), st[el].imag());
                                _g_elem[(size_t)t * n_e + el] += v;
                            }
                        }
                    g3[t] = _g_elem[(size_t)t * n_e + ref_e];
                    e3[t] = e;
                }
                // SELF-CALIBRATED ELEMENT SUM (hpp note). Once this PRN's cal is warm, every
                // header row (E/P/L/PH/RES -- same weights: same antennas) becomes the
                // calibrated weighted mean instead of the bare reference element: reference-
                // anchored phase, "one element" scale, noise down ~sqrt(N_healthy). The cal
                // updates AFTER the combine (causal: this record's weights come from previous
                // records only), from the PROMPT row against the header actually emitted --
                // the bootstrap that starts on the reference element and converges to MRC.
                // The element BLOCKS are untouched: they are the per-antenna measurement.
                std::complex<double> g_sky(0.0, 0.0); // LOO sky-phase-corrected prompt (slot 24/25)
                if (_elem_sum) {
                    gnss::ElemCal& ec = _cal[p];
                    if (c.reanchored == 1 && !_elem_hold_on_reanchor)
                        ec.reset(); // fresh acquisition: full reset (legacy behavior)
                    // HOLD across a carrier re-anchor by default: the per-element cal is
                    // E_e*conj(E_ref), which the COMMON carrier cancels out of -- so a carrier
                    // re-pin (reanchored==1) does not change the element gains, and resetting
                    // them there is exactly what kept ElemCal from ever warming under the
                    // near-every-record re-anchoring (measured 0%% warm). The L5 array is phase-
                    // coherent, so the held gains stay valid; update() refreshes any drift.
                    // NB: do NOT reset _anchor_warned on re-anchor -- the trackers can re-anchor
                    // every record, and re-arming the WARN there ballooned the node log to tens
                    // of GB. The "reference too weak" message is worth exactly once per PRN.
                    if (ec.warm()) {
                        for (int t = 0; t < n_rows_spec; ++t)
                            g3[t] = ec.combine(&_g_elem[(size_t)t * n_e]);
                        // The deep fold's input: same prompt, each element derotated by the phase
                        // of the OTHERS. Separate slot -- the header keeps the raw phase because
                        // that IS the ADR observable (gnssRecord.hpp REC_SKY_RE).
                        g_sky = ec.combine_split(&_g_elem[(size_t)1 * n_e]);
                    }
                    if (_elem_prev_ok[p] && wstart > _wstart_prev[p])
                        ec.update(&_g_elem[(size_t)1 * n_e],
                                  (double)(wstart - _wstart_prev[p]) / _sample_rate);
                    if (ec.anchor_moved()) {
                        if (!_anchor_warned[p]) {
                            WARN("elem_sum PRN {:d}: reference element {:d} too weak -- phase "
                                 "anchor moved to the strongest element (one-time phase step "
                                 "downstream)",
                                 _prns[p], _reference_element);
                            _anchor_warned[p] = 1;
                        }
                    }
                }
                // Per-channel PROMPT dump (diagnostic, see hpp): raw pre-rotation per-channel
                // correlations -- the cross-channel relative phases are the observable.
                if (_chan_dump && _prns[p] == _chan_dump_prn
                    && (++_chan_dump_ctr % _chan_dump_decim) == 0) {
                    const size_t prow = (size_t)(c.job0 + 1) * n_chan; // trial 1 = PROMPT
                    for (int ch = 0; ch < n_chan; ++ch)
                        if ((c.chan_mask >> ch) & 1ULL)
                            std::fprintf(_chan_dump, "%.6f %d %.6e %.6e %.6e\n", utc, ch,
                                         corr[2 * ((prow + ch) * n_e + ref_e)],
                                         corr[2 * ((prow + ch) * n_e + ref_e) + 1],
                                         energy[prow + ch]);
                }
                rec[1] = c.fcar_report;
                rec[2] = (float)c.cp_seed;
                rec[6] = c.n_owned;
                rec[gnss::REC_E_RE] = (float)g3[0].real();
                rec[gnss::REC_E_IM] = (float)g3[0].imag();
                rec[gnss::REC_E_ENERGY] = (float)e3[0];
                rec[gnss::REC_L_RE] = (float)g3[2].real();
                rec[gnss::REC_L_IM] = (float)g3[2].imag();
                rec[gnss::REC_L_ENERGY] = (float)e3[2];

                // Carrier NCO (pass-2 half): the command's fence re-anchor resets the phase
                // history exactly like the tracker's in-place reset; f_nco (ctrim + ff ramp)
                // changes the slope of phi, never jumps it.
                // reanchored 2 and 3 are the SAME event; they differ only in who subtracted.
                // 3 carries the step in c.dcyc because the assembler cannot compute it
                // accurately from fcar (see gnssGpuChain.hpp: differencing two 1.176 GHz
                // doubles leaves 0.4 rad on the table, the same order as the term itself).
                const bool repin = (c.reanchored == 2 || c.reanchored == 3);
                if (c.reanchored == 1 || (repin && !_fcar_prev_ok[p])) {
                    // FRESH acquisition: no phase history to preserve. Break the arc.
                    _phi[p] = 0.0;
                    _phi_cyc[p] = 0.0;
                    _a_prev_ok[p] = 0;
                    if (!_elem_hold_on_reanchor)
                        _elem_prev_ok[p] = 0; // legacy: also break element continuity
                    _phi_cmd_ok[p] = 0;
                } else if (repin) {
                    // PHASE-CONTINUOUS RE-PIN. Re-pinning f_ref steps the ABSOLUTELY-ANCHORED
                    // replica phase by df*t_abs -- thousands of cycles at soak age. The old code
                    // folded that step into an EXPORT-ONLY offset and then ZEROED the NCO, so the
                    // exported ADR survived but the DATA did not: the rotation applied to A moved
                    // by the step, i.e. every re-pin punched an effectively random phase jump into
                    // the very correlation the combiner deep-integrates. That is the residual C/N0
                    // sawtooth -- and it showed exactly where this explanation says it must, in the
                    // COHERENT estimator at precisely max_anchor_age_s (30.0 s on every strong sat,
                    // 0.2 dB on GPS to 1.6 dB on B1C) and NOT in the phase-blind incoherent one
                    // (2026-07-14, on-sky).
                    //
                    // The step is FOLDABLE: an NCO absorbs an arbitrary constant phase. Put it in
                    // the NCO instead of the export and the despread output is continuous THROUGH
                    // the re-pin -- the commanded phase stays exactly as continuous as before (the
                    // export algebra is unchanged: fcar*t - phi_cyc is invariant under this fold),
                    // so the validated ADR arcs are untouched.
                    //
                    // This is what makes a re-pin CHEAP, which is the real prize: with the phase
                    // step folded and the code-currency step already translated above, the replica
                    // can be re-pinned as often as we like. max_anchor_age_s: 0 re-pins EVERY
                    // record, so f_ref never goes stale and the within-record decoherence that
                    // grows with anchor age (the OTHER half of the sawtooth, ~(dop_rate*age*t_rec)^2
                    // -- negligible on GPS's 1 ms record, ~1 dB on B1C's 10 ms) never accumulates.
                    //
                    // ⚠️ THIS BRANCH WAS DEAD CODE ON CHORD UNTIL 2026-08-13. The only producer
                    // that ever set `reanchored` was cudaGnssTrack (the airspy chain); BOTH
                    // CHORD producers hardcoded 0 (cudaGnssChordTrack.cpp, cudaGnssInject.cpp),
                    // so the step below -- 109 to 1127 CYCLES per record at 3.37 days of uptime,
                    // measured on gal_e5a -- went straight into every correlation. With
                    // carrier-gain 0.0 making f_nco zero as well, _phi was identically zero and
                    // this stage applied NO derotation at all. That is the whole of the
                    // "per-record common phase is white in time" folklore: it is not the sky,
                    // it is this subtraction never being performed. See task #52.
                    const double t_pin = (double)wstart / _sample_rate;
                    const double dcyc = (c.reanchored == 3) ? c.dcyc
                                                            : (c.fcar - _fcar_prev[p]) * t_pin;
                    _phi_cyc[p] += dcyc;
                    _phi[p] = std::remainder(_phi[p] + 2.0 * M_PI * dcyc, 2.0 * M_PI);
                }
                // #72 DIAGNOSTIC PASS-THROUGH. Both are recorded by the despread as the values
                // it actually handed the kernel; this stage only copies them. Unconditional --
                // NOT inside the dt/_a_prev_ok guard below -- because the question they answer
                // is asked per record, including the first of an arc.
                rec[gnss::REC_ANG0] = (float)c.ang0;
                rec[gnss::REC_PHI_DDOP] = (float)c.phi_ddop;
                const double dt = (double)(wstart - _wstart_prev[p]) / _sample_rate;
                if (_a_prev_ok[p] && dt > 0.0) {
                    // Commanded-trim increment (slot 19, see gnssRecord.hpp): the trim the
                    // producer ACTUALLY applied, exported honestly in PrnCtl::ctrim_hz and
                    // integrated over this record so downstream can subtract the loop's
                    // transients from the ADR exactly. This used to be reconstructed from the
                    // identity ctrim = (f_nco + fcar_report - fcar)/2 -- valid only for the
                    // airspy tracker's f_nco = ctrim + ff convention; on the CHORD producers
                    // it evaluated to (ctrim - f_offset)/2, MHz-scale garbage (see the
                    // PrnCtl::ctrim_hz doc, gnssGpuChain.hpp).
                    rec[gnss::REC_TRIM_INC] = (float)(c.ctrim_hz * dt);
                    // ⚠️ THE MIDPOINT of the previous and current f_nco (e2e [4e],
                    // 2026-08-21, MEASURED): neither endpoint is right. dt spans
                    // [t_prev, t_now]; charging the NEW slope over that gap left a
                    // persistent |dctrim|*dt offset per command step (the "[4e] drip"),
                    // but charging the OLD slope leaves the same-size residue with the
                    // opposite lineage, because a step also moves the record's PHASE
                    // CENTROID (the prompt integrates over the window; its phase sits at
                    // window CENTER, the argument anchor at window START). The average is
                    // exact: bench commanded rms 7.8 vs control 7.5 mcyc under a
                    // 4x-live staircase, from ~20-26 for either endpoint. REC_TRIM_INC
                    // above deliberately keeps THIS record's ctrim: it documents what the
                    // producer applied to this record's despread -- a different (and
                    // correct) statement.
                    _phi[p] += 2.0 * M_PI * 0.5 * (_fnco_prev[p] + c.f_nco) * dt;
                    // Keep the ROTATION phase bounded, but track the NCO phase UNWRAPPED (in
                    // cycles) for the carrier-phase export. remainder() slides phi by whole
                    // multiples of 2*pi, which a rotation cannot see and a mod-1 phase export
                    // cannot see either -- but a per-record INCREMENT sees every one of them as
                    // a spurious +-1 cycle. The NCO wraps ~f_nco times a second, so the ADR bled
                    // about one cycle per wrap: measured as a per-satellite ADR-rate error of a
                    // few Hz, scaling with each satellite's carrier trim (2026-07-13).
                    _phi[p] = std::remainder(_phi[p], 2.0 * M_PI);
                    _phi_cyc[p] += 0.5 * (_fnco_prev[p] + c.f_nco) * dt;
                }
                _fnco_prev[p] = c.f_nco; // AFTER the charge: next record's gap rides this value
                const std::complex<double> rot = std::polar(1.0, -_phi[p]);
                // #72: PUBLISH THE CURRENCY. `rot` is about to be applied to slots 3/4 and to
                // every comb column, and its exponent has a per-instance ARBITRARY origin. Ship
                // it so a consumer can rotate every instance onto one common reference; see
                // gnss::REC_PHI0, and the SpecWindow's W.phi0[p] which does the same for
                // /get_spectrum. Written BEFORE the rotation is used, so it can never describe
                // a different record's phase than the one it rode out with.
                rec[gnss::REC_PHI0] = (float)_phi[p];
                const std::complex<double> g_corr = g3[1] * rot;
                rec[3] = (float)g_corr.real();
                rec[4] = (float)g_corr.imag();
                rec[5] = (float)e3[1];
                // PER-CHANNEL PROMPT SPECTRUM accumulation (hpp note; task #32). Same source
                // as the chan_dump above (trial 1 = PROMPT, per channel, pre-sum), same
                // element combination as the header (cal when warm, else the reference
                // element), and the SAME `rot` the record's prompt just got -- one common
                // phase per record, so the accumulation doesn't wind with the residual
                // carrier while the cross-channel RELATIVE phases (the delay observable)
                // pass through untouched. Only the covering-mask channels accumulate;
                // masked-off ones stay identically zero and the consumer skips zero-energy.
                if (!_spec_freq_ids.empty() && (int)_spec_freq_ids.size() == n_chan) {
                    const size_t prow = (size_t)(c.job0 + 1) * n_chan;
                    std::lock_guard<std::mutex> lk(_spec_mtx);
                    SpecWindow& W = spec_window_for(wstart);
                    // PUBLISH THE PHASE CURRENCY (task #52). The rotation applied below is
                    // exp(-i*_phi[p]), and _phi[p] is an accumulator that is ZEROED on a fresh
                    // acquisition and STEPPED on a continuous re-pin -- and the broker re-pins
                    // seeds every ~2 minutes. So the export's phase reference MOVES UNDERNEATH
                    // the consumer, on exactly the minutes timescale where the measured
                    // window-to-window coherence collapses to its random floor.
                    //
                    // Neither of my two earlier attempts was right: leaving it implicit (the
                    // original) hands the consumer a reference it cannot know, and subtracting
                    // a per-window origin (45fe3a438) throws away the within-window continuity
                    // that is the whole point. KV: "the original was also wrong."
                    //
                    // So publish it. A window's sum is exp(-i*phi0) * SUM_r G_r*exp(-i*dphi_r),
                    // so a consumer that knows phi0 can rotate any two windows onto a common
                    // reference exactly; and n_reanchor > 0 says the accumulator was reset or
                    // stepped MID-window, which no single constant can undo -- drop those.
                    // This is #45's (value, currency, epoch) discipline applied to a phase:
                    // the spectrum was a value whose currency was never transported.
                    if (W.nrec[p] == 0)
                        W.phi0[p] = _phi[p];
                    // ⚠️ ONLY AN **UNFOLDED** RESET BREAKS THE WINDOW (2026-08-14, #53).
                    // This counted `reanchored != 0`, but reanchored 2 and 3 are the
                    // CONTINUOUS re-pins whose phase step the block immediately above has
                    // just folded INTO _phi -- exactly so the accumulator stays continuous
                    // across them. Only 1 is a fresh acquisition, where there is no phase
                    // history, the NCO is reset and the arc genuinely breaks.
                    //
                    // Counting the folded ones made n_reanchor a constant: cudaGnssInject
                    // sets `reanchored = have_hist ? 3 : 1` on EVERY record (the Doppler is
                    // propagated per record, so the reference legitimately moves every
                    // record), so path B reported n_reanchor > 0 for every PRN in every
                    // window, forever. Measured on sky: 100% of (PRN, window) on all four
                    // chains, before AND after the seed-cadence work -- which is what
                    // finally showed the counter, not the receiver, was the problem.
                    // fleet_spectrum_aligned drops any PRN with n_reanchor > 0, so #53's
                    // aligned gather discarded 100% of its points on a benign condition.
                    if (c.reanchored == 1)
                        W.nreanchor[p] += 1;
                    // ⚠️ DO **NOT** REFERENCE THIS TO THE WINDOW'S OWN FIRST RECORD.
                    // I tried exactly that (45fe3a438) to remove the arbitrary per-instance
                    // origin of _phi[p], and it was a mistake: _phi[p] is CONTINUOUS IN TIME
                    // (it only resets on re-acquisition), so subtracting a per-window origin
                    // gives every window its own arbitrary constant and the exported phase
                    // becomes random FROM WINDOW TO WINDOW. Measured after that change:
                    // coherence of a fixed (PRN,instance,channel) across 7 consecutive windows
                    // was 0.38 against a 1/sqrt(7) = 0.378 random baseline -- i.e. destroyed.
                    // KV: "we'd be adding flat but random phases time-to-time."
                    // The cross-INSTANCE concern that motivated it was real but is not fixed
                    // here; a continuous common reference is worth more than a tidy origin.
                    // KEEP the raw _phi[p]: continuous in time, and its INCREMENTS are common
                    // across instances (same commanded carrier, and since #53 the same
                    // records), which is what a cross-window combine needs.
                    // (task #52).
                    // `_phi[p]` is an ACCUMULATOR whose zero is set at a per-instance
                    // re-acquisition ("FRESH acquisition: break the arc"), so its absolute
                    // value is arbitrary and DIFFERENT on every instance. Exporting
                    // g * exp(-i*_phi) therefore stamped each instance's spectrum with its own
                    // arbitrary constant -- which left INTRA-instance coherence untouched
                    // (a constant does not tilt a ramp) while destroying INTER-instance
                    // coherence. Measured on sky 2026-08-12 before this fix: intra 0.48-0.98
                    // (0.86-0.98 on strong satellites) against inter 0.05-0.46. I first read
                    // that as a physical per-instance instrumental phase; it is not -- the
                    // instances are compute assignments over ONE PFB, interleaved stride-16
                    // combs over the SAME 5972-6076 span, with no separate signal path to
                    // carry a calibration term. KV caught it.
                    //
                    // The accumulator itself is NECESSARY: without it the prompt winds with the
                    // residual carrier across the ~100 records in a window (1 Hz of residual is
                    // 6.6 rad over 1.05 s) and the window sum decoheres. Only its ORIGIN is
                    // wrong. Computing an absolute phase instead is not an option either --
                    // f_car * t_abs at days of uptime is ~1e8 cycles, the same precision trap
                    // as the code-phase transport disease.
                    //
                    // So subtract the phase at the window's OWN first record. The INCREMENT
                    // from there is identical on every instance (same commanded carrier, and
                    // since #53 the same records), so the arbitrary origin cancels exactly and
                    // every instance lands on one common reference. The record path keeps the
                    // raw _phi[p] and is untouched.
                    // ROWS: trial 0 = EARLY, 1 = PROMPT, 2 = LATE (see prow above). One lambda
                    // so all three taps get IDENTICALLY the same element combine and the same
                    // NCO rotation -- a discriminator built from taps combined even slightly
                    // differently is measuring the difference between the combines, not the
                    // code offset.
                    const size_t erow = (size_t)(c.job0 + 0) * n_chan;
                    const size_t lrow = (size_t)(c.job0 + 2) * n_chan;
                    const bool warm = _elem_sum && _cal[p].warm();
                    auto tap = [&](size_t row, int ch) -> std::complex<double> {
                        const size_t b = (row + ch) * n_e;
                        std::complex<double> v;
                        if (warm) {
                            for (int el = 0; el < n_e; ++el)
                                _spec_scratch[el] = {corr[2 * (b + el)], corr[2 * (b + el) + 1]};
                            v = _cal[p].combine(_spec_scratch.data());
                        } else {
                            v = {corr[2 * (b + ref_e)], corr[2 * (b + ref_e) + 1]};
                        }
                        return v * rot;
                    };
                    for (int ch = 0; ch < n_chan; ++ch) {
                        if (!((c.chan_mask >> ch) & 1ULL))
                            continue;
                        std::complex<double> g = tap(prow, ch);
                        const size_t k = (size_t)p * n_chan + ch;
                        W.re[k] += g.real();
                        W.im[k] += g.imag();
                        W.energy[k] += energy[prow + ch];
                        // THE SAME VALUE, PER RECORD (gnssRecord.hpp comb block). The window
                        // ring above is what /get_spectrum serves; this is the identical
                        // quantity written out un-accumulated, because a cross-RECORD rate fit
                        // cannot be done on a window sum -- the sum is exactly what a residual
                        // rate destroys.
                        if (_chan_export) {
                            float* cb = out + gnss::chan_offset(p, ch, n_prn, n_chan,
                                                                _n_elements);
                            const std::complex<double> ge = tap(erow, ch);
                            const std::complex<double> gl = tap(lrow, ch);
                            cb[gnss::CHAN_RE] = (float)g.real();
                            cb[gnss::CHAN_IM] = (float)g.imag();
                            cb[gnss::CHAN_ENERGY] = (float)energy[prow + ch];
                            cb[gnss::CHAN_E_RE] = (float)ge.real();
                            cb[gnss::CHAN_E_IM] = (float)ge.imag();
                            cb[gnss::CHAN_E_ENERGY] = (float)energy[erow + ch];
                            cb[gnss::CHAN_L_RE] = (float)gl.real();
                            cb[gnss::CHAN_L_IM] = (float)gl.imag();
                            cb[gnss::CHAN_L_ENERGY] = (float)energy[lrow + ch];
                        }
                    }
                    W.nrec[p] += 1;
                    if (W.w0 < 0)
                        W.w0 = wstart;
                    W.w1 = wstart;
                }
                // Head segment: SAME NCO rotation as the prompt (it is the same correlation,
                // restricted to the hops before the code-period boundary), so head + tail
                // reconstructs P exactly and the combiner can wipe each side with its own
                // overlay chip.
                // SKY-PHASE-CORRECTED PROMPT: same NCO rotation as the prompt it is a version of,
                // so the combiner can deep-integrate it in exactly the prompt's currency. Zero
                // when the cal is cold / elem_sum is off, which consumers read as "absent".
                if (std::norm(g_sky) > 0.0) {
                    const std::complex<double> gs = g_sky * rot;
                    rec[gnss::REC_SKY_RE] = (float)gs.real();
                    rec[gnss::REC_SKY_IM] = (float)gs.imag();
                }
                const std::complex<double> gh_corr = g3[3] * rot;
                rec[gnss::REC_PH_RE] = (float)gh_corr.real();
                rec[gnss::REC_PH_IM] = (float)gh_corr.imag();
                rec[gnss::REC_PH_ENERGY] = (float)e3[3];
                // PEEL RESIDUAL (slots 20-23): the same prompt correlation taken on the voltage
                // after this PRN's own waveform was subtracted, and its head segment. SAME NCO
                // rotation as the prompt -- it is the same correlation, so the combiner can
                // deep-integrate it with the same per-segment overlay wipe. Left at zero when the
                // chain does not peel, which every existing consumer already reads as "absent".
                if (n_rows_spec > ROW_RES_PH) {
                    const std::complex<double> gr = g3[ROW_RES_P] * rot;
                    const std::complex<double> grh = g3[ROW_RES_PH] * rot;
                    rec[gnss::REC_RES_RE] = (float)gr.real();
                    rec[gnss::REC_RES_IM] = (float)gr.imag();
                    rec[gnss::REC_RES_PH_RE] = (float)grh.real();
                    rec[gnss::REC_RES_PH_IM] = (float)grh.imag();
                }
                // ELEMENT BLOCKS. Every antenna gets the SAME NCO rotation: the NCO is a per-PRN
                // model of the code/carrier, not a per-antenna quantity, so rotating each element
                // by `rot` preserves exactly the inter-element phase differences -- which are the
                // measurement. Rotating per element would divide out the very thing being mapped.
                if (_n_elements > 0) {
                    for (int el = 0; el < _n_elements; ++el) {
                        float* eb = out + gnss::elem_offset(p, el, _n_elements);
                        const std::complex<double> ge = _g_elem[(size_t)1 * n_e + el] * rot;
                        const std::complex<double> gE = _g_elem[(size_t)0 * n_e + el] * rot;
                        const std::complex<double> gL = _g_elem[(size_t)2 * n_e + el] * rot;
                        const std::complex<double> gH = _g_elem[(size_t)3 * n_e + el] * rot;
                        eb[gnss::ELEM_P_RE] = (float)ge.real();
                        eb[gnss::ELEM_P_IM] = (float)ge.imag();
                        eb[gnss::ELEM_E_RE] = (float)gE.real();
                        eb[gnss::ELEM_E_IM] = (float)gE.imag();
                        eb[gnss::ELEM_L_RE] = (float)gL.real();
                        eb[gnss::ELEM_L_IM] = (float)gL.imag();
                        eb[gnss::ELEM_PH_RE] = (float)gH.real();
                        eb[gnss::ELEM_PH_IM] = (float)gH.imag();
                        if (n_rows_spec > ROW_RES_PH) {
                            const std::complex<double> gr =
                                _g_elem[(size_t)ROW_RES_P * n_e + el] * rot;
                            const std::complex<double> grh =
                                _g_elem[(size_t)ROW_RES_PH * n_e + el] * rot;
                            eb[gnss::ELEM_RES_RE] = (float)gr.real();
                            eb[gnss::ELEM_RES_IM] = (float)gr.imag();
                            eb[gnss::ELEM_RES_PH_RE] = (float)grh.real();
                            eb[gnss::ELEM_RES_PH_IM] = (float)grh.imag();
                        }
                    }
                }

                _a_prev[p] = (e3[1] > 0.0) ? g_corr / e3[1] : std::complex<double>(0.0, 0.0);
                _a_prev_ok[p] = 1;
                _elem_prev_ok[p] = 1;
                _wstart_prev[p] = wstart;

                // COMMANDED CARRIER PHASE (cycles mod 1), the GPU twin of the CPU tracker's
                // export -- see gnssRecord.hpp: replica f_ref*t_abs + the NCO's phi. Adding
                // the combiner's measured arg(A) reconstructs the received carrier phase,
                // with the re-pin's replica-phase step cancelling instead of slipping.
                // phi enters NEGATED: f_ref is physical-signed, the NCO is in the r2c-flipped
                // internal convention (see GnssChannelizedTracker for the on-sky measurement
                // that settled this -- a satellite with its trim pinned at the clamp).
                // (no _phi_fix term any more: the re-pin step is folded into _phi_cyc itself,
                // which leaves fcar*t_abs - phi_cyc continuous on its own.)
                const double t_abs_w = (double)wstart / _sample_rate;
                const double phi_cmd_cyc = c.fcar * t_abs_w - _phi_cyc[p];
                // the INCREMENT (gnssRecord.hpp): bounded, float-exact, unwrap-free
                rec[gnss::REC_CPHASE] =
                    _phi_cmd_ok[p] ? (float)(phi_cmd_cyc - _phi_cmd_prev[p]) : 0.0f;
                _phi_cmd_prev[p] = phi_cmd_cyc;
                _phi_cmd_ok[p] = 1;
                _fcar_prev[p] = c.fcar;
                _fcar_prev_ok[p] = 1;
            }

            if (out_buf->metadata_pool) {
                out_buf->allocate_new_metadata_object(frame_out);
                get_gnss_chan_metadata(out_buf, frame_out)->sample_seq = wstart;
            }
            out_buf->mark_frame_full(unique_name, frame_out);
            frame_out++;
        }
        in_buf->mark_frame_empty(unique_name, frame_in);
        frame_in++;
    }
}

GnssGpuRecordAssemble::SpecWindow& GnssGpuRecordAssemble::spec_window_for(int64_t wstart) {
    // Caller holds _spec_mtx.
    if (_spec_win_samples <= 0)
        return _spec_ring[0]; // legacy single accumulator: one open window, reset on read
    // FLOOR division, not C's truncation-toward-zero. wstart should never be negative, but a
    // silent sign convention change upstream must not quietly put two instances on different
    // sides of zero -- that is precisely the class of bug this endpoint exists to remove.
    int64_t idx = wstart / _spec_win_samples;
    if (wstart < 0 && idx * _spec_win_samples != wstart)
        --idx;
    SpecWindow& W = _spec_ring[(size_t)(((idx % (int64_t)_spec_ring.size())
                                         + (int64_t)_spec_ring.size())
                                        % (int64_t)_spec_ring.size())];
    if (W.idx != idx) {
        // First record of this window (or the ring wrapped past the old occupant): clear.
        std::fill(W.re.begin(), W.re.end(), 0.0);
        std::fill(W.im.begin(), W.im.end(), 0.0);
        std::fill(W.energy.begin(), W.energy.end(), 0.0);
        std::fill(W.nrec.begin(), W.nrec.end(), 0);
        std::fill(W.phi0.begin(), W.phi0.end(), 0.0);
        std::fill(W.nreanchor.begin(), W.nreanchor.end(), 0);
        W.idx = idx;
        W.w0 = W.w1 = -1;
    }
    if (idx > _spec_max_idx)
        _spec_max_idx = idx;
    return W;
}

void GnssGpuRecordAssemble::spectrum_callback(kotekan::connectionInstance& conn) {
    // ADDRESSABLE, IDEMPOTENT READ (task #53). `?window=N` returns exactly window N; no
    // argument returns the newest COMPLETE one. Nothing is reset, so polling is free of side
    // effects and a second consumer no longer steals anyone's data.
    //
    // EVERY reply -- success or refusal -- carries available:[lo,hi], because that is what
    // lets the broker resynchronise without guessing. `too_old` means those records are gone
    // and the broker must JUMP (and log how many windows it skipped, so falling behind is
    // visible rather than a silent permanent resync); `not_yet` is NORMAL, not an error --
    // instances lag each other by a few records and the caller should simply re-ask that
    // instance, which now works because the ring still holds the window when it completes.
    if (_spec_freq_ids.empty()) {
        conn.send_error("per-channel spectrum export is OFF: this stage's config has no "
                        "'channel_ids' (regenerate the node config with a "
                        "gen_chord_gnss_config.py that wires it -- task #32)",
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    const size_t nc = _spec_freq_ids.size();
    const bool legacy = (_spec_win_samples <= 0);

    // Requested index: absent -> newest COMPLETE window. "Complete" means a LATER window has
    // been opened, which is the only evidence available locally that no more records are
    // coming for this one -- there is no end-of-window event to wait on.
    bool want_specific = false;
    int64_t want = -1;
    for (const auto& kv : conn.get_query()) {
        if (kv.first != "window")
            continue;
        want_specific = true;
        try {
            want = std::stoll(kv.second);
        } catch (const std::exception&) {
            conn.send_error("get_spectrum: 'window' must be an integer window index",
                            kotekan::HTTP_RESPONSE::BAD_REQUEST);
            return;
        }
    }

    std::vector<double> re, im, en, phi0;
    std::vector<int> nrec, nre;
    int64_t w0 = -1, w1 = -1, idx = -1, lo = -1, hi = -1;
    const char* status = "ok";
    {
        std::lock_guard<std::mutex> lk(_spec_mtx);
        if (legacy) {
            // Legacy path keeps the old reset-on-read semantics exactly, so a not-yet-
            // regenerated node behaves as it always did rather than half-changing.
            SpecWindow& W = _spec_ring[0];
            re.swap(W.re);
            im.swap(W.im);
            en.swap(W.energy);
            nrec.swap(W.nrec);
            phi0 = W.phi0;
            nre = W.nreanchor;
            w0 = W.w0;
            w1 = W.w1;
            W.re.assign((size_t)_prns.size() * nc, 0.0);
            W.im.assign((size_t)_prns.size() * nc, 0.0);
            W.energy.assign((size_t)_prns.size() * nc, 0.0);
            W.nrec.assign(_prns.size(), 0);
            W.w0 = W.w1 = -1;
        } else {
            hi = _spec_max_idx - 1; // newest COMPLETE
            for (const auto& W : _spec_ring)
                if (W.idx >= 0 && W.idx <= hi && (lo < 0 || W.idx < lo))
                    lo = W.idx;
            idx = want_specific ? want : hi;
            if (_spec_max_idx < 0 || hi < 0 || lo < 0) {
                status = "not_yet"; // nothing has completed since start-up
            } else if (idx > hi) {
                status = "not_yet";
            } else if (idx < lo) {
                status = "too_old";
            } else {
                const SpecWindow& W =
                    _spec_ring[(size_t)(((idx % (int64_t)_spec_ring.size())
                                         + (int64_t)_spec_ring.size())
                                        % (int64_t)_spec_ring.size())];
                if (W.idx != idx) {
                    // In range but the slot holds someone else: a gap (no records landed in
                    // that window at all). Report it as too_old rather than inventing zeros --
                    // "I never had it" and "here is an empty window" are different facts.
                    status = "too_old";
                } else {
                    re = W.re;
                    im = W.im;
                    en = W.energy;
                    nrec = W.nrec;
                    phi0 = W.phi0;
                    nre = W.nreanchor;
                    w0 = W.w0;
                    w1 = W.w1;
                }
            }
        }
    }

    nlohmann::json reply;
    reply["freq_ids"] = _spec_freq_ids;
    reply["sample_rate"] = _sample_rate;
    reply["status"] = status;
    reply["addressable"] = !legacy; // false -> this node predates #53; do NOT trust alignment
    reply["window_samples"] = (double)_spec_win_samples;
    reply["window"] = idx;
    reply["available"] = nlohmann::json::array({lo, hi});
    reply["wstart0"] = w0; // absolute sample of the first / last record in this window --
    reply["wstart1"] = w1; // the cross-instance alignment key, same clock as fleet hops
    if (std::string(status) != "ok") {
        reply["prns"] = nlohmann::json::array();
        conn.send_json_reply(reply);
        return;
    }
    nlohmann::json prns = nlohmann::json::array();
    for (size_t p = 0; p < _prns.size(); ++p) {
        if (nrec[p] <= 0)
            continue; // silence, not zeros (get_records convention)
        nlohmann::json chans = nlohmann::json::array();
        for (size_t ch = 0; ch < nc; ++ch) {
            const size_t k = p * nc + ch;
            // Energy-normalized like get_records' re/im: A = G/E is directly comparable
            // across instances, and E rides along as the ML combining weight.
            if (en[k] > 0.0)
                chans.push_back({re[k] / en[k], im[k] / en[k], en[k]});
            else
                chans.push_back({0.0, 0.0, 0.0}); // masked-off channel: weight 0, skipped
        }
        prns.push_back({{"prn", _prns[p]}, {"n_rec", nrec[p]},
                        {"phi0", phi0.empty() ? 0.0 : phi0[p]},
                        {"n_reanchor", nre.empty() ? 0 : nre[p]},
                        {"chan", chans}});
    }
    reply["prns"] = prns;
    conn.send_json_reply(reply);
}


// PATH B endpoint: inject a per-element complex gain prior into every PRN's ElemCal so the
// coherent combine starts warm on (re)acquisition. Parse FIRST, lock LAST: main_thread takes
// _gain_mtx only for the swap, never across a parse.
void GnssGpuRecordAssemble::set_elem_gain_callback(kotekan::connectionInstance& conn,
                                                   nlohmann::json& request) {
    if (!_elem_sum) {
        conn.send_error("elem_sum is OFF on this stage: no per-element cal to seed",
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    std::vector<std::complex<double>> gains;
    try {
        auto& arr = request.at("gain");   // [[re,im], ...], length n_elements; [] or all-zero clears
        if (!arr.is_array())
            throw std::runtime_error("'gain' must be an array of [re, im] pairs");
        gains.reserve(arr.size());
        for (auto& e : arr) {
            if (!e.is_array() || e.size() != 2)
                throw std::runtime_error("each gain must be [re, im]");
            gains.emplace_back(e[0].get<double>(), e[1].get<double>());
        }
    } catch (const std::exception& ex) {
        conn.send_error(std::string("set_elem_gain: ") + ex.what(),
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    if (!gains.empty() && (int)gains.size() != _n_elements) {
        conn.send_error("set_elem_gain: gain length != n_elements",
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    if (gains.empty())   // explicit "clear the prior": a full-length zero vector -> seed() clears
        gains.assign((size_t)_n_elements, std::complex<double>(0.0, 0.0));
    {
        std::lock_guard<std::mutex> lk(_gain_mtx);
        _pending_gain.swap(gains);
        _pending_gain_set = true;
    }
    conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
}

void GnssGpuRecordAssemble::set_reference_element_callback(kotekan::connectionInstance& conn,
                                                           nlohmann::json& request) {
    // Body: {"element": N}. Staged only; main_thread applies at the next frame boundary
    // (see the swap block there for the exact semantics). Validation mirrors the
    // construction-time FATAL: the same bounds, answered as a 400 instead of a crash.
    int el = -1;
    try {
        el = request.at("element").get<int>();
    } catch (const std::exception& e) {
        conn.send_error(std::string("set_reference_element: bad payload (want {\"element\": N}): ")
                            + e.what(),
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    if (el < 0 || el >= _n_elements) {
        conn.send_error(fmt::format("set_reference_element: element {:d} outside "
                                    "[0, n_elements={:d})",
                                    el, _n_elements),
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    int prev;
    {
        std::lock_guard<std::mutex> lk(_gain_mtx);
        _pending_ref = el;
        prev = _reference_element;
    }
    nlohmann::json reply{{"reference_element", prev},
                         {"pending", el},
                         {"applies", "next frame boundary"},
                         {"rewarm_s", _elem_sum ? 3.0 * _elem_sum_tau_s : 0.0}};
    conn.send_json_reply(reply);
}
