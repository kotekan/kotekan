#include "GnssChordVoltageTap.hpp"

#include "GnssChanMetadata.hpp"
#include "StageFactory.hpp"
#include "chordMetadata.hpp"
#include "kotekanLogging.hpp"
#include "prometheusMetrics.hpp"
#include "visUtil.hpp" // for frameID

#include <cstring> // for memcpy
#include <functional>

using kotekan::Config;
using kotekan::bufferContainer;
using kotekan::Stage;
using kotekan::prometheus::Metrics;

REGISTER_KOTEKAN_STAGE(GnssChordVoltageTap);

GnssChordVoltageTap::GnssChordVoltageTap(Config& config, const std::string& unique_name,
                                         bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container,
          std::bind(&GnssChordVoltageTap::main_thread, this)) {
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);
    out_buf = get_buffer("out_buf");
    out_buf->register_producer(unique_name);

    _chan_ids = config.get<std::vector<int>>(unique_name, "chan_ids");
    _n_elements = config.get<int>(unique_name, "n_elements");
    _element_offset = config.get_default<int>(unique_name, "element_offset", 0);
    _frame_chan_stride = config.get<int>(unique_name, "frame_chan_stride");
    _frame_elem_stride = config.get<int>(unique_name, "frame_elem_stride");
    _n_hops = config.get<int>(unique_name, "samples_per_data_set");
    _fft_length = config.get_default<int>(unique_name, "fft_length", 16384);

    // ── #8: RF-PATH HEALTH. Absent `band_power_chans` = OFF, and OFF means the pass below
    // never executes -- not "runs and reports nothing". The generator arms this.
    _bp_chans = config.get_default<std::vector<int>>(unique_name, "band_power_chans", {});
    _bp_period_s = config.get_default<double>(unique_name, "band_power_period_s", 10.0);
    _bp_hop_stride = config.get_default<int>(unique_name, "band_power_hop_stride", 32);

    if (_chan_ids.empty())
        FATAL_ERROR("GnssChordVoltageTap: chan_ids is empty -- nothing to tap.");
    for (int c : _chan_ids)
        if (c < 0 || c >= _frame_chan_stride)
            FATAL_ERROR("GnssChordVoltageTap: chan_id {:d} outside [0, {:d})", c,
                        _frame_chan_stride);
    for (int c : _bp_chans)
        if (c < 0 || c >= _frame_chan_stride)
            FATAL_ERROR("GnssChordVoltageTap: band_power_chans entry {:d} outside [0, {:d})", c,
                        _frame_chan_stride);
    if (_bp_hop_stride < 1)
        FATAL_ERROR("GnssChordVoltageTap: band_power_hop_stride {:d} < 1", _bp_hop_stride);
    if (_bp_period_s < 0.0)
        FATAL_ERROR("GnssChordVoltageTap: band_power_period_s {:f} < 0", _bp_period_s);

    const int elem0_cfg = _element_offset.load();
    if (elem0_cfg < 0 || elem0_cfg + _n_elements > _frame_elem_stride)
        FATAL_ERROR("GnssChordVoltageTap: elements [{:d}, {:d}) outside the frame's {:d}",
                    elem0_cfg, elem0_cfg + _n_elements, _frame_elem_stride);

    // ---- LIVE ELEMENT SELECTION + LIVENESS -------------------------------------------------
    // Antennas come and go. On 2026-08-07 element 0 -- the search's reference -- was dark, and
    // because the search correlates a SINGLE element it despread literal zeros for a whole
    // session: no detections, and a best peak BELOW the pure-noise ceiling. Diagnosing that
    // meant dumping a buffer and decoding base64 by hand; re-pointing it meant regenerating
    // seven configs and restarting six nodes. Both should be one request.
    //
    // set_element re-points the tap. element_power reports MEAN POWER PER ELEMENT over the last
    // frame this tap copied, which is what "which antennas are alive?" actually asks. Note the
    // deliberate limit: power distinguishes DARK from NOT-DARK, and nothing more. A
    // self-oscillating amplifier is loud, not silent, so a healthy-looking power is necessary
    // and not sufficient -- judge a candidate by the SEARCH SNR it yields, not by this number.
    kotekan::restServer::instance().register_post_callback(
        unique_name + "/set_element",
        [this, unique_name](kotekan::connectionInstance& conn, nlohmann::json& json) {
            int e;
            try {
                e = json.at("element").get<int>();
            } catch (const std::exception& ex) {
                conn.send_error(fmt::format("expected {{\"element\": <int>}}: {:s}", ex.what()),
                                kotekan::HTTP_RESPONSE::BAD_REQUEST);
                return;
            }
            if (e < 0 || e + _n_elements > _frame_elem_stride) {
                conn.send_error(fmt::format("element {:d} + n_elements {:d} outside the frame's "
                                            "{:d}",
                                            e, _n_elements, _frame_elem_stride),
                                kotekan::HTTP_RESPONSE::BAD_REQUEST);
                return;
            }
            const int old = _element_offset.exchange(e);
            INFO("GnssChordVoltageTap[{:s}]: element_offset {:d} -> {:d} (live)", unique_name,
                 old, e);
            nlohmann::json r;
            r["element_offset"] = e;
            r["previous"] = old;
            conn.send_json_reply(r);
        });

    kotekan::restServer::instance().register_get_callback(
        unique_name + "/element_power",
        [this](kotekan::connectionInstance& conn) {
            nlohmann::json r;
            std::lock_guard<std::mutex> lk(_pow_lock);
            r["element_power"] = _elem_power;   // index = ABSOLUTE element, not tap-relative
            r["element_offset"] = _element_offset.load();
            r["frames"] = _pow_frames;
            conn.send_json_reply(r);
        });

    // ── #8: GET <unique_name>/rf_stats -- "is the RF path healthy, and in WHICH band?" ─────
    // Deliberately a NEW endpoint rather than extra keys on element_power: that one has a
    // consumer (scripts/gnss/elemsweep.py) and a documented contract, and widening a served
    // shape under a live reader is how the C/N0 pairing bugs happened.
    //
    // `cost_ms` is served on purpose. The whole safety argument for this feature is that a
    // decimated in-place pass is cheap; a number that says so is worth more than my estimate
    // of it, and if it ever grows this is where it shows.
    kotekan::restServer::instance().register_get_callback(
        unique_name + "/rf_stats", [this](kotekan::connectionInstance& conn) {
            nlohmann::json r;
            std::lock_guard<std::mutex> lk(_rf_lock);
            r["enabled"] = !_bp_chans.empty();
            r["chans"] = _bp_chans;             // LOCAL channel indices, as configured
            r["power"] = _bp_power;             // mean |x|^2 per monitored channel
            r["clip_lo"] = _bp_clip_lo;         // fraction of nibbles at -8, per channel
            r["clip_hi"] = _bp_clip_hi;         // fraction at +7, per channel
            r["elem_power"] = _bp_elem_pow;     // per ABSOLUTE element, over the monitored set
            r["elem_clip"] = _bp_elem_clip;     // per ABSOLUTE element, (lo+hi)
            r["passes"] = _bp_passes;
            r["period_s"] = _bp_period_s;
            r["hop_stride"] = _bp_hop_stride;
            r["cost_ms"] = _bp_cost_ms;
            r["fpga_seq"] = _bp_seq;            // WHICH frame this describes -- one frame, not
                                                // an average over the period
            r["age_s"] = _bp_last_s > 0.0 ? current_time() - _bp_last_s : -1.0;
            conn.send_json_reply(r);
        });

    // Both frames are sized by config with nothing linking them to these strides; a stale value
    // means reading or writing past the end. Die at construction instead.
    const size_t in_need =
        (size_t)_n_hops * _frame_chan_stride * _frame_elem_stride; // 1 byte per complex sample
    const size_t out_need = (size_t)_n_hops * _chan_ids.size() * _n_elements;
    if ((size_t)in_buf->frame_size < in_need)
        FATAL_ERROR("GnssChordVoltageTap: in_buf frame {:d} B < {:d} hops x {:d} chan x {:d} elem "
                    "({:d} B)",
                    (size_t)in_buf->frame_size, _n_hops, _frame_chan_stride, _frame_elem_stride,
                    in_need);
    if ((size_t)out_buf->frame_size < out_need)
        FATAL_ERROR("GnssChordVoltageTap: out_buf frame {:d} B < {:d} hops x {:d} chan x {:d} elem "
                    "({:d} B)",
                    (size_t)out_buf->frame_size, _n_hops, _chan_ids.size(), _n_elements, out_need);
}

void GnssChordVoltageTap::main_thread() {
    frameID in_id(in_buf);
    frameID out_id(out_buf);

    auto& dropped_total =
        Metrics::instance().add_counter("kotekan_gnss_tap_dropped_frames_total", unique_name);
    uint64_t n_dropped = 0;
    uint64_t n_no_seq = 0;

    const int n_chan = (int)_chan_ids.size();

    while (!stop_thread) {
        uint8_t* in = in_buf->wait_for_full_frame(unique_name, in_id);
        if (in == nullptr)
            break;

        // ── #8: RF-PATH HEALTH, LOW CADENCE ─────────────────────────────────────────────
        // At most one pass per _bp_period_s, reading a decimated slice of the frame we are
        // ALREADY holding.
        //
        // ⚠️ DELIBERATELY ABOVE THE DROP CHECK. This measures the RF PATH, which is upstream
        // of and independent of whether the GNSS branch can keep up -- and a branch that is
        // backing up is exactly when you most want to know whether the input still looks sane.
        // Placed below the drop, the monitor would go blind in precisely the situation it
        // exists to describe, and would do so silently.
        //
        // ⚠️ WE DELIBERATELY DO NOT COPY THE FRAME OUT FIRST, and that is the CHEAP choice
        // rather than a corner cut. The instinct is "copy, release, compute at leisure" -- but
        // a frame here is 8192 x 384 x 128 = 402 MB, so the copy alone is ~40 ms of DRAM
        // traffic against a 41.94 ms frame period: one frame's worth of memory bandwidth, per
        // pass, to avoid holding the frame. The decimated read below touches
        // n_chans x n_elem x (n_hops / stride) bytes -- ~0.5 MB at the defaults, ~1000x less --
        // so computing in place holds the frame for far LESS time than copying it would.
        // The frame is marked empty immediately after this block either way.
        //
        // The cost is served as cost_ms rather than asserted here. Measure the statistic before
        // trusting the loop it lives in.
        if (!_bp_chans.empty()) {
            const double t_now = current_time();
            if (_bp_last_s == 0.0 || t_now - _bp_last_s >= _bp_period_s) {
                const double t_start = current_time();
                const int nbc = (int)_bp_chans.size();
                const int nel = _frame_elem_stride;
                std::vector<double> pw((size_t)nbc, 0.0), clo((size_t)nbc, 0.0),
                    chi((size_t)nbc, 0.0);
                std::vector<double> epw((size_t)nel, 0.0), ecl((size_t)nel, 0.0);
                long nhop = 0;
                for (int m = 0; m < _n_hops; m += _bp_hop_stride) {
                    const size_t hop = (size_t)m * _frame_chan_stride * _frame_elem_stride;
                    for (int ci = 0; ci < nbc; ++ci) {
                        const uint8_t* row = in + hop + (size_t)_bp_chans[ci] * _frame_elem_stride;
                        for (int e = 0; e < nel; ++e) {
                            // 4+4b offset encoding, same decode as element_power above:
                            // real = high nibble - 8, imag = low nibble - 8, both in [-8, +7].
                            const uint8_t byte = row[e];
                            const int re = (int)(byte >> 4) - 8;
                            const int im = (int)(byte & 0x0F) - 8;
                            const double p = (double)(re * re + im * im);
                            pw[(size_t)ci] += p;
                            epw[(size_t)e] += p;
                            // THE RAILS ARE COUNTED APART, and -8 is not pedantry: it is the
                            // value n2k's negate_4bit silently corrupts (CorrelatorKernel.hpp),
                            // so a run of -8 is a CORRECTNESS hazard where a run of +7 is only
                            // a headroom one. A single "clip fraction" would hide that.
                            const int lo = (re == -8) + (im == -8);
                            const int hi = (re == 7) + (im == 7);
                            clo[(size_t)ci] += lo;
                            chi[(size_t)ci] += hi;
                            ecl[(size_t)e] += lo + hi;
                        }
                    }
                    ++nhop;
                }
                if (nhop > 0) {
                    // Two nibbles per sample, so a clip FRACTION divides by 2x the sample count.
                    const double n_per_chan = (double)nhop * (double)nel;
                    const double n_per_elem = (double)nhop * (double)nbc;
                    for (int ci = 0; ci < nbc; ++ci) {
                        pw[(size_t)ci] /= n_per_chan;
                        clo[(size_t)ci] /= 2.0 * n_per_chan;
                        chi[(size_t)ci] /= 2.0 * n_per_chan;
                    }
                    for (int e = 0; e < nel; ++e) {
                        epw[(size_t)e] /= n_per_elem;
                        ecl[(size_t)e] /= 2.0 * n_per_elem;
                    }
                    int64_t seq = -1;
                    if (metadata_is_chord(in_buf, in_id)) {
                        auto md = get_chord_metadata(in_buf, in_id);
                        if (md != nullptr && md->has_fpga_seq_num())
                            seq = md->get_fpga_seq_num();
                    }
                    const double cost = (current_time() - t_start) * 1e3;
                    std::lock_guard<std::mutex> lk(_rf_lock);
                    _bp_power.swap(pw);
                    _bp_clip_lo.swap(clo);
                    _bp_clip_hi.swap(chi);
                    _bp_elem_pow.swap(epw);
                    _bp_elem_clip.swap(ecl);
                    _bp_cost_ms = cost;
                    _bp_seq = seq;
                    _bp_last_s = t_now;
                    ++_bp_passes;
                }
            }
        }


        // DROP RATHER THAN BLOCK. This is the whole reason the tap owns the valve behaviour:
        // in_buf is produced by the production pipeline, so waiting here for a downstream GNSS
        // stage would hold the frame and back-pressure the F-engine ingest.
        if (!out_buf->is_frame_empty(out_id)) {
            ++n_dropped;
            if (n_dropped == 1 || n_dropped % 100 == 0)
                WARN("GnssChordVoltageTap: output full, dropped {:d} frames (GNSS branch only; "
                     "the science pipeline is unaffected)",
                     n_dropped);
            dropped_total.inc();
            in_buf->mark_frame_empty(unique_name, in_id++);
            continue;
        }

        uint8_t* out = out_buf->wait_for_empty_frame(unique_name, out_id);
        if (out == nullptr)
            break;

        // [hop][frame_chan][elem] -> [hop][chan][elem]. The element run is contiguous in both,
        // so each (hop, channel) is a single memcpy of n_elements bytes.
        const int elem_off = _element_offset.load();   // one read: a mid-frame change is fine,
                                                       // but must not split THIS frame's copy
        for (int m = 0; m < _n_hops; ++m) {
            const size_t in_hop = (size_t)m * _frame_chan_stride * _frame_elem_stride;
            const size_t out_hop = (size_t)m * n_chan * _n_elements;
            for (int c = 0; c < n_chan; ++c)
                std::memcpy(out + out_hop + (size_t)c * _n_elements,
                            in + in_hop + (size_t)_chan_ids[c] * _frame_elem_stride
                                + elem_off,
                            (size_t)_n_elements);
        }

        // PER-ELEMENT POWER over the WHOLE input frame -- every element, not just the tapped
        // ones, because the question this answers is "where should I point the tap?". Sampled
        // every _pow_stride'th hop on the first channel: ~250 hops of 8192 is a 0.2% cost and
        // still 1750 samples per element, far more than needed to separate dark from live.
        // 4+4b offset encoding: real = high nibble - 8, imag = low nibble - 8.
        {
            std::vector<double> pw((size_t)_frame_elem_stride, 0.0);
            long nsamp = 0;
            for (int m = 0; m < _n_hops; m += _pow_stride) {
                const size_t in_hop = (size_t)m * _frame_chan_stride * _frame_elem_stride
                                      + (size_t)_chan_ids[0] * _frame_elem_stride;
                for (int e = 0; e < _frame_elem_stride; ++e) {
                    const uint8_t byte = (uint8_t)in[in_hop + (size_t)e];
                    const int re = (int)(byte >> 4) - 8, im = (int)(byte & 0x0F) - 8;
                    pw[(size_t)e] += (double)(re * re + im * im);
                }
                ++nsamp;
            }
            if (nsamp > 0) {
                std::lock_guard<std::mutex> lk(_pow_lock);
                _elem_power.assign(pw.begin(), pw.end());
                for (double& v : _elem_power)
                    v /= (double)nsamp;
                ++_pow_frames;
            }
        }

        // CHORD counts 5.12 us PFB frames; the GNSS chain anchors replicas to an absolute
        // pre-channelization SAMPLE index. One multiply, done here so nothing downstream has to
        // know CHORD's convention.
        out_buf->allocate_new_metadata_object(out_id);
        GnssChanMetadata* out_md = get_gnss_chan_metadata(out_buf, out_id);
        out_md->sample_seq = -1;
        if (metadata_is_chord(in_buf, in_id)) {
            auto in_md = get_chord_metadata(in_buf, in_id);
            if (in_md != nullptr && in_md->has_fpga_seq_num())
                out_md->sample_seq = in_md->get_fpga_seq_num() * (int64_t)_fft_length;
        }
        if (out_md->sample_seq < 0 && (n_no_seq++ % 100) == 0)
            WARN("GnssChordVoltageTap: no fpga_seq_num on the input frame -- the replica anchor "
                 "is unset and the despread cannot be trusted ({:d} so far).",
                 n_no_seq);

        in_buf->mark_frame_empty(unique_name, in_id++);
        out_buf->mark_frame_full(unique_name, out_id++);
    }
}
