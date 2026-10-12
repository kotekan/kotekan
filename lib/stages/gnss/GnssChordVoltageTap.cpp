#include "GnssChordVoltageTap.hpp"

#include "GnssChanMetadata.hpp"
#include "StageFactory.hpp"
#include "chordMetadata.hpp"
#include "gnssChanPower.hpp" // for ChanPower
#include "kotekanLogging.hpp"
#include "prometheusMetrics.hpp"
#include "visUtil.hpp" // for frameID

#include <cstring> // for memcpy
#include <functional>

using kotekan::bufferContainer;
using kotekan::Config;
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

    // RF-path health: with `band_power_chans` absent, the pass in main_thread never runs.
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

    // set_element re-points the tap at runtime, so a dark element can be replaced without a
    // config change. element_power reports mean power per element over the last frame this
    // tap copied. Power separates dark from live and nothing more: a self-oscillating amplifier
    // is loud, so judge a candidate element by the search SNR it gives.
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
            INFO("GnssChordVoltageTap[{:s}]: element_offset {:d} -> {:d} (live)", unique_name, old,
                 e);
            nlohmann::json r;
            r["element_offset"] = e;
            r["previous"] = old;
            conn.send_json_reply(r);
        });

    kotekan::restServer::instance().register_get_callback(
        unique_name + "/element_power", [this](kotekan::connectionInstance& conn) {
            nlohmann::json r;
            std::lock_guard<std::mutex> lk(_pow_lock);
            r["element_power"] = _elem_power; // indexed by absolute element, not tap-relative
            r["element_offset"] = _element_offset.load();
            r["frames"] = _pow_frames;
            conn.send_json_reply(r);
        });

    // GET <unique_name>/rf_stats: band power and clipping of the monitored channels. A separate
    // endpoint so that element_power, which scripts/gnss/site/elemsweep.py reads, keeps its
    // shape. cost_ms is the measured wall time of the last pass, which must stay small because
    // it holds the input frame.
    kotekan::restServer::instance().register_get_callback(
        unique_name + "/rf_stats", [this](kotekan::connectionInstance& conn) {
            nlohmann::json r;
            std::lock_guard<std::mutex> lk(_rf_lock);
            r["enabled"] = !_bp_chans.empty();
            r["chans"] = _bp_chans; // local channel indices, as configured
            // From the frame's chord metadata, not the config: nodes hold different channel
            // subsets, so only the absolute freq_id names a band off the node. Parallel to
            // `chans`; empty until a chord-metadata frame has been measured, which a consumer
            // must read as unknown, never as freq_id 0.
            r["freq_ids"] = _bp_freq_ids;
            r["power"] = _bp_power;         // mean |x|^2 per monitored channel
            r["clip_lo"] = _bp_clip_lo;     // fraction of nibbles at -8, per channel
            r["clip_hi"] = _bp_clip_hi;     // fraction at +7, per channel
            r["elem_power"] = _bp_elem_pow; // per absolute element, over the monitored set
            r["elem_clip"] = _bp_elem_clip; // per absolute element, fraction at -8 or +7
            r["passes"] = _bp_passes;
            r["period_s"] = _bp_period_s;
            r["hop_stride"] = _bp_hop_stride;
            r["cost_ms"] = _bp_cost_ms;
            r["fpga_seq"] = _bp_seq; // the one frame these figures describe
            r["age_s"] = _bp_last_s > 0.0 ? current_time() - _bp_last_s : -1.0;
            conn.send_json_reply(r);
        });

    // Frame sizes come from the buffer config, independently of these strides; check them here
    // rather than read or write past the end of a frame.
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

        // RF-path health: at most one pass per _bp_period_s, over a decimated slice of the
        // frame already held.
        //
        // The pass sits above the drop check because it measures the RF path, which does not
        // depend on whether the GNSS branch keeps up; below the drop it would go blind exactly
        // when the branch is backing up.
        //
        // It reads the frame in place rather than copying it out: copying a 402 MB frame takes
        // about one frame period of memory bandwidth, while the decimated read touches
        // n_chans x n_elem x (n_hops / stride) bytes, 32 KB per channel for 128 elements and
        // 8192 hops at the default stride. Its wall time is served as rf_stats cost_ms.
        if (!_bp_chans.empty()) {
            const double t_now = current_time();
            if (_bp_last_s == 0.0 || t_now - _bp_last_s >= _bp_period_s) {
                const double t_start = current_time();
                const int nbc = (int)_bp_chans.size();
                const int nel = _frame_elem_stride;
                std::vector<double> pw((size_t)nbc, 0.0), clo((size_t)nbc, 0.0),
                    chi((size_t)nbc, 0.0);
                std::vector<double> epw((size_t)nel, 0.0), ecl((size_t)nel, 0.0);
                std::vector<double> pwe((size_t)nbc * nel, 0.0); // [chan][elem], for ChanPower
                long nhop = 0;
                for (int m = 0; m < _n_hops; m += _bp_hop_stride) {
                    const size_t hop = (size_t)m * _frame_chan_stride * _frame_elem_stride;
                    for (int ci = 0; ci < nbc; ++ci) {
                        const uint8_t* row = in + hop + (size_t)_bp_chans[ci] * _frame_elem_stride;
                        for (int e = 0; e < nel; ++e) {
                            // 4+4b offset encoding:
                            // real = high nibble - 8, imag = low nibble - 8, both in [-8, +7].
                            const uint8_t byte = row[e];
                            const int re = (int)(byte >> 4) - 8;
                            const int im = (int)(byte & 0x0F) - 8;
                            const double p = (double)(re * re + im * im);
                            pw[(size_t)ci] += p;
                            epw[(size_t)e] += p;
                            pwe[(size_t)ci * nel + e] += p;
                            // The rails are counted apart: -8 is outside the range n2k's
                            // negate_4bit handles (CorrelatorKernel.hpp), so -8 is a correctness
                            // hazard for the N^2 correlator, where +7 only marks lost headroom.
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
                    // Two nibbles per sample, so a clip fraction divides by 2x the sample count.
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
                    std::vector<int> fids;
                    if (metadata_is_chord(in_buf, in_id)) {
                        auto md = get_chord_metadata(in_buf, in_id);
                        if (md != nullptr && md->has_fpga_seq_num())
                            seq = md->get_fpga_seq_num();
                        // Absolute freq_id per monitored channel, from the frame just
                        // measured. coarse_freq is indexed by local channel position, as
                        // _bp_chans is, so this is a lookup with no comb arithmetic. Unless
                        // every entry is in range, publish nothing: a wrong band name is worse
                        // than none.
                        if (md != nullptr && md->has_coarse_freq()) {
                            const std::vector<int> cf = md->get_coarse_freq();
                            bool ok = true;
                            for (int c : _bp_chans)
                                if (c < 0 || (size_t)c >= cf.size())
                                    ok = false;
                            if (ok)
                                for (int c : _bp_chans)
                                    fids.push_back(cf[(size_t)c]);
                        }
                    }
                    const double cost = (current_time() - t_start) * 1e3;
                    std::lock_guard<std::mutex> lk(_rf_lock);
                    if (!fids.empty())
                        _bp_freq_ids.swap(fids); // keep the last known map if this frame had none
                    _bp_power.swap(pw);
                    // GnssGpuRecordAssemble weights its cross-channel sums by this power, kept
                    // per element so it can average over the inputs it actually sums.
                    if ((int)_bp_freq_ids.size() == nbc) {
                        for (double& v : pwe)
                            v /= (double)nhop;
                        gnss::ChanPower::instance().publish(_bp_freq_ids, pwe, nel);
                    }
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


        // Drop rather than block: in_buf belongs to the science pipeline, so waiting here for a
        // downstream GNSS stage would hold the frame and back-pressure the F-engine ingest.
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
        const int elem_off = _element_offset.load(); // read once, so set_element cannot split
                                                     // this frame's copy
        for (int m = 0; m < _n_hops; ++m) {
            const size_t in_hop = (size_t)m * _frame_chan_stride * _frame_elem_stride;
            const size_t out_hop = (size_t)m * n_chan * _n_elements;
            for (int c = 0; c < n_chan; ++c)
                std::memcpy(out + out_hop + (size_t)c * _n_elements,
                            in + in_hop + (size_t)_chan_ids[c] * _frame_elem_stride + elem_off,
                            (size_t)_n_elements);
        }

        // Per-element power over every element of the input frame, not only the tapped ones,
        // so element_power shows where the tap could point. Sampled every _pow_stride'th hop
        // on the first tapped channel (256 samples per element at 8192 hops), enough to
        // separate dark from live. 4+4b offset encoding: real = high nibble - 8,
        // imag = low nibble - 8.
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
        // pre-channelization sample index. Converted here so nothing downstream needs CHORD's
        // convention; -1 when the frame carries no fpga_seq_num.
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
