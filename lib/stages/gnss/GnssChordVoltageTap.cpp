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

    if (_chan_ids.empty())
        FATAL_ERROR("GnssChordVoltageTap: chan_ids is empty -- nothing to tap.");
    for (int c : _chan_ids)
        if (c < 0 || c >= _frame_chan_stride)
            FATAL_ERROR("GnssChordVoltageTap: chan_id {:d} outside [0, {:d})", c,
                        _frame_chan_stride);
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
