#include "frameWindowGate.hpp"

#include "GnssChanMetadata.hpp"
#include "StageFactory.hpp"
#include "chordMetadata.hpp"
#include "kotekanLogging.hpp"
#include "metadata.hpp"
#include "restServer.hpp"

#include "fmt.hpp"
#include "json.hpp"

#include <chrono>
#include <cstring>
#include <exception>
#include <stdexcept>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::prometheus::Metrics;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(FrameWindowGate);

FrameWindowGate::FrameWindowGate(Config& config, const std::string& unique_name,
                                 bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&FrameWindowGate::main_thread, this)) {

    _in_bufs = get_buffer_array("in_bufs");
    _out_bufs = get_buffer_array("out_bufs");
    if (_in_bufs.empty() || _in_bufs.size() != _out_bufs.size())
        throw std::runtime_error(fmt::format(
            fmt("FrameWindowGate[{:s}]: in_bufs ({:d}) and out_bufs ({:d}) must pair up"),
            unique_name, _in_bufs.size(), _out_bufs.size()));
    for (size_t i = 0; i < _in_bufs.size(); ++i) {
        if (_in_bufs[i]->frame_size != _out_bufs[i]->frame_size)
            throw std::runtime_error(fmt::format(
                fmt("FrameWindowGate[{:s}]: frame_size mismatch on pair {:d} ({:d} != {:d})"),
                unique_name, i, _in_bufs[i]->frame_size, _out_bufs[i]->frame_size));
        _in_bufs[i]->register_consumer(unique_name);
        _out_bufs[i]->register_producer(unique_name);
    }

    _clock_buf = config.get_default<int>(unique_name, "clock_buf", 0);
    if (_clock_buf < 0 || (size_t)_clock_buf >= _in_bufs.size())
        throw std::runtime_error(
            fmt::format(fmt("FrameWindowGate[{:s}]: clock_buf {:d} out of range"), unique_name,
                        _clock_buf));
    const std::string src = config.get_default<std::string>(unique_name, "clock_source", "metadata");
    if (src == "metadata")
        _clock_from_frame = false;
    else if (src == "frame")
        _clock_from_frame = true;
    else
        throw std::runtime_error(fmt::format(
            fmt("FrameWindowGate[{:s}]: clock_source must be \"metadata\" or \"frame\""),
            unique_name));
    _clock_offset = config.get_default<size_t>(unique_name, "clock_offset", 0);
    if (_clock_from_frame && _clock_offset + sizeof(int64_t) > _in_bufs[_clock_buf]->frame_size)
        throw std::runtime_error(fmt::format(
            fmt("FrameWindowGate[{:s}]: clock_offset {:d} outside the {:d}-byte clock frame"),
            unique_name, _clock_offset, _in_bufs[_clock_buf]->frame_size));

    auto& rest = kotekan::restServer::instance();
    rest.register_post_callback(
        unique_name + "/arm", [this, unique_name](kotekan::connectionInstance& conn,
                                                  nlohmann::json& json) {
            int64_t s, e;
            try {
                s = json.at("start_seq").get<int64_t>();
                e = json.value("end_seq", (int64_t)0);
            } catch (const std::exception& ex) {
                conn.send_error(fmt::format("expected {{\"start_seq\": <int64>, \"end_seq\": "
                                            "<int64, optional>}}: {:s}",
                                            ex.what()),
                                kotekan::HTTP_RESPONSE::BAD_REQUEST);
                return;
            }
            if (e > 0 && e <= s) {
                conn.send_error("end_seq must exceed start_seq (or be <= 0 for open-ended)",
                                kotekan::HTTP_RESPONSE::BAD_REQUEST);
                return;
            }
            int64_t last;
            {
                std::lock_guard<std::mutex> lk(_mtx);
                _armed = true;
                _start_seq = s;
                _end_seq = e;
                last = _last_seq;
            }
            INFO("FrameWindowGate[{:s}]: armed [{:d}, {:d}) (last seen seq {:d})", unique_name,
                 s, e, last);
            nlohmann::json r;
            r["armed"] = true;
            r["start_seq"] = s;
            r["end_seq"] = e;
            r["last_seq"] = last;
            conn.send_json_reply(r);
        });
    rest.register_post_callback(
        unique_name + "/disarm",
        [this, unique_name](kotekan::connectionInstance& conn, nlohmann::json&) {
            uint64_t passed;
            {
                std::lock_guard<std::mutex> lk(_mtx);
                _armed = false;
                passed = _passed;
            }
            INFO("FrameWindowGate[{:s}]: disarmed by request ({:d} frames passed in total)",
                 unique_name, passed);
            nlohmann::json r;
            r["armed"] = false;
            r["passed"] = passed;
            conn.send_json_reply(r);
        });
    rest.register_get_callback(unique_name + "/status", [this](kotekan::connectionInstance& conn) {
        nlohmann::json r;
        std::lock_guard<std::mutex> lk(_mtx);
        r["armed"] = _armed;
        r["start_seq"] = _start_seq;
        r["end_seq"] = _end_seq;
        r["last_seq"] = _last_seq;
        r["seq_per_frame"] = _seq_per_frame;
        r["passed"] = _passed;
        r["dropped"] = _dropped;
        r["dropped_in_window"] = _dropped_in_window;
        conn.send_json_reply(r);
    });
}

FrameWindowGate::~FrameWindowGate() {
    auto& rest = kotekan::restServer::instance();
    rest.remove_json_callback(unique_name + "/arm");
    rest.remove_json_callback(unique_name + "/disarm");
    rest.remove_get_callback(unique_name + "/status");
}

int64_t FrameWindowGate::read_seq(const uint8_t* frame, int frame_id) const {
    if (_clock_from_frame) {
        int64_t v;
        std::memcpy(&v, frame + _clock_offset, sizeof(v));
        return v;
    }
    auto meta = _in_bufs[_clock_buf]->get_metadata(frame_id);
    if (auto cm = std::dynamic_pointer_cast<chordMetadata>(meta))
        return cm->get_fpga_seq_num();
    if (auto gm = std::dynamic_pointer_cast<GnssChanMetadata>(meta))
        return gm->sample_seq;
    return -1;
}

void FrameWindowGate::copy_frame(Buffer* src, int src_id, Buffer* dst, int dst_id) {
    // The frame goes by copy, never by swap: a sibling consumer (the live pipeline) still
    // owns the source frame.
    std::memcpy(dst->frames[dst_id], src->frames[src_id], src->frame_size);
    auto sm = src->get_metadata(src_id);
    if (sm && dst->metadata_pool) {
        dst->allocate_new_metadata_object(dst_id);
        dst->metadata[dst_id]->deepCopy(sm);
    }
}

void FrameWindowGate::main_thread() {
    const size_t n = _in_bufs.size();
    std::vector<int> in_id(n, 0), out_id(n, 0);

    auto& passed_total = Metrics::instance().add_counter(
        "kotekan_framewindowgate_passed_frames_total", unique_name);
    auto& dropped_total = Metrics::instance().add_counter(
        "kotekan_framewindowgate_dropped_frames_total", unique_name);
    auto& window_dropped_total = Metrics::instance().add_counter(
        "kotekan_framewindowgate_window_dropped_frames_total", unique_name);
    auto& armed_gauge =
        Metrics::instance().add_gauge("kotekan_framewindowgate_armed", unique_name);

    bool in_window_prev = false;
    int64_t prev_seq = -1;
    auto last_warn = std::chrono::steady_clock::now();

    while (!stop_thread) {
        std::vector<uint8_t*> frames(n);
        bool eof = false;
        for (size_t i = 0; i < n; ++i) {
            frames[i] = _in_bufs[i]->wait_for_full_frame(unique_name, in_id[i]);
            if (frames[i] == nullptr)
                eof = true;
        }
        if (eof)
            break;

        const int64_t seq = read_seq(frames[_clock_buf], in_id[_clock_buf]);

        bool armed, in_window, past_end;
        int64_t start, end;
        {
            std::lock_guard<std::mutex> lk(_mtx);
            if (prev_seq >= 0 && seq > prev_seq)
                _seq_per_frame = seq - prev_seq;
            _last_seq = seq;
            armed = _armed;
            start = _start_seq;
            end = _end_seq;
            past_end = armed && end > 0 && seq >= end;
            in_window = armed && seq >= start && !past_end;
            if (past_end)
                _armed = false;
        }
        prev_seq = seq;
        armed_gauge.set(in_window ? 1 : 0);

        if (in_window && !in_window_prev)
            INFO("FrameWindowGate[{:s}]: window OPEN at seq {:d} (armed [{:d}, {:d}))",
                 unique_name, seq, start, end);
        if (past_end || (!in_window && in_window_prev)) {
            uint64_t p, dw;
            {
                std::lock_guard<std::mutex> lk(_mtx);
                p = _passed;
                dw = _dropped_in_window;
            }
            INFO("FrameWindowGate[{:s}]: window CLOSED at seq {:d}: {:d} frames passed, {:d} "
                 "dropped inside windows so far",
                 unique_name, seq, p, dw);
        }
        in_window_prev = in_window;

        if (in_window) {
            bool room = true;
            for (size_t i = 0; i < n; ++i)
                room = room && _out_bufs[i]->is_frame_empty(out_id[i]);
            if (room) {
                // Cannot block: every output was just seen empty.
                for (size_t i = 0; i < n; ++i) {
                    uint8_t* f = _out_bufs[i]->wait_for_empty_frame(unique_name, out_id[i]);
                    if (f == nullptr) {
                        eof = true;
                        break;
                    }
                    copy_frame(_in_bufs[i], in_id[i], _out_bufs[i], out_id[i]);
                    _out_bufs[i]->mark_frame_full(unique_name, out_id[i]);
                    out_id[i] = (out_id[i] + 1) % _out_bufs[i]->num_frames;
                }
                if (eof)
                    break;
                passed_total.inc();
                std::lock_guard<std::mutex> lk(_mtx);
                ++_passed;
            } else {
                window_dropped_total.inc();
                uint64_t dw;
                {
                    std::lock_guard<std::mutex> lk(_mtx);
                    dw = ++_dropped_in_window;
                }
                auto now = std::chrono::steady_clock::now();
                if (dw == 1 || now - last_warn >= std::chrono::seconds(10)) {
                    WARN("FrameWindowGate[{:s}]: output full inside the window at seq {:d}; "
                         "{:d} in-window frames lost so far (the sink is not keeping up)",
                         unique_name, seq, dw);
                    last_warn = now;
                }
            }
        } else {
            dropped_total.inc();
            std::lock_guard<std::mutex> lk(_mtx);
            ++_dropped;
        }

        for (size_t i = 0; i < n; ++i) {
            _in_bufs[i]->mark_frame_empty(unique_name, in_id[i]);
            in_id[i] = (in_id[i] + 1) % _in_bufs[i]->num_frames;
        }
    }
}
