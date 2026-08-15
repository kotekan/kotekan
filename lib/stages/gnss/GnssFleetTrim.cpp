#include "GnssFleetTrim.hpp"

#include "StageFactory.hpp"
#include "kotekanLogging.hpp"
#include "visUtil.hpp" // for frameID, current_time

#include "json.hpp"

#include <functional>

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssFleetTrim);

GnssFleetTrim::GnssFleetTrim(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&GnssFleetTrim::main_thread, this)),
    _dll(config.get_default<int>(unique_name, "n_win", 4),
         config.get_default<int>(unique_name, "min_instances", 2),
         config.get_default<int>(unique_name, "max_open_win", 8)) {
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    kotekan::restServer::instance().register_get_callback(
        unique_name + "/get_dll",
        std::bind(&GnssFleetTrim::dll_callback, this, std::placeholders::_1));
    kotekan::restServer::instance().register_get_callback(
        unique_name + "/get_stats",
        std::bind(&GnssFleetTrim::stats_callback, this, std::placeholders::_1));
}

void GnssFleetTrim::main_thread() {
    INFO("GnssFleetTrim[{:s}]: observing (F1 -- NO actuation). {:d} windows averaged "
         "({:.0f} ms at 4 records/window), min_instances {:d}.",
         unique_name, _dll.n_win(), _dll.n_win() * 4 * 10.4857, _dll.min_instances());

    frameID in_id(in_buf);
    while (!stop_thread) {
        // TIMED WAIT, like the gather beside us: a fleet that goes silent must still let the
        // stats endpoint say so rather than leaving this thread parked with nothing in the log.
        struct timespec deadline;
        clock_gettime(CLOCK_REALTIME, &deadline);
        deadline.tv_sec += 1;
        const int rc = in_buf->wait_for_full_frame_timeout(unique_name, in_id, deadline);
        if (rc < 0)
            break;
        if (rc > 0)
            continue;
        uint8_t* frame = in_buf->frames[in_id];
        if (frame == nullptr)
            break;

        const double t0 = current_time();
        std::string chain, inst;
        gnss::FoldStatus st;
        {
            std::lock_guard<std::mutex> lk(_mtx);
            st = _dll.fold(frame, (size_t)in_buf->frame_size, &chain, &inst);
            _frames++;
            if (st == gnss::FoldStatus::BAD_HEADER)
                _bad_frames++;
            else if (st == gnss::FoldStatus::LATE)
                _late_frames++;
            _fold_s += current_time() - t0;
        }
        if (st == gnss::FoldStatus::BAD_HEADER && (_bad_frames % 100) == 1)
            ERROR("GnssFleetTrim[{:s}]: rejected a frame ({:d} so far) -- a sender is on a "
                  "different build, max_prn or records_per_frame. Folding it would parse the "
                  "rows at the wrong stride and every number would be plausible and wrong.",
                  unique_name, _bad_frames);

        in_buf->mark_frame_empty(unique_name, in_id++);
    }
}

void GnssFleetTrim::dll_callback(kotekan::connectionInstance& conn) {
    nlohmann::json reply = nlohmann::json::object();
    std::lock_guard<std::mutex> lk(_mtx);
    for (const auto& cv : _dll.chains()) {
        nlohmann::json prns = nlohmann::json::object();
        for (const auto& pv : cv.second.row) {
            const gnss::FleetDllRow& s = pv.second;
            prns[std::to_string(pv.first)] = {
                {"disc", s.disc},   {"q", s.q},         {"e_pow", s.e_pow},
                {"p_pow", s.p_pow}, {"l_pow", s.l_pow}, {"n_src", s.n_src},
                {"n_chan", s.n_chan}, {"n_rec", s.n_rec}, {"hop", s.hop},
                {"win", s.win},     {"n_updates", s.n_updates}, {"src", "comb_cpp"}};
        }
        reply[cv.first] = prns;
    }
    conn.send_json_reply(reply);
}

void GnssFleetTrim::stats_callback(kotekan::connectionInstance& conn) {
    nlohmann::json reply;
    std::lock_guard<std::mutex> lk(_mtx);
    nlohmann::json chains = nlohmann::json::object();
    for (const auto& cv : _dll.chains()) {
        const gnss::FleetDll::Chain& c = cv.second;
        chains[cv.first] = {{"frames", c.n_frames},
                            {"windows_closed", c.n_closed},
                            {"late_frames", c.n_late},
                            {"forced_closes", c.n_forced},
                            {"open_windows", c.open.size()},
                            {"ring", c.closed.size()},
                            {"newest_win", c.newest},
                            {"prns", c.row.size()}};
    }
    reply["chains"] = chains;
    reply["frames"] = _frames;
    reply["bad_frames"] = _bad_frames;
    reply["late_frames"] = _late_frames;
    // THE BUDGET. This is a second consumer of the gather's buffer, so time spent here is time
    // the broker's own copy is not being handed out. Served as microseconds per frame so "is it
    // affordable?" is answerable from this endpoint alone: 60 senders x 23.84 frames/s means
    // the whole fleet costs fold_us_per_frame x 1430 us per second of wall clock.
    reply["fold_s"] = _fold_s;
    reply["fold_us_per_frame"] = _frames ? 1e6 * _fold_s / (double)_frames : 0.0;
    reply["n_win"] = _dll.n_win();
    reply["min_instances"] = _dll.min_instances();
    conn.send_json_reply(reply);
}
