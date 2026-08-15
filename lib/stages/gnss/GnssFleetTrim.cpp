#include "GnssFleetTrim.hpp"

#include "StageFactory.hpp"
#include "kotekanLogging.hpp"
#include "restClient.hpp"
#include "visUtil.hpp" // for frameID, current_time

#include "json.hpp"

#include <chrono>
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
    kotekan::restServer::instance().register_post_callback(
        unique_name + "/set_policy",
        std::bind(&GnssFleetTrim::policy_callback, this, std::placeholders::_1,
                  std::placeholders::_2));

    // THE ACTUATOR'S TARGETS ARRIVE WITH THE POLICY, not from config. The broker already
    // owns the tracker endpoint list (--trackers, brace-expanded), it is the thing that knows
    // which instances are serving which chain right now, and a second copy in the gather's
    // yaml is one more thing to drift -- adding a node would silently leave it out. So
    // /set_policy carries them and this stage holds no deployment knowledge at all.
    _post_every = std::max(1, config.get_default<int>(unique_name, "post_every_n_windows", 1));
    const int nthr = std::max(1, config.get_default<int>(unique_name, "post_threads", 4));
    _sent_gen.assign((size_t)nthr, 0);
    for (int i = 0; i < nthr; ++i)
        _post_threads.emplace_back(&GnssFleetTrim::post_loop, this, i);
    INFO("GnssFleetTrim[{:s}]: {:d} poster thread(s), blocking sends, posting every {:d} "
         "window(s). NOTHING IS TRIMMED until the broker POSTs /set_policy with targets.",
         unique_name, nthr, _post_every);
}

GnssFleetTrim::~GnssFleetTrim() {
    // Wake the posters and JOIN them before anything they touch is destroyed. The async
    // restClient path could not offer this: a reply landing after the stage is gone is a
    // use-after-free with no join to prevent it.
    {
        std::lock_guard<std::mutex> lk(_pend_mtx);
        _pend_gen++;
    }
    _pend_cv.notify_all();
    for (auto& t : _post_threads)
        if (t.joinable())
            t.join();
}

void GnssFleetTrim::main_thread() {
    INFO("GnssFleetTrim[{:s}]: {:s}. {:d} windows averaged ({:.0f} ms at 4 records/window), "
         "min_instances {:d}. Nothing is trimmed until the broker POSTs /set_policy -- this "
         "stage arms no PRN of its own.",
         unique_name, _targets.empty() ? "OBSERVING (no post targets configured)" : "ACTUATING",
         _dll.n_win(), _dll.n_win() * 4 * 10.4857, _dll.min_instances());

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
        // ACTUATE OUTSIDE THE FOLD LOCK, and only when a window actually closed. Driving this
        // off frames instead would post 10-12 times per window (once per instance) for the
        // same trim.
        // Re-derive the per-update leak from the MEASURED close rate before acting on it.
        rearm();
        post_trims();

        if (st == gnss::FoldStatus::BAD_HEADER && (_bad_frames % 100) == 1)
            ERROR("GnssFleetTrim[{:s}]: rejected a frame ({:d} so far) -- a sender is on a "
                  "different build, max_prn or records_per_frame. Folding it would parse the "
                  "rows at the wrong stride and every number would be plausible and wrong.",
                  unique_name, _bad_frames);

        in_buf->mark_frame_empty(unique_name, in_id++);
    }
}

GnssFleetTrim::Target GnssFleetTrim::parse_target(const std::string& url,
                                                  const std::string& chain) {
    // http://host:port/path -- deliberately hand-parsed and strict. A URL that ALMOST parses
    // is worse than one that does not: it posts somewhere plausible. Throws, so the caller
    // answers 400 rather than taking the process down -- this runs in a REST callback, and a
    // FATAL_ERROR here would let a bad broker payload kill the whole fleet's telemetry.
    const std::string pfx = "http://";
    if (url.rfind(pfx, 0) != 0)
        throw std::runtime_error("target '" + url + "' must start with http://");
    const size_t hb = pfx.size();
    const size_t sl = url.find('/', hb);
    if (sl == std::string::npos)
        throw std::runtime_error("target '" + url + "' has no path");
    const std::string hostport = url.substr(hb, sl - hb);
    const size_t co = hostport.find(':');
    if (co == std::string::npos)
        throw std::runtime_error("target '" + url + "' has no port -- state it rather than "
                                 "relying on a default that differs per stage");
    Target t;
    t.host = hostport.substr(0, co);
    t.port = (unsigned short)std::stoi(hostport.substr(co + 1));
    t.path = url.substr(sl);
    t.url = url;
    t.chain = chain;
    return t;
}

void GnssFleetTrim::policy_callback(kotekan::connectionInstance& conn, nlohmann::json& request) {
    // Parse first, lock last -- the fold path takes _mtx on every frame.
    std::map<std::string, PolicyReq> got;
    std::vector<Target> tg;
    try {
        const auto& chains = request.at("chains");
        for (auto it = chains.begin(); it != chains.end(); ++it) {
            PolicyReq p;
            const auto& v = it.value();
            for (const auto& a : v.at("armed"))
                p.armed.insert(a.get<int>());
            p.gain = v.value("gain", p.gain);
            p.clamp = v.value("clamp", p.clamp);
            p.spacing = v.value("spacing", p.spacing);
            // EXACTLY ONE of the two leak forms, and it must be stated. Defaulting the
            // conversion silently is how a loop ends up running at 8x the intended bandwidth
            // and being blamed on the combine.
            const bool has_s = v.contains("leak_per_s"), has_u = v.contains("leak");
            if (has_s == has_u)
                throw std::runtime_error("give exactly one of leak_per_s or leak");
            if (has_s)
                p.leak_per_s = v.at("leak_per_s").get<double>();
            else {
                p.leak_per_s = -1.0;
                p.leak = v.at("leak").get<double>();
            }
            if (p.gain <= 0.0 || p.clamp <= 0.0 || p.spacing <= 0.0)
                throw std::runtime_error("gain, clamp and spacing must be positive");
            // The tracker endpoints for this chain. Absent = observe this chain, command
            // nothing -- which is how a chain is armed for measurement without actuation.
            if (v.contains("targets"))
                for (const auto& u : v.at("targets"))
                    tg.push_back(parse_target(u.get<std::string>(), it.key()));
            got[it.key()] = p;
        }
    } catch (const std::exception& e) {
        conn.send_error(std::string("bad policy payload: ") + e.what(),
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    {
        std::lock_guard<std::mutex> lk(_mtx);
        // REPLACE, never merge: the broker publishes the whole armed set every cycle, so a
        // merge would leave a PRN armed forever after policy stopped naming it -- the
        // latched-forever failure again, in the arming rather than the seed.
        _policy = got;
        _policy_posts++;
    }
    {
        // Guarded by _pend_mtx because the poster threads read it. REPLACE wholesale: the
        // broker publishes the full list every cycle, so an instance it stops naming must
        // stop being commanded.
        std::lock_guard<std::mutex> lk(_pend_mtx);
        _targets = std::move(tg);
    }
    rearm();
    conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
}

void GnssFleetTrim::rearm() {
    std::lock_guard<std::mutex> lk(_mtx);
    // THE MEASURED CLOSE RATE, not the nominal 23.84. They differ: ~3% of frames arrive after
    // their window closed and are dropped, senders come and go, and a chain with no senders
    // closes nothing at all. Converting a per-second leak with a rate the loop is not actually
    // running at silently rescales the loop's bandwidth.
    uint64_t closed = 0;
    for (const auto& cv : _dll.chains())
        closed += cv.second.n_closed;
    const double now = current_time();
    if (_first_close_t == 0.0 && closed > 0) {
        _first_close_t = now;
        _first_close_n = closed;
    }
    const double dt = now - _first_close_t;
    if (_first_close_t > 0.0 && dt > 1.0)
        _close_hz = (double)(closed - _first_close_n) / dt / std::max<size_t>(1, _dll.chains().size());

    for (const auto& pv : _policy) {
        gnss::TrimPolicy tp;
        tp.gain = pv.second.gain;
        tp.clamp = pv.second.clamp;
        tp.spacing = pv.second.spacing;
        if (pv.second.leak_per_s >= 0.0)
            // Until a rate has been measured, fall back to the wire's nominal 23.84 Hz rather
            // than to a per-update number nobody chose. Stated, not silent.
            tp.leak = pv.second.leak_per_s / (_close_hz > 0.1 ? _close_hz : 23.84);
        else
            tp.leak = pv.second.leak;
        tp.leak = std::max(0.0, std::min(1.0, tp.leak));
        _dll.set_armed(pv.first, pv.second.armed, tp);
    }
}

void GnssFleetTrim::post_trims() {
    // PUBLISH ONLY. The fold thread builds the payload and hands it over; the poster threads do
    // every byte of I/O. See the header: this stage is a second consumer of telem_buf, so a
    // network wait here would back-pressure the BROKER's telemetry too.
    std::map<std::string, nlohmann::json> pend;
    {
        std::lock_guard<std::mutex> lk(_mtx);
        for (const auto& cv : _dll.chains()) {
            // ⚠️ PER CHAIN. This was a fleet-wide sum, so one chain closing a window commanded
            // all five (measured 5x the intended request rate). A chain only gets a new
            // command when ITS OWN window closed.
            uint64_t& seen = _closed_seen[cv.first];
            const uint64_t closed = cv.second.n_closed;
            const bool due = (closed / (uint64_t)_post_every) != (seen / (uint64_t)_post_every);
            const bool moved = closed != seen;
            seen = closed;
            if (!moved || !due)
                continue;
            if (cv.second.trim.empty())
                continue;
            nlohmann::json rows = nlohmann::json::array();
            for (const auto& tv : cv.second.trim) {
                if (!cv.second.armed.count(tv.first))
                    continue; // disarmed: its trim stands, but we stop commanding it
                rows.push_back({{"prn", tv.first},
                                {"trim_chips", tv.second.trim},
                                {"win", tv.second.last_win}});
            }
            if (!rows.empty())
                pend[cv.first] = std::move(rows);
        }
        if (!pend.empty())
            _post_rounds++;
    }
    if (pend.empty())
        return;
    {
        std::lock_guard<std::mutex> lk(_pend_mtx);
        // REPLACE, NEVER APPEND. The trim is absolute, so the newest payload supersedes any the
        // posters have not sent yet. Queueing would deliver stale corrections late, which is
        // strictly worse than skipping them.
        _pending = std::move(pend);
        _pend_gen++;
    }
    _pend_cv.notify_all();
}

void GnssFleetTrim::post_loop(int slot) {
    uint64_t last = 0;
    while (!stop_thread) {
        std::map<std::string, nlohmann::json> work;
        std::vector<Target> tgt;
        uint64_t gen = 0;
        {
            std::unique_lock<std::mutex> lk(_pend_mtx);
            _pend_cv.wait_for(lk, std::chrono::milliseconds(200),
                              [&] { return _pend_gen != last || stop_thread; });
            if (stop_thread)
                break;
            if (_pend_gen == last)
                continue;
            gen = _pend_gen;
            work = _pending; // a copy: the fold thread may replace it while we send
            tgt = _targets;  // ditto -- /set_policy can replace the list mid-round
        }
        last = gen;
        for (size_t i = (size_t)slot; i < tgt.size(); i += _post_threads.size()) {
            const Target& t = tgt[i];
            auto it = work.find(t.chain);
            if (it == work.end())
                continue;
            const restClient::restReply r = restClient::instance().make_request_blocking(
                t.path, it->second, t.host, t.port, 0, 1);
            std::lock_guard<std::mutex> lk(_mtx);
            _post_reqs++;
            if (r.first)
                _post_ok++;
            else {
                _post_fail++;
                _post_last_err = t.url + ": " + r.second;
            }
        }
        {
            std::lock_guard<std::mutex> lk(_pend_mtx);
            if ((size_t)slot < _sent_gen.size())
                _sent_gen[(size_t)slot] = gen;
        }
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
    reply["close_hz_measured"] = _close_hz;
    reply["policy_posts"] = _policy_posts;
    { nlohmann::json a = nlohmann::json::object();
      for (const auto& cv : _dll.chains())
          a[cv.first] = {{"armed", cv.second.armed}, {"leak_per_update", cv.second.policy.leak},
                         {"gain", cv.second.policy.gain}};
      reply["policy"] = a; }
    reply["post_targets"] = _targets.size();
    reply["post_threads"] = _post_threads.size();
    reply["post_every_n_windows"] = _post_every;
    reply["post_rounds"] = _post_rounds;
    reply["post_requests"] = _post_reqs;
    reply["post_ok"] = _post_ok;
    reply["post_fail"] = _post_fail;
    reply["post_last_err"] = _post_last_err;
    reply["n_win"] = _dll.n_win();
    reply["min_instances"] = _dll.min_instances();
    conn.send_json_reply(reply);
}
