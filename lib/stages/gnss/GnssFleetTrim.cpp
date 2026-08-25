#include "GnssFleetTrim.hpp"

#include <fstream>

#include "StageFactory.hpp"
#include "kotekanLogging.hpp"
#include "visUtil.hpp" // for frameID, current_time

#include "json.hpp"

#include <algorithm>
#include <chrono>
#include <functional>

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

/// Connect with a bounded wait. A blocking connect() has no timeout short of the kernel's,
/// which is minutes -- far longer than the 42 ms window this loop lives in.
int connect_to(const struct sockaddr_in& addr, int timeout_ms) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
        return -1;
    ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
    int rc = ::connect(fd, (const struct sockaddr*)&addr, sizeof(addr));
    if (rc < 0 && errno == EINPROGRESS) {
        struct pollfd pfd = {fd, POLLOUT, 0};
        if (::poll(&pfd, 1, timeout_ms) != 1) {
            ::close(fd);
            return -1;
        }
        int err = 0;
        socklen_t len = sizeof(err);
        if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) < 0 || err != 0) {
            ::close(fd);
            return -1;
        }
    } else if (rc < 0) {
        ::close(fd);
        return -1;
    }
    ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL, 0) & ~O_NONBLOCK);
    struct timeval tv = {timeout_ms / 1000, (timeout_ms % 1000) * 1000};
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)); // 200-byte posts at 24 Hz
    return fd;
}

/// One HTTP/1.1 POST on a KEPT-ALIVE connection. Returns false and leaves *fd == -1 on any
/// error, so the caller simply reconnects next round -- there is no state to repair and
/// nothing that can throw or exit.
bool http_post(int* fd, const struct sockaddr_in& addr, const std::string& host,
               const std::string& path, const std::string& body, int timeout_ms,
               std::string* err) {
    for (int attempt = 0; attempt < 2; ++attempt) { // one free retry: a kept-alive socket the
        if (*fd < 0)                                // peer closed fails on the first write
            *fd = connect_to(addr, timeout_ms);
        if (*fd < 0) {
            *err = "connect: " + std::string(std::strerror(errno));
            return false;
        }
        std::string req = "POST " + path + " HTTP/1.1\r\nHost: " + host
                          + "\r\nContent-Type: application/json\r\nContent-Length: "
                          + std::to_string(body.size()) + "\r\nConnection: keep-alive\r\n\r\n"
                          + body;
        size_t sent = 0;
        bool wrote = true;
        while (sent < req.size()) {
            const ssize_t w = ::send(*fd, req.data() + sent, req.size() - sent, MSG_NOSIGNAL);
            if (w <= 0) {
                wrote = false;
                break;
            }
            sent += (size_t)w;
        }
        if (!wrote) {
            ::close(*fd);
            *fd = -1;
            if (attempt == 0)
                continue; // stale keep-alive: reconnect and send once more
            *err = "send: " + std::string(std::strerror(errno));
            return false;
        }
        // ⚠️ DRAIN THE WHOLE RESPONSE, OR KEEP-ALIVE POISONS THE NEXT REQUEST.
        // The first version read 512 bytes once and moved on. kotekan's restServer sends CORS
        // headers (enable_cors), which push the reply past that, so the remainder sat in the
        // socket and the NEXT request parsed it as ITS status line -- surfacing as
        // "status: -Max-Age: 2520", the tail of Access-Control-Max-Age. 13% of posts failed
        // against the real fleet, and it looked exactly like flaky trackers. A reused
        // connection must be left byte-clean.
        std::string resp;
        char buf[1024];
        size_t hdr_end = std::string::npos;
        bool dead = false;
        while (hdr_end == std::string::npos) {
            const ssize_t n = ::recv(*fd, buf, sizeof(buf), 0);
            if (n <= 0) {
                dead = true;
                break;
            }
            resp.append(buf, (size_t)n);
            hdr_end = resp.find("\r\n\r\n");
            if (resp.size() > 64 * 1024) { // a reply this large is not one of ours
                dead = true;
                break;
            }
        }
        if (!dead) {
            // Content-Length is what makes the body's end knowable. restServer always sets it
            // (send_empty_reply included, as 0), so a missing one means this is not the peer
            // we think it is -- drop the connection rather than guess where the body ends.
            size_t want = std::string::npos;
            const size_t cl = resp.find("Content-Length:");
            if (cl != std::string::npos && cl < hdr_end)
                want = (size_t)std::strtoul(resp.c_str() + cl + 15, nullptr, 10);
            if (want == std::string::npos)
                dead = true;
            else {
                const size_t have = resp.size() - (hdr_end + 4);
                for (size_t got = have; got < want;) {
                    const ssize_t n = ::recv(*fd, buf, std::min(sizeof(buf), want - got), 0);
                    if (n <= 0) {
                        dead = true;
                        break;
                    }
                    got += (size_t)n;
                }
            }
        }
        if (dead) {
            ::close(*fd);
            *fd = -1;
            if (attempt == 0)
                continue;
            *err = "recv: " + std::string(std::strerror(errno));
            return false;
        }
        if (resp.compare(0, 12, "HTTP/1.1 200") != 0 && resp.compare(0, 12, "HTTP/1.0 200") != 0) {
            *err = "status: " + resp.substr(0, std::min<size_t>(resp.size(), 40));
            return false; // a 4xx is the payload's fault; the connection is still clean
        }
        return true;
    }
    return false;
}

} // namespace

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssFleetTrim);

GnssFleetTrim::GnssFleetTrim(Config& config, const std::string& unique_name,
                             bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&GnssFleetTrim::main_thread, this)),
    _dll(config.get_default<int>(unique_name, "n_win", 4),
         config.get_default<int>(unique_name, "min_instances", 2),
         config.get_default<int>(unique_name, "max_open_win", 8),
         config.get_default<double>(unique_name, "sig_k", 3.0),
         // THE BROKER'S DEPTH, NOT THE LOOP'S. 0 follows n_win; production wants this at the
         // broker's `telem-windows` (32) so /get_taps serves the same integration length the
         // Python arm computed for itself.
         config.get_default<int>(unique_name, "taps_win", 0)) {
    _trim_state_file = config.get_default<std::string>(unique_name, "trim_state_file", "");
    _trim_state_max_age_s =
        config.get_default<double>(unique_name, "trim_state_max_age_s", 300.0);
    _trim_state_save_s = config.get_default<double>(unique_name, "trim_state_save_s", 2.0);
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    kotekan::restServer::instance().register_get_callback(
        unique_name + "/get_dll",
        std::bind(&GnssFleetTrim::dll_callback, this, std::placeholders::_1));
    kotekan::restServer::instance().register_get_callback(
        unique_name + "/get_taps",
        std::bind(&GnssFleetTrim::taps_callback, this, std::placeholders::_1));
    kotekan::restServer::instance().register_get_callback(
        unique_name + "/get_rec_taps",
        std::bind(&GnssFleetTrim::rec_taps_callback, this, std::placeholders::_1));
    kotekan::restServer::instance().register_get_callback(
        unique_name + "/get_stats",
        std::bind(&GnssFleetTrim::stats_callback, this, std::placeholders::_1));
    kotekan::restServer::instance().register_post_callback(
        unique_name + "/set_policy",
        std::bind(&GnssFleetTrim::policy_callback, this, std::placeholders::_1,
                  std::placeholders::_2));
    kotekan::restServer::instance().register_post_callback(
        unique_name + "/adjust_trim",
        std::bind(&GnssFleetTrim::adjust_callback, this, std::placeholders::_1,
                  std::placeholders::_2));

    // THE ACTUATOR'S TARGETS ARRIVE WITH THE POLICY, not from config. The broker already
    // owns the tracker endpoint list (--trackers, brace-expanded), it is the thing that knows
    // which instances are serving which chain right now, and a second copy in the gather's
    // yaml is one more thing to drift -- adding a node would silently leave it out. So
    // /set_policy carries them and this stage holds no deployment knowledge at all.
    // Generous against the broker's ~12 s policy cycle: this is a DEAD-THREAD detector, not a
    // freshness gate, and expiring a chain that merely posted late would disarm a healthy loop.
    _policy_ttl_s = std::max(5.0, config.get_default<double>(unique_name, "policy_ttl_s", 60.0));
    _post_every = std::max(1, config.get_default<int>(unique_name, "post_every_n_windows", 1));
    _post_timeout_ms = std::max(20, config.get_default<int>(unique_name, "post_timeout_ms", 200));
    const int nthr = std::max(1, config.get_default<int>(unique_name, "post_threads", 4));
    _sent_gen.assign((size_t)nthr, 0);
    _n_post_threads = nthr; // BEFORE the threads exist -- see the header note
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

    load_trims();
    _trim_saved_at = current_time();

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
        // OUTSIDE THE FOLD LOCK and off the frame path's critical section. Saving on every
        // window close would be ~24 writes/s for a file nothing reads until the next restart.
        if (_trim_state_save_s > 0.0
            && current_time() - _trim_saved_at >= _trim_state_save_s) {
            _trim_saved_at = current_time();
            save_trims();
        }

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
    // RESOLVE NOW, ONCE. See the Target note: per-request DNS at ~1430/s took the resolver
    // down and the process with it. A name that will not resolve is a policy error, answered
    // with 400, not something to rediscover on every post.
    std::memset(&t.addr, 0, sizeof(t.addr));
    t.addr.sin_family = AF_INET;
    t.addr.sin_port = htons(t.port);
    if (::inet_pton(AF_INET, t.host.c_str(), &t.addr.sin_addr) != 1) {
        struct addrinfo hints;
        std::memset(&hints, 0, sizeof(hints));
        hints.ai_family = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        struct addrinfo* res = nullptr;
        if (::getaddrinfo(t.host.c_str(), nullptr, &hints, &res) != 0 || res == nullptr)
            throw std::runtime_error("cannot resolve host '" + t.host + "' in '" + url + "'");
        t.addr.sin_addr = ((struct sockaddr_in*)res->ai_addr)->sin_addr;
        ::freeaddrinfo(res);
    }
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
            // ⚠️ THE GAIN SCALES WITH RATE TOO -- the lesson the leak taught did not go far
            // enough. Loop BANDWIDTH is gain x rate, and against this loop's ~0.3-0.5 s
            // measurement round trip (4-window disc average + actuation + record + fold), the
            // per-update gain that was stable at 3.1 Hz (0.25 -> 0.78/s) is a limit cycle at
            // 23.84 Hz (6/s): measured on sky 2026-08-15, trim swinging +-1 chip at 5-10 s
            // period, disc anti-correlated and lagging, q hitting 3.3 and being thrown off.
            // "Same gain, faster rate" holds for SLEW AUTHORITY and fails for STABILITY. So
            // policy states gain_per_s (the bandwidth) and the conversion uses the measured
            // rate, exactly as for the leak.
            const bool hgs = v.contains("gain_per_s"), hgu = v.contains("gain");
            if (hgs && hgu)
                throw std::runtime_error("give at most one of gain_per_s or gain");
            if (hgs)
                p.gain_per_s = v.at("gain_per_s").get<double>();
            else if (hgu) {
                p.gain_per_s = -1.0;
                p.gain = v.at("gain").get<double>();
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
    const double now = current_time();
    {
        std::lock_guard<std::mutex> lk(_mtx);
        // REPLACE PER CHAIN, never merge WITHIN one: the broker publishes a chain's whole armed
        // set every cycle, so replacing that chain's entry expires a PRN it stopped naming --
        // the latched-forever failure, in the arming rather than the seed. But each chain thread
        // POSTs only its OWN chain, so replacing the whole map disarms every chain that did not
        // happen to post last. See the header note: with one armed chain that is invisible.
        for (auto& kv : got) {
            _policy[kv.first] = kv.second;
            _policy_seen[kv.first] = now;
        }
        _policy_posts++;
    }
    {
        // Guarded by _pend_mtx because the poster threads read it. Targets are chain-tagged
        // (parse_target carries the key) and matched per chain in post_trims, so they replace
        // with the SAME per-chain scope as the policy above: drop this post's chains, keep the
        // rest. A flat wholesale replace here is the identical clobber one layer down.
        std::lock_guard<std::mutex> lk(_pend_mtx);
        std::vector<Target> keep;
        for (auto& t : _targets)
            if (!got.count(t.chain))
                keep.push_back(std::move(t));
        for (auto& t : tg)
            keep.push_back(std::move(t));
        _targets = std::move(keep);
    }
    rearm();
    conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
}

void GnssFleetTrim::adjust_callback(kotekan::connectionInstance& conn, nlohmann::json& request) {
    // #92: same payload shape as /set_policy ({"chains": {chain: {prn: delta_chips}}}),
    // parse first, lock last. Every refusal is counted and echoed back per PRN so the
    // broker's log can say WHICH handover was dropped -- a silently refused adjustment
    // re-creates exactly the sawtooth this endpoint exists to end.
    std::map<std::string, std::map<int, double>> got;
    try {
        const auto& chains = request.at("chains");
        for (auto it = chains.begin(); it != chains.end(); ++it)
            for (auto pit = it.value().begin(); pit != it.value().end(); ++pit)
                got[it.key()][std::stoi(pit.key())] = pit.value().get<double>();
    } catch (const std::exception& e) {
        conn.send_error(std::string("bad adjust payload: ") + e.what(),
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    nlohmann::json rep;
    int n_ok = 0, n_ref = 0;
    {
        std::lock_guard<std::mutex> lk(_mtx);
        for (const auto& cv : got)
            for (const auto& pv : cv.second) {
                const bool ok = _dll.adjust_trim(cv.first, pv.first, pv.second);
                rep[cv.first][std::to_string(pv.first)] = ok ? "adjusted" : "refused";
                (ok ? n_ok : n_ref)++;
            }
        _adjust_ok += n_ok;
        _adjust_refused += n_ref;
    }
    // Rare by construction (one per seed re-base, not per window) -- a line each is signal.
    for (const auto& cv : got)
        for (const auto& pv : cv.second)
            INFO("GnssFleetTrim[{:s}]: /adjust_trim {:s} PRN {:d} by {:+.3f} chips -> {:s}",
                 unique_name, cv.first, pv.first, pv.second,
                 rep[cv.first][std::to_string(pv.first)].get<std::string>());
    conn.send_json_reply(rep);
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

    // A CHAIN THAT HAS GONE SILENT STOPS BEING COMMANDED. Now that the policy is replaced per
    // chain rather than wholesale, this is what keeps the old anti-latch guarantee: a dead
    // broker chain thread must not leave its last armed set standing forever. It lives HERE
    // rather than in policy_callback because the sweep has to run when NOTHING is posting --
    // the whole-broker-death case, which is exactly when a POST-driven sweep never fires.
    // Erasing the policy is sufficient to disarm: post_trims skips any target whose chain has
    // no entry, so the (chain-tagged) targets left behind send nothing.
    for (auto it = _policy_seen.begin(); it != _policy_seen.end();) {
        if (now - it->second > _policy_ttl_s) {
            WARN("GnssFleetTrim[{:s}]: chain '{:s}' has POSTed no policy for {:.0f}s -- "
                 "DISARMED. Its broker chain thread is dead or cannot reach us; trims already "
                 "at the trackers expire on their own TTL.",
                 unique_name, it->first, now - it->second);
            _policy.erase(it->first);
            it = _policy_seen.erase(it);
            _policy_expired++;
        } else
            ++it;
    }

    for (const auto& pv : _policy) {
        gnss::TrimPolicy tp;
        tp.gain = pv.second.gain;
        tp.clamp = pv.second.clamp;
        tp.spacing = pv.second.spacing;
        const double hz = _close_hz > 0.1 ? _close_hz : 23.84;
        if (pv.second.gain_per_s >= 0.0)
            tp.gain = pv.second.gain_per_s / hz;
        if (pv.second.leak_per_s >= 0.0)
            // Until a rate has been measured, fall back to the wire's nominal 23.84 Hz rather
            // than to a per-update number nobody chose. Stated, not silent.
            tp.leak = pv.second.leak_per_s / hz;
        else
            tp.leak = pv.second.leak;
        tp.leak = std::max(0.0, std::min(1.0, tp.leak));
        _dll.set_armed(pv.first, pv.second.armed, tp);
        // ADOPT ANY RESTORED TRIM THE BROKER HAS JUST ARMED. This is the acceptance step: the
        // saved value is a proposal until policy names the PRN, so this stage still acts on
        // nothing the broker did not arm, and the trim cannot leak away unarmed while it waits.
        auto ri = _restored.find(pv.first);
        if (ri != _restored.end()) {
            for (int prn : pv.second.armed) {
                auto ti = ri->second.find(prn);
                if (ti == ri->second.end())
                    continue;
                _restored_offered++;
                if (_dll.adopt_trim(pv.first, prn, ti->second))
                    _restored_adopted++;
                ri->second.erase(ti); // offered once, either way
            }
            if (ri->second.empty())
                _restored.erase(ri);
        }
    }
}

void GnssFleetTrim::save_trims() {
    if (_trim_state_file.empty())
        return;
    nlohmann::json j;
    {
        std::lock_guard<std::mutex> lk(_mtx);
        nlohmann::json chains = nlohmann::json::object();
        for (const auto& cv : _dll.trim_snapshot()) {
            nlohmann::json pj = nlohmann::json::object();
            for (const auto& tv : cv.second)
                pj[std::to_string(tv.first)] = tv.second;
            chains[cv.first] = pj;
        }
        j["chains"] = chains;
    }
    j["saved_unix"] = current_time();
    j["note"] = "GnssFleetTrim standing code trims, chips. Adopted on ARMING, not on load.";
    // ⚠️ WRITE-THEN-RENAME. A restart that lands mid-write would otherwise read a truncated
    // file, and the failure mode of a HALF-PARSED trim map is adopting some satellites'
    // corrections and not others -- worse than adopting none, and silent.
    const std::string tmp = _trim_state_file + ".tmp";
    {
        std::ofstream fh(tmp, std::ios::trunc);
        if (!fh) {
            WARN_NON_OO("GnssFleetTrim[{:s}]: cannot write trim store {:s}", unique_name, tmp);
            return;
        }
        fh << j.dump();
    }
    if (std::rename(tmp.c_str(), _trim_state_file.c_str()) != 0)
        WARN_NON_OO("GnssFleetTrim[{:s}]: cannot rename {:s} -> {:s}", unique_name, tmp,
                    _trim_state_file);
}

void GnssFleetTrim::load_trims() {
    if (_trim_state_file.empty())
        return;
    std::ifstream fh(_trim_state_file);
    if (!fh) {
        INFO("GnssFleetTrim[{:s}]: no trim store at {:s} -- starting from zero trims, which "
             "costs the fleet a pull-in (q ~1 until the loop re-acquires).",
             unique_name, _trim_state_file);
        return;
    }
    nlohmann::json j;
    try {
        fh >> j;
    } catch (const std::exception& e) {
        ERROR("GnssFleetTrim[{:s}]: trim store {:s} is unreadable ({:s}) -- ignoring it. "
              "Starting from zero is correct here; half a trim map is not.",
              unique_name, _trim_state_file, e.what());
        return;
    }
    const double age = current_time() - j.value("saved_unix", 0.0);
    _restored_age_s = age;
    // ⚠️ AGE IS A REFUSAL, NOT A WARNING. A trim corrects the BROKER'S MODEL, and after a long
    // outage the model this file describes is not the model that will be republished. Adopting
    // a stale correction commands a code step at exactly the moment nothing is verified.
    if (age < 0.0 || age > _trim_state_max_age_s) {
        WARN_NON_OO("GnssFleetTrim[{:s}]: trim store is {:.0f} s old (bound {:.0f}) -- REFUSED. "
                    "The fleet will pull in from zero.",
                    unique_name, age, _trim_state_max_age_s);
        return;
    }
    int n = 0;
    // ⚠️ NAMED, not `j.value(...).items()`: value() returns a TEMPORARY and iterating it
    // dangles the moment the full expression ends. -Werror caught it; it would have been a
    // read of freed memory producing plausible trims.
    const nlohmann::json chains = j.value("chains", nlohmann::json::object());
    for (const auto& cv : chains.items()) {
        for (const auto& pv : cv.value().items()) {
            _restored[cv.key()][std::stoi(pv.key())] = pv.value().get<double>();
            n++;
        }
    }
    INFO("GnssFleetTrim[{:s}]: trim store {:.1f} s old, {:d} standing trim(s) on {:d} chain(s) "
         "HELD pending arming -- they are adopted when the broker's /set_policy names each PRN, "
         "never at load (an unarmed trim leaks to erasure in ~5.6 s).",
         unique_name, age, n, (int)_restored.size());
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
                // Disarmed PRNs with a standing trim are still here: integrate() decays them
                // through the leak and removes them when negligible, so the release is a
                // ramp, not the 4 s TTL step it was.
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
    std::map<size_t, int> fds;    // target index -> socket, this thread's alone
    std::map<size_t, int> skips;  // rounds still to skip (backoff)
    std::map<size_t, int> nfails; // consecutive failures
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
        for (size_t i = (size_t)slot; i < tgt.size(); i += (size_t)_n_post_threads) {
            const Target& t = tgt[i];
            auto it = work.find(t.chain);
            if (it == work.end())
                continue;
            // One socket per target, owned by THIS thread and kept alive across rounds. Each
            // thread takes a disjoint stride of the target list, so no socket is shared.
            // ⚠️ BACK OFF A FAILING TARGET. A wedged tracker costs its timeout on EVERY
            // round, and threads serve a stride GROUP -- so without this one dead node drags
            // its stride-mates from 23.8 Hz down to the timeout rate. Measured here: the hung
            // target held its thread ~400 ms per round (send, timeout, retry, timeout).
            // Exponential, capped, reset by a single success, so a node that comes back
            // rejoins within a couple of seconds.
            int& skip = skips[i];
            if (skip > 0) {
                --skip;
                continue;
            }
            int& fd = fds[i];
            std::string err;
            const bool ok = http_post(&fd, t.addr, t.host, t.path, it->second.dump(),
                                      _post_timeout_ms, &err);
            int& nfail = nfails[i];
            if (ok)
                nfail = 0;
            else
                skip = std::min(48, 1 << std::min(5, ++nfail)); // <= ~2 s at 23.8 Hz
            std::lock_guard<std::mutex> lk(_mtx);
            _post_reqs++;
            if (ok)
                _post_ok++;
            else {
                _post_fail++;
                _post_last_err = t.url + ": " + err;
            }
        }
        {
            std::lock_guard<std::mutex> lk(_pend_mtx);
            if ((size_t)slot < _sent_gen.size())
                _sent_gen[(size_t)slot] = gen;
        }
    }
    for (auto& f : fds)
        if (f.second >= 0)
            ::close(f.second);
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
        // ⚠️ THE TRIM RIDES THE SAME REPLY (#76). This endpoint served the discriminator but
        // not the integrator: the broker could see the error signal and NOT the standing
        // correction it had commanded, so every consumer of the seed (the escape referee,
        // the innovation) judged the model as if untrimmed -- while up to clamp chips of
        // command stood at the trackers. The trim map is walked separately from `row`
        // because they genuinely differ: a disarmed PRN decaying through the leak has a trim
        // and no fresh row, and a PRN with signal but not yet armed has a row and no trim.
        // A MISSING trim_chips key therefore means "no standing trim", which is not the same
        // statement as trim_chips == 0.0 (an armed integrator passing through zero).
        for (const auto& tv : cv.second.trim) {
            nlohmann::json& row = prns[std::to_string(tv.first)];
            row["trim_chips"] = tv.second.trim;
            row["trim_steps"] = tv.second.n_steps;
            row["trim_railed"] = tv.second.n_railed;
            row["trim_skipped"] = tv.second.n_skipped;
            row["trim_win"] = tv.second.last_win;
            row["armed"] = cv.second.armed.count(tv.first) != 0;
        }
        reply[cv.first] = prns;
    }
    conn.send_json_reply(reply);
}

/// `<unique_name>/get_taps` -- the per-instance, per-channel taps over the served window depth.
///
/// WHAT THIS REPLACES. combdll.instance_taps builds the identical object in Python by walking
/// every (window, instance, record, PRN, channel) of the gathered stream: ~140k channel-tuples
/// per chain per cycle, ~700k across the fleet, each allocating Python complex objects.
/// Profiled live it is ~18% of chain CPU -- and the broker is pinned at 100% of ONE core by
/// the GIL, where cycle time IS the sum of the five chains' Python CPU. The frames are already
/// here. The reduction is ~6k numbers per chain against the 46 MB/s it reduces.
///
/// ⚠️ MEASUREMENTS ONLY. Presence, the noise floor, the deep gate and the arming verdict are
/// POLICY and stay on the broker's cycle (see the class header). This endpoint invents no gate
/// and drops no satellite: a PRN with a live comb on one instance appears here, and whether
/// that is enough is the broker's call, exactly as it is when it walks the frames itself.
void GnssFleetTrim::taps_callback(kotekan::connectionInstance& conn) {
    nlohmann::json reply = nlohmann::json::object();
    std::lock_guard<std::mutex> lk(_mtx);
    for (const auto& cv : _dll.taps()) {
        nlohmann::json pj = nlohmann::json::object();
        for (const auto& pv : cv.second) {
            nlohmann::json ij = nlohmann::json::object();
            for (const auto& iv : pv.second) {
                nlohmann::json cj = nlohmann::json::object();
                for (const auto& ch : iv.second.chan)
                    cj[std::to_string(ch.first)] = {ch.second[0], ch.second[1], ch.second[2],
                                                    ch.second[3]};
                ij[iv.first] = {{"e", iv.second.e},         {"p", iv.second.p},
                                {"l", iv.second.l},         {"n_chan", iv.second.n_chan},
                                {"n_rec", iv.second.n_rec}, {"hop", iv.second.hop},
                                {"chan", cj}};
            }
            pj[std::to_string(pv.first)] = ij;
        }
        reply[cv.first] = pj;
    }
    reply["taps_win"] = _dll.taps_win();
    conn.send_json_reply(reply);
}

/// `<unique_name>/get_rec_taps` -- the PER-RECORD three powers, summed across instances.
///
/// The served C/N0's input (#57). combdll.prompt_cn0 builds this by walking the gathered
/// frames a SECOND time, after the comb DLL has already walked them for its own reduction --
/// the same ~140k channel-tuples per chain per cycle, twice.
///
/// Rows are [win, slot, prn, n_inst, e, p, l], flat and time-ordered. Flat rather than nested
/// because it is a SERIES: nesting it by window invites a consumer to reduce it, and the whole
/// point of this estimator is that it fits and averages nothing upstream of its own q gate.
///
/// ⚠️ MEASUREMENTS ONLY, again. The probe anchor, the Gamma-mean debias, the q gate and the
/// clip are statistics with judgement in them and stay on the broker's cycle.
void GnssFleetTrim::rec_taps_callback(kotekan::connectionInstance& conn) {
    nlohmann::json reply = nlohmann::json::object();
    nlohmann::json hpr = nlohmann::json::object();
    std::lock_guard<std::mutex> lk(_mtx);
    for (const auto& cv : _dll.rec_series()) {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& r : cv.second)
            arr.push_back({r.win, r.slot, r.prn, r.n_inst, r.e, r.p, r.l});
        reply[cv.first] = arr;
    }
    for (const auto& cv : _dll.chains())
        hpr[cv.first] = cv.second.hops_per_record;
    reply["hops_per_record"] = hpr;
    reply["taps_win"] = _dll.taps_win();
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
    reply["adjust_ok"] = _adjust_ok;         // #92 handover adjustments applied
    reply["adjust_refused"] = _adjust_refused;
    // THE BUDGET. This is a second consumer of the gather's buffer, so time spent here is time
    // the broker's own copy is not being handed out. Served as microseconds per frame so "is it
    // affordable?" is answerable from this endpoint alone: 60 senders x 23.84 frames/s means
    // the whole fleet costs fold_us_per_frame x 1430 us per second of wall clock.
    reply["fold_s"] = _fold_s;
    // THE TRIM STORE, VISIBLE. Persistence that cannot be checked is persistence nobody will
    // trust after the first surprise -- and the failure that matters here is silent: a store
    // that loads and is never adopted looks exactly like one that works. `offered` counts
    // what arming actually asked about, so offered > 0 with adopted == 0 is the real alarm.
    reply["trim_store"] = {{"file", _trim_state_file},
                           {"age_s_at_load", _restored_age_s},
                           {"offered", _restored_offered},
                           {"adopted", _restored_adopted},
                           {"pending", (int)_restored.size()}};
    reply["fold_us_per_frame"] = _frames ? 1e6 * _fold_s / (double)_frames : 0.0;
    reply["close_hz_measured"] = _close_hz;
    reply["policy_posts"] = _policy_posts;
    // A RISING policy_expired MEANS A BROKER CHAIN THREAD DIED. Served next to the per-chain
    // last-seen age so "which chain stopped talking, and how long ago" is one poll, not a log
    // grep -- the arming is now per chain and so is its failure mode.
    reply["policy_expired"] = _policy_expired;
    reply["policy_ttl_s"] = _policy_ttl_s;
    { nlohmann::json a = nlohmann::json::object();
      const double now = current_time();
      for (const auto& sv : _policy_seen)
          a[sv.first] = now - sv.second;
      reply["policy_age_s"] = a; }
    // WHAT THE BROKER ASKED FOR, straight off _policy -- deliberately NOT off _dll.chains()
    // like `policy` below, which only lists chains that have delivered frames. The per-chain
    // clobber this guards against is a property of the POST path alone, so the gate for it has
    // to be readable with no data flowing at all.
    { nlohmann::json a = nlohmann::json::object();
      for (const auto& pv : _policy)
          a[pv.first] = pv.second.armed.size();
      reply["policy_armed_requested"] = a; }
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
