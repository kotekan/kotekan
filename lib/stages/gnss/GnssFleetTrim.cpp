#include "GnssFleetTrim.hpp"

#include "StageFactory.hpp"
#include "kotekanLogging.hpp"
#include "visUtil.hpp" // for frameID, current_time

#include "json.hpp"

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

/// Start a non-blocking connect and return at once; the poster checks it with connect_poll()
/// on later rounds. A lost SYN or SYN-ACK is recovered only by the kernel's SYN retransmit
/// (after 1 s, then 2 s more), far longer than a post timeout, so a connect that had to finish
/// within one would fail whenever a handshake segment is lost, and on a lossy link the backoff
/// would hold the target past the trackers' trim TTL. Left in flight, the handshake completes
/// on its own while the thread serves its other targets.
int connect_start(const struct sockaddr_in& addr) {
    const int fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
        return -1;
    ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL, 0) | O_NONBLOCK);
    if (::connect(fd, (const struct sockaddr*)&addr, sizeof(addr)) < 0 && errno != EINPROGRESS) {
        ::close(fd);
        return -1;
    }
    return fd;
}

/// 1: established (socket now blocking, with the post timeout on send and recv);
/// 0: still in flight; -1: failed and closed (errno says why).
int connect_poll(int fd, int timeout_ms) {
    struct pollfd pfd = {fd, POLLOUT, 0};
    if (::poll(&pfd, 1, 0) != 1)
        return 0;
    int err = 0;
    socklen_t len = sizeof(err);
    if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len) < 0 || err != 0) {
        ::close(fd);
        errno = err ? err : ECONNABORTED;
        return -1;
    }
    ::fcntl(fd, F_SETFL, ::fcntl(fd, F_GETFL, 0) & ~O_NONBLOCK);
    struct timeval tv = {timeout_ms / 1000, (timeout_ms % 1000) * 1000};
    ::setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    ::setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    int one = 1;
    ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one)); // 200-byte posts at 24 Hz
    return 1;
}

/// One HTTP/1.1 POST on an established kept-alive connection; true on a 200 reply. Any
/// transport error closes the socket and sets *fd = -1, so the caller reconnects on a later
/// round. Nothing here throws or exits.
bool http_post(int* fd, const std::string& host, const std::string& path, const std::string& body,
               std::string* err) {
    {
        std::string req =
            "POST " + path + " HTTP/1.1\r\nHost: " + host
            + "\r\nContent-Type: application/json\r\nContent-Length: " + std::to_string(body.size())
            + "\r\nConnection: keep-alive\r\n\r\n" + body;
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
            *err = "send: " + std::string(std::strerror(errno));
            return false;
        }
        // Read the whole response, headers and body, so the kept-alive connection is left
        // byte-clean: an unread tail (with enable_cors, restServer's headers alone are long)
        // would be parsed as the next request's status line.
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
            // Only Content-Length marks the body's end. restServer always sends it (0 for
            // send_empty_reply), so without it drop the connection rather than guess.
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
            *err = "recv: " + std::string(std::strerror(errno));
            return false;
        }
        if (resp.compare(0, 12, "HTTP/1.1 200") != 0 && resp.compare(0, 12, "HTTP/1.0 200") != 0) {
            *err = "status: " + resp.substr(0, std::min<size_t>(resp.size(), 40));
            return false; // response fully read, so the connection is still usable
        }
        return true;
    }
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
         config.get_default<int>(unique_name, "taps_win", 0)) {
    _trim_state_file = config.get<std::string>(unique_name, "trim_state_file");
    if (_trim_state_file.empty())
        FATAL_ERROR("GnssFleetTrim[{:s}]: trim_state_file must name the trim store", unique_name);
    _trim_state_max_age_s = config.get_default<double>(unique_name, "trim_state_max_age_s", 300.0);
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
        unique_name + "/set_policy", std::bind(&GnssFleetTrim::policy_callback, this,
                                               std::placeholders::_1, std::placeholders::_2));

    // Post targets arrive with /set_policy, not from config: the broker owns the tracker
    // endpoint list, and a copy here would drift from it.
    // policy_ttl_s detects a dead broker chain thread, not a late post, so it is generous
    // against the ~12 s policy cycle.
    _policy_ttl_s = std::max(5.0, config.get_default<double>(unique_name, "policy_ttl_s", 60.0));
    if (config.get_default<int>(unique_name, "post_every_n_windows", 1) != 1)
        FATAL_ERROR("GnssFleetTrim[{:s}]: post_every_n_windows must be 1; a chain's trims are "
                    "posted on every close of its window",
                    unique_name);
    const int nthr = std::max(1, config.get_default<int>(unique_name, "post_threads", 4));
    _n_post_threads = nthr; // set before the first poster reads it
    for (int i = 0; i < nthr; ++i)
        _post_threads.emplace_back(&GnssFleetTrim::post_loop, this, i);
    INFO("GnssFleetTrim[{:s}]: {:d} poster thread(s), blocking sends, posting on every window "
         "close. Nothing is trimmed until the broker posts /set_policy with targets.",
         unique_name, nthr);
}

GnssFleetTrim::~GnssFleetTrim() {
    // Wake the posters and join them before the members they use are destroyed.
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
    INFO("GnssFleetTrim[{:s}]: {:d} windows averaged ({:.0f} ms at 4 records/window), "
         "min_instances {:d}. Nothing is trimmed until the broker POSTs /set_policy -- this "
         "stage arms no PRN of its own.",
         unique_name, _dll.n_win(), _dll.n_win() * 4 * 10.4857, _dll.min_instances());

    load_trims();
    _trim_saved_at = current_time();

    frameID in_id(in_buf);
    while (!stop_thread) {
        // Timed wait, so the loop still checks stop_thread when no frames arrive.
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
        gnss::FoldStatus st;
        {
            std::lock_guard<std::mutex> lk(_mtx);
            st = _dll.fold(frame, (size_t)in_buf->frame_size);
            _frames++;
            if (st == gnss::FoldStatus::BAD_HEADER)
                _bad_frames++;
            else if (st == gnss::FoldStatus::LATE)
                _late_frames++;
            _fold_s += current_time() - t0;
        }
        // Outside the fold lock. rearm() reconverts the per-second gain and leak with the
        // measured close rate; post_trims() posts a chain only when its own window closed, not
        // once per frame.
        rearm();
        post_trims();
        // On a timer rather than per window close: the store is read only at the next start.
        if (current_time() - _trim_saved_at >= _trim_state_save_s) {
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
    // http://host:port/path, with scheme, explicit port and path all required: a URL that
    // almost parses would post somewhere plausible. Throws rather than FATAL_ERROR: this runs
    // in the /set_policy callback, which answers 400, and a bad payload must not stop the
    // gather process.
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
        throw std::runtime_error("target '" + url
                                 + "' has no port -- state it rather than "
                                   "relying on a default that differs per stage");
    Target t;
    t.host = hostport.substr(0, co);
    t.port = (unsigned short)std::stoi(hostport.substr(co + 1));
    t.path = url.substr(sl);
    t.url = url;
    t.chain = chain;
    // Resolve once, here (see Target::addr); a name that does not resolve fails the policy
    // post with 400.
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
            p.clamp = v.value("clamp", p.clamp);
            p.spacing = v.value("spacing", p.spacing);
            // Absent or 0: FleetDll steps no PRN of this chain, it only leaks.
            p.p_floor_abs = v.value("p_floor_abs", 0.0);
            if (p.p_floor_abs < 0.0)
                throw std::runtime_error("p_floor_abs must be >= 0");
            // Both are required, per second: a silent default would run the loop at a
            // bandwidth nobody chose. Loop bandwidth is gain x update rate, and against the
            // loop's ~0.3-0.5 s measurement round trip (disc average, actuation, record, fold)
            // a per-update gain of 0.25 at 23.84 Hz (6/s) limit-cycles, so rearm() converts
            // both with the measured rate. 0 is allowed: the broker freezes the loop with it.
            p.gain_per_s = v.at("gain_per_s").get<double>();
            p.leak_per_s = v.at("leak_per_s").get<double>();
            if (!(p.gain_per_s >= 0.0) || !(p.leak_per_s >= 0.0))
                throw std::runtime_error("gain_per_s and leak_per_s must be >= 0");
            if (p.clamp <= 0.0 || p.spacing <= 0.0)
                throw std::runtime_error("clamp and spacing must be positive");
            // This chain's tracker endpoints. Absent: the chain is armed and integrated but
            // nothing is posted for it.
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
        // Replace per chain: a posted chain's entry is replaced wholesale, so a PRN it stopped
        // naming is disarmed, and chains absent from this post keep theirs (each broker chain
        // thread posts only its own chain).
        for (auto& kv : got) {
            _policy[kv.first] = kv.second;
            _policy_seen[kv.first] = now;
        }
        _policy_posts++;
    }
    {
        // Targets replace with the same per-chain scope as the policy: drop this post's
        // chains' targets, keep the rest. _pend_mtx because the posters read the list.
        std::lock_guard<std::mutex> lk(_pend_mtx);
        std::vector<Target> keep;
        for (auto& t : _targets)
            if (!got.count(t.chain))
                keep.push_back(std::move(t));
        for (auto& t : tg)
            keep.push_back(std::move(t));
        // Sort by url: the posters stride this list by index, so a target's position, and
        // hence its thread, must depend only on the set of targets, not on which chain posted
        // last. Otherwise sockets would be torn down and reopened on other threads every post.
        std::stable_sort(keep.begin(), keep.end(),
                         [](const Target& a, const Target& b) { return a.url < b.url; });
        _targets = std::move(keep);
    }
    rearm();
    conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
}

void GnssFleetTrim::rearm() {
    std::lock_guard<std::mutex> lk(_mtx);
    // Measured window close rate: closes per second since the first close, averaged over the
    // chains FleetDll holds. A chain that closes nothing (no senders, or a policy-only entry)
    // lowers it below the nominal 23.84 Hz.
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
        _close_hz =
            (double)(closed - _first_close_n) / dt / std::max<size_t>(1, _dll.chains().size());

    // Late-frame warning: a chain whose frames keep arriving late is either not followed by
    // the fold (a frame0 move FleetDll did not re-anchor on) or fed by a sender on a different
    // time axis.
    for (const auto& cv : _dll.chains()) {
        const uint64_t late = cv.second.n_late;
        uint64_t& prev = _late_seen[cv.first];
        const double dtl = now - _late_seen_t;
        if (_late_seen_t > 0.0 && dtl > 1.0) {
            const double rate = (double)(late - prev) / dtl;
            if (rate > 0.5 * std::max(1.0, _close_hz))
                WARN("GnssFleetTrim[{:s}]: chain '{:s}' is DROPPING {:.1f} frames/s as LATE "
                     "(close rate {:.1f}/s) -- the fold is not following this stream. Check "
                     "the F-engine axis and /get_dll's hop; epoch resets so far: {:d}.",
                     unique_name, cv.first, rate, _close_hz, cv.second.n_epoch_reset);
        }
        prev = late;
    }
    if (_late_seen_t == 0.0 || now - _late_seen_t > 1.0)
        _late_seen_t = now;

    // Drop the policy of a chain that has not posted for policy_ttl_s (a dead broker chain
    // thread). The sweep runs here, per frame, rather than in policy_callback so it also fires
    // when nothing posts. It removes the chain from _policy only: FleetDll keeps the chain's
    // last armed set and policy, and its targets stay in _targets.
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
        tp.clamp = pv.second.clamp;
        tp.spacing = pv.second.spacing;
        tp.p_floor_abs = pv.second.p_floor_abs;
        // Until a close rate has been measured, convert with the nominal 23.84 Hz.
        const double hz = _close_hz > 0.1 ? _close_hz : 23.84;
        tp.gain = pv.second.gain_per_s / hz;
        tp.leak = std::max(0.0, std::min(1.0, pv.second.leak_per_s / hz));
        _dll.set_armed(pv.first, pv.second.armed, tp);
        // Offer each restored trim once, when the policy arms its PRN, so a saved trim is
        // applied only to a PRN the broker armed; FleetDll::adopt_trim decides.
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
    // Write then rename, so a restart never reads a partly written store.
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
    // A trim corrects the broker's model, which changes over a long outage, so an old store is
    // refused rather than adopted.
    if (age < 0.0 || age > _trim_state_max_age_s) {
        WARN_NON_OO("GnssFleetTrim[{:s}]: trim store is {:.0f} s old (bound {:.0f}) -- REFUSED. "
                    "The fleet will pull in from zero.",
                    unique_name, age, _trim_state_max_age_s);
        return;
    }
    int n = 0;
    // A named copy: a range-for over j.value(...).items() would iterate a destroyed
    // temporary.
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
    // Builds the payloads and hands them to the posters; no network I/O on the fold thread,
    // which would back-pressure telem_buf (see the class note).
    std::map<std::string, nlohmann::json> pend;
    {
        std::lock_guard<std::mutex> lk(_mtx);
        for (const auto& cv : _dll.chains()) {
            // Post a chain only when its own window count moved.
            uint64_t& seen = _closed_seen[cv.first];
            const uint64_t closed = cv.second.n_closed;
            const bool moved = closed != seen;
            seen = closed;
            if (!moved)
                continue;
            if (cv.second.trim.empty())
                continue;
            nlohmann::json rows = nlohmann::json::array();
            for (const auto& tv : cv.second.trim) {
                // Disarmed PRNs stay in `trim` while FleetDll decays them through the leak,
                // so the tracker sees their release as a ramp.
                rows.push_back({{"prn", tv.first}, {"trim_chips", tv.second.trim}});
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
        // Replace, never queue: the trim is absolute, so the newest payload supersedes any not
        // yet sent. _pending then holds only the chains posted by this call.
        _pending = std::move(pend);
        _pend_gen++;
    }
    _pend_cv.notify_all();
}

void GnssFleetTrim::post_loop(int slot) {
    uint64_t last = 0;
    // Per-target state is keyed by url, not list index: a kept-alive socket belongs to one
    // host, and /set_policy can rebuild the list between rounds.
    struct Conn {
        int fd = -1;
        bool connecting = false; // fd is a handshake in flight, not yet a connection
        std::chrono::steady_clock::time_point since;
    };
    // A handshake gets this long before it counts as a failure: enough for the kernel's SYN
    // retransmits at 1 s and 3 s, and inside the trackers' trim TTL, so a target that comes
    // back is served before its trims expire.
    constexpr auto kConnectBudget = std::chrono::milliseconds(3500);
    std::map<std::string, Conn> conns; // target url -> socket, this thread's alone
    std::map<std::string, int> skips;  // rounds still to skip (backoff)
    std::map<std::string, int> nfails; // consecutive failures
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
            // One kept-alive socket per target, owned by this thread; the threads' strides
            // are disjoint, so no socket is shared.
            // Back off a failing target exponentially, reset by one success: a wedged tracker
            // costs its timeout every round, which would slow every target in this stride.
            int& skip = skips[t.url];
            if (skip > 0) {
                --skip;
                continue;
            }
            Conn& c = conns[t.url];
            std::string err;
            bool ok = false;
            if (c.fd < 0) {
                c.fd = connect_start(t.addr);
                c.connecting = true;
                c.since = std::chrono::steady_clock::now();
                if (c.fd < 0)
                    err = "connect: " + std::string(std::strerror(errno));
            }
            if (c.fd >= 0 && c.connecting) {
                const int r = connect_poll(c.fd, _post_timeout_ms);
                if (r > 0)
                    c.connecting = false;
                else if (r < 0) {
                    c.fd = -1;
                    c.connecting = false;
                    err = "connect: " + std::string(std::strerror(errno));
                } else if (std::chrono::steady_clock::now() - c.since < kConnectBudget) {
                    continue; // in flight: costs this thread nothing and counts as nothing
                } else {
                    ::close(c.fd);
                    c.fd = -1;
                    c.connecting = false;
                    err = "connect: timed out";
                }
            }
            if (c.fd >= 0)
                ok = http_post(&c.fd, t.host, t.path, it->second.dump(), &err);
            int& nfail = nfails[t.url];
            if (ok)
                nfail = 0;
            else
                skip = std::min(48, 1 << std::min(5, ++nfail)); // <= 32 rounds, ~1.3 s
            std::lock_guard<std::mutex> lk(_mtx);
            _post_reqs++;
            if (ok)
                _post_ok++;
            else {
                _post_fail++;
                _post_last_err = t.url + ": " + err;
            }
        }
        // A target that left this thread's stride (list changed) takes its socket with it:
        // close here rather than hold a connection nothing will use again.
        for (auto f = conns.begin(); f != conns.end();) {
            bool mine = false;
            for (size_t i = (size_t)slot; i < tgt.size() && !mine; i += (size_t)_n_post_threads)
                mine = tgt[i].url == f->first;
            if (mine) {
                ++f;
                continue;
            }
            if (f->second.fd >= 0)
                ::close(f->second.fd);
            skips.erase(f->first);
            nfails.erase(f->first);
            f = conns.erase(f);
        }
    }
    for (auto& f : conns)
        if (f.second.fd >= 0)
            ::close(f.second.fd);
}

void GnssFleetTrim::dll_callback(kotekan::connectionInstance& conn) {
    nlohmann::json reply = nlohmann::json::object();
    std::lock_guard<std::mutex> lk(_mtx);
    for (const auto& cv : _dll.chains()) {
        nlohmann::json prns = nlohmann::json::object();
        for (const auto& pv : cv.second.row) {
            const gnss::FleetDllRow& s = pv.second;
            prns[std::to_string(pv.first)] = {{"disc", s.disc},
                                              {"q", s.q},
                                              {"e_pow", s.e_pow},
                                              {"p_pow", s.p_pow},
                                              {"l_pow", s.l_pow},
                                              {"n_src", s.n_src},
                                              {"n_chan", s.n_chan},
                                              {"n_rec", s.n_rec},
                                              {"hop", s.hop},
                                              {"win", s.win},
                                              {"n_updates", s.n_updates}};
        }
        // The integrator state rides the same reply, so the broker sees the standing
        // correction beside the discriminator. `trim` is walked separately from `row`: a
        // disarmed PRN decaying through the leak can have a trim and no row, and an unarmed
        // PRN with signal has a row and no trim. A missing trim_chips means no standing trim,
        // which differs from trim_chips == 0 (an armed integrator passing through zero).
        for (const auto& tv : cv.second.trim) {
            nlohmann::json& row = prns[std::to_string(tv.first)];
            row["trim_chips"] = tv.second.trim;
            row["trim_steps"] = tv.second.n_steps;
            row["trim_railed"] = tv.second.n_railed;
            row["trim_skipped"] = tv.second.n_skipped;
            row["trim_win"] = tv.second.last_win;
            row["armed"] = cv.second.armed.count(tv.first) != 0;
        }
        // The chain's window-gate floor, so a climbing trim_skipped can be read from here.
        prns["_p_floor_abs"] = cv.second.policy.p_floor_abs;
        reply[cv.first] = prns;
    }
    conn.send_json_reply(reply);
}

/// `<unique_name>/get_taps`: the lobe-coherent and per-channel taps over the newest taps_win
/// windows, the C++ counterpart of combdll.lobe_taps, read by combdll.taps_from_rest so the
/// broker does not walk every (window, sender, record, PRN, channel) in Python.
///
/// Shape: {chain: {prn: {e, p, l, n_chan, n_rec, n_inst, xcoh, n_xcoh, hop,
/// chan: {freq_id: [e, p, l, n_rec]}}}, taps_win}. No sender appears: the coherence unit is the
/// lobe (gnssFleetDll.hpp), and `n_inst` is the completeness of the best record behind the
/// mean, never an operand.
///
/// Measurements only: presence, the noise floor, the deep gate and arming stay in the broker.
/// Records are filtered only by the `min_instances` completeness gate, as in the Python arm;
/// channels two senders both carried in a window are left out.
void GnssFleetTrim::taps_callback(kotekan::connectionInstance& conn) {
    nlohmann::json reply = nlohmann::json::object();
    std::lock_guard<std::mutex> lk(_mtx);
    for (const auto& cv : _dll.taps()) {
        nlohmann::json pj = nlohmann::json::object();
        for (const auto& pv : cv.second) {
            const gnss::FleetDll::LobeTap& t = pv.second;
            nlohmann::json cj = nlohmann::json::object();
            // JSON has no NaN: xcoh is null when no record reached two senders
            const nlohmann::json xc =
                std::isfinite(t.xcoh) ? nlohmann::json(t.xcoh) : nlohmann::json();
            for (const auto& ch : t.chan)
                cj[std::to_string(ch.first)] = {ch.second[0], ch.second[1], ch.second[2],
                                                ch.second[3]};
            pj[std::to_string(pv.first)] = {
                {"e", t.e},         {"p", t.p},           {"l", t.l},   {"n_chan", t.n_chan},
                {"n_rec", t.n_rec}, {"n_inst", t.n_inst}, {"xcoh", xc}, {"n_xcoh", t.n_xcoh},
                {"hop", t.hop},     {"chan", cj}};
        }
        reply[cv.first] = pj;
    }
    reply["taps_win"] = _dll.taps_win();
    conn.send_json_reply(reply);
}

/// `<unique_name>/get_rec_taps`: every record's lobe-coherent E/P/L over the newest taps_win
/// windows, the input of the served C/N0 (combdll.recs_from_rest, for combdll.prompt_cn0),
/// plus each chain's hops_per_record.
///
/// Rows are [win, slot, prn, n_inst, n_chan, e, p, l], flat and time-ordered: the estimator
/// gates and averages records itself, so nothing is reduced here. The probe anchor, the
/// Gamma-mean debias, the q gate and the clip stay in the broker.
void GnssFleetTrim::rec_taps_callback(kotekan::connectionInstance& conn) {
    nlohmann::json reply = nlohmann::json::object();
    nlohmann::json hpr = nlohmann::json::object();
    std::lock_guard<std::mutex> lk(_mtx);
    for (const auto& cv : _dll.rec_series()) {
        nlohmann::json arr = nlohmann::json::array();
        for (const auto& r : cv.second)
            arr.push_back({r.win, r.slot, r.prn, r.n_inst, r.n_chan, r.e, r.p, r.l});
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
                            {"dup_chan", c.n_dup_chan},
                            // >0: the F-engine's frame0 moved and the fold re-anchored;
                            // a steady rise points at the time axis, not the fold.
                            {"epoch_resets", c.n_epoch_reset},
                            {"open_windows", c.open.size()},
                            {"ring", c.closed.size()},
                            {"newest_win", c.newest},
                            {"prns", c.row.size()}};
    }
    reply["chains"] = chains;
    reply["frames"] = _frames;
    reply["bad_frames"] = _bad_frames;
    reply["late_frames"] = _late_frames;
    // Fold cost: fold_us_per_frame x the fleet's frame rate (senders x 23.84 Hz) is the fold's
    // microseconds per second, which must stay well under 1e6 or telem_buf backs up.
    reply["fold_s"] = _fold_s;
    // Trim store: `offered` counts restored trims whose PRN was armed, so offered > 0 with
    // adopted == 0 means the store loads but every trim in it was refused.
    reply["trim_store"] = {{"file", _trim_state_file},
                           {"age_s_at_load", _restored_age_s},
                           {"offered", _restored_offered},
                           {"adopted", _restored_adopted},
                           {"pending", (int)_restored.size()}};
    reply["fold_us_per_frame"] = _frames ? 1e6 * _fold_s / (double)_frames : 0.0;
    reply["close_hz_measured"] = _close_hz;
    reply["policy_posts"] = _policy_posts;
    // A rising policy_expired means a broker chain thread stopped posting. An expired chain is
    // removed from policy_age_s and policy_armed_requested but stays in `policy`.
    reply["policy_expired"] = _policy_expired;
    reply["policy_ttl_s"] = _policy_ttl_s;
    {
        nlohmann::json a = nlohmann::json::object();
        const double now = current_time();
        for (const auto& sv : _policy_seen)
            a[sv.first] = now - sv.second;
        reply["policy_age_s"] = a;
    }
    // What the broker currently asks for, read from _policy: chains whose policy expired are
    // gone here but stay in `policy` below, which reports FleetDll's state.
    {
        nlohmann::json a = nlohmann::json::object();
        for (const auto& pv : _policy)
            a[pv.first] = pv.second.armed.size();
        reply["policy_armed_requested"] = a;
    }
    {
        nlohmann::json a = nlohmann::json::object();
        for (const auto& cv : _dll.chains())
            a[cv.first] = {{"armed", cv.second.armed},
                           {"leak_per_update", cv.second.policy.leak},
                           {"gain", cv.second.policy.gain}};
        reply["policy"] = a;
    }
    reply["post_targets"] = _targets.size();
    reply["post_threads"] = _post_threads.size();
    reply["post_rounds"] = _post_rounds;
    reply["post_requests"] = _post_reqs;
    reply["post_ok"] = _post_ok;
    reply["post_fail"] = _post_fail;
    reply["post_last_err"] = _post_last_err;
    reply["n_win"] = _dll.n_win();
    reply["min_instances"] = _dll.min_instances();
    conn.send_json_reply(reply);
}
