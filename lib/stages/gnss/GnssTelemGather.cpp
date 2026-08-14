#include "GnssTelemGather.hpp"

#include "StageFactory.hpp"
#include "gnssTelem.hpp"
#include "kotekanLogging.hpp"
#include "visUtil.hpp" // for frameID, current_time

#include "json.hpp"

#include <algorithm> // for std::remove
#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <functional>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#ifndef MSG_NOSIGNAL
#define MSG_NOSIGNAL 0
#endif

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssTelemGather);

GnssTelemGather::GnssTelemGather(Config& config, const std::string& unique_name,
                                 bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&GnssTelemGather::main_thread, this)) {
    in_buf = get_buffer("in_buf");
    in_buf->register_consumer(unique_name);

    _serve_host = config.get_default<std::string>(unique_name, "serve_host", "127.0.0.1");
    _serve_port = config.get_default<int>(unique_name, "serve_port", 11061);
    _send_timeout_ms = config.get_default<int>(unique_name, "send_timeout_ms", 200);

    kotekan::restServer::instance().register_get_callback(
        unique_name + "/get_stats",
        std::bind(&GnssTelemGather::stats_callback, this, std::placeholders::_1));
}

GnssTelemGather::~GnssTelemGather() {
    _accepting = false;
    if (_listen_fd >= 0) {
        ::shutdown(_listen_fd, SHUT_RDWR);
        ::close(_listen_fd);
        _listen_fd = -1;
    }
    if (_accept_thread.joinable())
        _accept_thread.join();
    std::lock_guard<std::mutex> lk(_client_mtx);
    for (int fd : _clients)
        ::close(fd);
    _clients.clear();
}

void GnssTelemGather::accept_loop() {
    while (_accepting && !stop_thread) {
        struct pollfd pfd = {_listen_fd, POLLIN, 0};
        const int r = ::poll(&pfd, 1, 500);
        if (r <= 0)
            continue;
        struct sockaddr_in peer;
        socklen_t len = sizeof(peer);
        const int fd = ::accept(_listen_fd, (struct sockaddr*)&peer, &len);
        if (fd < 0)
            continue;
        int one = 1;
        // Frames are 16 kB and the reader wants them promptly; Nagle would coalesce them into
        // the next write's latency for no benefit on a local link.
        ::setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
        {
            std::lock_guard<std::mutex> lk(_client_mtx);
            _clients.push_back(fd);
        }
        INFO("GnssTelemGather[{:s}]: client connected from {:s} (fd {:d})", unique_name,
             inet_ntoa(peer.sin_addr), fd);
    }
}

bool GnssTelemGather::send_all(int fd, const uint8_t* p, size_t n) {
    size_t sent = 0;
    const double deadline = current_time() + 1e-3 * _send_timeout_ms;
    while (sent < n) {
        const ssize_t w = ::send(fd, p + sent, n - sent, MSG_NOSIGNAL | MSG_DONTWAIT);
        if (w > 0) {
            sent += (size_t)w;
            continue;
        }
        if (w < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) {
            const double left = deadline - current_time();
            if (left <= 0.0)
                return false; // too slow: the caller drops the client rather than half-write
            struct pollfd pfd = {fd, POLLOUT, 0};
            if (::poll(&pfd, 1, (int)(1e3 * left) + 1) <= 0)
                return false;
            continue;
        }
        return false; // peer closed, or a real error
    }
    return true;
}

void GnssTelemGather::broadcast(const uint8_t* frame, size_t n) {
    std::vector<int> dead;
    {
        std::lock_guard<std::mutex> lk(_client_mtx);
        if (_clients.empty())
            return;
        const uint32_t len = (uint32_t)n;
        for (int fd : _clients) {
            if (!send_all(fd, (const uint8_t*)&len, sizeof(len)) || !send_all(fd, frame, n))
                dead.push_back(fd);
        }
        if (!dead.empty()) {
            for (int fd : dead) {
                _clients.erase(std::remove(_clients.begin(), _clients.end(), fd), _clients.end());
                ::close(fd);
                _client_drops++;
            }
        }
    }
    for (int fd : dead)
        WARN("GnssTelemGather[{:s}]: dropped client fd {:d} -- it could not take a frame within "
             "{:d} ms. A frame is delivered whole or not at all; a partial write would "
             "desynchronise the stream silently.",
             unique_name, fd, _send_timeout_ms);
}

void GnssTelemGather::main_thread() {
    // Listener first, so a broker that starts before us just retries rather than seeing a
    // half-open port.
    _listen_fd = ::socket(AF_INET, SOCK_STREAM, 0);
    if (_listen_fd < 0) {
        FATAL_ERROR("GnssTelemGather[{:s}]: socket(): {:s}", unique_name, strerror(errno));
        return;
    }
    int one = 1;
    ::setsockopt(_listen_fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    struct sockaddr_in addr;
    std::memset(&addr, 0, sizeof(addr));
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)_serve_port);
    addr.sin_addr.s_addr = inet_addr(_serve_host.c_str());
    if (::bind(_listen_fd, (struct sockaddr*)&addr, sizeof(addr)) < 0
        || ::listen(_listen_fd, 8) < 0) {
        FATAL_ERROR("GnssTelemGather[{:s}]: cannot listen on {:s}:{:d}: {:s}", unique_name,
                    _serve_host, _serve_port, strerror(errno));
        return;
    }
    _accepting = true;
    _accept_thread = std::thread(&GnssTelemGather::accept_loop, this);
    INFO("GnssTelemGather[{:s}]: serving telemetry frames on {:s}:{:d}", unique_name, _serve_host,
         _serve_port);

    frameID in_id(in_buf);
    while (!stop_thread) {
        uint8_t* frame = in_buf->wait_for_full_frame(unique_name, in_id);
        if (frame == nullptr)
            break;

        const auto* h = (const gnss::TelemHeader*)frame;
        // A sender built against a different record layout, or something that is not one of our
        // frames at all, must be REJECTED rather than forwarded: the broker would parse the
        // rows at the wrong stride and every number it produced would be plausible and wrong.
        const bool ok = h->magic == gnss::TELEM_MAGIC && h->version == gnss::TELEM_VERSION
                        && h->n_row == gnss::RECORD_FLOATS && h->n_rec > 0
                        && h->n_rec <= gnss::TELEM_MAX_REC && h->n_prn > 0
                        && gnss::telem_frame_bytes(h->n_rec, h->n_prn)
                               == (size_t)in_buf->frame_size;
        if (!ok) {
            if ((_bad_frames++ % 100) == 0)
                ERROR("GnssTelemGather[{:s}]: rejecting frame (magic {:#x} v{:d} n_rec {:d} "
                      "n_prn {:d} n_row {:d} vs RECORD_FLOATS {:d}, buffer frame {:d} B) -- {:d} "
                      "so far. A sender is on a different build or a different max_prn.",
                      unique_name, h->magic, h->version, h->n_rec, h->n_prn, h->n_row,
                      gnss::RECORD_FLOATS, (size_t)in_buf->frame_size, _bad_frames);
            in_buf->mark_frame_empty(unique_name, in_id++);
            continue;
        }

        {
            const std::string chain(h->chain, strnlen(h->chain, gnss::TELEM_NAME));
            const std::string inst(h->inst, strnlen(h->inst, gnss::TELEM_NAME));
            std::lock_guard<std::mutex> lk(_stat_mtx);
            Sender& s = _senders[chain + "/" + inst];
            // SEQUENCE GAPS, not a rate. A sender missing every fourth frame still shows a
            // healthy-looking rate; the sender's own counter is the only thing that can say a
            // frame was lost, and it is on the wire for exactly this reason.
            if (s.frames > 0 && h->seq > s.last_seq + 1)
                s.gaps += h->seq - s.last_seq - 1;
            s.frames++;
            s.last_seq = h->seq;
            s.last_win = (int64_t)h->win;
            s.last_utc = h->utc0;
            s.last_rx = current_time();
            s.n_present = h->present;
        }

        broadcast(frame, (size_t)in_buf->frame_size);
        in_buf->mark_frame_empty(unique_name, in_id++);
    }
}

void GnssTelemGather::stats_callback(kotekan::connectionInstance& conn) {
    nlohmann::json reply;
    const double now = current_time();
    {
        std::lock_guard<std::mutex> lk(_stat_mtx);
        nlohmann::json senders = nlohmann::json::array();
        // THE ALIGNMENT CHECK, served rather than inferred: every sender's most recent window
        // index, side by side. Equal (or within one) across a chain is the transport working;
        // a spread is the thing this stage exists to make visible.
        for (const auto& kv : _senders)
            senders.push_back({{"key", kv.first},
                               {"frames", kv.second.frames},
                               {"gaps", kv.second.gaps},
                               {"last_seq", kv.second.last_seq},
                               {"last_win", kv.second.last_win},
                               {"last_utc", kv.second.last_utc},
                               {"age_s", now - kv.second.last_rx},
                               {"present", kv.second.n_present}});
        reply["senders"] = senders;
        reply["bad_frames"] = _bad_frames;
        reply["client_drops"] = _client_drops;
    }
    {
        std::lock_guard<std::mutex> lk(_client_mtx);
        reply["clients"] = _clients.size();
    }
    reply["frame_bytes"] = (size_t)in_buf->frame_size;
    reply["serve"] = _serve_host + ":" + std::to_string(_serve_port);
    conn.send_json_reply(reply);
}
