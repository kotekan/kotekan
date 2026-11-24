#include "GnssNavSymbolSend.hpp"

#include "StageFactory.hpp"   // for REGISTER_KOTEKAN_STAGE
#include "kotekanLogging.hpp" // for FATAL_ERROR, INFO, WARN
#include "visUtil.hpp"        // for current_time

#include <arpa/inet.h>   // for inet_pton, htons
#include <cerrno>        // for errno, EINPROGRESS, EINTR, ETIMEDOUT
#include <chrono>        // for duration
#include <cstring>       // for memcpy, strerror
#include <fcntl.h>       // for fcntl, F_GETFL, F_SETFL, O_NONBLOCK
#include <functional>    // for bind
#include <iterator>      // for next
#include <netinet/in.h>  // for sockaddr_in, IPPROTO_TCP
#include <netinet/tcp.h> // for TCP_NODELAY
#include <poll.h>        // for poll, pollfd, POLLOUT
#include <stdexcept>     // for runtime_error
#include <sys/socket.h>  // for socket, connect, send, setsockopt
#include <sys/time.h>    // for timeval
#include <thread>        // for thread
#include <unistd.h>      // for close

using kotekan::bufferContainer;
using kotekan::Config;
using kotekan::Stage;

REGISTER_KOTEKAN_STAGE(GnssNavSymbolSend);

GnssNavSymbolSend::GnssNavSymbolSend(Config& config, const std::string& unique_name,
                                     bufferContainer& buffer_container) :
    Stage(config, unique_name, buffer_container, std::bind(&GnssNavSymbolSend::main_thread, this)) {
    _in_bufs = get_buffer_array("in_bufs");
    for (Buffer* b : _in_bufs)
        b->register_consumer(unique_name);
    const auto sys = config.get<std::vector<std::string>>(unique_name, "sys");
    _sig = config.get<std::vector<int>>(unique_name, "sig");
    if (sys.size() != _in_bufs.size() || _sig.size() != _in_bufs.size())
        FATAL_ERROR("GnssNavSymbolSend[{:s}]: sys and sig need one entry per in_buf", unique_name);
    for (const auto& s : sys) {
        if (s.size() != 1)
            FATAL_ERROR("GnssNavSymbolSend[{:s}]: sys entries are one letter, not '{:s}'",
                        unique_name, s);
        _sys.push_back(s[0]);
    }
    _server_ip = config.get_default<std::string>(unique_name, "server_ip", "127.0.0.1");
    _server_port = config.get_default<int>(unique_name, "server_port", 12080);
    _reconnect_s = config.get_default<double>(unique_name, "reconnect_time", 5.0);
    _send_timeout_s = config.get_default<double>(unique_name, "send_timeout_s", 1.0);
    _heartbeat_s = config.get_default<double>(unique_name, "heartbeat_s", 1.0);
    _anchor_ttl_s = config.get_default<double>(unique_name, "anchor_ttl_s", 600.0);
    _queue_max = (size_t)config.get_default<int>(unique_name, "queue_max", 4096);

    kotekan::restServer::instance().register_post_callback(
        unique_name + "/set_anchors", std::bind(&GnssNavSymbolSend::set_anchors_callback, this,
                                                std::placeholders::_1, std::placeholders::_2));
    kotekan::restServer::instance().register_get_callback(
        unique_name + "/get_status",
        std::bind(&GnssNavSymbolSend::get_status_callback, this, std::placeholders::_1));
}

GnssNavSymbolSend::~GnssNavSymbolSend() {
    kotekan::restServer::instance().remove_json_callback(unique_name + "/set_anchors");
    kotekan::restServer::instance().remove_get_callback(unique_name + "/get_status");
    close_connection();
}

void GnssNavSymbolSend::read_input(size_t idx) {
    Buffer* buf = _in_bufs[idx];
    int id = 0;
    while (!stop_thread) {
        const uint8_t* frame = buf->wait_for_full_frame(unique_name, id);
        if (frame == nullptr)
            break;
        int32_t n_rows = 0;
        std::memcpy(&n_rows, frame, sizeof(n_rows));
        {
            std::lock_guard<std::mutex> lk(_m);
            const double now = current_time();
            for (int32_t r = 0; r < n_rows; ++r) {
                gnss::NavSymRow row;
                std::memcpy(&row, frame + 8 + (size_t)r * sizeof(row), sizeof(row));
                auto it = _anchors.find(Key{idx, row.prn, row.stream_id});
                if (it == _anchors.end()) {
                    _unanchored[{idx, row.prn}] = row.stream_id;
                    continue;
                }
                _unanchored.erase({idx, row.prn});
                const gnss::NavSymAnchor& a = it->second.first;
                it->second.second = now;
                Pending p{idx,
                          row.prn,
                          row.stream_id,
                          a.sv + (row.first - a.k),
                          {},
                          row.t_dec,
                          row.t0 + (double)row.first * row.sym_dt};
                for (int i = 0; i < row.n && i < gnss::NAV_SYM_MAX; ++i)
                    p.signs.push_back((int8_t)(row.sign[i] * a.sign));
                if (_queue.size() >= _queue_max) {
                    _queue.pop_front();
                    ++_n_dropped_queue;
                }
                _queue.push_back(std::move(p));
            }
        }
        _cv.notify_one();
        buf->mark_frame_empty(unique_name, id);
        id = (id + 1) % buf->num_frames;
    }
}

bool GnssNavSymbolSend::try_connect() {
    const int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0)
        return false;
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons((uint16_t)_server_port);
    if (inet_pton(AF_INET, _server_ip.c_str(), &addr.sin_addr) != 1)
        FATAL_ERROR("GnssNavSymbolSend[{:s}]: server_ip '{:s}' is not an IPv4 address", unique_name,
                    _server_ip);
    // A non-blocking connect, so that an unreachable server holds the writer for at most the
    // send timeout.
    const int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    int rc = connect(fd, (sockaddr*)&addr, sizeof(addr));
    if (rc < 0 && errno == EINPROGRESS) {
        pollfd pfd{fd, POLLOUT, 0};
        int err = ETIMEDOUT;
        socklen_t len = sizeof(err);
        if (poll(&pfd, 1, (int)(_send_timeout_s * 1000.0)) == 1)
            getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &len);
        rc = err == 0 ? 0 : -1;
        errno = err; // the connect's own error, for the caller's log
    }
    if (rc < 0) {
        close(fd);
        return false;
    }
    fcntl(fd, F_SETFL, flags);
    timeval tv{(time_t)_send_timeout_s,
               (suseconds_t)((_send_timeout_s - (double)(time_t)_send_timeout_s) * 1e6)};
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
    const int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
    _fd = fd;
    _seq = 0;
    return true;
}

bool GnssNavSymbolSend::send_all(const std::vector<uint8_t>& bytes) {
    size_t sent = 0;
    while (sent < bytes.size()) {
        const ssize_t n = send(_fd, bytes.data() + sent, bytes.size() - sent, MSG_NOSIGNAL);
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            return false;
        sent += (size_t)n;
    }
    return true;
}

void GnssNavSymbolSend::close_connection() {
    if (_fd >= 0)
        close(_fd);
    _fd = -1;
}

void GnssNavSymbolSend::main_thread() {
    std::vector<std::thread> readers;
    for (size_t i = 0; i < _in_bufs.size(); ++i)
        readers.emplace_back(&GnssNavSymbolSend::read_input, this, i);

    double t_retry = 0.0, t_last = current_time();
    long long n_fail = 0;
    while (!stop_thread) {
        if (_fd < 0 && current_time() >= t_retry) {
            if (try_connect()) {
                INFO("GnssNavSymbolSend[{:s}]: connected to {:s}:{:d}", unique_name, _server_ip,
                     _server_port);
                std::lock_guard<std::mutex> lk(_m);
                ++_n_connects;
                n_fail = 0;
            } else {
                if (n_fail++ % 60 == 0)
                    WARN("GnssNavSymbolSend[{:s}]: {:s}:{:d} unreachable ({:s}); dropping "
                         "records until it answers",
                         unique_name, _server_ip, _server_port, std::strerror(errno));
                t_retry = current_time() + _reconnect_s;
            }
        }
        std::deque<Pending> batch;
        {
            std::unique_lock<std::mutex> lk(_m);
            _cv.wait_for(lk, std::chrono::duration<double>(_heartbeat_s / 4.0),
                         [this] { return !_queue.empty() || stop_thread; });
            batch.swap(_queue);
            // Forget the anchors of streams that have gone.
            const double now = current_time();
            for (auto it = _anchors.begin(); it != _anchors.end();)
                it = now - it->second.second > _anchor_ttl_s ? _anchors.erase(it) : std::next(it);
        }
        bool ok = true;
        for (const Pending& p : batch) {
            if (_fd < 0 || !ok) {
                std::lock_guard<std::mutex> lk(_m);
                ++_n_dropped_down;
                continue;
            }
            const auto rec =
                gnss::nsym_record(_seq, _sys[p.idx], (uint8_t)_sig[p.idx], (uint16_t)p.prn,
                                  (uint32_t)(p.stream_id & 0xFFFFFFFF), true, p.sym0, p.signs,
                                  p.t_dec, current_time(), p.rx_utc);
            ok = send_all(rec);
            if (ok) {
                ++_seq;
                t_last = current_time();
                std::lock_guard<std::mutex> lk(_m);
                ++_n_records;
                _n_symbols += (long long)p.signs.size();
            }
        }
        if (_fd >= 0 && ok && current_time() - t_last >= _heartbeat_s) {
            ok = send_all(
                gnss::nsym_record(_seq, 0, 0, 0, 0, false, 0, {}, 0.0, current_time(), 0.0));
            if (ok) {
                ++_seq;
                t_last = current_time();
                std::lock_guard<std::mutex> lk(_m);
                ++_n_heartbeats;
            }
        }
        if (_fd >= 0 && !ok) {
            WARN("GnssNavSymbolSend[{:s}]: send to {:s}:{:d} failed ({:s}); reconnecting",
                 unique_name, _server_ip, _server_port, std::strerror(errno));
            close_connection();
            t_retry = current_time() + _reconnect_s;
        }
    }
    for (auto& t : readers)
        t.join();
}

void GnssNavSymbolSend::set_anchors_callback(kotekan::connectionInstance& conn,
                                             nlohmann::json& request) {
    try {
        std::lock_guard<std::mutex> lk(_m);
        const double now = current_time();
        for (const auto& a : request) {
            const int sig = a.at("sig").get<int>();
            size_t idx = 0;
            while (idx < _sig.size() && _sig[idx] != sig)
                ++idx;
            if (idx == _sig.size())
                throw std::runtime_error("no input sends sig " + std::to_string(sig));
            const int sign = a.at("sign").get<int>();
            if (sign != 1 && sign != -1)
                throw std::runtime_error("sign must be +1 or -1");
            _anchors[Key{idx, a.at("prn").get<int>(), a.at("stream_id").get<int64_t>()}] = {
                gnss::NavSymAnchor{a.at("k").get<int64_t>(), a.at("sv").get<int64_t>(), sign}, now};
        }
    } catch (const std::exception& e) {
        conn.send_error(std::string("bad anchors: ") + e.what(),
                        kotekan::HTTP_RESPONSE::BAD_REQUEST);
        return;
    }
    conn.send_empty_reply(kotekan::HTTP_RESPONSE::OK);
}

void GnssNavSymbolSend::get_status_callback(kotekan::connectionInstance& conn) {
    nlohmann::json out;
    {
        std::lock_guard<std::mutex> lk(_m);
        const double now = current_time();
        out["server"] = _server_ip + ":" + std::to_string(_server_port);
        out["connected"] = _fd >= 0;
        out["connects"] = _n_connects;
        out["seq"] = _seq;
        out["records"] = _n_records;
        out["symbols"] = _n_symbols;
        out["heartbeats"] = _n_heartbeats;
        out["dropped_queue"] = _n_dropped_queue;
        out["dropped_down"] = _n_dropped_down;
        nlohmann::json anchors = nlohmann::json::array();
        for (const auto& [key, v] : _anchors)
            anchors.push_back({{"sig", _sig[std::get<0>(key)]},
                               {"prn", std::get<1>(key)},
                               {"stream_id", std::get<2>(key)},
                               {"k", v.first.k},
                               {"sv", v.first.sv},
                               {"sign", v.first.sign},
                               {"idle_s", now - v.second}});
        out["anchors"] = anchors;
        nlohmann::json un = nlohmann::json::array();
        for (const auto& [key, sid] : _unanchored)
            un.push_back({{"sig", _sig[key.first]}, {"prn", key.second}, {"stream_id", sid}});
        out["unanchored"] = un;
    }
    conn.send_json_reply(out);
}
