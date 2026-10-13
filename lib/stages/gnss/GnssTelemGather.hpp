#ifndef GNSS_TELEM_GATHER_HPP
#define GNSS_TELEM_GATHER_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "restServer.hpp"

#include <atomic>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

/**
 * @class GnssTelemGather
 * @brief Gather-host side of the tracker-to-broker telemetry transport: every instance's
 *        telemetry frames in, one local byte stream out to the Python broker.
 *
 * Sits behind a single @c bufferRecv that all senders connect to; each frame carries its own
 * chain and instance tags, so one listener serves the whole fleet. Each frame's wire header is
 * validated and the frame is written verbatim, length-prefixed, to every connected client.
 *
 * This stage does no collation. Every frame carries the absolute window index, so the broker
 * groups instances by exact integer match. @ref GnssFleetTrim is a second consumer of the same
 * buffer that does collate, for the fleet-wide code loop (docs/CHORD_FAST_TRIM.md).
 *
 * Wire protocol to the broker: a stream of `[uint32 little-endian length][length bytes]`, where
 * the bytes are the telemetry frame as it arrived (gnssTelem.hpp). Frame sizes differ between
 * senders; the prefix lets a reader split the stream without parsing headers.
 *
 * A client that cannot take a frame within 200 ms is disconnected rather than left
 * with a partial frame, which would offset every later read without any error.
 *
 * @par buffers
 * @buffer in_buf  telemetry frames as received (producer: bufferRecv)
 *
 * @conf serve_host      String, default "127.0.0.1". Bind address for the broker stream.
 * @conf serve_port      Int, default 11061.
 * @conf stale_after_s   Double, default 5.0, > 0. A sender silent this long is stale: logged
 *                       once, left out of the alignment spread, and live again on its next frame.
 *
 * @par REST
 * `<unique_name>/get_stats` -- per-sender frame counts, sequence gaps, last window and age;
 * per-chain window spread over live senders, with stale senders named; bad frames, client
 * drops and hangups, and the client count. Loss is reported as sequence gaps rather than a
 * rate, because a rate can look right while frames are missing.
 */
class GnssTelemGather : public kotekan::Stage {
public:
    GnssTelemGather(kotekan::Config& config, const std::string& unique_name,
                    kotekan::bufferContainer& buffer_container);
    ~GnssTelemGather() override;
    void main_thread() override;

private:
    Buffer* in_buf;

    std::string _serve_host;
    int _serve_port = 11061;
    /// A client that cannot take a write (the length prefix or the frame) within this long is
    /// dropped.
    static constexpr int send_deadline_ms = 200;
    double _stale_after_s = 5.0;

    /// Listener and accepted clients. @c _clients is guarded by @c _client_mtx; it is touched by
    /// the accept thread, the main thread and the REST callback.
    int _listen_fd = -1;
    std::vector<int> _clients;
    std::mutex _client_mtx;
    std::thread _accept_thread;
    std::atomic<bool> _accepting{false};

    /// Per-sender bookkeeping, keyed "chain/inst", guarded by @c _stat_mtx.
    struct Sender {
        uint64_t frames = 0;
        uint64_t gaps = 0; ///< missed frames inferred from the sender's own seq counter
        uint64_t last_seq = 0;
        int64_t last_win = -1;
        double last_utc = 0.0;
        double last_rx = 0.0;   ///< host time of the last frame from this sender
        uint32_t n_present = 0; ///< record-slot bitmask (`present`) of the last frame
        bool stale = false;     ///< silent for longer than stale_after_s
        uint64_t stalls = 0;    ///< times this sender has gone stale
    };
    std::map<std::string, Sender> _senders;
    std::mutex _stat_mtx;

    uint64_t _bad_frames = 0;     ///< failed the magic/version/shape check
    uint64_t _client_drops = 0;   ///< clients dropped on a failed or timed-out write
    uint64_t _client_hangups = 0; ///< clients reaped after their peer closed
    double _last_reap = 0.0;      ///< rate limit for reap_clients: it polls every client fd

    /// Mark senders silent for stale_after_s as stale, logging each transition once. Runs on
    /// the 1 s wait timeout as well as per frame, so a fleet that goes silent is still
    /// reported. A stale sender is marked live again when its next frame arrives.
    void sweep_stale();

    /// Close clients whose peer has gone away. A failed write drops a client only while frames
    /// flow; with no senders nothing is written, and closed sockets would accumulate in
    /// CLOSE_WAIT until accept() fails with EMFILE. Runs at most once a second, from the same
    /// heartbeat as sweep_stale().
    void reap_clients();

    void accept_loop();
    /// Write the whole buffer or fail. Returns false if the client should be dropped.
    bool send_all(int fd, const uint8_t* p, size_t n);
    void broadcast(const uint8_t* frame, size_t n);
    void stats_callback(kotekan::connectionInstance& conn);
};

#endif // GNSS_TELEM_GATHER_HPP
