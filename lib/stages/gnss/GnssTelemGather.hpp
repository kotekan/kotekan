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
 * @brief GATHER-HOST SIDE of the frame-synced tracker->broker transport (task #59): every
 *        instance's telemetry frames in, one local byte stream out to the Python broker.
 *
 * Sits behind a single @c bufferRecv that all senders connect to (the payload carries its own
 * chain/instance tags, so one listener serves the whole fleet and there is no per-chain port
 * map to wire up crooked). For each frame it validates the wire header and writes the frame
 * verbatim, length-prefixed, to every connected client.
 *
 * ⚠️ IT COLLATES NOTHING, DELIBERATELY. What this stage exists to provide is IDENTITY and
 * DELIVERY -- which sender, which window, with no inference -- and those ride in the payload.
 * Every frame already carries the absolute window index that makes collation an exact integer
 * grouping, so a collator here would buy the broker nothing it cannot do itself.
 *
 * ⚠️ THAT IS NOT THE SAME AS "no C++ may collate" (updated 2026-08-15, task #51).
 * @ref GnssFleetTrim now rides beside this stage as a second consumer of the same buffer and
 * does collate -- because the CODE LOOP needs a fleet-wide view at frame rate, and this process
 * is the only one that has one (a tracker instance sees ~7 of the fleet's ~105 channels). The
 * rule that survives is ONE PLACE WITH THE FLEET-WIDE VIEW OWNS THE LOOP: policy -- who is
 * armed, every gate, the clock, the ephemeris -- stays in the Python broker, and only the
 * discriminator, the integrator and the actuation post moved here. docs/CHORD_FAST_TRIM.md.
 *
 * WIRE PROTOCOL TO THE BROKER: a stream of `[uint32 little-endian length][length bytes]`, where
 * the bytes are the telemetry frame exactly as it arrived (gnssTelem.hpp). The length prefix is
 * redundant with the fixed frame size on purpose: it lets a client resynchronise, and it lets
 * the frame size change without every reader needing to be rebuilt at the same instant.
 *
 * A client that cannot keep up is DISCONNECTED, never half-written: a frame is delivered whole
 * or the connection is closed. A partially written frame would desynchronise the stream for
 * good, and silently -- the reader would keep parsing, just offset, which is the failure this
 * whole change exists to stop tolerating.
 *
 * @par buffers
 * @buffer in_buf  telemetry frames as received (producer: bufferRecv)
 *
 * @conf serve_host   String, default "127.0.0.1". Bind address for the broker stream. The
 *                      broker runs on this host; do not widen it without a reason.
 * @conf serve_port   Int, default 11061.
 * @conf send_timeout_ms Int, default 200. How long a single frame may take to hand to one
 *                      client before that client is dropped.
 * @conf stale_after_s   Double, default 5.0. A sender silent this long is STALE: it is
 *                      announced once in the log, excluded from the alignment verdict, and
 *                      rejoins on its own. 0 disables the check.
 *
 * @par REST
 * `<unique_name>/get_stats` -- per-sender frame counts, last window, sequence gaps, and the
 * client list. This is the "is the transport healthy?" endpoint; it deliberately reports
 * SEQUENCE GAPS rather than a rate, because a rate that looks right can still be missing every
 * fourth frame.
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
    int _send_timeout_ms = 200;
    /// A sender silent this long is STALE. Nothing here blocks on a sender -- the gather has no
    /// barrier by construction -- but until this existed nothing NOTICED one stopping either,
    /// which is worse: cx43's GPU-0 chain died 2026-08-14 and only a hand poll found it.
    double _stale_after_s = 5.0;

    /// Listener + accepted clients. Touched by the accept thread and the main thread.
    int _listen_fd = -1;
    std::vector<int> _clients;
    std::mutex _client_mtx;
    std::thread _accept_thread;
    std::atomic<bool> _accepting{false};

    /// Per-sender bookkeeping, keyed "chain/inst".
    struct Sender {
        uint64_t frames = 0;
        uint64_t gaps = 0;      ///< missed frames inferred from the sender's own seq counter
        uint64_t last_seq = 0;
        int64_t last_win = -1;
        double last_utc = 0.0;
        double last_rx = 0.0;   ///< host time of the last frame from this sender
        uint32_t n_present = 0; ///< record slots filled in the last frame
        bool stale = false;     ///< silent for longer than stale_after_s
        uint64_t stalls = 0;    ///< times this sender has gone stale (churn is itself a signal)
    };
    std::map<std::string, Sender> _senders;
    std::mutex _stat_mtx;

    uint64_t _bad_frames = 0;   ///< failed the magic/version/geometry check
    uint64_t _client_drops = 0; ///< clients disconnected for being too slow

    /// Mark senders stale/live and log each TRANSITION once. Called on a timer, not only when
    /// a frame arrives -- a fleet that goes completely silent must still be reported, and that
    /// is exactly the case a frame-driven loop cannot see.
    void sweep_stale();

    void accept_loop();
    /// Write the whole buffer or fail. Returns false if the client should be dropped.
    bool send_all(int fd, const uint8_t* p, size_t n);
    void broadcast(const uint8_t* frame, size_t n);
    void stats_callback(kotekan::connectionInstance& conn);
};

#endif // GNSS_TELEM_GATHER_HPP
