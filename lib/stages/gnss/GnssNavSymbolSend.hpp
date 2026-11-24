/**
 * @file
 * @brief The nav-symbol feed: GnssNavSymbols' decisions to a TCP stream of NSYM records.
 *  - GnssNavSymbolSend : public kotekan::Stage
 */

#ifndef GNSS_NAV_SYMBOL_SEND_STAGE_HPP
#define GNSS_NAV_SYMBOL_SEND_STAGE_HPP

#include "Config.hpp"          // for Config
#include "Stage.hpp"           // for Stage
#include "buffer.hpp"          // for Buffer
#include "bufferContainer.hpp" // for bufferContainer
#include "gnssNavSym.hpp"      // for NavSymAnchor
#include "restServer.hpp"      // for connectionInstance

#include "json.hpp" // for json

#include <condition_variable> // for condition_variable
#include <cstdint>            // for int64_t, uint8_t, uint32_t
#include <deque>              // for deque
#include <map>                // for map
#include <mutex>              // for mutex
#include <string>             // for string
#include <tuple>              // for tuple
#include <vector>             // for vector

/**
 * @class GnssNavSymbolSend
 * @brief Each anchored stream's symbols, as GnssNavSymbols decides them, to a TCP server as NSYM
 *        records (gnss::nsym_record): the nav-symbol feed to CHORD.
 *
 * GnssNavSymbols decides each symbol up to one sign per stream and counts it in the stream. A
 * stream is sent once it has an anchor: one of its symbols, its satellite-time index (CHORD's
 * key: floor(t / T_sym), t the satellite clock's transmit time in s since 1980-01-06 on the
 * system's own scale) and the stream's sign, which nav_symbol_anchor.py finds from the decoded
 * messages and posts here. Symbol i then goes out as index sv + (i - k) with sign x sign, and
 * the records are flagged resolved. A row of an unanchored stream is dropped.
 *
 * One reader thread per input takes its frames as they fill; the stage's thread writes the
 * records, one per row, numbered per connection, and a heartbeat after @c heartbeat_s with
 * nothing to send. A send that fails or takes over @c send_timeout_s closes the connection,
 * which is retried every @c reconnect_time; meanwhile the records are dropped and counted, and
 * the inputs are never held back.
 *
 * @par REST
 * POST `<unique_name>/set_anchors`: [{sig, prn, stream_id, k, sv, sign}], each replacing the
 * anchor of that stream. GET `<unique_name>/get_status`: the connection, counts, the anchors and
 * the streams seen without one.
 *
 * @buffer in_bufs GnssNavSymbols out_bufs, one per signal.
 *     @buffer_format gnss::nav_sym_frame_bytes(n_prn) (gnssNavSym.hpp)
 *     @buffer_metadata none
 *
 * @conf sys            Array<String>. Each input's constellation letter (G, E, C, R, S, W).
 * @conf sig            Array<Int>. Each input's NSYM signal code.
 * @conf server_ip      String (default "127.0.0.1").
 * @conf server_port    Int (default 12080).
 * @conf reconnect_time Double (default 5). Seconds between connection attempts.
 * @conf send_timeout_s Double (default 1). A send slower than this closes the connection.
 * @conf heartbeat_s    Double (default 1). Idle time before a heartbeat.
 * @conf queue_max      Int (default 4096). Records held for the writer; the oldest beyond are
 *                      dropped.
 * @conf anchor_ttl_s   Double (default 600). An anchor whose stream has sent nothing this long
 *                      is forgotten.
 */
class GnssNavSymbolSend : public kotekan::Stage {
public:
    GnssNavSymbolSend(kotekan::Config& config, const std::string& unique_name,
                      kotekan::bufferContainer& buffer_container);
    ~GnssNavSymbolSend() override;
    void main_thread() override;

private:
    /// One input's frames to the queue, until the stage stops.
    void read_input(size_t idx);
    void set_anchors_callback(kotekan::connectionInstance& conn, nlohmann::json& request);
    void get_status_callback(kotekan::connectionInstance& conn);
    bool try_connect();
    bool send_all(const std::vector<uint8_t>& bytes);
    void close_connection();

    /// A record waiting for the writer.
    struct Pending {
        size_t idx;
        int prn;
        int64_t stream_id;
        int64_t sym0;
        std::vector<int8_t> signs;
        double t_dec;
        double rx_utc;
    };
    using Key = std::tuple<size_t, int, int64_t>; ///< (input, prn, stream_id)

    std::vector<Buffer*> _in_bufs;
    std::vector<char> _sys;
    std::vector<int> _sig;
    std::string _server_ip;
    int _server_port;
    double _reconnect_s, _send_timeout_s, _heartbeat_s, _anchor_ttl_s;
    size_t _queue_max;

    std::mutex _m; ///< guards everything below
    std::condition_variable _cv;
    std::map<Key, std::pair<gnss::NavSymAnchor, double>> _anchors; ///< and when last used
    std::map<std::pair<size_t, int>, int64_t> _unanchored;         ///< streams seen without one
    std::deque<Pending> _queue;
    int _fd = -1;
    uint32_t _seq = 0;
    long long _n_records = 0, _n_symbols = 0, _n_heartbeats = 0;
    long long _n_dropped_queue = 0, _n_dropped_down = 0, _n_connects = 0;
};

#endif // GNSS_NAV_SYMBOL_SEND_STAGE_HPP
