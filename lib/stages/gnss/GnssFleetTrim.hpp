#ifndef GNSS_FLEET_TRIM_HPP
#define GNSS_FLEET_TRIM_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "gnssFleetDll.hpp"
#include "restServer.hpp"

#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

/**
 * @class GnssFleetTrim
 * @brief The fleet-wide fast code loop: forms the fleet code discriminator from the gathered
 *        telemetry, integrates it, and posts the trims to the tracker instances.
 *
 * The discriminator's authority is a step per update, not a rate:
 *
 *     tau  = -clamp(disc, +-1)/4 * (spacing/0.5)   =>  |tau| <= 0.25 chips at 0.5 spacing
 *     step = gain * tau                            =>  <= 0.0625 chips at gain 0.25
 *
 * Holding against CHORD's ~0.12 chips/s of code drift therefore needs ~2 updates per second,
 * which the broker's ~12 s policy cycle cannot supply. tau is bounded by construction, so no
 * gain or clamp helps; the update rate is the only lever. A tracker instance sees only its own
 * channels, so the loop runs in the gather process, the one place that sees every instance at
 * frame rate.
 *
 * Policy stays in the Python broker: ephemeris, clock, PRN assignment, presence and quality
 * gates, and which PRNs are armed. This stage forms the fleet E/P/L, runs the integrator and
 * posts the trim; it arms no PRN of its own. See docs/CHORD_FAST_TRIM.md.
 *
 * Each sender sends one telemetry frame per window, 4 records x 2048 hops x 5.12 us = 41.94 ms,
 * so each chain closes windows, and steps its integrator, at 23.84 Hz: 12x the break-even rate.
 *
 * This stage is a second consumer of the gather's telem_buf, so a slow pass back-pressures the
 * buffer and bufferRecv drops frames for the broker too. Work per frame is bounded, the fold
 * cost is served as `fold_us_per_frame`, and the fold thread does no network I/O.
 *
 * The arithmetic is in gnss::FleetDll (gnssFleetDll.hpp), so scripts/gnss/fleetdll.cpp can run
 * it offline and scripts/gnss/fleetdll_gate.py can compare it with combdll.fleet_dll_comb on
 * the same frames. This class is buffer handling, REST and the poster.
 *
 * @par buffers
 * @buffer in_buf  telemetry frames as received (producer: bufferRecv; shared with the gather)
 *
 * @conf n_win                 Int, default 4. Windows averaged into one discriminator
 *                             (4 windows = 16 records = 168 ms).
 * @conf min_instances         Int, default 2. Senders a record must have reached to be
 *                             averaged in; a completeness gate, never an operand.
 * @conf taps_win              Int, default 0. Window depth served by get_taps and get_rec_taps;
 *                             0 follows n_win. Set it to the broker's `--telem-windows`.
 * @conf trim_state_file       String, required. Trim store path, saved every 2 s.
 * @conf trim_state_max_age_s  Double, default 300. A trim store older than this is refused.
 * @conf policy_ttl_s          Double, default 60, at least 5. A chain's policy is dropped after
 *                             this long without a /set_policy for it.
 * @conf post_every_n_windows  Int, default 1, and 1 is the only value accepted: a chain's
 *                             trims are posted on every close of its window.
 * @conf post_threads          Int, default 4. Poster threads.
 *
 * @par REST
 * `<unique_name>/get_dll`      -- per chain, per PRN: disc, q, e/p/l_pow, n_src, n_chan, n_rec,
 *                                 hop, win, n_updates, and for a PRN with a standing trim the
 *                                 integrator state (trim_chips, trim_steps, trim_railed,
 *                                 trim_skipped, trim_win, armed); per chain, `_p_floor_abs`.
 *                                 No presence: that is policy.
 * `<unique_name>/get_taps`     -- per chain, per PRN: lobe-coherent and per-channel taps.
 * `<unique_name>/get_rec_taps` -- per chain: the per-record lobe-coherent E/P/L series.
 * `<unique_name>/get_stats`    -- counters, fold cost, trim store, policy and poster state.
 * `<unique_name>/set_policy`   -- POST {"chains": {chain: {armed, gain_per_s, leak_per_s,
 *                                 clamp, spacing, p_floor_abs, targets}}}.
 *
 * @author Keith Vanderlinde
 */
class GnssFleetTrim : public kotekan::Stage {
public:
    GnssFleetTrim(kotekan::Config& config, const std::string& unique_name,
                  kotekan::bufferContainer& buffer_container);
    ~GnssFleetTrim() override;

    void main_thread() override;

private:
    /// One tracker instance's /set_trim endpoint, parsed by parse_target when a /set_policy
    /// names it; a malformed URL fails that request with 400.
    struct Target {
        std::string host;
        unsigned short port = 0;
        std::string path;
        std::string url;   ///< as posted; the sort key and the key for per-target poster state
        std::string chain; ///< which chain's payload it takes
        /// Resolved by parse_target on each /set_policy, never per post: lookups block, and at
        /// one post per instance per window they would stall the posters and overload the
        /// resolver.
        struct sockaddr_in addr;
    };

    /// One chain's /set_policy request, before rearm() converts the per-second gain and leak to
    /// per-update values (gnss::TrimPolicy) with the measured window close rate.
    struct PolicyReq {
        std::set<int> armed;
        double clamp = 3.0, spacing = 0.5;
        double gain_per_s = 0.0;  ///< the bandwidth, 1/s
        double leak_per_s = 0.0;  ///< 1/s
        double p_floor_abs = 0.0; ///< absolute prompt-power floor; 0 = leak only, no steps
    };

    void dll_callback(kotekan::connectionInstance& conn);
    void taps_callback(kotekan::connectionInstance& conn);
    void rec_taps_callback(kotekan::connectionInstance& conn);

    /// Trim store. The integrator is the only state here that cannot be rebuilt from the
    /// stream, so it is saved to trim_state_file and restored after a restart; without it every
    /// tracker is left off-peak by its standing trim. See FleetDll::trim_snapshot().
    void save_trims();
    void load_trims();
    /// Restored trims, chain -> PRN -> chips, held until a /set_policy arms the PRN. Not
    /// applied at load: until its first policy a chain runs the default per-update leak, which
    /// erases an unarmed 1-chip trim in ~5.6 s, inside the ~12 s policy cycle.
    std::map<std::string, std::map<int, double>> _restored;
    std::string _trim_state_file;
    double _trim_state_max_age_s = 300.0;
    static constexpr double _trim_state_save_s = 2.0; ///< seconds between trim store saves
    double _trim_saved_at = 0.0;
    double _restored_age_s = -1.0;
    int _restored_adopted = 0, _restored_offered = 0;
    void stats_callback(kotekan::connectionInstance& conn);
    void policy_callback(kotekan::connectionInstance& conn, nlohmann::json& request);
    static Target parse_target(const std::string& url, const std::string& chain);
    void post_trims();
    void post_loop(int slot);
    void rearm();

    Buffer* in_buf;

    std::mutex _mtx; ///< guards _dll, the policy and trim-store state, and the counters
    gnss::FleetDll _dll;
    uint64_t _frames = 0, _bad_frames = 0, _late_frames = 0;
    double _fold_s = 0.0; ///< total seconds in the fold step, including waits for _mtx

    // ---- Actuator ----
    // Every instance tracking a chain needs the same trims, but each has its own /set_trim
    // endpoint, so a post is one request per instance.
    //
    // restClient is not usable here: make_request keeps only a pointer to the caller's
    // callback, make_request_blocking exits the process on timeout, and both send
    // `Connection: close`. The poster uses plain sockets with explicit timeouts, addresses
    // resolved at policy time, kept-alive connections, and no path that ends the process,
    // which also runs the gather.
    //
    // The fold thread never sends: post_trims publishes the newest payloads and the poster
    // threads do the I/O. The trim is absolute, so a poster that falls behind sends whatever
    // _pending holds when it next looks and skips the rest; nothing is queued.
    std::vector<Target> _targets; ///< one per tracker instance, sorted by url; guarded by _pend_mtx
    std::vector<std::thread> _post_threads;
    /// Poster stride: thread i serves targets i, i + n, ... Set before any poster starts;
    /// post_loop must not use _post_threads.size(), which grows while the first posters run.
    int _n_post_threads = 1;
    std::mutex _pend_mtx;
    std::condition_variable _pend_cv;
    std::map<std::string, nlohmann::json> _pending; ///< chain -> payload, from the last post_trims
    uint64_t _pend_gen = 0;                         ///< bumped when _pending is replaced
    std::vector<uint64_t> _sent_gen;                ///< per thread slot, last generation sent
    /// Per-request budget on an established connection: wider than the kernel's minimum RTO
    /// (200 ms) so a single lost segment recovers instead of costing the socket, short enough
    /// that a wedged tracker cannot hold a poster thread past the point where its trim would
    /// be stale anyway. Handshakes are not charged against it (see post_loop).
    static constexpr int _post_timeout_ms = 450;
    /// chain -> n_closed at the last post_trims, so a chain is posted only when its own window
    /// closed.
    std::map<std::string, uint64_t> _closed_seen;
    uint64_t _post_reqs = 0, _post_ok = 0, _post_fail = 0, _post_rounds = 0;
    std::string _post_last_err;

    // ---- Policy ----
    // The broker's policy cycle posts, per chain, the armed PRNs, the loop constants and the
    // targets; every judgement (presence, floors, the deep gate, the clock, the ephemeris)
    // stays there.
    //
    // gain and leak are per update, so at fixed constants the loop's closed-loop and noise
    // bandwidths scale with the update rate. The broker states gain_per_s and leak_per_s, and
    // rearm() divides them by the measured window close rate, served as close_hz_measured.
    //
    // Each broker chain thread posts only its own chain, so policies and targets are replaced
    // per chain: a chain's entry is replaced wholesale on every post, so a PRN it stops naming
    // is disarmed, and chains absent from the post are untouched. _policy_seen and
    // policy_ttl_s drop the policy of a chain that stops posting.
    std::map<std::string, PolicyReq> _policy;
    std::map<std::string, double> _policy_seen; ///< chain -> wall time of its last /set_policy
    double _policy_ttl_s = 60.0;                ///< drop a chain's policy after this silence
    double _first_close_t = 0.0; ///< wall time of the first window close, for the rate
    uint64_t _first_close_n = 0;
    double _close_hz = 0.0; ///< window closes/s per chain, averaged since the first close
    std::map<std::string, uint64_t> _late_seen; ///< chain -> n_late at the previous rearm()
    double _late_seen_t = 0.0;
    uint64_t _policy_posts = 0;
    uint64_t _policy_expired = 0; ///< chains dropped for silence: a broker chain thread stopped
};

#endif // GNSS_FLEET_TRIM_HPP
