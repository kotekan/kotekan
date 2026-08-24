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
#include <mutex>
#include <thread>
#include <vector>
#include <map>
#include <string>

/**
 * @class GnssFleetTrim
 * @brief THE FAST CODE LOOP, fleet-wide, in C++ (task #51). MILESTONE F1: IT OBSERVES ONLY --
 *        it forms the fleet discriminator every window and actuates nothing.
 *
 * WHY THIS EXISTS, in one line of arithmetic. The code discriminator's authority is a step
 * PER UPDATE, not a rate:
 *
 *     tau  = -clamp(disc, +-1)/4 * (spacing/0.5)   =>  |tau| <= 0.25 chips, ALWAYS
 *     step = dll_gain * tau                        =>  <= 0.0625 chips at gain 0.25
 *
 * against a measured 0.121 chips/s of code drift on CHORD. Break-even is therefore 1.94
 * UPDATES PER SECOND, and the broker's policy cycle delivers one every ~12 s -- 23x too slow.
 * The loop pulls in and cannot hold, the prompt tap sits off-peak, and every statistic derived
 * from it (prompt power, C/N0 coherent AND incoherent, the deep fold, every coherence number)
 * is computed on noise. One fault, a dozen symptoms. No gain and no clamp fixes it: tau is
 * clamped BY CONSTRUCTION, so the update RATE is the only lever.
 *
 * WHY HERE AND NOT IN THE TRACKER. cudaGnssChordTrack already carries this loop (`code_trim`),
 * already at frame rate, and it is default OFF for a reason that has not changed: an instance
 * sees ~7 of the fleet's ~105 channels and one reference element. It has the authority and not
 * the observability. The rule worth preserving is not "the loop lives in the broker" but ONE
 * PLACE WITH THE FLEET-WIDE VIEW OWNS THE LOOP -- and the gather host is the only other place
 * that has every instance at frame rate.
 *
 * WHY THE POLICY DOES NOT COME WITH IT. Everything that is a judgement stays in the Python
 * broker on its 12 s cycle: ephemeris, sky and visibility, the clock solve and the joint state,
 * PRN assignment, presence and quality gates, who is armed, publishing and the archive. This
 * stage gets three jobs and no others -- form the fleet E/P/L, run the integrator, post the
 * trim -- and it invents no gate and chooses no PRN. See docs/CHORD_FAST_TRIM.md 3.
 *
 * RATES (docs/CHORD_FAST_TRIM.md 2). A frame is 4 records x 2048 hops x 5.12 us = 41.94 ms, so
 * frames arrive at 23.84 Hz and RECORDS at 95.4 Hz. There is no 50 Hz measurement to be had;
 * there does not need to be. Each frame carries four records, which are four legitimate
 * sequential integrator steps, and because the limit is a step per update those four fold into
 * ONE post with no loss of authority: 95.4 steps/s, 23.84 posts/s, a 49x margin on the drift.
 *
 * ⚠️ THIS STAGE MUST NEVER STALL. It is a SECOND CONSUMER of the gather's telem_buf, so a slow
 * pass here back-pressures the buffer and bufferRecv starts dropping frames FOR THE BROKER TOO.
 * Bounded work per frame, and `fold_us_per_frame` is served so "is it affordable?" is
 * answerable without a second measurement -- a silently-slow controller looks exactly like a
 * working one.
 *
 * THE ARITHMETIC IS NOT HERE. It is gnss::FleetDll (gnssFleetDll.hpp), so `scripts/gnss/fleetdll`
 * can drive the SHIPPED code against a byte-identical fixture offline in under a second and the
 * gate can compare it with combdll.fleet_dll_comb on the same bytes. This class is buffer, REST
 * and log, and nothing else.
 *
 * @par buffers
 * @buffer in_buf  telemetry frames as received (producer: bufferRecv; shared with the gather)
 *
 * @conf n_win          Int, default 4. Windows averaged into one discriminator. 4 windows =
 *                        16 records = 168 ms, matching --fast-trim-windows on the Python arm.
 * @conf min_instances  Int, default 2. Instances required before a PRN gets a discriminator.
 * @conf max_open_win   Int, default 8. Windows held open per chain before the oldest is
 *                        force-closed, so a sender that dies mid-window cannot pin memory.
 *
 * @par REST
 * `<unique_name>/get_dll`   -- per chain, per PRN: disc, q, e/p/l_pow, n_src, n_chan, n_rec,
 *                              hop, win. The SAME SHAPE combdll.fleet_dll_comb produces, minus
 *                              presence (that is policy, and it stays in Python).
 * `<unique_name>/get_stats` -- frames, windows closed, late frames, forced closes, fold budget.
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
    /// One tracker instance's /set_trim. Parsed from a URL at construction, so a malformed
    /// endpoint fails loudly at startup rather than as a silent stream of failed posts.
    struct Target {
        std::string host;
        unsigned short port = 0;
        std::string path;
        std::string url;   ///< as configured, for the log and get_stats
        std::string chain; ///< which chain's payload it takes
        /// ⚠️ RESOLVED ONCE, AT POLICY TIME. restClient's own header warns "Prefer numerical,
        /// because the DNS lookup is blocking" -- and at ~1430 posts/s it is not a slowdown,
        /// it is an outage: on 2026-08-15 the gather's log went "Nameserver 127.0.0.53:53 has
        /// failed: request timed out. All nameservers have failed" and the process died. Six
        /// hostnames do not change while a policy stands; resolving them per request is pure
        /// self-harm.
        struct sockaddr_in addr;
    };

    /// What the policy cycle asked for, before the rate conversion. Kept separate from
    /// gnss::TrimPolicy because `leak_per_s` is what the broker states and `leak` is what the
    /// integrator uses, and conflating them is exactly the 30x-bandwidth trap.
    struct PolicyReq {
        std::set<int> armed;
        double gain = 0.25, clamp = 3.0, spacing = 0.5;
        double leak_per_s = -1.0; ///< <0 means "leak was given per-update instead"
        double leak = 0.05;
        double gain_per_s = -1.0; ///< the BANDWIDTH; <0 means per-update `gain` was given
    };

    void dll_callback(kotekan::connectionInstance& conn);
    void taps_callback(kotekan::connectionInstance& conn);
    void rec_taps_callback(kotekan::connectionInstance& conn);

    /// THE TRIM STORE. The integrator is the only state here that cannot be rebuilt from the
    /// stream, and a gather restart is the only thing that loses it -- measured 2026-08-23 as
    /// q 2.0-3.7 -> ~1.0 fleet-wide for minutes. See gnssFleetDll.hpp trim_snapshot().
    void save_trims();
    void load_trims();
    /// Restored trims awaiting the broker's first /set_policy for that PRN. NOT applied at
    /// load: an unarmed trim leaks to erasure in ~5.6 s and the policy cycle is ~11 s.
    std::map<std::string, std::map<int, double>> _restored;
    std::string _trim_state_file;
    double _trim_state_max_age_s = 300.0;
    double _trim_state_save_s = 2.0;
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

    std::mutex _mtx; ///< guards _dll and the counters (the REST threads read them)
    gnss::FleetDll _dll;
    uint64_t _frames = 0, _bad_frames = 0, _late_frames = 0;
    double _fold_s = 0.0; ///< total seconds folding -- the budget this must not blow

    // ---- THE ACTUATOR (F2) ---------------------------------------------------------------
    //
    // ⚠️ THE POST COUNT IS NOT WHAT I FIRST CLAIMED. Every instance tracking a chain despreads
    // the same PRNs and so needs the same trims, and each instance has its OWN endpoint path --
    // 10-12 per chain, ~60 across five. They share only 6 host processes, but a shared HOST is
    // not a shared ENDPOINT, so they cannot be batched into 6 requests as I first wrote. At one
    // post per window that is ~1430 requests/s, and restClient sends `Connection: close`, so
    // each is its own TCP connection. `post_every_n_windows` decimates the ACTUATION without
    // touching the integrator: the trim moves at most gain*0.25 = 0.0625 chips per step, so
    // posting every 4th window costs ~0.02 chips of lag against a 0.121 chips/s drift --
    // nothing -- and cuts the request rate by four.
    //
    // ⚠️ NEITHER restClient ENTRY POINT IS USABLE HERE, and both were tried on sky.
    //
    //   make_request (async): stores a POINTER to the caller's std::function and never owns
    //     it -- cleanup() frees only the internal pair. A stack-local callback is dangling by
    //     the time the reply lands on libevent's thread. Killed the gather within seconds,
    //     silently: last log line "restClient: libevent version", then nothing.
    //
    //   make_request_blocking: FATAL_ERROR_NON_OO on timeout -- "This might leave the
    //     restClient in an abnormal state. Exiting..." It does not return an error, it exits
    //     the process. So ONE slow tracker takes the fleet's telemetry down with it. That is
    //     exactly what happened at 12:41 on 2026-08-15, compounded by per-request DNS.
    //
    // Both failures are the same shape: this stage back-pressures every chain's telemetry, so
    // it cannot host a dependency that reserves the right to call exit(). The poster below is
    // ~100 lines of plain socket with explicit timeouts, addresses resolved once, persistent
    // connections, and NO path that can terminate the process. Slower to write, and the only
    // version that has not taken the instrument down.
    //
    // THE FOLD THREAD NEVER SENDS. It publishes the newest payload and moves on; these threads
    // do the I/O. And because the trim is ABSOLUTE, a thread that falls behind simply picks up
    // the newest payload -- SKIP, NEVER QUEUE. A queue would deliver stale corrections late,
    // which is worse than delivering none.
    std::vector<Target> _targets; ///< flat, one per tracker instance; guarded by _pend_mtx
    std::vector<std::thread> _post_threads;
    /// ⚠️ THE STRIDE, FIXED BEFORE ANY THREAD STARTS. post_loop used to read
    /// _post_threads.size() -- a vector still being emplace_back'd by the constructor while
    /// the earlier threads were already running -- so every thread read a DIFFERENT stride and
    /// the target list was covered raggedly. Measured on sky 2026-08-15: 4 targets, 4 threads,
    /// and exactly 2 requests per round. It fails SILENTLY: the targets that are served look
    /// perfect, and the ones that are not simply never appear.
    int _n_post_threads = 1;
    std::mutex _pend_mtx;
    std::condition_variable _pend_cv;
    std::map<std::string, nlohmann::json> _pending; ///< chain -> newest payload
    uint64_t _pend_gen = 0;                         ///< bumped when _pending is replaced
    std::vector<uint64_t> _sent_gen;                ///< per thread slot, last generation sent
    int _post_every = 1;
    /// Per-request budget. 200 ms is ~5 frames: long enough that a busy tracker is not
    /// declared dead, short enough that a wedged one cannot hold a poster thread past the
    /// point where its trim would be stale anyway.
    int _post_timeout_ms = 200;
    /// PER CHAIN, not fleet-wide. Summing across chains made any chain's window close trigger
    /// a post round for EVERY chain: measured 116 posts/s/path against the 23.8 intended, a
    /// clean 5x for the five chains (2026-08-15). Harmless in correctness -- the trim is
    /// absolute and idempotent -- and 5x the request rate, which is exactly the sort of thing
    /// that is invisible unless the measurement is made against a stated expectation.
    std::map<std::string, uint64_t> _closed_seen;
    uint64_t _post_reqs = 0, _post_ok = 0, _post_fail = 0, _post_rounds = 0;
    std::string _post_last_err;

    // ---- THE POLICY SEAM (F3) -------------------------------------------------------------
    //
    // The broker's 12 s cycle POSTs who is armed and with what constants; this stage converts
    // and applies. Everything that is a JUDGEMENT -- presence, floors, the deep gate, the
    // clock, the ephemeris -- stays there. See docs/CHORD_FAST_TRIM.md 4.
    //
    // ⚠️ THE LEAK CONVERSION LIVES HERE AND NOWHERE ELSE. `leak` is PER UPDATE, so the loop's
    // closed-loop and noise bandwidths both scale with the update rate: at unchanged constants,
    // moving 3.1 -> 23.8 Hz is ~8x the bandwidth. The broker therefore states `leak_per_s` and
    // this converts with the MEASURED close rate (not the nominal 23.84, which is what the wire
    // would deliver if nothing were ever late -- 3% are). The measured rate is served in
    // get_stats so the conversion is checkable rather than trusted.
    //
    // ⚠️ PER CHAIN, AND THAT IS NOT COSMETIC. Each broker chain thread POSTs a payload naming
    // ONLY its own chain (`{"chains": {telem_chain: {...}}}`), so a wholesale `_policy = got`
    // means the last chain to POST disarms every other one. With a single armed chain that is
    // invisible; the moment a second is armed the two clobber each other at the policy cadence
    // and BOTH fall to a duty cycle set by who posted last. Found before arming gal_e5a/bds_b2a
    // (#49), by reading the POST payload rather than the endpoint -- it would have presented as
    // "arming the new chains broke gps_l5".
    //
    // The anti-latch property that motivated "replace, never merge" is preserved, just at chain
    // granularity: a chain's OWN entry is still replaced wholesale every post, so a PRN it stops
    // naming stops being armed. What replace-everything additionally bought -- a chain that goes
    // silent stops being commanded -- is now bought explicitly by _policy_seen + policy_ttl_s,
    // because a silent chain is a broker thread that died and its trims must expire.
    std::map<std::string, PolicyReq> _policy;
    std::map<std::string, double> _policy_seen; ///< chain -> wall time of its last /set_policy
    double _policy_ttl_s = 60.0;                ///< drop a chain's policy after this silence
    double _first_close_t = 0.0; ///< wall time of the first window close, for the rate
    uint64_t _first_close_n = 0;
    double _close_hz = 0.0;      ///< MEASURED window closes/s, fleet-wide
    uint64_t _policy_posts = 0;
    uint64_t _policy_expired = 0; ///< chains dropped for silence; a rising count is a dead thread
};

#endif // GNSS_FLEET_TRIM_HPP
