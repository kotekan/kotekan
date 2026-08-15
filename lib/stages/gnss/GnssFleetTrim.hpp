#ifndef GNSS_FLEET_TRIM_HPP
#define GNSS_FLEET_TRIM_HPP

#include "Config.hpp"
#include "Stage.hpp"
#include "buffer.hpp"
#include "bufferContainer.hpp"
#include "gnssFleetDll.hpp"
#include "restServer.hpp"

#include <cstdint>
#include <mutex>
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
    ~GnssFleetTrim() override = default;

    void main_thread() override;

private:
    void dll_callback(kotekan::connectionInstance& conn);
    void stats_callback(kotekan::connectionInstance& conn);

    Buffer* in_buf;

    std::mutex _mtx; ///< guards _dll and the counters (the REST threads read them)
    gnss::FleetDll _dll;
    uint64_t _frames = 0, _bad_frames = 0, _late_frames = 0;
    double _fold_s = 0.0; ///< total seconds folding -- the budget this must not blow
};

#endif // GNSS_FLEET_TRIM_HPP
