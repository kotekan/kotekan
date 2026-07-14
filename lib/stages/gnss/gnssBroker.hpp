/**
 * @file
 * @brief Broker assignment: map GNSS targets onto correlator workers.
 *
 * The planning brain of the distributed-band pipeline. Given the local PFB
 * channel grid (@ref gnssBandPlan.hpp), a set of correlator *workers* (each
 * owning a contiguous channel range, correlating one signal, optionally a PRN
 * subset for load splitting), and a target list (signal + PRN, with a
 * visibility flag), it decides which worker measures which PRN.
 *
 * The rule is carrier-aware: a target goes to a worker whose channel range
 * holds *all* of the target's covering channels (@ref covering_channels), so
 * acquisition + despread stay local to that worker -- no cross-worker (and, when
 * distributed, no cross-node) signal assembly. Among the workers that qualify,
 * the least-loaded is chosen for balance. A target whose covering channels fit
 * in no single worker is reported unplaced (locally avoidable by choosing the
 * channel split between carriers; remotely it is the straddle case).
 *
 * Pure logic: the same mechanism whether workers are local stages or remote
 * nodes. Hosting is local for now -- workers are correlator instances in one
 * process; Phase 3 adds the network transport that lets workers live on
 * different nodes.
 */

#ifndef GNSS_BROKER_HPP
#define GNSS_BROKER_HPP

#include "gnssBandPlan.hpp" // for ChannelBand, freq_id_t

#include <map>    // for map
#include <string> // for string
#include <vector> // for vector

namespace gnss {

/// A correlator instance available to take work.
struct Worker {
    std::string name;   ///< stage/instance name (assignment key)
    std::string signal; ///< SignalDescriptor name it correlates
    freq_id_t chan_lo;  ///< owned channel range, inclusive [chan_lo, chan_hi]
    freq_id_t chan_hi;
    int prn_lo; ///< accepted PRN range, inclusive [prn_lo, prn_hi]
    int prn_hi;
};

/// A satellite signal to measure.
struct Target {
    std::string signal; ///< SignalDescriptor name
    int prn;
    bool visible; ///< above the horizon / selected (set by the visibility mask)
};

/// Outcome of an assignment pass.
struct Assignment {
    std::map<std::string, std::vector<int>> worker_prns; ///< worker name -> assigned PRNs (sorted)
    std::vector<Target> unplaced;                        ///< no single worker could take these
};

/**
 * Assign visible targets to workers. A target is placed on the least-loaded
 * worker that (a) correlates its signal, (b) accepts its PRN, and (c) owns every
 * channel covering the carrier at the worst-case Doppler. Invisible targets are
 * skipped; unplaceable visible targets are returned in @c unplaced. Deterministic
 * (workers are considered in name order).
 *
 * @param workers      Available correlator instances.
 * @param targets      Signals + PRNs to measure, with visibility.
 * @param channels     The local PFB channel grid.
 * @param max_doppler_hz Worst-case |Doppler| for covering-channel selection.
 */
Assignment assign_targets(const std::vector<Worker>& workers, const std::vector<Target>& targets,
                          const std::vector<ChannelBand>& channels, double max_doppler_hz);

} // namespace gnss

#endif // GNSS_BROKER_HPP
