/**
 * @file
 * @brief Process-wide rendezvous between the GNSS search and track stages.
 */

#ifndef GNSS_TRACK_STATE_HPP
#define GNSS_TRACK_STATE_HPP

#include <map>
#include <mutex>
#include <set>
#include <string>
#include <vector>

namespace gnss {

/// A satellite hand-off from @ref GnssAcquire (search) to @ref GnssTracker.
/// The tracker re-finds the code phase every block with its own parallel-code-
/// phase FFT, so a lock only needs the PRN plus a coarse Doppler to centre the
/// narrow tracking window; a few-Hz-stale Doppler (snapshot age) is well inside
/// the window. No code phase / sample anchor is needed.
struct Seed {
    int prn;
    double doppler_hz; ///< acquisition Doppler (parabola-refined), Hz
    float snr;         ///< acquisition peak/mean (logging / hysteresis)
};

/// Shared state for one search+track pair that name the same @c state_group.
/// The search pushes Seeds for newly acquired PRNs; the tracker drains them at
/// a block boundary and publishes back the PRN set it currently holds so the
/// search can drop those from its candidate list. Lock contention is trivial
/// (a handful of ints per acquisition / per integration window), so a single
/// mutex is plenty.
class TrackState {
public:
    /// search -> tracker: hand off a freshly acquired PRN. A newer seed for a
    /// PRN already pending replaces the older one.
    void promote(const Seed& s);
    /// tracker: atomically take and clear all pending seeds.
    std::vector<Seed> take_pending();
    /// tracker -> search: publish the PRNs the tracker currently owns.
    void set_owned(const std::set<int>& owned);
    /// search: PRNs to skip (already owned by the tracker).
    std::set<int> owned() const;

private:
    mutable std::mutex _mtx;
    std::vector<Seed> _pending;
    std::set<int> _owned;
};

/// Look up (or lazily create) the shared TrackState for a named group. The
/// returned reference is stable for the life of the process.
TrackState& track_state(const std::string& group);

} // namespace gnss

#endif // GNSS_TRACK_STATE_HPP
