/**
 * @file
 * @brief Shared per-PRN GNSS record layout.
 *
 * One fixed-width record per tracked PRN per emitted integration, written as a
 * homogeneous float32 block so rawFileWrite and the python/visibility readers
 * can walk it without a metadata schema. Field order (float32 slots):
 *
 *   0 prn            PRN id (1..32, exactly representable as float)
 *   1 doppler_hz     FLL-refined when locked, else parabola-refined
 *   2 code_phase_chips
 *   3 peak_amp       sqrt of the integrated correlation peak
 *   4 peak_re        complex peak (phase-tracking modes only, else 0)
 *   5 peak_im
 *   6 snr            peak/mean of the correlation surface
 *   7 nav_sign       cumulative +-1 nav-bit sign (0 if unlocked)
 *   8 phase_cont     unwrapped nav-stripped carrier phase, rad
 *   9,10 utc         float64 UNIX seconds, type-punned across two float slots
 *
 * Shared by GpsReplicaCorrelator (the original fused stage) and the split
 * GnssTracker so a single reader handles both.
 */

#ifndef GNSS_RECORD_HPP
#define GNSS_RECORD_HPP

namespace gnss {

/// float32 slots per per-PRN record (last two hold the float64 UTC stamp).
constexpr int RECORD_FLOATS = 11;
/// Slot index of the trailing float64 UTC timestamp.
constexpr int RECORD_UTC_SLOT = 9;

} // namespace gnss

#endif // GNSS_RECORD_HPP
