#ifndef GNSS_RECORD_HPP
#define GNSS_RECORD_HPP
/**
 * @file gnssRecord.hpp
 * @brief The GNSS record layout -- the single source of truth for the distributed pipeline's
 *        per-PRN record slots (tracker output, combiner input/output, BeamCube input).
 *
 * A record is RECORD_FLOATS float32 slots per PRN per window. Two flavours share the size and
 * the header slots (0-2) + UTC (9-10, an aliased double):
 *
 *  TRACKER record (GnssChannelizedTracker -> combiner/BeamCube; one per subband):
 *    0 PRN   1 Doppler_Hz   2 code_phase_chips
 *    3 P.re  4 P.im         -- PROMPT correlation, un-normalized coherent sum G (the combiner
 *                              sums G and energy across subbands, then normalizes: A = G/E)
 *    5 P_energy             -- prompt replica energy E
 *    6 n_chan               -- covering channels this subband contributed
 *    7 E_energy  8 L_energy -- early/late replica energies (see 11-14)
 *    9,10 UTC               -- capture UTC (double aliased over two floats)
 *    11 E.re 12 E.im 13 L.re 14 L.im
 *      -- EARLY/LATE correlations: the same record despread at code phase -+ dll_spacing
 *         (default 0.5 chip). Dumb outputs for the low-rate DLL: the broker forms the
 *         discriminator (|E|^2-|L|^2)/(|E|^2+|L|^2) from combiner-aggregated values and closes
 *         the code loop at ~Hz. The tracker itself makes NO alignment decisions.
 *
 *  COMBINER record (GnssCoherentCombiner -> record/viewer; full-band, one per emit):
 *    0 PRN   1 Doppler_Hz   2 code_phase_chips
 *    3 |A|_incoh  4 <A>.re  5 <A>.im  6 |<A>|_coh  7 n_chan  8 deep |A|
 *    9,10 UTC
 *    11 <|E|^2>  12 <|L|^2>  13 DLL discriminator  14 spare
 *
 * The monolithic ground-truth path (GpsReplicaCorrelator + gps_mono_watch.py) keeps its own
 * frozen 11-float layout and does NOT include this header. Python tools parse these constants
 * from this file (the gnssSignal.hpp pattern); keep the literals machine-readable.
 */

namespace gnss {

constexpr int RECORD_FLOATS = 15;  ///< float32 slots per PRN per record
constexpr int RECORD_UTC_SLOT = 9; ///< capture-UTC double aliased at slots 9-10

// Shared header slots
constexpr int REC_PRN = 0;
constexpr int REC_DOPPLER = 1;
constexpr int REC_CP = 2;

// Tracker-record slots
constexpr int REC_P_RE = 3;
constexpr int REC_P_IM = 4;
constexpr int REC_P_ENERGY = 5;
constexpr int REC_NCHAN = 6;
constexpr int REC_E_ENERGY = 7;
constexpr int REC_L_ENERGY = 8;
constexpr int REC_E_RE = 11;
constexpr int REC_E_IM = 12;
constexpr int REC_L_RE = 13;
constexpr int REC_L_IM = 14;

// Combiner-record slots
constexpr int CMB_AMP_INCOH = 3;
constexpr int CMB_MEAN_RE = 4;
constexpr int CMB_MEAN_IM = 5;
constexpr int CMB_AMP_COH = 6;
constexpr int CMB_NCHAN = 7;
constexpr int CMB_DEEP = 8;
constexpr int CMB_E_POW = 11;
constexpr int CMB_L_POW = 12;
constexpr int CMB_DLL_DISC = 13;

} // namespace gnss
#endif // GNSS_RECORD_HPP
