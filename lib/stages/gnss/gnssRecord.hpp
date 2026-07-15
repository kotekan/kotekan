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
 *    15 dphi_cmd           -- COMMANDED carrier-phase INCREMENT, cycles (see below)
 *    16 PH.re 17 PH.im 18 PH_energy  19 (reserved)
 *      -- HEAD SEGMENT of the prompt: the same P correlation restricted to the hops BEFORE
 *         the code-period boundary inside this window. Windows are hop-grid-aligned, not
 *         code-period-aligned, so every record straddles one period boundary at offset
 *         fraction f -- and the secondary overlay / nav symbol FLIPS SIGN exactly there,
 *         mid-record. Summed blind, a record straddling a chip TRANSITION cancels to
 *         |2f-1| (f~0.5 nulls it: the 2026-07-15 "bistable" -- E1C lost 12/25 records to
 *         CS25's 12 transitions, B1C ~49%). With the head exported (tail = P - PH), the
 *         combiner wipes each segment with ITS OWN overlay chip (head gets chip n, tail
 *         chip n+1) and the cancellation never happens. PH == P means "no boundary in
 *         this window / not segmented" (the CPU tracker's compatibility default): tail = 0
 *         and every wipe reduces exactly to the unsegmented behaviour.
 *         PH_energy/P_energy = f, the measured boundary fraction (free diagnostic).
 *
 *  COMBINER record (GnssCoherentCombiner -> record/viewer; full-band, one per emit):
 *    0 PRN   1 Doppler_Hz   2 code_phase_chips
 *    3 |A|_incoh  4 <A>.re  5 <A>.im  6 |<A>|_coh  7 n_chan  8 deep |A|
 *    9,10 UTC
 *    11 <|E|^2>  12 <|L|^2>  13 DLL discriminator  14 carrier residual (Hz, full-band
 *       cross-record phase walk -- the shared carrier loop's observable; broker closes it)
 *    15 phase arc id        -- increments on every cycle slip / phase-continuity break; the
 *       accumulated carrier phase itself is a DOUBLE and ships via REST get_status (adr_cycles),
 *       not here: a float32 mantissa quantizes ~1e6 cycles of ADR to ~0.06 cycles, useless.
 *
 *  CARRIER PHASE (slot 15, tracker -> combiner). The tracker removes a known carrier phase
 *  from the data: the replica's own 2*pi*f_ref*t (ABSOLUTELY anchored at sample 0) plus the
 *  NCO's phi. The combiner measures what's left, arg(A), so the received carrier phase is
 *      Phi_rx(t) = Phi_cmd(t) - arg(A)/2pi,   Phi_cmd = f_ref*t_abs - phi/(2*pi)  [cycles]
 *  (both minus signs because the NCO and the despread residual live in the r2c-flipped
 *  internal convention while f_ref is physical-signed -- see GnssCoherentCombiner for the
 *  on-sky measurement that pinned each of them down).
 *
 *  Slot 15 carries the per-record INCREMENT of Phi_cmd (cycles since this PRN's previous
 *  record), NOT Phi_cmd itself. Two reasons, both learned the hard way:
 *    * Phi_cmd is ~1e7 cycles at soak age -- a float32 quantizes that to ~1 cycle, useless.
 *      The obvious dodge (ship it mod 1 cycle) forces the combiner to UNWRAP, i.e. to guess
 *      the integer part from a predicted rate. The only rate it has is the reported Doppler,
 *      which is wrong by 2*carrier_trim -- 2 whole cycles per 10 ms B1C record at the trim
 *      clamp. It then unwraps to the wrong integer, adopts that as its rate, and stays
 *      self-consistently wrong forever (measured: BeiDou C24 reading exactly +99.5 Hz off,
 *      = one cycle per 10 ms record).
 *    * The increment is bounded and small (Doppler * record period: ~5 cycles at L1, ~50 at
 *      B1C), so a float32 holds it to ~1e-6 cycles and NOTHING needs unwrapping. The tracker
 *      keeps Phi_cmd continuous through f_ref re-pins (the phase-continuity fold), so the
 *      increment stays small across those too -- which is what makes this possible at all.
 *  0 means "no previous record for this PRN": the arc starts here.
 *
 * The monolithic ground-truth path (GpsReplicaCorrelator + gps_mono_watch.py) keeps its own
 * frozen 11-float layout and does NOT include this header. Python tools parse these constants
 * from this file (the gnssSignal.hpp pattern); keep the literals machine-readable.
 */

namespace gnss {

constexpr int RECORD_FLOATS = 20;  ///< float32 slots per PRN per record
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
constexpr int REC_CPHASE = 15; ///< commanded carrier-phase INCREMENT, cycles since this
                               ///< PRN's previous record (0 = arc start). See the header note.
constexpr int REC_PH_RE = 16;     ///< prompt HEAD segment (hops before the code-period boundary)
constexpr int REC_PH_IM = 17;     ///< -- see the header note; tail = P - PH
constexpr int REC_PH_ENERGY = 18; ///< head replica energy; /P_energy = boundary fraction f

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
constexpr int CMB_CARRIER_RESID = 14;
constexpr int CMB_ARC = 15; ///< phase arc id: ++ on every cycle slip / continuity break
constexpr int CMB_HEAD_FRAC = 16; ///< window-mean boundary fraction f = <PH_energy>/<P_energy>
                                  ///< (0/1 = period boundary at the window edge -- benign;
                                  ///< ~0.5 = the old bistable's null zone; now fixed, diagnostic)

} // namespace gnss
#endif // GNSS_RECORD_HPP
