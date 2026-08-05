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
 *      -- SLOT-2 CURRENCY CONTRACT (2026-07-31): slot 2 is the sample-0-anchored code phase
 *         expressed in THE SAME RECORD'S slot-1 carrier: cp_phys(t_rec) = slot2 +
 *         t_abs*f_chip*(1 + sign*slot1/f_carrier) mod L, EXACTLY. The tracker re-expresses
 *         its internal f_ref-currency despread command into slot 1's reported carrier at
 *         emit (the export-currency translation in both trackers); consumers MUST pair
 *         slot 2 with slot 1 from the same record (the combiner forwards both from its
 *         reference record, so status doppler_hz/code_phase_chips are a valid pair).
 *         Before this, slot 2 carried the raw f_ref-currency command and every downstream
 *         reconstruction inherited a t_abs*f_chip*(ctrim)/f_c error (~25 chips/Hz at soak
 *         age) -- the absolute-VTEC blocker.
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
 *    16 PH.re 17 PH.im 18 PH_energy  19 trim_inc (REC_TRIM_INC, commanded-trim increment)
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
 *    20 RES.re 21 RES.im  22 RES_PH.re 23 RES_PH.im
 *      -- PEEL RESIDUAL prompt (and its head segment): the SAME prompt correlation, taken on
 *         the voltage AFTER this PRN's own reconstructed waveform was subtracted
 *         (docs/gnss_voltage_peel_live.md). Slots 3/4/16/17 continue to carry the FULL
 *         (un-peeled) correlation -- the fused peel adds its own known contribution back in
 *         closed form, exactly, so every existing consumer is untouched. These slots are the
 *         NEW product: 20*log10(|P| / |RES|) is the achieved peel depth, and it is honest
 *         per-record because the subtracted gain is FEED-FORWARD, i.e. statistically
 *         independent of this record's noise. (A gain fitted IN-window subtracts its own
 *         noise back along R and the residual reads ~0 = infinite depth -- the degeneracy
 *         that makes GnssVoltagePeel's own peel_db unusable.)
 *         Energies are shared with the prompt (same replica), so none are duplicated.
 *         The HEAD slots are REQUIRED, not decorative: the residual carries the same overlay
 *         sign flip at the code-period boundary, so a blind deep integration of RES cancels
 *         to |2f-1| exactly like the prompt did before the 2026-07-15 segmentation fix.
 *         All four are ZERO when no peel ran -- the compatibility default.
 *
 *  COMBINER record (GnssCoherentCombiner -> record/viewer; full-band, one per emit):
 *    0 PRN   1 Doppler_Hz   2 code_phase_chips
 *    3 |A|_incoh  4 <A>.re  5 <A>.im  6 |<A>|_coh  7 n_chan  8 deep |A|
 *    9,10 UTC
 *    11 <|E|^2>  12 <|L|^2>  13 DLL discriminator  14 carrier residual (Hz, full-band
 *       cross-record phase walk -- the shared carrier loop's observable; broker closes it)
 *    17 deep |A| of the PEEL RESIDUAL  18 incoherent |A| of the peel residual
 *       -- the same two estimators as slots 8 and 3, run on the residual prompt (slots 20-23).
 *          peel depth (dB) = 20*log10(CMB_DEEP / CMB_PEEL_DEEP). Zero when no peel ran.
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

/// float32 slots per PRN per record.
///
/// ⚠️ This is ALSO a yaml constant: every record buffer is sized `n_prn * record_floats *
/// sizeof_float`, and NOTHING reads this header to check that. Bumping it means bumping
/// `record_floats:` in `config/gnss_node.yaml` (the generator source) and re-running the
/// generators, plus the hand-maintained bench configs. A config left behind under-sizes the
/// frame and the producers write past the end of it -- so the record producers now assert the
/// frame is big enough at construction (see GnssChannelizedTracker / GnssGpuRecordAssemble /
/// GnssCoherentCombiner) and die loudly instead of corrupting memory.
constexpr int RECORD_FLOATS = 24;
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
constexpr int REC_TRIM_INC = 19;  ///< commanded carrier-TRIM phase increment, cycles since this
                                  ///< PRN's previous record (0 = arc start / not written).
                                  ///< The broker-commanded ctrim component of f_nco only --
                                  ///< reconstructed in the assembler as
                                  ///< (f_nco + fcar_report - fcar)/2 -- integrated so the
                                  ///< combiner can carry trim_cycles beside adr_cycles and
                                  ///< downstream TEC can subtract the KNOWN loop transients
                                  ///< from the ADR exactly (docs/adr_trim_subtraction.md; the
                                  ///< broker-log journal's +-1 s timing error becomes exact).
                                  ///< Bounded (trim <= clamp 100 Hz x t_rec) = float32-exact,
                                  ///< the same increment design as slot 15.
constexpr int REC_RES_RE = 20;    ///< PEEL RESIDUAL prompt correlation -- the prompt despread on
constexpr int REC_RES_IM = 21;    ///< the voltage after this PRN's waveform was subtracted.
constexpr int REC_RES_PH_RE = 22; ///< residual prompt HEAD segment (same boundary as REC_PH_*);
constexpr int REC_RES_PH_IM = 23; ///< required, or the residual's deep integration self-cancels.
                                  ///< All four zero = no peel ran. Energies are the prompt's
                                  ///< (same replica). See the header note + the peel design doc.

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
constexpr int CMB_PEEL_DEEP = 17; ///< deep |A| of the PEEL RESIDUAL (the CMB_DEEP estimator run on
                                  ///< slots 20-23) -> depth_dB = 20*log10(CMB_DEEP/CMB_PEEL_DEEP)
constexpr int CMB_PEEL_INCOH = 18;///< incoherent |A| of the peel residual (the CMB_AMP_INCOH twin).
                                  ///< Both zero when no peel ran.
constexpr int CMB_HOP_SLOT = 19;  ///< absolute HOP index (int64 aliased at slots 19-20) the E/P/L
                                  ///< window ends on = GnssChanMetadata::sample_seq / fft_len.
                                  ///< -1 (or 0 from an older writer) = unset. int64 and not a
                                  ///< float because the whole point is an EXACT integer key: the
                                  ///< fleet DLL groups instances by equal hop, and the hop reaches
                                  ///< ~1.7e10/day -- past float32's 2^24 by three orders of
                                  ///< magnitude, so a float slot would quantize the key into
                                  ///< collisions. UTC at slot 9 stays: overlay_apply and coh_span
                                  ///< use it, and this is additive.
                                  ///< COMBINER RECORDS ONLY -- tracker records use 19-23 for
                                  ///< REC_TRIM_INC / REC_RES_*, and the two flavours never share
                                  ///< a buffer. See docs/CHORD_GNSS_SHARED_DLL.md.
constexpr int CMB_COH_FRAC = 21;  ///< coherence fraction |sum A|/sum|A| of the WINNING deep
                                  ///< stream (after rate/phase-track derotation). THE chopping-
                                  ///< independent coherence measure: deep_snr scales with the
                                  ///< record count when phase-limited (8.20.24), this does not.
                                  ///< exp(-sigma_phi^2/2) for Gaussian wander; 0 = not computed
                                  ///< (non-plain deep paths, for now). COMBINER RECORDS ONLY.

// ---------------------------------------------------------------------------------------
// ELEMENT AXIS (CHORD)
// ---------------------------------------------------------------------------------------
/// The airspy node is single-antenna: one record per PRN, and the correlations above ARE the
/// measurement. CHORD despreads N antennas against the same M references, so a record carries
/// an antenna axis the layout above has no room for.
///
/// LAYOUT. Per PRN: the existing @ref RECORD_FLOATS header, followed by @c n_elem blocks of
/// @ref ELEM_FLOATS:
///
///     [PRN 0: 24 header floats][elem 0: 12][elem 1: 12]...[elem n-1: 12][PRN 1: ...]
///
/// so the frame is `n_prn * (RECORD_FLOATS + n_elem * ELEM_FLOATS)` floats, and element e of
/// PRN p starts at `p * (RECORD_FLOATS + n_elem*ELEM_FLOATS) + RECORD_FLOATS + e*ELEM_FLOATS`
/// (see @ref elem_offset). @c n_elem == 0 reproduces the airspy layout byte-for-byte, so every
/// existing consumer and every existing config keeps working untouched.
///
/// WHAT IS *NOT* HERE, DELIBERATELY. Only the per-antenna quantities live in an element block.
/// The code/carrier model (PRN, Doppler, code phase), the commanded NCO increments, the
/// covering-channel count and -- importantly -- the replica ENERGIES are element-INDEPENDENT:
/// one replica is correlated against every antenna, so E, P, L and P_HEAD energies are shared
/// and stay in the header. That is also the M x M block path B gets free from the N^2 kernel
/// (docs/gnss_chord_framework.md), which is why the split is drawn here: when the standalone
/// N x M correlator is replaced by synthetic lanes, the N x M block fills the element blocks
/// and the M x M block fills the header energies, with no schema change.
///
/// THE HEADER'S CORRELATION SLOTS carry the REFERENCE ELEMENT (config @c reference_element),
/// not element 0 blindly and not a blind sum. The broker closes the DLL and carrier loops off
/// those slots, so they must be a phase-coherent view; an incoherent element sum would destroy
/// the carrier loop, and a coherent sum needs the per-element phases that are the very thing
/// being measured. Pick the reference as a healthy high-gain feed.
/// The promised upgrade landed 2026-08-05 (config @c elem_sum, gnssElemCal.hpp): the assembler
/// SELF-CALIBRATES those phases from the satellite (bootstrap MRC, reference-anchored phase
/// convention, "one element" scale) and writes the calibrated weighted mean instead --
/// per-record SNR up ~sqrt(N_healthy), same observable, and byte-identical to the historical
/// output until each PRN's cal warms (~3 tau) or when elem_sum is off (the default).
constexpr int ELEM_FLOATS = 12;

constexpr int ELEM_P_RE = 0;  ///< prompt correlation for this antenna
constexpr int ELEM_P_IM = 1;
constexpr int ELEM_E_RE = 2;  ///< early / late, same taps as REC_E_*/REC_L_*
constexpr int ELEM_E_IM = 3;
constexpr int ELEM_L_RE = 4;
constexpr int ELEM_L_IM = 5;
constexpr int ELEM_PH_RE = 6; ///< prompt HEAD segment (hops before the code-period boundary);
constexpr int ELEM_PH_IM = 7; ///< tail = P - PH, exactly as in the header. Required for deep
                              ///< integration across an overlay/nav sign flip.
constexpr int ELEM_RES_RE = 8;     ///< PEEL RESIDUAL prompt, per antenna. RESERVED -- the peel
constexpr int ELEM_RES_IM = 9;     ///< is deferred until acq/track and beam mapping are up
constexpr int ELEM_RES_PH_RE = 10; ///< (a single-PRN replica is likely sub-quantization at
constexpr int ELEM_RES_PH_IM = 11; ///< 4+4b). Written zero until then; slots held so the
                                   ///< record size does not change when the peel lands.

// COMBINER element blocks reuse the same ELEM_FLOATS stride -- one geometry for both record
// types, so a buffer sized for one fits the other and record_stride() is the only size rule in
// the system. The SLOT MEANINGS differ, exactly as they do for the shared 24-float header
// (REC_* vs CMB_*): the tracker's blocks hold per-record correlations, the combiner's hold
// integrated per-antenna amplitudes.
constexpr int CMB_ELEM_AMP_INCOH = 0; ///< sqrt<|A_e|^2> -- THE BEAM MAP VALUE for this antenna.
                                      ///< Incoherent, so it needs no per-antenna phase model and
                                      ///< grows like sqrt(K); this is what a transit traces out.
constexpr int CMB_ELEM_MEAN_RE = 1;   ///< <A_e> coherent mean: carries the per-antenna PHASE,
constexpr int CMB_ELEM_MEAN_IM = 2;   ///< i.e. the complex gain the array calibration wants.
constexpr int CMB_ELEM_AMP_COH = 3;   ///< |<A_e>| -- coherent amplitude. Below the incoherent one
                                      ///< by the amount the phase wandered over the window, so
                                      ///< the RATIO is a per-antenna coherence diagnostic.
// 4..11 reserved (per-antenna deep integration, peel residual).

/// Floats per PRN for a record carrying @c n_elem antennas (@c n_elem 0 => the airspy layout).
constexpr int record_stride(int n_elem) {
    return RECORD_FLOATS + n_elem * ELEM_FLOATS;
}

/// Float offset of element @c e of PRN @c p within the frame.
constexpr int elem_offset(int p, int e, int n_elem) {
    return p * record_stride(n_elem) + RECORD_FLOATS + e * ELEM_FLOATS;
}

/// Float offset of PRN @c p's header within the frame.
constexpr int record_offset(int p, int n_elem) {
    return p * record_stride(n_elem);
}

// Layout invariants, checked at compile time so a careless edit above cannot silently
// change the frame geometry that the config's `record_floats` has to match.
static_assert(record_stride(0) == RECORD_FLOATS,
              "n_elem == 0 must reproduce the single-antenna layout exactly");
static_assert(elem_offset(0, 0, 4) == RECORD_FLOATS,
              "element 0 must start immediately after the PRN header");
static_assert(elem_offset(0, 1, 4) - elem_offset(0, 0, 4) == ELEM_FLOATS,
              "consecutive elements must be ELEM_FLOATS apart");
static_assert(record_offset(1, 4) == record_stride(4),
              "consecutive PRNs must be one stride apart");
static_assert(elem_offset(1, 0, 4) == record_stride(4) + RECORD_FLOATS,
              "PRN 1's first element must follow PRN 1's header");
// The last element of PRN p must end exactly where PRN p+1 begins -- no gap, no overlap.
static_assert(elem_offset(0, 3, 4) + ELEM_FLOATS == record_offset(1, 4),
              "element blocks must tile the record exactly");

} // namespace gnss
#endif // GNSS_RECORD_HPP
