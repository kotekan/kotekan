#ifndef GNSS_RECORD_HPP
#define GNSS_RECORD_HPP
/**
 * @file gnssRecord.hpp
 * @brief GNSS record layout (the per-PRN record slots shared by the tracker output, the
 *        combiner input and output and the telemetry rows) and the beam-cube frame layout.
 *
 * A record is RECORD_FLOATS float32 slots per PRN per window. Two flavours share the size, the
 * header slots 0-2 and the capture UTC (slots 9-10, an aliased double):
 *
 *  TRACKER record (GnssGpuRecordAssemble -> combiner, GnssTelemPack):
 *    0 PRN   1 Doppler_Hz   2 code_phase_chips
 *      -- slot 2 is the generator's sample-0-anchored code-phase argument at the Doppler in
 *         slot 1 of the same record: cp_phys = ChannelizedReplicaBank::phase_from_arg(slot2,
 *         wstart, slot1), i.e. slot2 + t_abs*f_chip*(1 + sign*slot1/f_carrier) mod L for
 *         comb_mult 1. The argument means nothing with any other Doppler, so consumers must pair
 *         slots 1 and 2 from one record; the combiner forwards both from its reference record.
 *    3 P.re  4 P.im         -- prompt correlation, un-normalized coherent sum G (the combiner
 *                              sums G and energy across subbands, then normalizes: A = G/E)
 *    5 P_energy             -- prompt replica energy E
 *    6 n_chan               -- covering channels this subband contributed
 *    7 E_energy  8 L_energy -- early/late replica energies (see 11-14)
 *    9,10 UTC               -- capture UTC (double aliased over two floats)
 *    11 E.re 12 E.im 13 L.re 14 L.im
 *      -- early/late correlations: the same record despread at code phase -+ dll_spacing
 *         (default 0.5 chip). The tracker makes no alignment decision from them; the combiner
 *         (CMB_DLL_DISC) and the broker's fleet DLL form the discriminator downstream.
 *    15 dphi_cmd            -- commanded carrier-phase increment, cycles (see below)
 *    16 PH.re 17 PH.im 18 PH_energy  19 trim_inc (REC_TRIM_INC)
 *      -- head segment of the prompt: the same correlation restricted to the hops before the
 *         code-period boundary inside this window. Windows are hop-aligned, not
 *         code-period-aligned, so a record straddles a period boundary at fraction f, and the
 *         secondary overlay or nav symbol can flip sign exactly there. Summed blind, such a
 *         record cancels to |2f-1| of its amplitude; with the head exported (tail = P - PH) the
 *         combiner wipes each segment with its own overlay chip. PH == P means no boundary in
 *         this window (tail = 0). PH_energy/P_energy = f.
 *    20 RES.re 21 RES.im  22 RES_PH.re 23 RES_PH.im
 *      -- peel residual prompt and its head segment: the same correlation on the voltage after
 *         this PRN's own reconstructed waveform was subtracted. Slots 3/4/16/17 carry the
 *         un-peeled correlation (the peel's own contribution is added back analytically).
 *         20*log10(|P| / |RES|) is the achieved peel depth, and it is unbiased per record
 *         because the subtracted gain is feed-forward: a gain fitted to this record would
 *         subtract its own noise and read as infinite depth. The head slots are needed for the
 *         same overlay sign flip as the prompt's. All four are zero when no peel ran.
 *    24-31: see REC_SKY_*, REC_ANG0, REC_PHI_DDOP, REC_PHI0 and REC_PROJ_* below.
 *
 *  COMBINER record (GnssCoherentCombiner output; full band, one per emit):
 *    0 PRN   1 Doppler_Hz   2 code_phase_chips
 *    3 |A|_incoh  4 <A>.re  5 <A>.im  6 |<A>|_coh  7 n_chan  8 deep |A|
 *    9,10 UTC
 *    11 <|E|^2>  12 <|L|^2>  13 DLL discriminator  14 carrier residual (Hz, full-band
 *       cross-record phase walk: the shared carrier loop's observable; the broker closes it)
 *    15 phase arc id        -- incremented on every cycle slip or phase-continuity break. The
 *       accumulated carrier phase itself is a double served by REST get_status (adr_cycles):
 *       float32 would quantize ~1e6 cycles of ADR to ~0.06 cycles.
 *    16 head fraction (CMB_HEAD_FRAC)
 *    17 deep |A| of the peel residual  18 incoherent |A| of the peel residual
 *       -- the slot-8 and slot-3 estimators run on the residual prompt (tracker slots 20-23).
 *          peel depth (dB) = 20*log10(CMB_DEEP / CMB_PEEL_DEEP). Zero when no peel ran.
 *    19,20 absolute hop (CMB_HOP_SLOT, int64)  21 coherence fraction (CMB_COH_FRAC)
 *
 *  CARRIER PHASE (slot 15, tracker -> combiner). The tracker removes a known carrier phase from
 *  the data: the replica's own 2*pi*f_ref*t, anchored at absolute sample 0, plus the
 *  assembler's NCO phase phi. The combiner measures what is left, arg(A), so the received
 *  carrier phase is
 *      Phi_rx(t) = Phi_cmd(t) - arg(A)/2pi,   Phi_cmd = f_ref*t_abs - phi/(2*pi)  [cycles]
 *  (both minus signs because the NCO and the despread residual are in the r2c-flipped internal
 *  convention while f_ref is physical-signed).
 *
 *  Slot 15 carries the per-record increment of Phi_cmd (cycles since this PRN's previous
 *  record), not Phi_cmd itself, which is far beyond float32 resolution; shipping it mod 1 cycle
 *  instead would force the combiner to unwrap it from the reported Doppler, which is not the
 *  applied carrier. The assembler folds each f_ref re-pin into phi, so the increment is
 *  continuous across re-pins. Its float32 precision follows its size, f_ref x record length:
 *  where f_ref is the RF carrier (CHORD, ~1.2 GHz) it is 1e7 cycles or more and float32 rounds
 *  it to whole cycles. GnssCoherentCombiner's adr_cycles sums slot 15 as shipped and carries
 *  that rounding; python/scripts/gnss/gnss_broker/fleetadr.py rebuilds the commanded advance
 *  from slot 1 and the record's hop instead. 0 means no previous record for this PRN: the arc
 *  starts here.
 *
 * config/gnss_record_layout.py parses RECORD_FLOATS, ELEM_FLOATS, CHAN_FLOATS,
 * CUBE_CHAIN_CHARS and CUBE_HEADER_BYTES out of this file by regex: keep them single-line
 * definitions with literal values (CUBE_HEADER_BYTES as `N + CUBE_CHAIN_CHARS`).
 */

#include <cstddef> // for size_t: this header must not rely on an earlier include
#include <cstdint> // for the fixed-width types of the cube frame layout

namespace gnss {

/// float32 slots per PRN per record header.
///
/// Also a yaml constant: every record buffer is sized `n_prn * record_floats * sizeof_float`.
/// The config generator reads it through config/gnss_record_layout.py, and
/// python/scripts/gnss/gnss_broker/telem.py mirrors it, so changing it means regenerating the
/// configs, updating that mirror and bumping TELEM_VERSION (the telemetry row grows). The record
/// producers (GnssGpuRecordAssemble, GnssCoherentCombiner) check the frame size at construction
/// and stop rather than write past a frame sized for an older value.
constexpr int RECORD_FLOATS = 32;
constexpr int RECORD_UTC_SLOT = 9; ///< capture-UTC double aliased at slots 9-10

// Hop order: every hop-indexed GNSS array (the tap output, the synthesis `wave`
// [3*n_job][n_chan][n_hops], and the record windows these rows summarize) stores hop `mh` at
// index `mh`, in time order. It must not be permuted to coalesce stores: path B's N^2
// correlator consumes hops as its time axis, and the pack kernel (wave -> synthetic 4+4 lanes)
// assumes wave index == correlator time sample. Coalescing fixes belong inside a kernel
// (shared-memory exchange), not in the layout. See docs/gnss_gpu_search.md section 11.3.

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
constexpr int REC_CPHASE = 15;    ///< commanded carrier-phase increment, cycles since this
                                  ///< PRN's previous record (0 = arc start). See the file note.
constexpr int REC_PH_RE = 16;     ///< prompt head segment (hops before the code-period boundary);
constexpr int REC_PH_IM = 17;     ///< tail = P - PH. See the file note.
constexpr int REC_PH_ENERGY = 18; ///< head replica energy; / P_energy = boundary fraction f
constexpr int REC_TRIM_INC = 19;  ///< applied carrier-trim phase increment, cycles since this
                                  ///< PRN's previous record (0 = arc start / not written):
                                  ///< PrnCtl::ctrim_hz x the record spacing. The combiner sums
                                  ///< it into get_status trim_cycles, on the arc of adr_cycles.
constexpr int REC_RES_RE = 20;    ///< peel residual prompt correlation: the prompt despread on
constexpr int REC_RES_IM = 21;    ///< the voltage after this PRN's waveform was subtracted.
constexpr int REC_RES_PH_RE = 22; ///< residual prompt head segment (same boundary as REC_PH_*);
constexpr int REC_RES_PH_IM = 23; ///< needed, or the residual's deep integration self-cancels.
                                  ///< All four zero = no peel ran. Energies are the prompt's
                                  ///< (same replica). See the file note.
constexpr int REC_SKY_RE = 24;    ///< Sky-phase-corrected prompt (ElemCal::combine_split): one
constexpr int REC_SKY_IM = 25;    ///< half of the elements derotated by the phase of the other,
                                  ///< disjoint half, then rotated like slots 3/4. Input to the deep
                                  ///< folds. A separate slot because the header prompt's phase
                                  ///< is the ADR observable, and the sky phase is part of what
                                  ///< ADR measures. Zero (absent) when the cal is cold or
                                  ///< elem_sum is off.

/// Replica carrier phase anchor, radians in [0, 2*pi): DespreadJob::ang0 as GnssCudaDespread
/// built it for this record (last_ang0), copied rather than recomputed, so it reports what the
/// kernel used; NaN if no job was built. ang0 depends on the absolute sample index
/// (n0 ~ 1e15), where ~5e-7 Hz of carrier moves it by a radian, far below what the float32
/// @ref REC_DOPPLER resolves, so this slot is how instances compare it. float32 holds it to
/// ~1e-7 rad.
constexpr int REC_ANG0 = 26;

/// The comb's phase reference, radians in [-pi, pi]: the assembler's carrier NCO phase
/// `_phi[p]` at this record, i.e. the rotation `exp(-i*phi0)` applied to slots 3/4 and to every
/// comb column. The NCO's zero is set at each instance's fresh acquisition, so phi0 differs
/// arbitrarily between instances: multiply by `exp(+i*phi0)` to return to the replica's own
/// reference (anchored at absolute sample 0, common to all instances) before combining
/// instances within a record. The rotation stays in the export because it removes the carrier
/// trim and the re-pin phase steps, keeping values continuous from record to record, so do not
/// undo it across records. The /get_spectrum windows and the beam cube publish the same phi0.
constexpr int REC_PHI0 = 28;

/// Bright-satellite projection cost on this record (GnssGpuRecordAssemble::proj_slot_diag): the
/// fraction of this satellite's live element weights lying in the projected subspace, cos^2 in
/// [0, 1] averaged over its covering channels with a basis in force; the projection leaves it
/// (1 - cost)^2 of its prompt power. -1 = not applicable: no basis in force on this record, the
/// satellite is itself a projection source, it is unsteered, or its cal is cold. The broker's
/// fleet DLL and gnss::FleetDll weight the row by 1 - cost.
constexpr int REC_PROJ_COST = 29;

/// Bright-satellite projection rank on this record: the largest k over this instance's channels
/// (GnssGpuRecordAssemble::proj_prepare_record), 0 when no basis is in force. The same value on
/// every PRN row of the record.
constexpr int REC_PROJ_K = 30;

/// Bright-satellite projection sources on this record: the OR over this instance's channels of
/// where its basis directions came from: 1 its own row of the source satellite, 2 a sibling
/// chain's row, 4 the gated probe stack. 0 when no basis is in force. Same value on every row.
constexpr int REC_PROJ_SRC = 31;

/// Phi-cache staleness, Hz: doppler_now - the Doppler this PRN's channelizer filter (Phi) was
/// built at; NaN if no Phi was built. GnssCudaDespread rebuilds a PRN's Phi only when its
/// Doppler moves further than `refresh_hz` (default 100), so this is per-(instance, PRN) state
/// and a candidate for a phase intercept that differs per instance. A spread across instances
/// comparable to refresh_hz is the sign of it.
constexpr int REC_PHI_DDOP = 27;

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
constexpr int CMB_ARC = 15;        ///< phase arc id: ++ on every cycle slip / continuity break
constexpr int CMB_HEAD_FRAC = 16;  ///< window-mean boundary fraction f = <PH_energy>/<P_energy>
                                   ///< (0 or 1: the period boundary sits at a window edge).
                                   ///< Diagnostic only.
constexpr int CMB_PEEL_DEEP = 17;  ///< deep |A| of the peel residual (the CMB_DEEP estimator run on
                                   ///< slots 20-23) -> depth_dB = 20*log10(CMB_DEEP/CMB_PEEL_DEEP)
constexpr int CMB_PEEL_INCOH = 18; ///< incoherent |A| of the peel residual (the CMB_AMP_INCOH
                                   ///< twin). Both zero when no peel ran.
constexpr int CMB_HOP_SLOT = 19;   ///< absolute hop index (int64 aliased at slots 19-20) labelling
                                   ///< the E/P/L window: GnssChanMetadata::sample_seq / fft_len of
                                   ///< its first record (block mode) or newest record (rolling).
                                   ///< -1 = unset. int64 because it is an exact integer key: the
                                   ///< fleet DLL groups instances by equal hop, and the hop
                                   ///< reaches ~1.7e10/day, far past float32's 2^24.
                                   ///< Combiner records only: tracker records use 19-23 for
                                   ///< REC_TRIM_INC / REC_RES_*, and the two flavours never share
                                   ///< a buffer. See docs/CHORD_GNSS_SHARED_DLL.md.
constexpr int CMB_COH_FRAC = 21;   ///< coherence fraction |sum A|/sum|A| of the winning deep
                                   ///< stream (after rate/phase-track derotation). Unlike
                                   ///< deep_snr, which grows with the record count when
                                   ///< phase-limited, it does not depend on how the span is
                                   ///< chopped. exp(-sigma_phi^2/2) for Gaussian phase wander;
                                   ///< 0 = not computed (the non-plain deep paths).
                                   ///< Combiner records only.

// ---------------------------------------------------------------------------------------
// ELEMENT AXIS (CHORD)
// ---------------------------------------------------------------------------------------
/// A single-antenna record carries one set of correlations per PRN. CHORD despreads N antennas
/// against the same M replicas, so its record carries an antenna axis as well.
///
/// Layout. Per PRN: the @ref RECORD_FLOATS header, followed by @c n_elem blocks of
/// @ref ELEM_FLOATS:
///
///     [PRN 0: RECORD_FLOATS header][elem 0: 12][elem 1: 12]...[elem n-1: 12][PRN 1: ...]
///
/// @ref record_stride and @ref elem_offset compute the offsets; @c n_elem == 0 is the
/// single-antenna layout.
///
/// Only per-antenna quantities live in an element block. The code/carrier model (PRN, Doppler,
/// code phase), the commanded NCO increments, the covering-channel count and the replica
/// energies are element-independent: one replica is correlated against every antenna, so the
/// E, P, L and P_HEAD energies are shared and stay in the header. On path B
/// (GnssN2RecordAssemble) the N^2 correlator's antenna x replica tiles fill the element blocks,
/// and the header energies are cudaGnssInject's replica energies, not the quantised
/// replica x replica tiles.
///
/// The header's correlation slots carry the reference element (config @c reference_element, a
/// healthy high-gain feed), not a blind element sum: the carrier loop and ADR are taken from
/// them, so they must be one phase-coherent view, and a coherent sum needs the per-element
/// phases being measured. With config @c elem_sum (default off) the assembler self-calibrates
/// those phases from the satellite (gnssElemCal.hpp: bootstrap MRC, reference-anchored phase,
/// one-element scale) and, once a PRN's cal is warm, writes the calibrated weighted mean
/// instead: the same observable with per-record SNR up ~sqrt(N_healthy).
constexpr int ELEM_FLOATS = 12;

constexpr int ELEM_P_RE = 0; ///< prompt correlation for this antenna
constexpr int ELEM_P_IM = 1;
constexpr int ELEM_E_RE = 2; ///< early / late, same taps as REC_E_*/REC_L_*
constexpr int ELEM_E_IM = 3;
constexpr int ELEM_L_RE = 4;
constexpr int ELEM_L_IM = 5;
constexpr int ELEM_PH_RE = 6;  ///< prompt head segment (hops before the code-period boundary);
constexpr int ELEM_PH_IM = 7;  ///< tail = P - PH, exactly as in the header. Required for deep
                               ///< integration across an overlay/nav sign flip.
constexpr int ELEM_RES_RE = 8; ///< peel residual prompt and its head segment, per antenna
constexpr int ELEM_RES_IM = 9; ///< (as REC_RES_*). Zero when no peel ran.
constexpr int ELEM_RES_PH_RE = 10;
constexpr int ELEM_RES_PH_IM = 11;

// Combiner element blocks use the same ELEM_FLOATS stride, so one geometry serves both record
// types and record_stride() is the only size rule. The slot meanings differ, as they do for the
// header (REC_* vs CMB_*): the tracker's blocks hold per-record correlations, the combiner's
// hold integrated per-antenna amplitudes A_e = G_e/E.
constexpr int CMB_ELEM_AMP_INCOH = 0; ///< sqrt<|A_e|^2>: the beam-map value for this antenna.
                                      ///< Incoherent, so it needs no per-antenna phase model;
                                      ///< this is what a transit traces out.
constexpr int CMB_ELEM_MEAN_RE = 1;   ///< <A_e> coherent mean: carries the per-antenna phase,
constexpr int CMB_ELEM_MEAN_IM = 2;   ///< i.e. the complex gain the array calibration wants.
constexpr int CMB_ELEM_AMP_COH = 3;   ///< |<A_e>|: coherent amplitude. Below the incoherent one
                                      ///< by the amount the phase wandered over the window, so
                                      ///< the ratio is a per-antenna coherence diagnostic.
// 4..11 unused in combiner element blocks.

// ---------------------------------------------------------------------------------------
// PER-CHANNEL BLOCK: THE COMB, UNSUMMED
// ---------------------------------------------------------------------------------------
/// GnssGpuRecordAssemble sums each PRN's covering channels into one complex prompt (slots 3/4),
/// which loses the frequency axis: a delay is a phase ramp across frequency, and with the comb
/// collapsed the broker can only fit a free constant per instance instead of deriving it from
/// the ramp. With config @c chan_export the assembler also emits each channel's E, P and L,
/// appended after the PRN records in the same frame:
///
///     [PRN 0 record][PRN 1 record]...[PRN n-1 record][chan block]
///     chan block = [PRN 0: ch 0..n_chan-1][PRN 1: ...],  CHAN_FLOATS each
///
/// Appended rather than interleaved, so a consumer that indexes by record_stride() and a
/// configured n_prn reads the records unchanged. A consumer that infers n_prn from frame_size
/// would count the comb as records; give it an explicit `n_prn`.
///
/// The values are per record: rotated by the same NCO phase as slots 3/4 (@ref REC_PHI0),
/// combined over elements like the header, and weighted per channel like the record sums.
/// Channels outside the PRN's covering mask are zero. /get_spectrum serves the same prompt
/// values summed over a window; the per-record values are what a cross-record rate fit needs.
constexpr int CHAN_FLOATS = 9;
constexpr int CHAN_RE = 0; ///< per-channel prompt, NCO-derotated + element-combined
constexpr int CHAN_IM = 1;
constexpr int CHAN_ENERGY = 2; ///< that channel's prompt replica energy (the ML weight)
constexpr int CHAN_E_RE = 3;   ///< early, same channel, same rotation, same element combine
constexpr int CHAN_E_IM = 4;
constexpr int CHAN_E_ENERGY = 5;
constexpr int CHAN_L_RE = 6; ///< late
constexpr int CHAN_L_IM = 7;
constexpr int CHAN_L_ENERGY = 8;

// ---------------------------------------------------------------------------------------
// BEAM-CUBE FRAME: one completed window, pushed when the next window opens
// ---------------------------------------------------------------------------------------
/// One frame carries exactly one window of the (subband x element) cube from one instance.
///
/// The address travels with the data: chain, gpu, window index and wstart are in the frame,
/// so the receiver never infers which instance or which window it holds.
///
/// Every sender uses the same frame size: one bufferRecv on one port receives the whole fleet,
/// and bufferRecv closes a connection whose frame size differs from its buffer. The frame is
/// sized for @c max_prn PRN slots and @c max_bins bins; unused bins and slots are zero, and a
/// PRN slot with no records this window has prn = 0.
///
/// Layout (little-endian, offsets in bytes):
///   0   int64   version (== CUBE_VERSION)
///   8   int64   window index (absolute, = floor(wstart / win_samples))
///   16  int64   wstart0, 24 int64 wstart1   -- first/last record's wstart in this window
///   32  int64   dropped_windows, cumulative -- windows lost to a full output buffer since
///                                              start; difference consecutive frames to size a gap
///   40  int64   n_prn, 48 int64 n_bin, 56 int64 n_elem   (actual, <= the max the frame holds)
///   64  double  win_samples, 72 double sample_rate
///   80  int32   gpu, 84 int32 bin_width
///   88  int32   max_prn, 92 int32 max_bins   -- the frame's own size (see below)
///   96  double  utc0: UTC (Unix seconds) of absolute F-engine sample 0, so that
///                utc(wstart) = utc0 + wstart / sample_rate. From the producer's frame0_utc.
///   104 char[CUBE_CHAIN_CHARS]  "<short hostname>/<stage>" (config beam_cube_chain),
///                                 NUL-padded. A stage name is unique only within a node, and
///                                 the receiver has one bufferRecv for the whole fleet, so the
///                                 node has to be in the frame.
///   then, all arrays sized by the max (zero-padded), in order:
///     double  phi0[max_prn]                  -- @ref REC_PHI0 at the window's first record
///     int32   freq_id_lo[max_bins], freq_id_hi[max_bins]
///     int32   prn[max_prn], n_rec[max_prn], n_reanchor[max_prn]
///     float   w[max_prn][max_bins]           -- (record, channel) term count
///     float   energy[max_prn][max_bins]
///     float   coh_re[max_prn][max_bins][n_elem], coh_im[...]   -- sum of rotated A_e
///     float   incoh[max_prn][max_bins][n_elem]                 -- sum of |A_e|^2
///
/// The arrays are laid out by the maxima, not the actual extents, so max_prn and max_bins make
/// an archive of frames self-delimiting without the config that produced it.
///
/// Bump CUBE_VERSION on any layout change; python/scripts/gnss/gnss_cube_read.py keeps the
/// header size of each version it reads. When a layout change alters frame_size, deploy the
/// archiver before the nodes.
///
/// float32 on the wire: the sums are accumulated in double, and 7 digits is ample for a power
/// sum.
///
/// The doubles come first: CUBE_HEADER_BYTES must stay a multiple of 8 (so must
/// CUBE_CHAIN_CHARS), which keeps phi0 8-aligned for every max_prn/max_bins. After the int32
/// arrays its alignment would depend on the maxima, and a misaligned double read through a cast
/// pointer is undefined behaviour that happens to work on x86.
constexpr int64_t CUBE_VERSION = 3;
constexpr int CUBE_CHAIN_CHARS = 48;
constexpr size_t CUBE_UTC0_OFFSET = 96;
constexpr size_t CUBE_CHAIN_OFFSET = 104;
constexpr size_t CUBE_HEADER_BYTES = 104 + CUBE_CHAIN_CHARS;

/// Bytes in a beam-cube frame sized for @c max_prn PRN slots, @c max_bins subband bins and
/// @c n_elem antennas. The one size rule: the stage, the buffer and any reader all call this.
constexpr size_t cube_frame_bytes(int max_prn, int max_bins, int n_elem) {
    return CUBE_HEADER_BYTES + (size_t)max_bins * 2 * sizeof(int32_t) // freq_id lo/hi
           + (size_t)max_prn * 3 * sizeof(int32_t)                    // prn, n_rec, n_reanchor
           + (size_t)max_prn * sizeof(double)                         // phi0
           + (size_t)max_prn * max_bins * 2 * sizeof(float)           // w, energy
           + (size_t)max_prn * max_bins * n_elem * 3 * sizeof(float); // coh_re/im, incoh
}

/// Floats per PRN for a record carrying @c n_elem antennas (@c n_elem 0 => single antenna).
constexpr int record_stride(int n_elem) {
    return RECORD_FLOATS + n_elem * ELEM_FLOATS;
}

/// Float offset of the appended per-channel block (0 channels => no block, frame unchanged).
constexpr size_t chan_block_offset(int n_prn, int n_elem) {
    return (size_t)n_prn * record_stride(n_elem);
}

/// Float offset of PRN @c p, channel @c ch within the frame.
constexpr size_t chan_offset(int p, int ch, int n_prn, int n_chan, int n_elem) {
    return chan_block_offset(n_prn, n_elem) + ((size_t)p * n_chan + ch) * CHAN_FLOATS;
}

/// Total floats in a frame carrying @c n_prn records plus the comb (@c n_chan 0 = no comb).
constexpr size_t frame_floats(int n_prn, int n_elem, int n_chan) {
    return chan_block_offset(n_prn, n_elem) + (size_t)n_prn * n_chan * CHAN_FLOATS;
}

static_assert(frame_floats(4, 2, 0) == chan_block_offset(4, 2),
              "n_chan 0 must reproduce the record-only frame exactly");
static_assert(chan_offset(0, 0, 4, 7, 2) == chan_block_offset(4, 2),
              "the comb starts immediately after the last PRN record");
static_assert(chan_offset(1, 0, 4, 7, 2) - chan_offset(0, 0, 4, 7, 2) == 7 * CHAN_FLOATS,
              "consecutive PRNs are n_chan channels apart");

/// Float offset of element @c e of PRN @c p within the frame.
constexpr int elem_offset(int p, int e, int n_elem) {
    return p * record_stride(n_elem) + RECORD_FLOATS + e * ELEM_FLOATS;
}

/// Float offset of PRN @c p's header within the frame.
constexpr int record_offset(int p, int n_elem) {
    return p * record_stride(n_elem);
}

// Layout invariants, checked at compile time so an edit above cannot silently change the frame
// geometry that the config's `record_floats` has to match.
static_assert(record_stride(0) == RECORD_FLOATS,
              "n_elem == 0 must reproduce the single-antenna layout exactly");
static_assert(elem_offset(0, 0, 4) == RECORD_FLOATS,
              "element 0 must start immediately after the PRN header");
static_assert(elem_offset(0, 1, 4) - elem_offset(0, 0, 4) == ELEM_FLOATS,
              "consecutive elements must be ELEM_FLOATS apart");
static_assert(record_offset(1, 4) == record_stride(4), "consecutive PRNs must be one stride apart");
static_assert(elem_offset(1, 0, 4) == record_stride(4) + RECORD_FLOATS,
              "PRN 1's first element must follow PRN 1's header");
// The last element of PRN p must end exactly where PRN p+1 begins: no gap, no overlap.
static_assert(elem_offset(0, 3, 4) + ELEM_FLOATS == record_offset(1, 4),
              "element blocks must tile the record exactly");

} // namespace gnss
#endif // GNSS_RECORD_HPP
