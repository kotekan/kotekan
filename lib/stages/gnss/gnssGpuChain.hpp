#ifndef GNSS_GPU_CHAIN_HPP
#define GNSS_GPU_CHAIN_HPP

#include <stddef.h>
#include <stdint.h>

/**
 * Frame layout of the GNSS GPU chain: cudaGnssInject writes the control block (this layout
 * without corr), GnssN2RecordAssemble the full frame, and GnssGpuRecordAssemble reads it.
 *
 * One frame carries everything the host assembler needs to build tracker records, the
 * per-record control decisions and the per-channel correlations, so the assembler keeps no
 * seed state of its own.
 *
 * Frame layout (all sections 16-byte aligned; offsets are functions of n_prn, n_chan, n_elem):
 *   [FrameHdr]
 *   [int64_t window_start   x MAX_REC]                 absolute sample of each record window
 *   [PrnCtl  x MAX_REC x n_prn]                        control per (record, PRN slot)
 *   [double2 corr   x max_jobs x n_chan x n_elem]      row-major
 *   [double  energy x max_jobs x n_chan]
 * A job row is one output row; an active PRN contributes n_rows_spec consecutive rows
 * (E, P, L, P_HEAD) starting at PrnCtl::job0. P_HEAD is the prompt restricted to the hops
 * before the code-period boundary inside the window (gnssRecord.hpp slots 16-18): the
 * secondary overlay or nav symbol can flip sign at that boundary, so the combiner needs the
 * two sides separately or a straddling record cancels.
 *
 * Rows are not kernel jobs: the GPU takes one job per active PRN per record (max_specs sizes
 * that array) and produces all of its rows. Everything downstream indexes rows (PrnCtl::job0,
 * FrameHdr::n_jobs, max_jobs).
 *
 * config/gen_chord_gnss_config.py (buffer sizes), scripts/gnss/viscap_read.py and
 * scripts/gnss/cube_e2e.py restate this layout; the generator reads sizeof(PrnCtl) through
 * config/gnss_record_layout.py, which parses the static_assert below.
 */
namespace gnss_gpu {

/// Maximum records per frame. A frame holds samples_per_data_set / hops_per_record records
/// (4 for CHORD's 8192-hop frames and 2048-hop records); cudaGnssChordTrackState rejects a
/// geometry that exceeds this.
constexpr int MAX_REC = 16;

struct FrameHdr {
    int32_t n_rec;       ///< record windows in this frame (<= MAX_REC)
    int32_t n_prn;       ///< PRN slots (layout stride)
    int32_t n_chan;      ///< channels per job row (layout stride)
    int32_t n_jobs;      ///< job rows written this frame: n_rows_spec per active PRN per record
                         ///< (<= max_jobs(n_prn, n_rows_spec))
    int64_t seq0;        ///< absolute sample index of the frame's first hop
    double utc0;         ///< capture UTC of sample 0 (the tracker's capture_utc0 convention)
    int32_t n_rows_spec; ///< output rows per active PRN per record: ROWS_PLAIN or ROWS_PEEL
                         ///< (rows 4/5 the peel residual prompt and its head segment,
                         ///< gnssRecord.hpp slots 20-23). The reader takes 0 as ROWS_PLAIN.
    int32_t _pad0;
    int64_t _pad1;
};
static_assert(sizeof(FrameHdr) == 48, "FrameHdr must stay 16-byte aligned");

/// Output rows per active PRN per record, with and without the peel.
constexpr int ROWS_PLAIN = 4; ///< E, P, L, P_HEAD
constexpr int ROWS_PEEL = 6;  ///< ... + residual P, residual P_HEAD
/// Row indices within a spec's block (add to PrnCtl::job0).
constexpr int ROW_E = 0;
constexpr int ROW_P = 1;
constexpr int ROW_L = 2;
constexpr int ROW_PH = 3;
constexpr int ROW_RES_P = 4;  ///< peel only
constexpr int ROW_RES_PH = 5; ///< peel only

/// Control for one (record window, PRN slot), as the producer ran it.
struct PrnCtl {
    uint8_t run;        ///< 1 = despread this window (the slot has a live seed)
    uint8_t reanchored; ///< f_ref moved at this window. 0 = no. 1 = fresh acquisition: no phase
                        ///< history to keep, so the assembler resets the NCO and breaks the arc.
                        ///< 3 = continuous re-pin: the replica phase stepped by @ref dcyc
                        ///< cycles (see there for why the producer computes it), which the
                        ///< assembler folds into the NCO. cudaGnssInject writes 1 and 3.
    /// The PRN in this slot for this record, stamped for every slot, running or not. Carrying
    /// it in the frame leaves one slot -> PRN map, the producer's, so a live slot swap
    /// (docs/CHORD_LIVE_PRN_RECONFIG.md) cannot mislabel records. uint16 fills the padding,
    /// keeping sizeof(PrnCtl) at 80.
    uint16_t prn;
    int32_t job0;       ///< first of this PRN's rows (E, P, L, P_HEAD); meaningless if !run
    float fcar_report;  ///< record slot 1 (physical-signed reported Doppler)
    float n_owned;      ///< record slot 6 (covering channels owned by this subband)
    double cp_seed;     ///< record slot 2: the code-phase argument handed to the despread this
                        ///< record, chips; meaningful only with the Doppler in fcar_report
                        ///< (gnssRecord.hpp slot-2 contract)
    double f_nco;       ///< NCO slope for this record, Hz, internal convention (cudaGnssInject
                        ///< sets the applied carrier trim)
    uint64_t chan_mask; ///< local covering-channel bits (for the assembler's cross-channel sum)
    /// Applied carrier trim over this record, Hz (the broker's carrier_trim_hz as the producer
    /// ran it); the assembler integrates it into record slot 19 (REC_TRIM_INC). Read it from
    /// here, not from (f_nco + fcar_report - fcar)/2: with cudaGnssInject's f_nco = ctrim and
    /// fcar = f_offset + fcar_report, that gives (ctrim - f_offset)/2.
    double ctrim_hz;
    /// Replica carrier phase anchor the kernel used this record, radians; copied into record
    /// slot gnss::REC_ANG0 (see there). NaN = no job was built; 0 is a legal value.
    double ang0;
    /// Phi-cache staleness, Hz (record slot gnss::REC_PHI_DDOP). NaN if no Phi was built.
    double phi_ddop;
    double fcar; ///< replica carrier f_ref (Hz): the assembler needs it to reconstruct the
                 ///< commanded carrier phase, f_ref*t_abs minus the NCO phase (record slot
                 ///< 15). Not derivable from fcar_report, which leaves out the carrier offset.
    /// Re-pin phase step for this record, cycles (reanchored == 3 only; 0 otherwise).
    ///
    /// The replica's carrier phase is anchored absolutely, 2*pi*(fcar + ctrim_hz)*t_abs, so a
    /// change in the applied carrier (Doppler plus trim) since the previous record steps it by
    /// d(carrier)*t_abs: many cycles at days of uptime, with a uniformly random fractional part
    /// if left unfolded.
    ///
    /// The producer computes it because only it can do so exactly. fcar includes f_offset, the
    /// sky carrier (1.176 GHz on E5a), and differencing two doubles that large loses an ulp of
    /// 2.4e-7 Hz, ~0.07 cycles (0.4 rad) after a few days. f_offset is constant, so the
    /// producer differences the applied Doppler and trim (~1e3 Hz, ulp 2e-13) instead.
    double dcyc;
};
/// The literal is deliberate: adding a field must change it, and config/gnss_record_layout.py
/// parses it to size the generated buffers. The layout needs a multiple of 16.
static_assert(sizeof(PrnCtl) == 80, "PrnCtl must stay 16-byte aligned");
static_assert(sizeof(PrnCtl) % 16 == 0, "PrnCtl must stay 16-byte aligned");

/// Maximum output rows (corr/energy) per frame: @c rows_spec per PRN per record.
constexpr int max_jobs(int n_prn, int rows_spec) {
    return rows_spec * n_prn * MAX_REC;
}
/// Maximum kernel jobs per frame: one per PRN per record, each producing @c rows_spec rows.
/// Sizes the DespreadJob arena.
constexpr int max_specs(int n_prn) {
    return n_prn * MAX_REC;
}
constexpr size_t off_winstart() {
    return sizeof(FrameHdr);
}
constexpr size_t off_prnctl() {
    return off_winstart() + sizeof(int64_t) * MAX_REC;
}
constexpr size_t off_corr(int n_prn) {
    return off_prnctl() + sizeof(PrnCtl) * MAX_REC * n_prn;
}
/// @param n_elem antennas per correlation row.
///
/// The correlation array carries the element axis, [jobs][n_chan][n_elem]; the energy array
/// does not, [jobs][n_chan]. One replica is correlated against every antenna, so its energy is
/// element-independent, as in the record schema (energies in the per-PRN header, correlations
/// in the element blocks).
constexpr size_t off_energy(int n_prn, int n_chan, int rows_spec, int n_elem) {
    return off_corr(n_prn) + sizeof(double) * 2 * max_jobs(n_prn, rows_spec) * n_chan * n_elem;
}
/// Total frame size; GnssN2RecordAssemble requires its out_buf frame_size to be at least this.
constexpr size_t frame_bytes(int n_prn, int n_chan, int rows_spec, int n_elem) {
    return off_energy(n_prn, n_chan, rows_spec, n_elem)
           + sizeof(double) * max_jobs(n_prn, rows_spec) * n_chan;
}

} // namespace gnss_gpu

#endif
