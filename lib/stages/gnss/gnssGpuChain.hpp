#ifndef GNSS_GPU_CHAIN_HPP
#define GNSS_GPU_CHAIN_HPP

#include <stddef.h>
#include <stdint.h>

/**
 * Shared frame layout for the phase-F GPU tracking chain (docs/gnss_gpu_migration.md §6):
 *
 *   cudaProcess [ cudaInputData -> cudaGnssTrack -> cudaOutputData ] -> GnssGpuRecordAssemble
 *
 * cudaGnssTrack runs the tracker's pass-1 control on the host (seeds -> commanded cp/carrier,
 * currency translation, quadratic code FF, f_ref fence) and the batched E/P/L despread on the
 * device (one launch per record window, read in place from an internal channel-major ring).
 * Everything the downstream host assembler needs to build tracker records -- the per-record
 * control decisions AND the raw per-channel correlations -- travels in ONE output frame with
 * this layout, so the assembler is stateless w.r.t. seeds (only NCO phase continuity lives
 * there) and the combiner/broker/viewer are untouched.
 *
 * Frame layout (all sections 16-byte aligned; offsets are functions of n_prn/n_chan below):
 *   [FrameHdr]
 *   [int64_t window_start   x MAX_REC]                 absolute sample of each record window
 *   [PrnCtl  x MAX_REC x n_prn]                        pass-1 control per (record, PRN slot)
 *   [double2 corr   x MAX_JOBS(n_prn) x n_chan]        kernel output, job-major
 *   [double  energy x MAX_JOBS(n_prn) x n_chan]
 * A "job" is one correlator trial; an active PRN contributes 4 consecutive job ROWS
 * (E, P, L, P_HEAD) starting at PrnCtl::job0. P_HEAD is the prompt restricted to the hops
 * BEFORE the code-period boundary inside the window (gnssRecord.hpp slots 16-18): the
 * secondary overlay flips sign at that boundary, so the combiner needs the two sides
 * separately or straddling records cancel (the 2026-07-15 "bistable").
 *
 * NB a "job row" is an OUTPUT row, no longer a kernel job: since 2026-07-16 the despread kernel
 * takes ONE job per active PRN and emits the whole quad from it (E/P/L share a carrier and a
 * data sample; P_HEAD is the prompt's own MAC gated on the hop). Rows are what everything
 * downstream indexes -- PrnCtl::job0, FrameHdr::n_jobs, max_jobs -- so this layout is unchanged;
 * only the kernel's INPUT array is per-PRN, and it is sized by max_specs.
 */
namespace gnss_gpu {

/// Records per GPU frame, sized for the worst backlog: the ring consumer drains to <1 record
/// each frame, so backlog <= (frame hops + record-1)/record = 11 at 1248-hop frames / 125-hop
/// records, with headroom for other geometries.
constexpr int MAX_REC = 16;

struct FrameHdr {
    int32_t n_rec;   ///< complete record windows despread in this frame (<= MAX_REC)
    int32_t n_prn;   ///< PRN slots (layout stride)
    int32_t n_chan;  ///< channels per job row (layout stride)
    int32_t n_jobs;  ///< total job ROWS written this frame (<= max_jobs(n_prn, n_rows_spec);
                     ///< n_rows_spec per active PRN per record -- see max_specs for the kernel's
                     ///< job count)
    int64_t seq0;    ///< absolute sample of the input frame that triggered this output
    double utc0;     ///< capture UTC of sample 0 (the tracker's capture_utc0 convention)
    int32_t n_rows_spec; ///< OUTPUT rows per active PRN per record: 4 (E, P, L, P_HEAD) normally,
                         ///< 6 when the chain peels -- rows 4/5 are then the PEEL RESIDUAL prompt
                         ///< and its head segment (gnssRecord.hpp slots 20-23). RUNTIME, not a
                         ///< constant, so a chain that does not peel keeps the smaller frame
                         ///< instead of every chain paying for one that does. 0 from an old
                         ///< writer reads as 4.
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

/// Pass-1 control for one (record window, PRN slot): exactly what the tracker's pass-2 needs.
struct PrnCtl {
    uint8_t run;        ///< active this window (seeded + covering channels in this subband)
    uint8_t reanchored; ///< f_ref moved at this window. 0 = no. 1 = FRESH acquisition (no phase
                        ///< history to keep: the assembler resets the NCO and breaks the arc).
                        ///< 2 = CONTINUOUS re-pin (fence or age): the replica phase stepped by
                        ///< df*t_abs, and the assembler folds that step INTO the NCO so the
                        ///< despread output never sees it. See GnssGpuRecordAssemble.
    uint16_t _pad0;
    int32_t job0;       ///< first of this PRN's 4 job rows (E,P,L,P_HEAD); -1 if !run
    float fcar_report;  ///< record slot 1 (physical-signed reported Doppler)
    float n_owned;      ///< record slot 6 (covering channels owned by this subband)
    double cp_seed;     ///< record slot 2: EXPORT-currency code phase, chips (gnssRecord.hpp
                        ///< slot-2 contract: paired with fcar_report). NOT the GPU despread
                        ///< command -- that keeps the f_ref currency and rides the Spec.
    double f_nco;       ///< NCO slope for this record (ctrim + ff, internal convention, Hz)
    uint64_t chan_mask; ///< local covering-channel bits (for the assembler's cross-channel sum)
    double energy_scale;///< reserved (1.0)
    double fcar;        ///< replica carrier f_ref (Hz): the assembler needs it to reconstruct the
                        ///< COMMANDED carrier phase f_ref*t_abs + phi/2pi (record slot 15). NOT
                        ///< derivable from fcar_report, which folds out the re-pin step on purpose.
    uint64_t _pad1;
};
static_assert(sizeof(PrnCtl) == 64, "PrnCtl must stay 16-byte aligned");

/// Output rows (corr/energy) per frame: @c rows_spec per active PRN per record.
constexpr int max_jobs(int n_prn, int rows_spec = ROWS_PLAIN) {
    return rows_spec * n_prn * MAX_REC;
}
/// Kernel jobs: ONE per active PRN per record -- each emits the rows above. Sizes the
/// DespreadJob arena only.
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
/// @param n_elem antennas per correlation (CHORD). 1 = the single-antenna airspy layout, which
///        is what every existing caller gets by default -- the expressions below collapse to
///        the originals term for term.
///
/// The CORRELATION array carries the element axis, [jobs][n_chan][n_elem]; the ENERGY array does
/// NOT, [jobs][n_chan]. One replica is correlated against every antenna, so its energy is
/// element-independent. That asymmetry is the same one the record schema encodes (energies in the
/// per-PRN header, correlations in the element blocks) and the same one path B will hand us for
/// free from the N^2 kernel as N x M versus M x M -- so it is worth keeping visible here rather
/// than padding the energy array to match.
constexpr size_t off_energy(int n_prn, int n_chan, int rows_spec = ROWS_PLAIN, int n_elem = 1) {
    return off_corr(n_prn) + sizeof(double) * 2 * max_jobs(n_prn, rows_spec) * n_chan * n_elem;
}
/// Total frame size -- the yaml epl buffer's frame_size must equal this (asserted at runtime).
constexpr size_t frame_bytes(int n_prn, int n_chan, int rows_spec = ROWS_PLAIN, int n_elem = 1) {
    return off_energy(n_prn, n_chan, rows_spec, n_elem)
           + sizeof(double) * max_jobs(n_prn, rows_spec) * n_chan;
}

} // namespace gnss_gpu

#endif
