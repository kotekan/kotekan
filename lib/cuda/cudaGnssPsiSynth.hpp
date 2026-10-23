/**
 * @file
 * @brief Prototype: code-phase-table ("Psi") replica synthesis for the CHORD waveform path.
 *
 * The production synthesis (gnss_waveform_kernel, cudaGnssChordDespread.cu) rebuilds every
 * replica sample from the chip sequence: per (PRN, channel, hop) it walks 80-212 chips of the
 * PFB span, reading two prefix-table entries per chip. Here the per-chip sum is tabulated once
 * per (PRN, channel, Doppler bucket) on a grid of G points per combined chip,
 *
 *     Psi[q] = S_B(q/G) = sum_d code[(floor(q/G) - d) mod L] * K_d(frac(q/G)),
 *
 * and the kernel evaluates S_B at an arbitrary code phase C = m + phi from four table taps:
 *
 *     S_B(C) ~= e^{-i theta_B phi inv_cps} sum_{j=-1..2} w_j(phi) e^{+i theta_B j inv_cps} Psi[m+j]
 *
 * (G = 1; with G > 1 the chip phase is C*G and inv_cps/G takes inv_cps's place). The ramp
 * e^{-i theta_B (phi - j) inv_cps} is exact: shifting the chip cells by phi*inv_cps taps under
 * the modulated prototype proto[k] e^{-i theta_B k} pulls that phasor out, leaving the
 * prototype shifted by a sub-chip fraction, and the prototype (bandwidth +-1/fft_len cycles per
 * sample, i.e. +-0.019 cycles per chip at 10.23 Mcps) is ~25x oversampled on a one-chip grid,
 * so a cubic Lagrange interpolation from whole-chip shifts is good to ~1e-5. theta_B must be the
 * (-pi, pi] branch of off_c - wc: any other branch puts a 2 pi inv_cps-cycle-per-chip ramp on the
 * function being interpolated. Only the near carrier image (B, at off_c - wc) is tabulated: the
 * far image (A, at off_c + wc, ~1.7 rad/sample) is a rect-chip-sampled-at-3.2 GS/s artefact that
 * the band-limited sky signal does not contain, and it is 1e-3..4e-2 of B per channel.
 *
 * Output contract, grid shape and job struct are gnss_waveform_kernel's, so the N x M
 * correlator and path B's pack consume the result unchanged. The replica is NOT the production
 * replica: it is the full-span, B-only, fractional-chip-edge model (see build_psi_table), which
 * differs from the 80-chip, two-image, sample-snapped production replica by ~1e-1 per sample
 * (span), with the correlation-level effect of ~1%. Compare at the correlation level, never per
 * sample (cudaGnssPsiBench.cpp).
 *
 * Tiled codes (the *_NH / *_CS descriptors, every production chain but two) are handled by the
 * Psi_plain / Psi_prev / sec[] split of PsiJob: a table over one primary period P plus the
 * lookback into the previous period, exact because the gather never reaches back more than one
 * period (n_chips < P). Plain codes are P = L, Ns = 1.
 */

#ifndef CUDA_GNSS_PSI_SYNTH_HPP
#define CUDA_GNSS_PSI_SYNTH_HPP

#include "cudaGnssDespreadKernel.hpp" // for DespreadJob, DespreadParams

#include <complex>
#include <cuda_runtime.h>
#include <stdint.h>
#include <vector>

namespace gnss {
class ChannelizedReplicaBank;
}

namespace gnss_cuda {

/// Per-job table geometry for @ref launch_psi_waveform. The code phase, carrier, spacing,
/// chan_mask and m_head come from the matching @ref DespreadJob (the production struct, so the
/// host builds one set of jobs for both kernels); only the tables differ.
struct PsiJob {
    /// [n_chan][G*P] float2 (or __half2 when @c half is set): Psi over one primary period, un-
    /// de-ramped (periodic in q). For a tiled code the first period's table divided by its sign.
    const void* psi;
    /// [n_chan][G*(n_chips-1)] or nullptr (plain code): Psi_prev[m'] = sum_{d>floor(m'/G)}
    /// prim[(m'-d) mod P] K_d, the part of the gather that reaches into the previous period.
    const void* psi_prev;
    const int8_t* sec; ///< [Ns] period signs of the tiled code (+-1); nullptr when Ns == 1
    int P;             ///< primary period, combined chips (= code_len for a plain code)
    int Ns;            ///< periods tiled into the code (code_len = Ns * P); 1 = plain
    int G;             ///< grid points per combined chip
    int n_prev;        ///< entries per channel of psi_prev, G*(n_chips-1); 0 when nullptr
    // Fixed-point per-hop phases (launch_psi_waveform's fixed = 1), filled by
    // @ref psi_fixed_setup from the DespreadJob: the code phase in grid units (chips * G) as
    // int64 in 2^-shift units, the carrier in 2^-64 turns.
    int shift = 0;
    long long cg_ref = 0;           ///< cp_ref * G * 2^shift
    long long dcg = 0;              ///< fft_len * cps * G * 2^shift (the per-hop advance)
    long long dsg = 0;              ///< ds * G * 2^shift
    unsigned long long ang0_t = 0;  ///< ang0 / 2pi in 2^-64 turns
    unsigned long long dturn_t = 0; ///< frac(wc * fft_len / 2pi) in 2^-64 turns
    float dcg_r = 0.f;              ///< dcg / (G*L*2^shift): the per-hop wrap-count estimate
    float cg_r = 0.f;               ///< cg_ref / (G*L*2^shift)
};

/**
 * @brief Synthesise E/P/L replicas from Psi tables: the same grid, block, output layout and
 *        energy rows as @ref launch_waveform.
 *
 * @param jobs    [n_job] device DespreadJob (ds, cp_ref, cps, inv_cps, wc, ang0, code_len,
 *                chan_mask, m_head read; the Phi pointers, n_chips and d_first ignored)
 * @param pjobs   [n_job] device PsiJob
 * @param theta   [n_job][n_chan] float: theta_B * inv_cps / G per (job, channel), theta_B
 *                reduced to (-pi, pi] on the host (see @ref psi_theta)
 * @param p       batch geometry (fft_len, n_hops, phi_half selects __half2 tables)
 * @param rot     nullptr: three table lookups per hop (E/P/L at C-ds, C, C+ds, each with its
 *                own energy row); else [n_job][n_chan] radians and E/L are the rotated prompt
 *                with the prompt's energy, exactly as launch_waveform's ROT contract
 * @param fixed   0: per-hop code and carrier phase in fp64 as the production kernel; 1: the
 *                per-hop advances in 64-bit fixed point (PsiJob's shift, 2^-64 turns), no
 *                fp64 in the hop loop
 * @param wave    out [3*n_job][n_chan][n_hops] float2
 * @param energy  out [4*n_job][n_chan] double rows E/P/L/P_HEAD
 */
cudaError_t launch_psi_waveform(const DespreadJob* jobs, const PsiJob* pjobs, const float* theta,
                                int n_job, int n_chan, const DespreadParams& p, const float* rot,
                                int fixed, float2* wave, double* energy, cudaStream_t stream);

/**
 * @brief Fused synthesis + pack: @ref launch_psi_waveform's replica written straight into the
 *        N^2 synthetic lanes, replacing launch_waveform + launch_pack44 and the wave buffer.
 *
 * jobs, pjobs, theta, rot, fixed and p are @ref launch_psi_waveform's (jobs indexed by spec);
 * slot2spec, n_slot, lanes_per_slot, n_chan, chan_map, frame_chan_stride, n_lanes, lane_base,
 * lane_pitch and synth are @ref launch_pack44's. The bytes written are those launch_psi_waveform
 * followed by launch_pack44 would write, for every byte of this record's rows: lane =
 * lanes_per_slot*slot + tl for every lane in [0, n_lanes), at
 * synth[((m*frame_chan_stride + chan_map[c])*lane_pitch + lane_base + lane)]; lanes whose slot
 * is >= n_slot or whose slot2spec is -1, uncovered channels (chan_mask) and the P_HEAD lane at
 * hops >= m_head are 0x88. The quantisation is gnss_pack44_kernel's: s = 7/(3*rms), rms =
 * sqrt(E_row/n_hops) (the P_HEAD lane: sqrt(E_row3/max(m_head,1))), qr = rn(s*w.x),
 * qi = rn(-s*w.y) (the conjugate), clamped to [-7,7], byte ((qr+8)<<4)|((qi+8)&0xf). Under
 * lanes_per_slot 2 lane 0 packs the prompt and lane 1 the prompt gated to hops < m_head with
 * row 3's scale; under 4 the lanes are E, P, L, P_HEAD.
 *
 * Grid: one block per (slot, channel) with the hop lanes of launch_psi_waveform (so the energy
 * rows are reduced in the same order and are bit-identical to that kernel's), plus sweeper
 * blocks that write the lanes beyond the last slot; blocks of inactive slots only write 0x88.
 * Record 0 synthesises its hops into registers (<= 4 hops per thread: more is refused with
 * cudaErrorInvalidValue), reduces the energy, broadcasts the scale through shared memory and
 * quantises; with energy_scale the scale is known up front and each hop is quantised inside
 * the synthesis loop. A hop's lanes_per_slot bytes go out as one 16- or 32-bit store when
 * synth, lane_base and lane_pitch are multiples of lanes_per_slot (byte stores otherwise).
 * Under lanes_per_slot 2 the 512-thread variant runs (PSI_PACK_T512, two blocks per SM; the
 * bytes and energy rows are the same), under 4 the 1024-thread one; see
 * @ref launch_psi_pack_opts. At 12 slots x 7 channels x 2048 hops on an A40: 28.7 us (lanes 2,
 * fp16 tables, fixed phases; 26.6 us for records with energy_scale) against 47.1 us for
 * launch_psi_waveform + launch_pack44 and 122.9 us for launch_waveform + launch_pack44
 * (fixtures/gpu_budget/psi_fused_cx52_20261021.txt).
 *
 * @param energy_scale nullptr: the scale comes from this launch's own energy reduction (record
 *                     0 of a frame). Else [4*n_spec][n_chan] double, the frame's record-0
 *                     energy rows (cudaGnssInject's d_energy0), used for the scale exactly as
 *                     launch_pack44 uses its energy argument, with THIS record's job.m_head
 *                     for the head lane.
 * @param energy       out [4*n_spec][n_chan] double rows E/P/L/P_HEAD as launch_psi_waveform
 *                     writes them (an uncovered channel's rows are 0), for every spec an
 *                     active slot maps to. rot != nullptr: E/L rows = P's (the ROT contract).
 *                     rot == nullptr with lanes_per_slot 4: exact E/L lookups with their own
 *                     energies. rot == nullptr with lanes_per_slot 2: the E/L rows are never
 *                     packed and their exact lookups would be wasted, so the E/L energy rows
 *                     are written EQUAL TO P's (this is the one departure from
 *                     launch_psi_waveform + launch_pack44, which would hold the exact E/L
 *                     energies in rows nobody reads).
 * @param lanes_per_slot 2 (P, P_HEAD) or 4 (E, P, L, P_HEAD); anything else is refused
 */
cudaError_t launch_psi_pack(const DespreadJob* jobs, const PsiJob* pjobs, const float* theta,
                            const float* rot, int fixed, const int* slot2spec, int n_slot,
                            int lanes_per_slot, int n_chan, const DespreadParams& p,
                            const int* chan_map, int frame_chan_stride, int n_lanes, int lane_base,
                            int lane_pitch, const double* energy_scale, double* energy,
                            unsigned char* synth, cudaStream_t stream);

/// Kernel variants of @ref launch_psi_pack_opts (cudaGnssPsiBench's fused bench measures them).
/// @ref launch_psi_pack picks PSI_PACK_T512 under lanes_per_slot 2 and the 1024-thread kernel
/// under 4, where the bench showed each winning.
enum : int {
    /// 512-thread blocks, two per SM, so one block's store drain overlaps another's synthesis;
    /// the 1024-thread kernel's reduction order is emulated, so the energy rows and the bytes
    /// are unchanged. Fixed-point phases and n_hops in [1024, 2048] only; else the 1024-thread
    /// kernel runs.
    PSI_PACK_T512 = 1,
};

/// @ref launch_psi_pack with explicit kernel variants (@c opts: PSI_PACK_*).
cudaError_t launch_psi_pack_opts(const DespreadJob* jobs, const PsiJob* pjobs, const float* theta,
                                 const float* rot, int fixed, const int* slot2spec, int n_slot,
                                 int lanes_per_slot, int n_chan, const DespreadParams& p,
                                 const int* chan_map, int frame_chan_stride, int n_lanes,
                                 int lane_base, int lane_pitch, const double* energy_scale,
                                 double* energy, unsigned char* synth, int opts,
                                 cudaStream_t stream);

/// The primary period of a tiled code for @ref build_psi_split: the smallest divisor P of @c L
/// with P >= n_chips + 1 (the PFB span must not reach back more than one period) such that
/// every period of @c code equals +-1 times the first; a period of zeros does not qualify. L
/// itself for a plain code (every code is a sign tiling of itself), 0 when no divisor
/// qualifies (L <= n_chips, or the code is not periodic at any admissible P).
int psi_primary_period(const int8_t* code, long L, int n_chips);

/// Table grid points per combined chip for a chip rate given as @c inv_cps samples per chip:
/// max(1, lround(inv_cps / 312.8)), i.e. 1 at 10.23 Mcps (312.8 samples per chip at 3.2 GS/s),
/// 2 at 5.115, 10 at 1.023 M/s combined, 20 at 0.5115. Holds the grid spacing near 313
/// samples, where the Lagrange interpolation of the ~25x-oversampled prototype is good to
/// ~1e-5 (see the file comment).
int psi_grid_for(double inv_cps);

/// Fill @c pj's fixed-point fields from @c j for a record of @c n_hops hops: the shift is the
/// largest that keeps the record's code-phase range (in grid units) below 2^62, so the per-hop
/// advance rounds to <= 2^-(shift+1) grid units (2^-46 at L5 geometry, 2^-39 for the 10.4M-chip
/// BDS B2A tiled code), accumulated over n_hops; the carrier advance is exact to 2^-64 turn per
/// hop. Computed in long double as GnssCudaDespread's references are.
void psi_fixed_setup(const DespreadJob& j, PsiJob& pj, int fft_len, int n_hops);

/// Host-side float2 -> __half2 conversion of @c n table entries into @c dst (n * 4 bytes).
void psi_to_half(const float2* src, void* dst, size_t n);

/// theta_B = off_c - wc reduced to (-pi, pi], radians per sample, for channel bin @c chan at
/// carrier rate @c wc (the job's wc, rad/sample). The far image's |theta_A| is returned through
/// @c theta_far when non-null, so a caller can refuse a chain whose far image is the near one
/// (above fs/2) or too strong (|theta_far| * inv_cps < ~50).
double psi_theta(int chan, int fft_len, double wc, double* theta_far = nullptr);

/// One channel set's Psi tables for one PRN at one Doppler, built from the bank.
struct PsiTable {
    int G = 1;
    int L = 0;       ///< combined code length (grid entries per channel = G*L)
    int n_chips = 0; ///< chips of the PFB span (the full span: the bank must have max_chips 0)
    double inv_cps = 0.0;
    double wc = 0.0;            ///< carrier the filter was built at, rad/sample
    std::vector<double> theta;  ///< [n_chan] theta_B reduced to (-pi, pi]
    std::vector<float2> psi;    ///< [n_chan][G*L]
    double build_seconds = 0.0; ///< wall time of the build (the amortised cost)
};

/**
 * @brief Build Psi for PRN slot @c prn_index of @c bank over @c chans at @c doppler_hz.
 *
 * PhiB comes from bank.hoprate_filter (full span: the bank must have max_chips 0, since a
 * centred truncation anchored to floor(C) is not interpolable). The per-chip kernel is
 *     frac = true:  K_d(phi) = Phi_lin((phi+d) inv + 1/2) - Phi_lin((phi+d-1) inv + 1/2),
 *                   Phi_lin the linear interpolation of the double prefix sum at the
 *                   fractional tap index, clamped to [0, Lf]: chip edges at their true
 *                   fractional sample positions, the edge tap weighted by the fraction of it
 *                   inside the chip. The +1/2 makes it the zero-mean-delay smoothing of the
 *                   rect model (whose exclusive edge floor(x)+1 averages to x + 1/2).
 *     frac = false: K_d(phi) = PhiB[t_d] - PhiB[t_{d-1}], t_d = clamp(floor((phi+d) inv)+1,
 *                   0, Lf): the production kernel's own sample-snapped cell boundaries, which
 *                   make S_B(C) a staircase in C (1.8e-4 at the carrier channel, 2e-2 at the
 *                   comb edges per one-sample shift) that no interpolation reproduces.
 * Psi[m*G + g] = sum_{d=0}^{n_chips-1} code[(m-d) mod L] K_d(g/G), accumulated in
 * complex<double> and stored float2. @c sample_rate gives cps exactly as the bank does.
 */
PsiTable build_psi_table(const gnss::ChannelizedReplicaBank& bank, int prn_index,
                         const std::vector<int>& chans, double doppler_hz, double sample_rate,
                         int G, bool frac = true);

/// The tiled-code split of a Psi table (PsiJob::psi, psi_prev, sec): one primary period plus
/// the lookback, for a code of Ns periods that differ only by a sign.
struct PsiSplit {
    int G = 1;
    int P = 0;  ///< primary period, combined chips
    int Ns = 0; ///< periods in the code (code_len = Ns * P)
    int n_chips = 0;
    int n_prev = 0; ///< G * (n_chips - 1) entries per channel of psi_prev
    double inv_cps = 0.0;
    double wc = 0.0;
    std::vector<double> theta;     ///< [n_chan]
    std::vector<float2> psi_plain; ///< [n_chan][G*P], the first period's table
    std::vector<float2> psi_prev;  ///< [n_chan][n_prev]
    std::vector<int8_t> sec;       ///< [Ns] sign of period q relative to the first
    double build_seconds = 0.0;
    /// The double accumulators psi_plain / psi_prev were rounded from, filled only when
    /// @ref build_psi_split_from is asked to keep them (cuda_gnss_psi_bench compares the FFT
    /// and the direct builds here, below float resolution).
    std::vector<std::complex<double>> plain_d, prev_d;
};

/// @ref build_psi_table's tables for a tiled code of primary period @c P: Psi_plain over the
/// first period (which defines sec[0] = +1), Psi_prev[m'] = sum_{d > floor(m'/G)}
/// prim[(m'-d) mod P] K_d, and sec[]. Throws unless every period is a sign of the first and
/// the PFB span is shorter than a period. The kernel reconstructs
/// Psi[q*P + m'] = sec[q] Psi_plain[m'] + (sec[q-1] - sec[q]) Psi_prev[m'] exactly.
/// @ref psi_split_code followed by @ref build_psi_split_from with the FFT method.
PsiSplit build_psi_split(const gnss::ChannelizedReplicaBank& bank, int prn_index,
                         const std::vector<int>& chans, double doppler_hz, double sample_rate,
                         int G, int P, bool frac = true);

/// A slot's code as @ref build_psi_split_from consumes it: the first primary period and the
/// sign of every period against it. This is the only part of a table build that reads the
/// code table, O(L), so a builder thread takes it under the lock that guards the bank's
/// set_prn and builds the rest from the copy while the bank's owner goes on using the bank
/// (hoprate_filter is const and allocates its own buffers).
struct PsiCodeSplit {
    int P = 0;                ///< primary period, combined chips
    int Ns = 0;               ///< periods in the code (L = Ns * P)
    std::vector<int8_t> prim; ///< [P] the first period (+-1, zeros in a TDM stream)
    std::vector<int8_t> sec;  ///< [Ns] sign of period q relative to the first
};

/// Split @c code (length @c L) at primary period @c P. Throws unless P divides L and every
/// period is +-1 times the first (the sign is read at the first non-zero chip).
PsiCodeSplit psi_split_code(const int8_t* code, long L, int P);

/// How @ref build_psi_split_from forms Psi_plain. The lookback table Psi_prev is always the
/// direct sum (G*(n_chips-1) entries per channel, 211 at L5 geometry).
enum class PsiBuildMethod : int {
    /// Psi_plain[m] = sum_d prim[(m-d) mod P] K_d is a circular convolution of period P, done
    /// as IFFT(FFT(prim) . FFT(K zero-padded to P)) in double (FFTW): one transform of the
    /// code per build plus two per (channel, grid point). The production path.
    FFT = 0,
    /// The direct O(G * n_chips * P) sum per channel: the reference the bench holds the FFT
    /// build against (agreement ~1e-13 relative in double; the float2 tables round the same).
    DIRECT = 1,
};

/// @ref build_psi_split from an already split code: reads the bank's geometry and filter
/// (const) and never its code table. @c keep_double fills PsiSplit::plain_d / prev_d.
PsiSplit build_psi_split_from(const gnss::ChannelizedReplicaBank& bank, int prn_index,
                              const std::vector<int>& chans, double doppler_hz, double sample_rate,
                              int G, const PsiCodeSplit& cs, bool frac = true,
                              PsiBuildMethod method = PsiBuildMethod::FFT,
                              bool keep_double = false);

} // namespace gnss_cuda

#endif // CUDA_GNSS_PSI_SYNTH_HPP
