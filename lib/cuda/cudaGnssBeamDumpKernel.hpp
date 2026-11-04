/**
 * @file
 * @brief Kernels for cudaGnssBeamDump: a self-calibrated beam per selected channel from the
 *        4+4b voltage frame, and the layout of the frame it writes.
 *
 * Per selected channel and per frame: the covariance of the 128 inputs over the frame's
 * samples (@ref launch_beam_cov), its top eigenvector after whitening each input by its own
 * power (@ref launch_beam_eig: warm-started power iterations, the phase referenced to one
 * input, the weights scaled so the beam reads in that input's units), and the beam itself
 * (@ref launch_beam_form). This is the offline method that recovered the GuoWang signal from
 * the 2026-10-30 baseband dump (docs/CHORD_GUOWANG_BEAMDUMP_PLAN.md), run on the node.
 *
 * Output frame (@ref BeamDumpHdr, @ref beam_dump_layout): the header, then per channel the
 * freq_id, the two largest whitened eigenvalues, the mean power per input, the 128 complex
 * weights, and the beam samples as half2 (float16 re, im) at the hop rate. The samples are
 * sum_i w_i x_i with x decoded as (real = high nibble, imaginary = low nibble, minus 8), i.e.
 * the F-engine's own convention, not conjugated to the sky convention. With targets
 * (cudaGnssBeamDump beam_lane_owner) the frame continues with, per target and channel, the
 * lane the beam was steered by (@ref BeamDumpTarget), its weights and its samples.
 */

#ifndef CUDA_GNSS_BEAM_DUMP_KERNEL_HPP
#define CUDA_GNSS_BEAM_DUMP_KERNEL_HPP

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace gnss_cuda {

/// Inputs per channel (two polarisations x 64 dishes); the kernels are written for this width.
static constexpr int BEAM_INPUTS = 128;

/// Frame header, 64 bytes. seq0 is at byte 0 so FrameWindowGate can read it as its clock.
struct BeamDumpHdr {
    int64_t seq0;      ///< absolute hop (F-engine seq) of sample 0
    int32_t magic;     ///< BEAM_DUMP_MAGIC
    int32_t version;   ///< BEAM_DUMP_VERSION
    int32_t n_chan;    ///< selected channels in this frame
    int32_t n_times;   ///< samples per channel (one frame)
    int32_t n_inputs;  ///< BEAM_INPUTS
    int32_t ref_input; ///< the phase-reference input
    int32_t conj;      ///< 0: samples in the F-engine's (re high, im low) convention
    int32_t eig_iters; ///< power iterations per frame
    int32_t n_targets; ///< tracked-satellite beams per channel (the target sections; 0 = none)
    int32_t reserved[5];
};
static constexpr int32_t BEAM_DUMP_MAGIC = 0x504d4442; // "BDMP" little-endian
/// Version 2 adds n_targets and the target sections; with n_targets 0 the frame is laid out
/// as version 1.
static constexpr int32_t BEAM_DUMP_VERSION = 2;

/// One (target, channel) of the target sections: the lane the beam was steered by.
struct BeamDumpTarget {
    int32_t prn;  ///< the lane's PRN; 0 = no lane this frame (weights and beam are zero)
    int32_t held; ///< 1: the assembler published its held direction (elem_dir_hold_s)
    float coh;    ///< LaneDirection coherence of the row
    float power;  ///< the row's EMA power, despread units
    float held_s; ///< seconds since the row was last coherent (0 = live)
    float age_s;  ///< frame mid-time minus the lane's track epoch: the re-steer interval
    float reserved[2];
};

/// Byte offsets of the frame's sections, each 16-byte aligned.
struct BeamDumpLayout {
    size_t off_freq;  ///< int32 [n_chan]
    size_t off_lam;   ///< float [n_chan][4]: lambda1, lambda2 (whitened units, noise = 1), 0, 0
    size_t off_pw;    ///< float [n_chan][BEAM_INPUTS]: mean |x|^2 per input
    size_t off_w;     ///< float2 [n_chan][BEAM_INPUTS]: beam weights
    size_t off_beam;  ///< half2 [n_chan][n_times]
    size_t off_tgt;   ///< BeamDumpTarget [n_targets][n_chan]
    size_t off_tw;    ///< float2 [n_targets][n_chan][BEAM_INPUTS]: the target beams' weights
    size_t off_tbeam; ///< half2 [n_targets][n_chan][n_times]
    size_t bytes;     ///< whole frame
};

inline size_t beam_dump_align(size_t off) {
    return (off + 15) & ~size_t(15);
}

inline BeamDumpLayout beam_dump_layout(int n_chan, int n_times, int n_targets = 0) {
    BeamDumpLayout L;
    L.off_freq = sizeof(BeamDumpHdr);
    L.off_lam = beam_dump_align(L.off_freq + sizeof(int32_t) * n_chan);
    L.off_pw = beam_dump_align(L.off_lam + sizeof(float) * 4 * n_chan);
    L.off_w = beam_dump_align(L.off_pw + sizeof(float) * BEAM_INPUTS * n_chan);
    L.off_beam = beam_dump_align(L.off_w + sizeof(float) * 2 * BEAM_INPUTS * n_chan);
    L.off_tgt = beam_dump_align(L.off_beam + (size_t)4 * n_chan * n_times);
    L.off_tw = beam_dump_align(L.off_tgt + sizeof(BeamDumpTarget) * (size_t)n_targets * n_chan);
    L.off_tbeam =
        beam_dump_align(L.off_tw + sizeof(float) * 2 * BEAM_INPUTS * (size_t)n_targets * n_chan);
    L.bytes = beam_dump_align(L.off_tbeam + (size_t)4 * n_targets * n_chan * n_times);
    return L;
}

/**
 * Covariance of the inputs per selected channel: cov[c][i][j] = sum_t x_i(t) conj(x_j(t)) over
 * the frame, as raw sums (exact in float: at most n_times * 98).
 * @param in        4+4b frame [n_times][n_freq][BEAM_INPUTS]
 * @param sel       local channel index of each selected channel [n_chan]
 * @param cov       out float2 [n_chan][BEAM_INPUTS][BEAM_INPUTS]
 */
void launch_beam_cov(const unsigned char* in, const int* sel, int n_chan, int n_freq, int n_times,
                     float2* cov, cudaStream_t stream);

/**
 * Whitened top eigenvector and beam weights per selected channel.
 * @param cov       from launch_beam_cov
 * @param n_times   the sum length (power normalisation)
 * @param ref       phase-reference input
 * @param n_iter    power iterations (warm-started from q1/q2)
 * @param q1,q2     persistent state float2 [n_chan][BEAM_INPUTS] (zeros = cold start)
 * @param w         out float2 [n_chan][BEAM_INPUTS]: beam = sum_i w_i x_i, in the reference
 *                  input's units
 * @param pw        out float [n_chan][BEAM_INPUTS]: mean |x|^2 per input
 * @param lam       out float [n_chan][4]
 * @param ref_dev   optional int [n_chan] on the device: per-channel reference input that
 *                  overrides ref (cudaDtvSubtract picks the strongest input this way)
 */
void launch_beam_eig(const float2* cov, int n_chan, int n_times, int ref, int n_iter, float2* q1,
                     float2* q2, float2* w, float* pw, float* lam, cudaStream_t stream,
                     const int* ref_dev = nullptr);

/**
 * The beam samples: out[c][t] = sum_i w[c][i] x_i(t) as half2.
 */
void launch_beam_form(const unsigned char* in, const int* sel, int n_chan, int n_freq, int n_times,
                      const float2* w, void* out, cudaStream_t stream);

} // namespace gnss_cuda

#endif
