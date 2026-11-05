/**
 * @file
 * @brief Kernels for cudaDtvSubtract: a transmitter timestream per selected channel, its lagged
 *        cross-correlations with the inputs, the lagged subtraction and the re-quantisation, and
 *        the layouts of the state it loads and the frame it writes.
 *
 * Model (docs/CHORD_DTV_SUBTRACTION_20261102.md): in a channel taken by one terrestrial
 * transmitter, input i sees x_i(t) = sum_tau g_i(tau) s(t - tau) + n_i(t): one timestream s
 * through a static impulse response per input (the direct path and the reflections, lags
 * -8..+40 hops). The timestream is a beam, s(t) = sum_i w_i x_i(t); g is fitted on the host from
 * R_xs,i(tau) = sum_t x_i(t) conj(s(t - tau)) and R_ss(tau) = sum_t s(t) conj(s(t - tau)) (the
 * normal equations of the least-squares fit); the residual r_i = x_i - g_i * s replaces x_i in
 * the voltage ring, re-quantised to 4+4 bits. A second source is the same on the residual.
 *
 * Per frame the command carries the last DTV_HIST samples of every timestream over (@ref
 * launch_dtv_shift), decodes the selected channels into float and forms source 0's timestream
 * (@ref launch_dtv_decode), then per active source takes R_ss (@ref launch_dtv_rss) and makes one
 * fused pass over the frame (@ref launch_dtv_fused: the lag correlations and the subtraction as
 * complex products on TF32 tensor cores, and the next source's timestream from the residual), and
 * re-quantises the residual in place (@ref launch_dtv_encode). The eigen start of a source's
 * weights uses @ref launch_dtv_cov with cudaGnssBeamDumpKernel's launch_beam_eig.
 */

#ifndef CUDA_DTV_SUBTRACT_KERNEL_HPP
#define CUDA_DTV_SUBTRACT_KERNEL_HPP

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace gnss_cuda {

static constexpr int DTV_INPUTS = 128;
/// Samples of s carried over from the previous frame: the largest positive lag.
static constexpr int DTV_HIST = 64;
static constexpr int DTV_MAX_LAGS = 64;
static constexpr int DTV_MAX_SRC = 4;

/// Frame header of the command's output, 64 bytes; seq0 at byte 0 is FrameWindowGate's clock.
/// Version 1: every channel has the lags lag_min .. lag_min + n_lag - 1. Version 2: each channel
/// has its own lag window (the lags section) and n_lag is the largest, the stride of the lag axis.
struct DtvOutHdr {
    int64_t seq0;     ///< absolute hop of sample 0
    int32_t magic;    ///< DTV_OUT_MAGIC
    int32_t version;  ///< DTV_OUT_VERSION or DTV_OUT_VERSION_LAGS
    int32_t n_chan;   ///< selected channels
    int32_t n_times;  ///< samples per frame
    int32_t n_inputs; ///< DTV_INPUTS
    int32_t n_lag;    ///< lags per source (version 2: the largest window, the stride)
    int32_t lag_min;  ///< lag of index 0 (lag j = lag_min + j, in hops; s(t - lag)); version 2: 0
    int32_t n_src;    ///< sources per channel (slots; see act)
    int32_t reserved[6];
};
static constexpr int32_t DTV_OUT_MAGIC = 0x4f565444; // "DTVO"
static constexpr int32_t DTV_OUT_VERSION = 1;
static constexpr int32_t DTV_OUT_VERSION_LAGS = 2;

/// Byte offsets of the output frame's sections, each 16-byte aligned.
struct DtvOutLayout {
    size_t off_freq;   ///< int32 [n_chan]
    size_t off_lags;   ///< int32 [n_chan][2]: lag_min, n_lag (version 2 only; empty in version 1)
    size_t off_act;    ///< int32 [n_chan][n_src]: 1 = source active (w set), 2 = also subtracting
    size_t off_pw_raw; ///< float [n_chan][DTV_INPUTS]: mean |x|^2 per input before subtraction
    size_t
        off_pw_res; ///< float [n_chan][DTV_INPUTS]: after every active source, before requantising
    size_t off_ps;  ///< float [n_chan][n_src]: mean |s|^2
    size_t off_w;   ///< float2 [n_chan][n_src][DTV_INPUTS]: the weights in use
    size_t off_rss; ///< float2 [n_chan][n_src][n_lag]: sum_t s(t) conj(s(t - d)), d = 0..n_lag-1
    size_t off_rxs; ///< float2 [n_chan][n_src][n_lag][DTV_INPUTS]: sum_t x_i(t) conj(s(t - lag))
    size_t bytes;
};
// A channel's rss and rxs beyond its own window are zero.

inline size_t dtv_align(size_t off) {
    return (off + 15) & ~size_t(15);
}

inline DtvOutLayout dtv_out_layout(int n_chan, int n_src, int n_lag, bool lags_section = false) {
    DtvOutLayout L;
    L.off_freq = sizeof(DtvOutHdr);
    L.off_lags = dtv_align(L.off_freq + sizeof(int32_t) * n_chan);
    L.off_act = lags_section ? dtv_align(L.off_lags + sizeof(int32_t) * 2 * n_chan) : L.off_lags;
    L.off_pw_raw = dtv_align(L.off_act + sizeof(int32_t) * n_chan * n_src);
    L.off_pw_res = dtv_align(L.off_pw_raw + sizeof(float) * DTV_INPUTS * n_chan);
    L.off_ps = dtv_align(L.off_pw_res + sizeof(float) * DTV_INPUTS * n_chan);
    L.off_w = dtv_align(L.off_ps + sizeof(float) * n_chan * n_src);
    L.off_rss = dtv_align(L.off_w + sizeof(float) * 2 * DTV_INPUTS * n_chan * n_src);
    L.off_rxs = dtv_align(L.off_rss + sizeof(float) * 2 * n_lag * n_chan * n_src);
    L.bytes = dtv_align(L.off_rxs + sizeof(float) * 2 * n_lag * DTV_INPUTS * n_chan * n_src);
    return L;
}

/// State file (scripts/gnss/dtv_state.py writes it): the header, then n_chan records, each a
/// record header followed by float2 w[n_src][DTV_INPUTS] and float2 g[n_src][n_lag][DTV_INPUTS].
/// Version 1: DtvStateRec, and every record has the header's lags. Version 2: DtvStateRecLags,
/// and each record has its own lag window (g has that record's n_lag lags); the header's n_lag
/// and lag_min are 0.
struct DtvStateHdr {
    int32_t magic;   ///< DTV_STATE_MAGIC
    int32_t version; ///< DTV_STATE_VERSION or DTV_STATE_VERSION_LAGS
    int32_t n_chan;
    int32_t n_inputs;
    int32_t n_lag;
    int32_t lag_min;
    int32_t n_src;
    int32_t reserved;
};
struct DtvStateRec {
    int32_t freq_id;
    int32_t n_src_set; ///< sources with weights in this record (the rest are inactive)
    int32_t subtract;  ///< 0: observe only (g ignored), 1: subtract
    int32_t reserved;
};
struct DtvStateRecLags {
    int32_t freq_id;
    int32_t n_src_set;
    int32_t subtract;
    int32_t lag_min;
    int32_t n_lag;
    int32_t reserved[3];
};
static constexpr int32_t DTV_STATE_MAGIC = 0x53565444; // "DTVS"
static constexpr int32_t DTV_STATE_VERSION = 1;
static constexpr int32_t DTV_STATE_VERSION_LAGS = 2;

/// The fused pass's g stride: n_lag rounded up to 16 (a whole number of 16 x 16 x 8 TF32 tiles).
inline int dtv_l16(int n_lag) {
    return (n_lag + 15) & ~15;
}

// The decoded channels x are float [n_chan][2][n_times][DTV_INPUTS], planar: each channel's real
// parts, then its imaginary parts.

/// Carry the last DTV_HIST samples of s over: s[c][k][0..DTV_HIST) = s[c][k][n_times..).
/// s is float2 [n_chan][n_src][DTV_HIST + n_times]. Before the frame's decode.
void launch_dtv_shift(float2* s, int n_chan, int n_src, int n_times, cudaStream_t stream);

/// The selected channels of a 4+4b frame into x, sum |x|^2 over the frame per input (pw_raw,
/// zeroed here), and the timestream of source 0 where it is active (s[c][0][DTV_HIST + t] = sum_i
/// w[c][0][i] x[c][t][i], ps[c][0] += sum |s|^2; the caller zeroes ps). in is [n_times][n_freq]
/// [DTV_INPUTS] bytes; sel the local channels; act int [n_chan][n_src].
void launch_dtv_decode(const unsigned char* in, const int* sel, const float2* w, const int* act,
                       int n_chan, int n_src, int n_freq, int n_times, float* x, float* pw_raw,
                       float2* s, float* ps, cudaStream_t stream);

/// s[c][k][DTV_HIST + t] = sum_i w[c][k][i] x[c][t][i] for the active (c, k); ps[c][k] += sum
/// |s|^2 over the frame (the caller zeroes ps). The decode and the fused pass form the
/// timestreams; this is for a source whose weights change mid-frame (an eigen start's end).
void launch_dtv_beam(const float* x, const float2* w, const int* act, int k, int n_chan, int n_src,
                     int n_times, float2* s, float* ps, cudaStream_t stream);

/// rss[c][k][d] += sum_t s(t) conj(s(t - d)) for d < n_lag_c, the lag differences of the normal
/// equations (the caller zeroes rss each frame). lags is int [n_chan][2]: lag_min_c, n_lag_c
/// (n_lag_c <= n_lag, the stride of the lag axis; -16 <= lag_min_c, lag_min_c + n_lag_c <=
/// DTV_HIST).
void launch_dtv_rss(const float2* s, const int* act, const int* lags, int k, int n_chan, int n_src,
                    int n_times, int n_lag, float2* rss, cudaStream_t stream);

/// One pass of source k over each channel's frame, in place, on TF32 tensor cores: with S[t][j]
/// = s[c][k][DTV_HIST + t - lag_j] (lag_j = lag_min_c + j, j < n_lag_c; 0 past the frame's end),
/// rxs[c][k][j][i] += sum_t x[c][t][i] conj(S[t][j]) where source k is active (act >= 1), then
/// x[c][t][i] -= sum_j g[c][k][j][i] S[t][j] where it subtracts (act == 2), then, where source k
/// + 1 is active, its timestream from the residual with its power (s[c][k+1], ps[c][k+1] +=, as
/// launch_dtv_beam). x is read and written once. gs is g de-interleaved: float [n_chan][n_src][2
/// (re, im)][dtv_l16(n_lag)][DTV_INPUTS]. rxs is accumulated (the caller zeroes it each frame).
/// A channel's work scales with its own window.
void launch_dtv_fused(float* x, float2* s, const float* gs, const float2* w, const int* act,
                      const int* lags, int k, int n_chan, int n_src, int n_times, int n_lag,
                      float2* rxs, float* ps, cudaStream_t stream);

/// Re-quantise x into the frame in place (round to the nearest integer, clipped to [-7, 7],
/// offset-encoded as the F-engine does) and pw_res[c][i] = sum |x|^2 over the frame before the
/// rounding (zeroed here). Channels with no subtracting source are left untouched.
void launch_dtv_encode(const float* x, const int* sel, const int* act, int n_chan, int n_src,
                       int n_freq, int n_times, unsigned char* out, float* pw_res,
                       cudaStream_t stream);

/// Per-source decay factors of the rolling fit's sums (dtvRollingFit.hpp).
struct DtvDecay {
    float lam[DTV_MAX_SRC];
};

/// The rolling fit's sums after a frame's correlations (rxs, rss, pw_raw and pw_res as the
/// output frame holds them): acc_rxs[c][k] = lam[k] acc_rxs[c][k] + rxs[c][k] and acc_rss
/// likewise; per channel, summed over the inputs, acc_pw[0][c] = lam[0] acc_pw[0][c] + pw_raw,
/// acc_pw[1][c] += pw_raw and acc_pw[2][c] += pw_res (the caller zeroes [1] and [2] after each
/// snapshot). acc_pw is float [3][n_chan].
void launch_dtv_roll_accum(const float2* rxs, const float2* rss, const float* pw_raw,
                           const float* pw_res, int n_chan, int n_src, int n_lag, DtvDecay lam,
                           float2* acc_rxs, float2* acc_rss, float* acc_pw, cudaStream_t stream);

/// cov[c][i][j] += sum_t x[c][t][i] conj(x[c][t][j]) (accumulated across frames for the eigen
/// start; the caller zeroes cov). Lag 0 only.
void launch_dtv_cov(const float* x, int n_chan, int n_times, float2* cov, cudaStream_t stream);

/// ref[c] = the input with the largest cov[c][i][i]: the eigen start's phase reference.
void launch_dtv_pick_ref(const float2* cov, int n_chan, int* ref, cudaStream_t stream);

} // namespace gnss_cuda

#endif
