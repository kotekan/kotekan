#ifndef CUDA_GNSS_DESPREAD_KERNEL_HPP
#define CUDA_GNSS_DESPREAD_KERNEL_HPP

#include <cuda_runtime.h>
#include <stdint.h>

/**
 * GPU GNSS batched despread: fused hop-rate replica generation + complex dot product.
 *
 * Port of gnss::ChannelizedReplicaBank::hoprate_stream (the closed-form per-hop channelized
 * replica from precomputed cumulative PFB filter tables) fused with gnss::channelized_despread
 * (data x conj(replica) MAC + replica energy). One launch despreads one record window for a
 * whole batch of (PRN x {Early,Prompt,Late}) correlator trials over the covering channels --
 * the CHORD-scale bulk compute (dish x PRN x subband), subband-local by construction.
 *
 * The Phi tables are Doppler-bucketed: built on the CPU (ChannelizedReplicaBank::hoprate_filter)
 * only when a sat's Doppler moves > refresh_hz, then uploaded; the kernel applies the EXACT
 * current code phase / carrier per job. Validated against the CPU path by
 * tests/cuda_gnss_despread_test (<=1e-5 relative).
 */
namespace gnss_cuda {

/// Per-(PRN x correlator) trial: everything that varies within a batch. Carries its own
/// (device) Phi table pointers + filter span so ONE launch can mix PRNs from different
/// Doppler buckets -- the G1c cross-PRN batch (one launch per record, all sats).
struct DespreadJob {
    double cp0;           ///< code phase (COMBINED-stream chips) at absolute sample 0 reference
    double cps;           ///< chips per sample incl. code Doppler: eff_chip_rate/fs*(1+sign*f/f_c)
    double wc;            ///< carrier angular rate: 2*pi*(f_offset + doppler)/fs
    int code_offset;      ///< this PRN's offset into the shared code table
    int code_len;         ///< combined-stream code length (chips)
    uint64_t chan_mask;   ///< bit ci set = channel ci is in this PRN's covering set (<=64 chans)
    const double2* phiA;  ///< [n_chan][Lf+1] cumulative filter table, this PRN's Doppler bucket
    const double2* phiB;  ///< (device pointers -- the tables live in the caller's per-PRN cache)
    int n_chips;          ///< chips spanned by this bucket's filter (gather depth per hop)
};

/// Batch-shared geometry.
struct DespreadParams {
    long long n0; ///< absolute sample index of the window's first hop reference (+fft_len-1)
    int fft_len;  ///< samples per hop
    int n_hops;   ///< hops per record (<=256)
    int Lf;       ///< filter length fft_len*num_taps (Phi tables have Lf+1 entries)
};

/**
 * Launch the fused despread.
 * @param data   [n_chan][n_hops] channelized voltage (device)
 * @param code   shared int8 code table (all PRNs concatenated; jobs carry offsets; device)
 * @param jobs   [n_batch] per-trial parameters incl. per-job Phi table pointers (device)
 * @param corr   out [n_batch][n_chan] complex correlations (device)
 * @param energy out [n_batch][n_chan] replica energies (device)
 * Cross-channel summation is left to the caller (tiny; host or follow-up kernel).
 */
cudaError_t launch_despread(const float2* data, const int8_t* code, const DespreadJob* jobs,
                            int n_batch, int n_chan, const DespreadParams& p, double2* corr,
                            double* energy, cudaStream_t stream);

} // namespace gnss_cuda

#endif
