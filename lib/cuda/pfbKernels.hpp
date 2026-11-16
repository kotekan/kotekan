// F-engine PFB kernels: the 4-tap polyphase fold that runs ahead of the window FFT.

#ifndef PFB_KERNELS_HPP
#define PFB_KERNELS_HPP

#include <cstddef>
#include <cstdint>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

/**
 * The device prototype table the fused fold kernel reads: pfb_prototype() computed in
 * double on the host, rounded once to __half on upload. The prototype is symmetric (w[k] ==
 * w[num_taps * fft_length - 1 - k], exactly, in float), so the table holds only its first
 * half, num_taps / 2 * fft_length values; the kernel mirror-reads the upper taps. One
 * allocation is shared process-wide per (gpu_id, fft_length, num_taps): cudaSparkFEngine is a plain
 * Stage, not a cudaProcess command, so the device-global named-memory claims (one owner per
 * region) cannot express the share; this registry can. The caller must have gpu_id current
 * (set_thread_device) and gets reused = true when an earlier caller already uploaded the
 * table. Tables are deliberately never freed: kotekan restarts by process, so they live
 * until exit.
 */
const __half* pfb_proto_shared(int gpu_id, size_t fft_length, int num_taps, bool& reused);

/**
 * 4-tap polyphase fold: out[n] = sum_q proto[q * n_samples + n] * in_q[n] for
 * n = 0 .. n_samples - 1, accumulated in fp32. in0..in3 are the last four windows of
 * unpacked samples, oldest first (in3 is the window being emitted); proto is the
 * pfb_proto_shared() table layout, 4 * n_samples __half on the device, widened to fp32
 * per read so this reference path folds the same weights as the fused fp16 path. The
 * n_samples-point real FFT of out gives the PFB channels under the same spectrum convention
 * as a plain FFT of in3 (fftWindowJob.hpp): a tone on a bin centre keeps its plain-FFT
 * amplitude.
 */
void gpu_pfb_fold(const float* in0, const float* in1, const float* in2, const float* in3,
                  const __half* proto, size_t n_samples, float* out, cudaStream_t stream);

/// The four windows a PFB output folds, as fftWindowJob.hpp lookup tables on the device,
/// oldest first (lut[3] is the window being emitted). A null table is a window of zeros:
/// the missing history right after startup or a realignment.
struct pfb_taps {
    const uint64_t* lut[4];
};

/**
 * Gather + unpack + fold for the fp16 four-step path: as gpu_unpack12_fp16_t
 * (fEngineKernels.hpp), but each sample is the 4-tap polyphase fold of the four windows
 * in taps weighted by proto (the symmetric prototype's taps 0 and 1, 2 * 2 * m1 * m2 __half
 * on the device as pfb_proto_shared stores them; taps 2 and 3 are read mirrored). The
 * weights widen to fp32, the fold accumulates in fp32 and rounds once to fp16. The fold
 * itself is an elementwise multiply-add; the tensor cores are in the FFT that consumes the
 * transposed output.
 */
void gpu_pfb_unpack12_fp16_t(pfb_taps taps, int samples_per_packet, size_t m1, size_t m2,
                             const __half* proto, __half2* out, cudaStream_t stream);

#endif // PFB_KERNELS_HPP
